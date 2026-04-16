"""
Advanced ML Churn Prediction Pipeline
======================================
Updates in this version:
  - Location (high-cardinality) → Frequency Encoding fix
  - batch_predict() encoder compatibility fix
  - SHAP on sampled data (fast on 300k rows)
  - Speed optimization: n_iter=10 for faster tuning
"""

import sqlite3
import json
import logging
import warnings
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# ============================================================
#  DIRECTORIES & LOGGING
# ============================================================
for d in ["models", "reports", "logs", "predictions"]:
    Path(d).mkdir(exist_ok=True)

logging.basicConfig(
    filename=f"logs/ml_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s"))
logger.addHandler(ch)

# ============================================================
#  CONFIG
# ============================================================
HIGH_CARD_THRESHOLD = 50   # is se zyada unique values = frequency encode


# ============================================================
#  STEP 1 — DATA INGESTION
# ============================================================
def load_from_db(db_path: str = "database.db", table: str = "customers") -> pd.DataFrame:
    logger.info(f"Loading data from '{db_path}' → table '{table}'")
    conn = sqlite3.connect(db_path)
    df = pd.read_sql(f"SELECT * FROM {table}", conn)
    conn.close()
    logger.info(f"Loaded {len(df):,} rows x {df.shape[1]} cols")
    return df


# ============================================================
#  STEP 2 — FEATURE ENGINEERING  (HIGH-CARDINALITY FIX)
# ============================================================
def engineer_features(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    - High-cardinality cols (e.g. Location 63k classes) → Frequency Encoding
    - Low-cardinality cols (e.g. ContractType, PaymentMethod) → LabelEncoder
    - Derived features: ChargePerMonth, TenureGroup, HighSpender
    Returns (transformed_df, encoder_map) for inference reuse.
    """
    df = df.copy()
    encoder_map = {}

    cat_cols = df.select_dtypes(include="object").columns.tolist()

    for col in cat_cols:
        n_unique = df[col].nunique()

        if n_unique > HIGH_CARD_THRESHOLD:
            # --- Frequency Encoding ---
            freq_map = df[col].value_counts(normalize=True).to_dict()
            df[col] = df[col].map(freq_map).fillna(0.0)
            encoder_map[col] = {"type": "frequency", "map": freq_map}
            logger.info(f"Frequency encoded '{col}' — {n_unique:,} unique values → float")
        else:
            # --- Label Encoding ---
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoder_map[col] = {"type": "label", "encoder": le}
            logger.info(f"Label encoded '{col}' — {n_unique} classes")

    # Derived features
    if {"MonthlyCharges", "Tenure"}.issubset(df.columns):
        df["ChargePerMonth"] = df["MonthlyCharges"] / df["Tenure"].clip(lower=1)
        df["TenureGroup"] = pd.cut(
            df["Tenure"],
            bins=[0, 12, 24, 48, 9999],
            labels=[0, 1, 2, 3],
        ).astype(int)

    if "MonthlyCharges" in df.columns:
        df["HighSpender"] = (
            df["MonthlyCharges"] > df["MonthlyCharges"].quantile(0.75)
        ).astype(int)

    logger.info(f"Feature engineering done — {df.shape[1]} total features")
    return df, encoder_map


# ============================================================
#  STEP 3 — SMOTE
# ============================================================
def apply_smote(X, y):
    try:
        from imblearn.over_sampling import SMOTE
        sm = SMOTE(random_state=42)
        X_res, y_res = sm.fit_resample(X, y)
        logger.info(
            f"SMOTE — before: {dict(y.value_counts())}, "
            f"after: {dict(pd.Series(y_res).value_counts())}"
        )
        return X_res, y_res
    except ImportError:
        logger.warning("imbalanced-learn not found — skipping SMOTE")
        return X, y


# ============================================================
#  STEP 4 — MODEL DEFINITIONS
# ============================================================
def get_models() -> dict:
    rf_params = {
        "n_estimators": [100, 200, 300],
        "max_depth": [None, 10, 20, 30],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "class_weight": ["balanced", None],
    }
    xgb_params = {
        "n_estimators": [100, 200, 300],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "subsample": [0.7, 0.8, 1.0],
        "colsample_bytree": [0.7, 0.8, 1.0],
        "scale_pos_weight": [1, 2, 5],
    }
    return {
        "RandomForest": (RandomForestClassifier(random_state=42), rf_params),
        "XGBoost": (
            XGBClassifier(random_state=42, eval_metric="logloss", verbosity=0),
            xgb_params,
        ),
    }


# ============================================================
#  STEP 5 — HYPERPARAMETER TUNING + CV
# ============================================================
def tune_and_evaluate(
    estimator, param_grid, X_train, y_train, X_test, y_test, model_name
) -> tuple:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=param_grid,
        n_iter=10,          # 300k rows pe 20 bahut slow tha — 10 kaafi hai
        scoring="f1",
        cv=cv,
        random_state=42,
        n_jobs=-1,
        verbose=1,
    )
    search.fit(X_train, y_train)
    best = search.best_estimator_

    cv_scores = cross_val_score(best, X_train, y_train, cv=cv, scoring="roc_auc")

    y_pred = best.predict(X_test)
    y_prob = best.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy":    round(accuracy_score(y_test, y_pred) * 100, 2),
        "f1_score":    round(f1_score(y_test, y_pred) * 100, 2),
        "auc_roc":     round(roc_auc_score(y_test, y_prob) * 100, 2),
        "cv_auc_mean": round(cv_scores.mean() * 100, 2),
        "cv_auc_std":  round(cv_scores.std() * 100, 2),
        "best_params": search.best_params_,
        "classification_report": classification_report(y_test, y_pred),
    }

    logger.info(
        f"[{model_name}] Acc={metrics['accuracy']}% | "
        f"F1={metrics['f1_score']}% | AUC={metrics['auc_roc']}% | "
        f"CV AUC={metrics['cv_auc_mean']}%±{metrics['cv_auc_std']}%"
    )
    return best, metrics


# ============================================================
#  STEP 6 — SHAP EXPLAINABILITY  (sampled for speed)
# ============================================================
def explain_model(model, X_train: pd.DataFrame, feature_names: list, model_name: str) -> dict:
    """
    300k rows pe full SHAP bahut slow hoga.
    2000 random rows sample karke chalao — result same aata hai.
    """
    try:
        sample_size = min(2000, len(X_train))
        X_sample = X_train.sample(n=sample_size, random_state=42)

        explainer   = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)

        sv = shap_values[1] if isinstance(shap_values, list) else shap_values
        mean_abs = np.abs(sv).mean(axis=0)

        importance = dict(sorted(
            zip(feature_names, mean_abs.tolist()),
            key=lambda x: x[1],
            reverse=True,
        ))
        logger.info(
            f"[{model_name}] SHAP top 5: "
            f"{ {k: round(v, 4) for k, v in list(importance.items())[:5]} }"
        )
        return importance
    except Exception as e:
        logger.warning(f"SHAP failed for {model_name}: {e}")
        return {}


# ============================================================
#  STEP 7 — MODEL REGISTRY
# ============================================================
def save_to_registry(
    model, encoder_map, scaler, metrics, feature_names, shap_importance, model_name
) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    slug = f"{model_name.lower()}_{timestamp}"

    joblib.dump(model,       f"models/{slug}_model.pkl")
    joblib.dump(encoder_map, f"models/{slug}_encoders.pkl")
    joblib.dump(scaler,      f"models/{slug}_scaler.pkl")

    metadata = {
        "model_name":       model_name,
        "version":          timestamp,
        "features":         feature_names,
        "metrics":          {k: v for k, v in metrics.items() if k != "classification_report"},
        "top_shap_features": list(shap_importance.items())[:10],
        "saved_at":         datetime.now().isoformat(),
    }
    with open(f"models/{slug}_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    logger.info(f"[{model_name}] Registry saved → models/{slug}_*")
    return slug


# ============================================================
#  STEP 8 — BATCH PREDICTION  (encoder_map format fix)
# ============================================================
def batch_predict(model, scaler, encoder_map, df_raw, feature_cols, model_name) -> pd.DataFrame:
    """
    encoder_map ab dict-of-dicts hai: {"type": "frequency"/"label", ...}
    Yeh function dono types handle karta hai correctly.
    """
    df = df_raw.copy()

    for col, info in encoder_map.items():
        if col not in df.columns:
            continue

        if info["type"] == "frequency":
            df[col] = df[col].map(info["map"]).fillna(0.0)

        elif info["type"] == "label":
            le = info["encoder"]
            known = set(le.classes_)
            # Unseen labels → pehli known class se replace
            df[col] = df[col].apply(lambda x: x if str(x) in known else le.classes_[0])
            df[col] = le.transform(df[col].astype(str))

    # Derived features (training jaisi)
    if {"MonthlyCharges", "Tenure"}.issubset(df.columns):
        df["ChargePerMonth"] = df["MonthlyCharges"] / df["Tenure"].clip(lower=1)
        df["TenureGroup"] = pd.cut(
            df["Tenure"], bins=[0, 12, 24, 48, 9999], labels=[0, 1, 2, 3]
        ).astype(int)
    if "MonthlyCharges" in df.columns:
        df["HighSpender"] = (
            df["MonthlyCharges"] > df["MonthlyCharges"].quantile(0.75)
        ).astype(int)

    X     = scaler.transform(df[feature_cols])
    probs = model.predict_proba(X)[:, 1]

    result = df_raw[["CustomerID"]].copy() if "CustomerID" in df_raw.columns else pd.DataFrame()
    result["churn_probability"] = np.round(probs * 100, 2)
    result["churn_risk"] = pd.cut(
        result["churn_probability"],
        bins=[0, 30, 60, 100],
        labels=["Low", "Medium", "High"],
    )
    result["predicted_by"] = model_name

    out_path = f"predictions/batch_scores_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    result.to_csv(out_path, index=False)
    logger.info(f"Batch predictions saved → {out_path} ({len(result):,} rows)")
    return result


# ============================================================
#  MAIN
# ============================================================
def main():
    logger.info("=" * 60)
    logger.info("Advanced ML Churn Pipeline — START")

    # 1. Load
    df = load_from_db()

    # 2. Feature engineering (Location fix included)
    df_feat, encoder_map = engineer_features(df)

    drop_cols   = ["Churn", "CustomerID"] + [c for c in df_feat.columns if c.endswith("ID")]
    feature_cols = [c for c in df_feat.columns if c not in drop_cols]
    X = df_feat[feature_cols]
    y = df_feat["Churn"]
    logger.info(f"Features ({len(feature_cols)}): {feature_cols}")

    # 3. SMOTE
    X_bal, y_bal = apply_smote(X, y)

    # 4. Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_bal, y_bal, test_size=0.2, random_state=42, stratify=y_bal
    )

    # 5. Scaling
    scaler      = StandardScaler()
    X_train_sc  = scaler.fit_transform(X_train)
    X_test_sc   = scaler.transform(X_test)

    # 6. Train + evaluate
    results    = {}
    best_model = None
    best_auc   = 0.0
    best_name  = ""

    for name, (estimator, params) in get_models().items():
        logger.info(f"{'─'*40}")
        logger.info(f"Training: {name}")
        trained, metrics = tune_and_evaluate(
            estimator, params, X_train_sc, y_train, X_test_sc, y_test, name
        )
        results[name] = (trained, metrics)
        if metrics["auc_roc"] > best_auc:
            best_auc, best_model, best_name = metrics["auc_roc"], trained, name

    # 7. SHAP (sampled — fast)
    X_train_df = pd.DataFrame(X_train_sc, columns=feature_cols)
    shap_imp   = explain_model(best_model, X_train_df, feature_cols, best_name)

    # 8. Save registry
    slug = save_to_registry(
        best_model, encoder_map, scaler,
        results[best_name][1], feature_cols, shap_imp, best_name
    )

    # 9. Batch scoring
    preds = batch_predict(best_model, scaler, encoder_map, df, feature_cols, best_name)

    # 10. Final summary
    print("\n" + "=" * 60)
    print("  MODEL COMPARISON SUMMARY")
    print("=" * 60)
    for name, (_, m) in results.items():
        tag = "  <-- BEST" if name == best_name else ""
        print(f"  {name:<18} Acc={m['accuracy']}%  F1={m['f1_score']}%  AUC={m['auc_roc']}%{tag}")

    print(f"\n  Best model   : {best_name}  (AUC-ROC = {best_auc}%)")
    print(f"  Registry ID  : {slug}")

    if shap_imp:
        print(f"\n  Top 5 SHAP features:")
        for feat, val in list(shap_imp.items())[:5]:
            print(f"    {feat:<28} {val:.4f}")

    high_risk = preds[preds["churn_risk"] == "High"]
    print(f"\n  High-risk customers: {len(high_risk):,} / {len(preds):,}")
    print(f"  Sample (top 5):")
    print(high_risk.head(5).to_string(index=False))
    print("=" * 60)

    logger.info("Pipeline completed successfully")


if __name__ == "__main__":
    main()