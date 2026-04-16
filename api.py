"""
Advanced Churn Prediction API
==============================
Upgrades over basic version:
  - Startup lifespan (modern FastAPI pattern)
  - Request validation with Pydantic v2 + field constraints
  - /health endpoint with model metadata
  - /predict/batch endpoint (bulk processing)
  - /model/info endpoint (SHAP features, metrics)
  - Preprocessing extracted to reusable function
  - Structured JSON error responses
  - Request logging middleware
  - CORS middleware
  - Response time header
  - XGBoost and RandomForest fallback support
"""

import glob
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

# ============================================================
#  LOGGING SETUP
# ============================================================
Path("logs").mkdir(exist_ok=True)
logging.basicConfig(
    filename=f"logs/api_{datetime.now().strftime('%Y%m%d')}.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
logger = logging.getLogger(__name__)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
console.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s"))
logger.addHandler(console)


# ============================================================
#  ARTIFACT LOADER
# ============================================================
def load_latest_artifacts() -> dict:
    """
    Loads the latest model, encoders, scaler, and metadata from the models/ folder.
    Attempts to load XGBoost by default, with a fallback to RandomForest.
    Returns a dictionary of artifacts; raises FileNotFoundError if none exist.
    """
    for prefix in ("xgboost", "randomforest"):
        files = glob.glob(f"models/{prefix}_*_model.pkl")
        if files:
            latest = max(files, key=os.path.getctime)
            slug   = latest.replace("_model.pkl", "")
            break
    else:
        raise FileNotFoundError(
            "No valid model files found in the models/ directory. "
            "Please run train_model.py first."
        )

    model    = joblib.load(f"{slug}_model.pkl")
    encoders = joblib.load(f"{slug}_encoders.pkl")
    scaler   = joblib.load(f"{slug}_scaler.pkl")

    meta_path = f"{slug}_metadata.json"
    metadata  = {}
    if Path(meta_path).exists():
        with open(meta_path) as f:
            metadata = json.load(f)

    logger.info(f"Artifacts successfully loaded from slug: {Path(slug).name}")
    return {
        "model":     model,
        "encoders":  encoders,
        "scaler":    scaler,
        "metadata":  metadata,
        "slug":      Path(slug).name,
        "loaded_at": datetime.now().isoformat(),
    }


# ============================================================
#  GLOBAL STATE
# ============================================================
ARTIFACTS: dict = {}

FEATURE_COLS = [
    "Age", "Tenure", "MonthlyCharges", "UsageHours", "SupportTickets",
    "Location", "ContractType", "PaymentMethod", "LifetimeValue",
    "IsCriticalRisk", "ChargePerMonth", "TenureGroup", "HighSpender",
]


# ============================================================
#  LIFESPAN HANDLER
# ============================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- Startup Logic ---
    global ARTIFACTS
    try:
        ARTIFACTS = load_latest_artifacts()
        logger.info("API Startup: Model artifacts are ready for inference.")
    except FileNotFoundError as e:
        logger.error(f"Startup Warning: {e}")
        ARTIFACTS = {}

    yield  # API is active

    # --- Shutdown Logic ---
    logger.info("API Lifecycle: Service is shutting down.")


# ============================================================
#  APP INITIALIZATION
# ============================================================
app = FastAPI(
    title="Churn Prediction AI API",
    description="Production-grade predictive analytics service featuring XGBoost/RandomForest and SHAP explainability.",
    version="3.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ============================================================
#  MIDDLEWARE: Performance Monitoring
# ============================================================
@app.middleware("http")
async def add_process_time(request: Request, call_next):
    start    = time.time()
    response = await call_next(request)
    elapsed  = round((time.time() - start) * 1000, 2)
    response.headers["X-Process-Time-Ms"] = str(elapsed)
    logger.info(f"{request.method} {request.url.path} -> Status: {response.status_code} ({elapsed}ms)")
    return response


# ============================================================
#  DATA SCHEMAS (Pydantic)
# ============================================================
class CustomerRequest(BaseModel):
    Age:            int   = Field(..., ge=18,  le=100,   description="Age of the customer")
    Tenure:         int   = Field(..., ge=0,   le=300,   description="Months of relationship with the company")
    MonthlyCharges: float = Field(..., ge=0.0, le=99999, description="Current monthly bill amount")
    UsageHours:     float = Field(..., ge=0.0,           description="Total usage hours in the last month")
    SupportTickets: int   = Field(..., ge=0,   le=100,   description="Number of support tickets raised")
    Location:       str   = Field(..., min_length=1,     description="Geographic location/city")
    ContractType:   str   = Field(...,                   description="Type of contract: month-to-month, one year, or two year")
    PaymentMethod:  str   = Field(...,                   description="Primary payment method")

    @field_validator("ContractType")
    @classmethod
    def validate_contract(cls, v):
        valid = {"month-to-month", "one year", "two year"}
        if v.lower() not in valid:
            raise ValueError(f"ContractType must be one of: {valid}")
        return v


class BatchRequest(BaseModel):
    customers: list[CustomerRequest] = Field(..., min_length=1, max_length=500)


class PredictionResponse(BaseModel):
    customer_index:     int
    churn_probability:  float
    churn_probability_pct: str
    risk_level:         str
    prediction:         str
    confidence:         str


class HealthResponse(BaseModel):
    status:       str
    model_loaded: bool
    model_slug:   Optional[str]
    loaded_at:    Optional[str]
    uptime_note:  str


# ============================================================
#  PREPROCESSING UTILITY
# ============================================================
def preprocess(df: pd.DataFrame, encoders: dict, scaler) -> np.ndarray:
    """
    Performs encoding, feature engineering, and scaling.
    Ensures data consistency between API requests and the training pipeline.
    """
    df = df.copy()

    # Apply Encoding
    for col, info in encoders.items():
        if col not in df.columns:
            continue
        if info["type"] == "frequency":
            df[col] = df[col].map(info["map"]).fillna(0.0)
        elif info["type"] == "label":
            le    = info["encoder"]
            known = set(le.classes_)
            df[col] = df[col].apply(lambda x: str(x) if str(x) in known else le.classes_[0])
            df[col] = le.transform(df[col].astype(str))

    # Calculate Derived Features
    df["LifetimeValue"]  = df["MonthlyCharges"] * df["Tenure"].clip(lower=1)
    df["ChargePerMonth"] = df["MonthlyCharges"] / df["Tenure"].clip(lower=1)
    df["TenureGroup"]    = pd.cut(
        df["Tenure"], bins=[0, 12, 24, 48, 9999], labels=[0, 1, 2, 3]
    ).astype(int)
    df["HighSpender"]    = (df["MonthlyCharges"] > 80).astype(int)
    df["IsCriticalRisk"] = (df["MonthlyCharges"] > 100).astype(int)

    return scaler.transform(df[FEATURE_COLS])


def make_prediction_response(prob: float, index: int = 0) -> PredictionResponse:
    risk  = "High" if prob > 0.6 else ("Medium" if prob > 0.3 else "Low")
    conf  = "High" if (prob > 0.75 or prob < 0.25) else ("Medium" if (prob > 0.6 or prob < 0.4) else "Low")
    return PredictionResponse(
        customer_index=index,
        churn_probability=round(prob, 4),
        churn_probability_pct=f"{prob * 100:.2f}%",
        risk_level=risk,
        prediction="Churn" if prob > 0.5 else "No Churn",
        confidence=conf,
    )


# ============================================================
#  API ENDPOINTS
# ============================================================

@app.get("/", tags=["General"])
def root():
    """Returns the API status and version information."""
    return {
        "api":     "Churn Prediction AI Service",
        "version": "3.0",
        "docs":    "/docs",
        "health":  "/health",
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
def health_check():
    """Provides a system health check and model loading status."""
    loaded = bool(ARTIFACTS)
    return HealthResponse(
        status      ="OK" if loaded else "Degraded",
        model_loaded=loaded,
        model_slug  =ARTIFACTS.get("slug"),
        loaded_at   =ARTIFACTS.get("loaded_at"),
        uptime_note ="Please run train_model.py if model_loaded is false." if not loaded else "System operational.",
    )


@app.get("/model/info", tags=["Model"])
def model_info():
    """Retrieves metrics, feature lists, and SHAP importance for the active model."""
    if not ARTIFACTS:
        raise HTTPException(status_code=503, detail="Model artifacts not initialized.")
    meta = ARTIFACTS.get("metadata", {})
    return {
        "model_name":         meta.get("model_name", "Unknown"),
        "version":            meta.get("version"),
        "features":           meta.get("features", FEATURE_COLS),
        "metrics":            meta.get("metrics", {}),
        "top_shap_features": meta.get("top_shap_features", []),
        "saved_at":           meta.get("saved_at"),
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict_single(data: CustomerRequest):
    """Predicts churn probability for a single customer profile."""
    if not ARTIFACTS:
        raise HTTPException(status_code=503, detail="Service Unavailable: Model not loaded.")

    try:
        df      = pd.DataFrame([data.model_dump()])
        X       = preprocess(df, ARTIFACTS["encoders"], ARTIFACTS["scaler"])
        prob    = float(ARTIFACTS["model"].predict_proba(X)[0, 1])
        return make_prediction_response(prob, index=0)

    except ValueError as e:
        raise HTTPException(status_code=422, detail=f"Preprocessing Error: {e}")
    except Exception as e:
        logger.error(f"Inference Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error during prediction.")


@app.post("/predict/batch", tags=["Prediction"])
def predict_batch(data: BatchRequest):
    """
    Processes a bulk list of customers (up to 500) for churn prediction.
    Returns individual results along with aggregate risk statistics.
    """
    if not ARTIFACTS:
        raise HTTPException(status_code=503, detail="Service Unavailable: Model not loaded.")

    try:
        records = [c.model_dump() for c in data.customers]
        df      = pd.DataFrame(records)
        X       = preprocess(df, ARTIFACTS["encoders"], ARTIFACTS["scaler"])
        probs   = ARTIFACTS["model"].predict_proba(X)[:, 1]

        predictions = [
            make_prediction_response(float(p), i).model_dump()
            for i, p in enumerate(probs)
        ]

        high   = sum(1 for p in probs if p > 0.6)
        medium = sum(1 for p in probs if 0.3 < p <= 0.6)
        low    = sum(1 for p in probs if p <= 0.3)

        # Dashboard ke requirements ke hisab se keys update ki gayi hain:
        return {
            "total_customers": len(predictions),
            "summary": {
                "high_risk":   high,
                "medium_risk": medium,
                "low_risk":    low,
                "avg_churn_probability": f"{np.mean(probs) * 100:.2f}%",
            },
            "predictions": predictions,
        }

    except ValueError as e:
        raise HTTPException(status_code=422, detail=f"Batch Preprocessing Error: {e}")
    except Exception as e:
        logger.error(f"Batch Inference Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error during batch processing.")


@app.post("/model/reload", tags=["Model"])
def reload_model():
    """
    Hot-reloads model artifacts without requiring a server restart. 
    Use this after training a new model.
    """
    global ARTIFACTS
    try:
        ARTIFACTS = load_latest_artifacts()
        return {
            "status":  "Success",
            "slug":    ARTIFACTS["slug"],
            "message": "Model artifacts reloaded successfully.",
        }
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Reload Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Artifact reload failed.")


# ============================================================
#  EXECUTION
# ============================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)