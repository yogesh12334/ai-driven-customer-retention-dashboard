"""
Advanced Churn Intelligence Dashboard
=======================================
Upgrades over basic version:
  - Custom CSS theming (dark industrial aesthetic)
  - @st.cache_data on all DB/file loads
  - os import fix + robust latest-file detection
  - Trend charts: Monthly Charges vs Churn scatter, Tenure histogram
  - SHAP feature importance bar chart from registry metadata
  - Retention ROI calculator
  - Tabbed layout for clean navigation
  - Error handling with actionable messages
  - [NEW] CSV Upload → API → Live Batch Predictions tab
"""

import glob
import json
import os
import sqlite3
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

# ============================================================
#  CONFIG LOAD
# ============================================================
import yaml

with open("config.yaml") as f:
    config = yaml.safe_load(f)

# ============================================================
#  PAGE CONFIG + CUSTOM CSS
# ============================================================

# --- CHANGE 1: page_icon and layout loaded from config ---
st.set_page_config(
    page_title=config['dashboard']['page_title'],
    page_icon=config['dashboard']['page_icon'],
    layout=config['dashboard']['layout'],
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  /* Font import */
  @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Sora:wght@300;400;600;700&display=swap');

  html, body, [class*="css"] {
    font-family: 'Sora', sans-serif;
  }

  /* Background */
  .stApp { background-color: #0d0f14; }

  /* Sidebar */
  section[data-testid="stSidebar"] {
    background-color: #111318;
    border-right: 1px solid #1e2330;
  }

  /* Metric cards */
  div[data-testid="metric-container"] {
    background: #141720;
    border: 1px solid #1e2330;
    border-radius: 12px;
    padding: 16px 20px;
  }
  div[data-testid="metric-container"] label {
    color: #5a6480 !important;
    font-size: 11px !important;
    text-transform: uppercase;
    letter-spacing: 1px;
    font-family: 'JetBrains Mono', monospace !important;
  }
  div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
    color: #e8eaf2 !important;
    font-size: 28px !important;
    font-weight: 700 !important;
  }

  /* Tabs */
  button[data-baseweb="tab"] {
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: #5a6480;
  }
  button[data-baseweb="tab"][aria-selected="true"] {
    color: #7c9ef8 !important;
    border-bottom: 2px solid #7c9ef8 !important;
  }

  /* Headers */
  h1 { color: #e8eaf2 !important; font-weight: 700 !important; }
  h2, h3 { color: #c4c9de !important; font-weight: 600 !important; }

  /* Divider */
  hr { border-color: #1e2330 !important; }

  /* Dataframe */
  .stDataFrame { border: 1px solid #1e2330 !important; border-radius: 8px; }

  /* Alert boxes */
  div[data-testid="stAlert"] { border-radius: 8px; }

  /* Number input */
  input[type="number"] { background: #141720 !important; color: #e8eaf2 !important; }

  /* Sidebar labels */
  .stMultiSelect label, .stSelectbox label {
    color: #5a6480 !important;
    font-size: 11px !important;
    text-transform: uppercase;
    letter-spacing: 1px;
    font-family: 'JetBrains Mono', monospace !important;
  }
</style>
""", unsafe_allow_html=True)

# Plotly dark template customised
PLOT_TEMPLATE = "plotly_dark"
PLOT_BG       = "#141720"
PAPER_BG      = "#0d0f14"
ACCENT_BLUE   = "#7c9ef8"
ACCENT_RED    = "#f87c7c"
ACCENT_GREEN  = "#7cf8b0"
ACCENT_AMBER  = "#f8c47c"

def apply_theme(fig):
    fig.update_layout(
        plot_bgcolor=PLOT_BG,
        paper_bgcolor=PAPER_BG,
        font=dict(family="Sora, sans-serif", color="#c4c9de"),
        margin=dict(t=30, l=10, r=10, b=10),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    return fig


# ============================================================
#  DATA LOADING (cached)
# ============================================================

# --- CHANGE 1 (prev): load_customers — db_path and table_name loaded from config ---
@st.cache_data(ttl=config['dashboard']['db_cache_ttl'])
def load_customers(db_path: str = None) -> pd.DataFrame:
    if db_path is None:
        db_path = config['database']['path']
    conn = sqlite3.connect(db_path)
    df   = pd.read_sql(f"SELECT * FROM {config['database']['table_name']}", conn)
    conn.close()
    return df


# --- CHANGE 2 (prev): load_predictions — predictions_dir loaded from config ---
@st.cache_data
def load_predictions() -> pd.DataFrame | None:
    path  = config['paths']['predictions_dir']
    files = glob.glob(f"{path}/*.csv")
    if not files:
        return None
    latest = max(files, key=os.path.getctime)
    return pd.read_csv(latest)


# --- CHANGE 3 (prev): load_shap_metadata — models_dir loaded from config ---
@st.cache_data
def load_shap_metadata() -> dict | None:
    """Load SHAP importance from the latest model metadata JSON."""
    path  = config['paths']['models_dir']
    files = glob.glob(f"{path}/*_metadata.json")
    if not files:
        return None
    latest = max(files, key=os.path.getctime)
    with open(latest) as f:
        return json.load(f)


# ============================================================
#  SIDEBAR
# ============================================================
with st.sidebar:
    # --- ONLY CHANGE: emoji hardcoded to fix encoding issue ---
    st.markdown(f"## 👑 {config['dashboard']['page_title']}")
    st.markdown(
        f"<span style='font-family:JetBrains Mono;font-size:11px;color:#5a6480'>{config['project']['name'].upper()} v{config['project']['version']}</span>",
        unsafe_allow_html=True,
    )
    st.divider()

    try:
        df_full = load_customers()
        data_ok = True
    except Exception as e:
        st.error(f"DB Error: {e}")
        data_ok = False
        df_full = pd.DataFrame()

    if data_ok and not df_full.empty:
        # Segment filter
        seg_options = sorted(df_full["Segment"].dropna().unique()) if "Segment" in df_full.columns else []
        segment = st.multiselect(
            "Segment",
            options=seg_options,
            default=seg_options,
        )

        # Contract filter
        ct_options = sorted(df_full["ContractType"].dropna().unique()) if "ContractType" in df_full.columns else []
        contract = st.multiselect(
            "Contract type",
            options=ct_options,
            default=ct_options,
        )

        # Churn filter
        churn_filter = st.radio(
            "Churn status",
            options=["All", "Churned", "Active"],
            horizontal=True,
        )

        st.divider()
        st.markdown(
            f"<p style='font-family:JetBrains Mono;font-size:11px;color:#5a6480'>"
            f"DATASET: {len(df_full):,} rows</p>",
            unsafe_allow_html=True,
        )


# ============================================================
#  FILTER LOGIC
# ============================================================
def apply_filters(df):
    if "Segment" in df.columns and segment:
        df = df[df["Segment"].isin(segment)]
    if "ContractType" in df.columns and contract:
        df = df[df["ContractType"].isin(contract)]
    if churn_filter == "Churned":
        df = df[df["Churn"] == 1]
    elif churn_filter == "Active":
        df = df[df["Churn"] == 0]
    return df


# ============================================================
#  MAIN CONTENT
# ============================================================

# --- CHANGE 3: Main title and description loaded from config ---
st.markdown(f"## {config['project']['name']}")
st.markdown(
    f"<p style='color:#5a6480;font-size:13px;margin-top:-12px'>{config['project']['description']}</p>",
    unsafe_allow_html=True,
)

if not data_ok or df_full.empty:
    st.error("Database not found. Run `pipeline_advanced.py` first to generate `database.db`.")
    st.stop()

df = apply_filters(df_full)

# ============================================================
#  KPI METRICS ROW
# ============================================================
total_cust      = len(df)
churn_rate      = df["Churn"].mean() * 100 if "Churn" in df.columns else 0
total_ltv       = df["LifetimeValue"].sum() if "LifetimeValue" in df.columns else 0
high_risk_count = df[df["IsCriticalRisk"] == 1].shape[0] if "IsCriticalRisk" in df.columns else 0
avg_monthly     = df["MonthlyCharges"].mean() if "MonthlyCharges" in df.columns else 0

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total customers",   f"{total_cust:,}")
c2.metric("Churn rate",        f"{churn_rate:.2f}%")
c3.metric("Lifetime value",    f"₹{total_ltv:,.0f}")
c4.metric("High-risk users",   f"{high_risk_count:,}")
c5.metric("Avg monthly charge",f"₹{avg_monthly:,.0f}")

st.divider()

# ============================================================
#  TABS
# ============================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "OVERVIEW",
    "DEEP DIVE",
    "PREDICTIONS",
    "RETENTION ROI",
    "LIVE SCORING",
])


# ------------------------------------------------------------------
#  TAB 1 — OVERVIEW
# ------------------------------------------------------------------
with tab1:
    col_a, col_b = st.columns(2)

    # Chart 1: Churn rate by contract type
    with col_a:
        st.markdown("#### Churn rate by contract type")
        if "ContractType" in df.columns:
            ct_df = (
                df.groupby("ContractType")["Churn"].mean().reset_index()
            )
            ct_df["Churn"] *= 100
            fig = px.bar(
                ct_df, x="ContractType", y="Churn",
                color="Churn",
                color_continuous_scale=["#7cf8b0", "#f8c47c", "#f87c7c"],
                labels={"Churn": "Churn rate (%)"},
                template=PLOT_TEMPLATE,
            )
            fig = apply_theme(fig)
            fig.update_coloraxes(showscale=False)
            st.plotly_chart(fig, use_container_width=True)

    # Chart 2: Segment pie
    with col_b:
        st.markdown("#### Customer segment distribution")
        if "Segment" in df.columns:
            fig = px.pie(
                df, names="Segment", hole=0.5,
                color_discrete_sequence=[ACCENT_BLUE, ACCENT_GREEN, ACCENT_AMBER, ACCENT_RED, "#c47cf8"],
                template=PLOT_TEMPLATE,
            )
            fig = apply_theme(fig)
            st.plotly_chart(fig, use_container_width=True)

    # Chart 3: Tenure distribution (churned vs active)
    st.markdown("#### Tenure distribution — churned vs active")
    if "Tenure" in df.columns:
        fig = px.histogram(
            df, x="Tenure", color="Churn",
            nbins=40,
            barmode="overlay",
            opacity=0.7,
            color_discrete_map={0: ACCENT_BLUE, 1: ACCENT_RED},
            labels={"Tenure": "Tenure (months)", "Churn": "Status"},
            template=PLOT_TEMPLATE,
        )
        fig = apply_theme(fig)
        fig.update_layout(bargap=0.05)
        st.plotly_chart(fig, use_container_width=True)


# ------------------------------------------------------------------
#  TAB 2 — DEEP DIVE
# ------------------------------------------------------------------
with tab2:
    col_c, col_d = st.columns(2)

    # Chart 4: Monthly Charges vs Churn scatter
    with col_c:
        st.markdown("#### Monthly charges vs churn probability")
        if {"MonthlyCharges", "Tenure", "Churn"}.issubset(df.columns):
            sample = df.sample(min(5000, len(df)), random_state=42)
            fig = px.scatter(
                sample, x="Tenure", y="MonthlyCharges",
                color=sample["Churn"].map({0: "Active", 1: "Churned"}),
                opacity=0.6, size_max=5,
                color_discrete_map={"Active": ACCENT_BLUE, "Churned": ACCENT_RED},
                labels={"color": "Status"},
                template=PLOT_TEMPLATE,
            )
            fig = apply_theme(fig)
            st.plotly_chart(fig, use_container_width=True)

    # Chart 5: Payment method churn
    with col_d:
        st.markdown("#### Churn rate by payment method")
        if "PaymentMethod" in df.columns:
            pm_df = (
                df.groupby("PaymentMethod")["Churn"].mean().reset_index()
            )
            pm_df["Churn"] *= 100
            fig = px.bar(
                pm_df.sort_values("Churn", ascending=True),
                x="Churn", y="PaymentMethod",
                orientation="h",
                color="Churn",
                color_continuous_scale=["#7cf8b0", "#f8c47c", "#f87c7c"],
                labels={"Churn": "Churn rate (%)", "PaymentMethod": ""},
                template=PLOT_TEMPLATE,
            )
            fig = apply_theme(fig)
            fig.update_coloraxes(showscale=False)
            st.plotly_chart(fig, use_container_width=True)

    # Chart 6: SHAP feature importance from model metadata
    st.markdown("#### SHAP feature importance (latest model)")
    meta = load_shap_metadata()
    if meta and meta.get("top_shap_features"):
        shap_data = pd.DataFrame(
            meta["top_shap_features"], columns=["Feature", "Importance"]
        ).sort_values("Importance")
        fig = px.bar(
            shap_data, x="Importance", y="Feature",
            orientation="h",
            color="Importance",
            color_continuous_scale=[ACCENT_BLUE, ACCENT_AMBER],
            template=PLOT_TEMPLATE,
        )
        fig = apply_theme(fig)
        fig.update_coloraxes(showscale=False)
        st.plotly_chart(fig, use_container_width=True)

        model_info = meta.get("model_name", "Unknown")
        auc        = meta.get("metrics", {}).get("auc_roc", "—")
        f1         = meta.get("metrics", {}).get("f1_score", "—")
        st.markdown(
            f"<p style='font-family:JetBrains Mono;font-size:11px;color:#5a6480'>"
            f"Model: {model_info} &nbsp;|&nbsp; AUC-ROC: {auc}% &nbsp;|&nbsp; F1: {f1}%</p>",
            unsafe_allow_html=True,
        )
    else:
        st.info("Run `train_model.py` first to generate SHAP data in `models/` folder.")

    # Critical risk table
    st.markdown("#### Critical-risk customer list")
    if "IsCriticalRisk" in df.columns:
        crit_cols = [c for c in ["CustomerID", "Segment", "MonthlyCharges", "Tenure", "LifetimeValue", "ContractType"] if c in df.columns]
        crit_df   = df[df["IsCriticalRisk"] == 1][crit_cols].sort_values("MonthlyCharges", ascending=False)
        st.dataframe(crit_df, use_container_width=True, height=320)
        st.caption(f"{len(crit_df):,} critical-risk customers shown")


# ------------------------------------------------------------------
#  TAB 3 — PREDICTIONS
# ------------------------------------------------------------------
with tab3:
    preds_df = load_predictions()

    if preds_df is None:
        st.warning("No prediction files found in `predictions/` folder. Run `train_model.py` to generate them.")
    else:
        st.markdown("#### Churn risk distribution")
        if "churn_risk" in preds_df.columns:
            risk_counts = preds_df["churn_risk"].value_counts().reset_index()
            risk_counts.columns = ["Risk level", "Count"]
            color_map = {"Low": ACCENT_GREEN, "Medium": ACCENT_AMBER, "High": ACCENT_RED}
            fig = px.bar(
                risk_counts, x="Risk level", y="Count",
                color="Risk level",
                color_discrete_map=color_map,
                template=PLOT_TEMPLATE,
            )
            fig = apply_theme(fig)
            st.plotly_chart(fig, use_container_width=True)

        st.divider()
        st.markdown("#### Individual customer lookup")

        search_id = st.number_input(
            "Enter Customer ID:",
            min_value=int(preds_df["CustomerID"].min()),
            max_value=int(preds_df["CustomerID"].max()),
            step=1,
            key="search_input",
        )

        if st.button("Check churn risk", type="primary"):
            result = preds_df[preds_df["CustomerID"] == search_id]

            if not result.empty:
                prob       = result["churn_probability"].values[0]
                risk_level = result["churn_risk"].values[0]
                model_used = result.get("predicted_by", pd.Series(["Unknown"])).values[0]

                # Color logic
                if prob > 60:
                    color, icon, status_msg = ACCENT_RED,   "HIGH RISK",   "Immediate retention action recommended."
                elif prob > 30:
                    color, icon, status_msg = ACCENT_AMBER, "MEDIUM RISK", "Monitor closely — consider proactive outreach."
                else:
                    color, icon, status_msg = ACCENT_GREEN, "LOW RISK",    "Customer appears stable."

                st.markdown(f"""
                <div style='
                  background:#141720;
                  border:1px solid {color}44;
                  border-left:4px solid {color};
                  border-radius:12px;
                  padding:24px 28px;
                  margin:16px 0;
                '>
                  <p style='font-family:JetBrains Mono;font-size:11px;color:#5a6480;margin:0 0 8px'>
                    CUSTOMER #{search_id} &nbsp;·&nbsp; MODEL: {model_used}
                  </p>
                  <p style='font-size:42px;font-weight:700;color:{color};margin:0 0 4px;font-family:Sora'>
                    {prob:.1f}%
                  </p>
                  <p style='font-size:13px;color:#5a6480;margin:0 0 16px;font-family:JetBrains Mono'>
                    CHURN PROBABILITY
                  </p>
                  <span style='
                    background:{color}22;
                    color:{color};
                    border:1px solid {color}55;
                    border-radius:20px;
                    padding:4px 14px;
                    font-size:11px;
                    font-family:JetBrains Mono;
                    letter-spacing:1px;
                  '>{icon}</span>
                  <p style='margin-top:16px;color:#8a92ac;font-size:13px'>{status_msg}</p>
                </div>
                """, unsafe_allow_html=True)

                # Gauge chart
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=prob,
                    number={"suffix": "%", "font": {"size": 32, "color": color}},
                    gauge={
                        "axis": {"range": [0, 100], "tickcolor": "#5a6480"},
                        "bar":  {"color": color, "thickness": 0.3},
                        "bgcolor": "#1a1e2b",
                        "steps": [
                            {"range": [0,  30], "color": "#0d1a14"},
                            {"range": [30, 60], "color": "#1a1700"},
                            {"range": [60, 100], "color": "#1a0d0d"},
                        ],
                        "threshold": {
                            "line": {"color": color, "width": 3},
                            "thickness": 0.8,
                            "value": prob,
                        },
                    },
                ))
                fig_gauge.update_layout(
                    paper_bgcolor=PAPER_BG,
                    height=260,
                    margin=dict(t=20, b=10, l=40, r=40),
                    font={"family": "Sora"},
                )
                st.plotly_chart(fig_gauge, use_container_width=True)

            else:
                st.error(f"Customer ID {search_id} not found in predictions database.")


# ------------------------------------------------------------------
#  TAB 4 — RETENTION ROI CALCULATOR
# ------------------------------------------------------------------
with tab4:
    st.markdown("#### Retention ROI calculator")
    st.markdown(
        "<p style='color:#5a6480;font-size:13px'>Estimate revenue saved by reducing churn rate.</p>",
        unsafe_allow_html=True,
    )

    rc1, rc2 = st.columns(2)
    with rc1:
        current_churn_rate   = st.slider("Current churn rate (%)",    0.0, 100.0, float(round(churn_rate, 1)), 0.5)
        target_churn_rate    = st.slider("Target churn rate (%)",     0.0, 100.0, max(0.0, float(round(churn_rate - 5, 1))), 0.5)
        avg_ltv_input        = st.number_input("Avg customer LTV (₹)", value=int(total_ltv / max(total_cust, 1)), step=1000)

    with rc2:
        customers_at_risk    = int(total_cust * current_churn_rate / 100)
        customers_saved      = int(total_cust * (current_churn_rate - target_churn_rate) / 100)
        revenue_saved        = customers_saved * avg_ltv_input
        intervention_cost    = st.number_input("Intervention cost per customer (₹)", value=500, step=100)
        total_intervention   = customers_saved * intervention_cost
        net_roi              = revenue_saved - total_intervention
        roi_pct              = (net_roi / max(total_intervention, 1)) * 100

    st.divider()
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Customers at risk",     f"{customers_at_risk:,}")
    m2.metric("Customers saved",       f"{customers_saved:,}")
    m3.metric("Revenue saved",         f"₹{revenue_saved:,.0f}")
    m4.metric("Net ROI",               f"₹{net_roi:,.0f}", delta=f"{roi_pct:.0f}% return")

    # Waterfall chart
    fig_wf = go.Figure(go.Waterfall(
        orientation="v",
        measure=["absolute", "relative", "relative", "total"],
        x=["Revenue at risk", "Intervention cost", "Revenue recovered", "Net gain"],
        y=[
            customers_at_risk * avg_ltv_input,
            -total_intervention,
            revenue_saved,
            net_roi,
        ],
        connector={"line": {"color": "#1e2330"}},
        increasing={"marker": {"color": ACCENT_GREEN}},
        decreasing={"marker": {"color": ACCENT_RED}},
        totals={"marker":    {"color": ACCENT_BLUE}},
        text=[
            f"₹{customers_at_risk * avg_ltv_input:,.0f}",
            f"-₹{total_intervention:,.0f}",
            f"₹{revenue_saved:,.0f}",
            f"₹{net_roi:,.0f}",
        ],
        textposition="outside",
    ))
    fig_wf.update_layout(
        plot_bgcolor=PLOT_BG,
        paper_bgcolor=PAPER_BG,
        font=dict(family="Sora", color="#c4c9de"),
        showlegend=False,
        margin=dict(t=30, l=10, r=10, b=10),
        yaxis=dict(tickprefix="₹", gridcolor="#1e2330"),
    )
    st.plotly_chart(fig_wf, use_container_width=True)


# ------------------------------------------------------------------
#  TAB 5 — LIVE SCORING  (NEW)
# ------------------------------------------------------------------

# --- CHANGE 4: REQUIRED_COLS loaded from config ---
REQUIRED_COLS = config['features']['input_cols']

with tab5:
    st.markdown("#### Live batch scoring via API")
    st.markdown(
        "<p style='color:#5a6480;font-size:13px'>"
        "Upload a CSV → columns will be validated → API will return predictions.</p>",
        unsafe_allow_html=True,
    )

    # --- API URL loaded from config ---
    api_url = st.text_input(
        "API base URL",
        value=config['dashboard']['api_default_url'],
        help="Where api.py is running. Default: http://localhost:8000",
    )

    # --- API health check button ---
    col_h1, col_h2 = st.columns([1, 4])
    with col_h1:
        if st.button("Check API status"):
            try:
                r = requests.get(f"{api_url}/health", timeout=5)
                if r.status_code == 200 and r.json().get("model_loaded"):
                    st.success("API online — model loaded")
                else:
                    st.warning("API reachable but model not loaded. Run train_model.py first.")
            except requests.exceptions.ConnectionError:
                st.error("API offline. Start with: `python api.py`")
            except Exception as e:
                st.error(f"Error: {e}")

    st.divider()

    # --- Step 1: CSV Upload ---
    st.markdown("##### Step 1 — Upload customer CSV")
    st.markdown(
        f"<p style='font-family:JetBrains Mono;font-size:11px;color:#5a6480'>"
        f"Required columns: {', '.join(REQUIRED_COLS)}</p>",
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "Drop CSV here",
        type=["csv"],
        help="CSV must contain these columns: " + ", ".join(REQUIRED_COLS),
    )

    if uploaded_file is not None:

        # --- Step 2: Read + validate columns ---
        try:
            upload_df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"CSV read error: {e}")
            st.stop()

        missing_cols = [c for c in REQUIRED_COLS if c not in upload_df.columns]

        if missing_cols:
            st.error(f"These columns are missing from the CSV: `{', '.join(missing_cols)}`")
            found_cols = [c for c in REQUIRED_COLS if c in upload_df.columns]
            st.markdown(
                f"<p style='font-family:JetBrains Mono;font-size:11px;color:#5a6480'>"
                f"Found: {', '.join(found_cols) or 'none'}</p>",
                unsafe_allow_html=True,
            )
        else:
            st.success(f"Column check passed — {len(upload_df):,} rows ready to score")

            # Preview
            st.markdown("##### Step 2 — Preview")
            st.dataframe(upload_df[REQUIRED_COLS].head(5), use_container_width=True)

            # --- CHANGE 6: Batch size limit loaded from config ---
            max_rows = config['batch']['max_rows']

            if len(upload_df) > max_rows:
                st.warning(
                    f"CSV contains {len(upload_df):,} rows. "
                    f"API accepts max {max_rows} — only the first {max_rows} rows will be scored."
                )
                upload_df = upload_df.head(max_rows)

            # --- Step 3: Send to API ---
            st.markdown("##### Step 3 — Score via API")
            if st.button("Run predictions", type="primary"):
                payload = {
                    "customers": upload_df[REQUIRED_COLS].to_dict(orient="records")
                }

                with st.spinner(f"Sending {len(upload_df):,} customers to the API..."):
                    try:
                        response = requests.post(
                            f"{api_url}/predict/batch",
                            json=payload,
                            timeout=60,
                        )

                        if response.status_code == 200:
                            api_result  = response.json()
                            predictions = api_result["predictions"]
                            summary     = api_result["summary"]

                            # Build results DataFrame
                            result_df = upload_df[REQUIRED_COLS].copy().reset_index(drop=True)
                            result_df["churn_probability_pct"] = [p["churn_probability_pct"] for p in predictions]
                            result_df["churn_probability"]     = [p["churn_probability"] * 100 for p in predictions]
                            result_df["risk_level"]            = [p["risk_level"] for p in predictions]
                            result_df["prediction"]            = [p["prediction"] for p in predictions]
                            result_df["confidence"]            = [p["confidence"] for p in predictions]

                            # Summary KPIs
                            st.divider()
                            st.markdown("##### Results")
                            k1, k2, k3, k4 = st.columns(4)
                            k1.metric("Total scored",   f"{api_result['total_customers']:,}")
                            k2.metric("High risk",      f"{summary['high_risk']:,}")
                            k3.metric("Medium risk",    f"{summary['medium_risk']:,}")
                            k4.metric("Avg churn prob", summary["avg_churn_probability"])

                            # Risk distribution chart
                            risk_counts = (
                                result_df["risk_level"]
                                .value_counts()
                                .reset_index()
                            )
                            risk_counts.columns = ["Risk level", "Count"]
                            color_map = {
                                "Low":    ACCENT_GREEN,
                                "Medium": ACCENT_AMBER,
                                "High":   ACCENT_RED,
                            }
                            fig_live = px.bar(
                                risk_counts,
                                x="Risk level", y="Count",
                                color="Risk level",
                                color_discrete_map=color_map,
                                template=PLOT_TEMPLATE,
                            )
                            fig_live = apply_theme(fig_live)
                            st.plotly_chart(fig_live, use_container_width=True)

                            # Full results table
                            st.markdown("##### Full scored results")
                            st.dataframe(
                                result_df.sort_values("churn_probability", ascending=False),
                                use_container_width=True,
                                height=400,
                            )

                            # Download button
                            csv_out = result_df.to_csv(index=False).encode("utf-8")
                            st.download_button(
                                label="Download results as CSV",
                                data=csv_out,
                                file_name=f"live_predictions_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv",
                            )

                        elif response.status_code == 503:
                            st.error("API model is not loaded. Run `train_model.py` first.")
                        elif response.status_code == 422:
                            detail = response.json().get("detail", "Unknown validation error")
                            st.error(f"Validation error: {detail}")
                        else:
                            st.error(f"API error {response.status_code}: {response.text[:300]}")

                    except requests.exceptions.ConnectionError:
                        st.error("Could not connect to the API. Start it with: `python api.py`")
                    except requests.exceptions.Timeout:
                        st.error("API request timed out — dataset too large or API is slow.")
                    except Exception as e:
                        st.error(f"Unexpected error: {e}")


# ============================================================
#  FOOTER
# ============================================================

# --- CHANGE 4 (prev): Footer — project name and version loaded from config ---
st.divider()
st.markdown(
    f"<p style='text-align:center;font-family:JetBrains Mono;font-size:11px;color:#2e3450'>"
    f"{config['project']['name'].upper()} v{config['project']['version']} &nbsp;·&nbsp; Powered by RandomForest + XGBoost &nbsp;·&nbsp; SHAP Explainability</p>",
    unsafe_allow_html=True,
)