import streamlit as st
import pandas as pd
import numpy as np
import os
import subprocess
import time
from joblib import load
import plotly.express as px

# ==========================
# PATH CONFIG
# ==========================
BASE_DIR = r"D:\minor in ai\Stock_Market_Sentiment_&_Trend_Forecasting"
SOURCE_DIR = os.path.join(BASE_DIR, "Source_code")
DATA_DIR = os.path.join(BASE_DIR, "dataset_Used")
MODEL_DIR = os.path.join(BASE_DIR, "Models")
EXPLAIN_DIR = os.path.join(BASE_DIR, "Explainability_Results")

os.makedirs(EXPLAIN_DIR, exist_ok=True)

COMPANIES = ["tesla", "apple", "amazon"]
FEATURES = [
    'Open', 'High', 'Low', 'Volume', 'Daily_Return',
    'MA_3', 'MA_7', 'Volatility_7',
    'sentiment_score_mean', 'Sentiment_MA_3', 'Sentiment_MA_7'
]

# ==========================
# Streamlit Config
# ==========================
st.set_page_config(page_title="Stock Sentiment & Trend Dashboard", layout="wide")
st.title("📈 Stock Market Sentiment & Trend Forecasting — Dynamic Dashboard")
st.markdown("Run pipeline to refresh live data, generate predictions, and view explainability outputs.")

# ==========================
# Sidebar Controls
# ==========================
st.sidebar.header("Controls")
selected_companies = st.sidebar.multiselect("Select companies to display", COMPANIES, default=COMPANIES)
run_pipeline_btn = st.sidebar.button("Run full pipeline (data_collection → preprocessing → sentiment → features → forecasting → SHAP)")
regenerate_shap_btn = st.sidebar.button("Regenerate SHAP only")
predict_button = st.sidebar.button("Generate Next 7 Day Predictions (dynamic)")
show_pred = st.sidebar.checkbox("Show predicted prices (chart & table)", value=True)
show_sentiment = st.sidebar.checkbox("Show sentiment charts", value=True)
show_shap = st.sidebar.checkbox("Show SHAP explainability", value=True)
show_top5 = st.sidebar.checkbox("Show top-5 SHAP table", value=True)
show_csv = st.sidebar.checkbox("View CSV Files (dataset_Used / Explainability_Results)", value=False)

if not selected_companies:
    st.sidebar.warning("Select at least one company to view results.")

# ==========================
# Run External Script
# ==========================
def run_script(script_name):
    python_path = r"D:\minor in ai\Stock_Market_Sentiment_&_Trend_Forecasting\venv_mia\Scripts\python.exe"
    script_path = os.path.join(SOURCE_DIR, script_name)
    if not os.path.exists(script_path):
        return False, f"Missing script: {script_path}"
    try:
        proc = subprocess.run([python_path, script_path], capture_output=True, text=True, check=False)
        return (proc.returncode == 0), proc
    except Exception as e:
        return False, str(e)

# ==========================
# Pipeline
# ==========================
if run_pipeline_btn:
    st.sidebar.info("Running full pipeline. Please wait...")
    scripts = [
        "data_collection.py",
        "data_preprocessing.py",
        "sentiment_analysis.py",
        "feature_engineering.py",
        "stock_forecasting.py",
        "shap_implementation.py"
    ]
    for script in scripts:
        with st.spinner(f"Running {script} ..."):
            ok, out = run_script(script)
            if not ok:
                st.error(f"{script} failed:\n{out}")
            else:
                st.success(f"{script} completed.")
        time.sleep(0.4)
    st.success("✅ Full pipeline finished. Data refreshed!")

if regenerate_shap_btn:
    st.sidebar.info("Regenerating SHAP visualizations...")
    ok, out = run_script("shap_implementation.py")
    st.success("SHAP regeneration finished!" if ok else f"Failed: {out}")

# ==========================
# Data Loaders
# ==========================
@st.cache_data
def load_features(company):
    fpath = os.path.join(DATA_DIR, f"{company}_features.csv")
    if not os.path.exists(fpath):
        return None
    df = pd.read_csv(fpath)
    return df

def predict_next_7_days(company, df_features):
    scaler_path = os.path.join(MODEL_DIR, f"{company}_scaler.pkl")
    model_path = os.path.join(MODEL_DIR, f"{company}_rf.pkl")
    if not os.path.exists(scaler_path) or not os.path.exists(model_path):
        return None
    scaler = load(scaler_path)
    model = load(model_path)
    last_row = df_features.iloc[-1:].copy().reset_index(drop=True)
    preds, dates = [], []
    last_date = pd.to_datetime(df_features['Date'].iloc[-1]) if 'Date' in df_features.columns else pd.Timestamp.today()

    for i in range(1, 8):
        X = pd.DataFrame(last_row[FEATURES].values, columns=FEATURES)
        X_scaled = scaler.transform(X)
        pred = model.predict(X_scaled)[0]
        preds.append(pred)
        dates.append((last_date + pd.Timedelta(days=i)).date())
        new_row = last_row.copy()
        new_row.loc[0, 'Open'] = new_row.loc[0, 'High'] = new_row.loc[0, 'Low'] = new_row.loc[0, 'Close'] = pred
        last_row = new_row
    return pd.DataFrame({"Date": dates, f"{company}_Predicted": preds})

# ==========================
# Load all selected data
# ==========================
all_pred_dfs = []
all_sentiment_dfs = {}
for comp in selected_companies:
    df = load_features(comp)
    if df is None:
        st.warning(f"Missing {comp}_features.csv")
        continue
    if predict_button or show_pred:
        pred_df = predict_next_7_days(comp, df)
        if pred_df is not None:
            all_pred_dfs.append(pred_df)
    all_sentiment_dfs[comp] = df

# ==========================
# Combined Predicted Price Chart
# ==========================
if show_pred and all_pred_dfs:
    st.markdown("---")
    st.header("📊 Combined Predicted Prices for All Companies")

    # Merge all prediction dataframes by Date
    merged_pred = all_pred_dfs[0]
    for df_ in all_pred_dfs[1:]:
        merged_pred = pd.merge(merged_pred, df_, on="Date", how="outer")

    fig_combined = px.line(merged_pred, x="Date", y=[c for c in merged_pred.columns if "Predicted" in c],
                           title="Next 7-Day Forecast Comparison")
    st.plotly_chart(fig_combined, use_container_width=True)

    st.subheader("Unified 7-Day Predictions Table")
    st.dataframe(merged_pred.set_index("Date"))

# ==========================
# Combined Sentiment Chart
# ==========================
if show_sentiment and all_sentiment_dfs:
    st.markdown("---")
    st.header("💬 Combined Sentiment Overview (Daily Average)")

    sentiment_merge = pd.DataFrame()
    for comp, df in all_sentiment_dfs.items():
        if 'Date' in df.columns and 'sentiment_score_mean' in df.columns:
            temp = df[['Date', 'sentiment_score_mean']].copy()
            temp['Date'] = pd.to_datetime(temp['Date'])
            temp.rename(columns={'sentiment_score_mean': f"{comp}_Sentiment"}, inplace=True)
            sentiment_merge = temp if sentiment_merge.empty else pd.merge(sentiment_merge, temp, on='Date', how='outer')

    if not sentiment_merge.empty:
        fig_sent = px.line(sentiment_merge, x='Date', y=[c for c in sentiment_merge.columns if "_Sentiment" in c],
                           title="Company-wise Sentiment Comparison")
        st.plotly_chart(fig_sent, use_container_width=True)
    else:
        st.info("Sentiment data not available for selected companies.")

# ==========================
# SHAP Explainability (choose one)
# ==========================
if show_shap:
    st.markdown("---")
    st.header("Explainability (SHAP) — Select Company")
    shap_company = st.radio("Choose company for SHAP visuals:", COMPANIES, horizontal=True)

    bar_img = os.path.join(EXPLAIN_DIR, f"{shap_company}_feature_importance_bar.png")
    summary_img = os.path.join(EXPLAIN_DIR, f"{shap_company}_summary_plot.png")
    local_force_img = os.path.join(EXPLAIN_DIR, f"{shap_company}_local_force_plot.png")
    local_decision_img = os.path.join(EXPLAIN_DIR, f"{shap_company}_local_decision_plot.png")

    cols = st.columns(2)
    with cols[0]:
        if os.path.exists(bar_img):
            st.image(bar_img, caption=f"{shap_company.upper()} - Feature Importance", use_column_width=True)
        if os.path.exists(local_force_img):
            st.image(local_force_img, caption=f"{shap_company.upper()} - Local Force Plot", use_column_width=True)
    with cols[1]:
        if os.path.exists(summary_img):
            st.image(summary_img, caption=f"{shap_company.upper()} - Summary Plot", use_column_width=True)
        if os.path.exists(local_decision_img):
            st.image(local_decision_img, caption=f"{shap_company.upper()} - Local Decision Plot", use_column_width=True)

# ==========================
# CSV Viewer
# ==========================
if show_csv:
    st.markdown("---")
    st.header("View CSV Files Generated by Pipeline")
    csv_dir = st.radio("Choose folder:", ("dataset_Used", "Explainability_Results"), horizontal=True)
    folder = DATA_DIR if csv_dir == "dataset_Used" else EXPLAIN_DIR
    try:
        csv_files = [f for f in os.listdir(folder) if f.lower().endswith(".csv")]
    except FileNotFoundError:
        csv_files = []

    if csv_files:
        selected_csv = st.selectbox("Select a CSV file:", csv_files)
        df_csv = pd.read_csv(os.path.join(folder, selected_csv))
        st.dataframe(df_csv.head(100))
    else:
        st.info(f"No CSV files found in {csv_dir}.")

st.markdown("---")
st.write("⚙️ Re-run the pipeline anytime from the sidebar to refresh live data, models, and SHAP visualizations.")
