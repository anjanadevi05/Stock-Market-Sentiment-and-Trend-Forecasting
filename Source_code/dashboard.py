# ==========================
# Imports
# ==========================
import streamlit as st
import pandas as pd
import numpy as np
import os
import subprocess
import time
from joblib import load
import plotly.express as px
from chatbot import marketbot_query


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
st.title("📈 Stock Market Sentiment & Trend Forecasting")

# ==========================
# Sidebar Controls
# ==========================
st.sidebar.header("Controls")
company = st.sidebar.selectbox("Select company", COMPANIES, index=0)
run_pipeline_btn = st.sidebar.button("Run full pipeline (collect → preprocess → features → sentiment → model → SHAP)")
regenerate_shap_btn = st.sidebar.button("Regenerate SHAP only")
predict_button = st.sidebar.button("Generate Next 7 Day Predictions (dynamic)")
show_pred = st.sidebar.checkbox("Show predicted prices (chart & table)", value=True)
show_sentiment = st.sidebar.checkbox("Show sentiment charts", value=True)
show_shap = st.sidebar.checkbox("Show SHAP images", value=True)
show_top5 = st.sidebar.checkbox("Show top-5 SHAP table", value=True)
show_csv = st.sidebar.checkbox("📂 View CSV Files Generated", value=False)

# ==========================
# Utility - Run Script (with virtual env)
# ==========================
def run_script(script_name):
    python_path = r"D:\minor in ai\Stock_Market_Sentiment_&_Trend_Forecasting\venv_mia\Scripts\python.exe"
    script_path = os.path.join(SOURCE_DIR, script_name)
    if not os.path.exists(script_path):
        st.error(f"Script not found: {script_path}")
        return False, f"Missing {script_name}"
    try:
        proc = subprocess.run([python_path, script_path], capture_output=True, text=True, check=False)
        return (proc.returncode == 0), proc
    except Exception as e:
        return False, str(e)

# ==========================
# PIPELINE (includes data_collection.py)
# ==========================
if run_pipeline_btn:
    st.sidebar.info("Running full dynamic pipeline. This may take a few minutes...")
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
        time.sleep(0.5)
    st.success("✅ Full pipeline finished. Data refreshed!")

# ==========================
# Regenerate SHAP only
# ==========================
if regenerate_shap_btn:
    st.sidebar.info("Regenerating SHAP visualizations...")
    ok, out = run_script("shap_implementation.py")
    if not ok:
        st.error("shap_implementation.py failed: " + str(out))
    else:
        st.success("SHAP regeneration finished.")

# ==========================
# Load Features
# ==========================
@st.cache_data
def load_features(company):
    fpath = os.path.join(DATA_DIR, f"{company}_features.csv")
    if not os.path.exists(fpath):
        st.error(f"Features CSV not found: {fpath}. Run full pipeline first.")
        return None
    df = pd.read_csv(fpath)
    return df

df = load_features(company)
if df is None:
    st.stop()

# ==========================
# Predict Next 7 Days
# ==========================
def predict_next_7_days(company, df_features):
    scaler_path = os.path.join(MODEL_DIR, f"{company}_scaler.pkl")
    model_path = os.path.join(MODEL_DIR, f"{company}_rf.pkl")
    if not os.path.exists(scaler_path) or not os.path.exists(model_path):
        st.warning("Model or scaler not found in Models/. Cannot generate predictions.")
        return None

    scaler = load(scaler_path)
    model = load(model_path)

    last_row = df_features.iloc[-1:].copy().reset_index(drop=True)
    preds, dates = [], []

    if 'Date' in df_features.columns:
        last_date = pd.to_datetime(df_features['Date'].iloc[-1])
    else:
        last_date = pd.Timestamp.today()

    for i in range(1, 8):
        X = last_row[FEATURES].values
        X_scaled = scaler.transform(X)
        pred = model.predict(X_scaled)[0]
        preds.append(pred)
        next_date = (last_date + pd.Timedelta(days=i)).date()
        dates.append(next_date)

        new_row = last_row.copy()
        new_row.loc[0, 'Open'] = pred
        new_row.loc[0, 'High'] = pred * 1.01
        new_row.loc[0, 'Low'] = pred * 0.99
        new_row.loc[0, 'Close'] = pred
        new_row.loc[0, 'Volume'] = last_row['Volume'].iloc[0]
        new_row.loc[0, 'Daily_Return'] = 0.0
        new_row.loc[0, 'MA_3'] = pred
        new_row.loc[0, 'MA_7'] = pred
        new_row.loc[0, 'Volatility_7'] = 0.0
        for s in ['sentiment_score_mean', 'Sentiment_MA_3', 'Sentiment_MA_7']:
            if s in new_row.columns:
                new_row.loc[0, s] = last_row[s].iloc[0]
        last_row = new_row

    return pd.DataFrame({"Date": dates, "Predicted_Close": preds})

# ==========================
# Generate Predictions
# ==========================
pred_df = None
if predict_button:
    with st.spinner("Generating next 7-day predictions..."):
        pred_df = predict_next_7_days(company, df)
    if pred_df is not None:
        st.success("✅ Predictions generated successfully.")

if show_pred and pred_df is None:
    pred_df = predict_next_7_days(company, df)

# ==========================
# Visualization: Prices
# ==========================
st.subheader(f"{company.upper()} — Historical & Predicted Prices")

if 'Date' in df.columns:
    df['Date'] = pd.to_datetime(df['Date'])
    fig = px.line(df, x='Date', y='Close', labels={'Close': 'Price'}, title="Historical Close")
    if show_pred and pred_df is not None:
        fig.add_scatter(x=pred_df['Date'], y=pred_df['Predicted_Close'], mode='lines+markers', name='Predicted')
    st.plotly_chart(fig, use_container_width=True)
else:
    st.write("⚠ No 'Date' column in feature CSV.")

if show_pred and pred_df is not None:
    st.subheader("Next 7-Day Predictions")
    st.table(pred_df)

# ==========================
# Sentiment Visualizations
# ==========================
if show_sentiment:
    st.subheader(f"{company.upper()} — Sentiment Overview")
    if 'sentiment_score_mean' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        fig_s = px.line(df, x='Date', y='sentiment_score_mean', title="Daily Avg Sentiment")
        st.plotly_chart(fig_s, use_container_width=True)
        if 'Sentiment_MA_3' in df.columns:
            fig_sma = px.line(df, x='Date', y=['Sentiment_MA_3','Sentiment_MA_7'], title="Sentiment Moving Averages")
            st.plotly_chart(fig_sma, use_container_width=True)
    else:
        st.info("Sentiment columns not found. Run pipeline to generate sentiment.")

# ==========================
# SHAP Explainability
# ==========================
if show_shap:
    st.subheader(f"{company.upper()} — Explainability (SHAP)")
    bar_img = os.path.join(EXPLAIN_DIR, f"{company}_feature_importance_bar.png")
    summary_img = os.path.join(EXPLAIN_DIR, f"{company}_summary_plot.png")
    local_force_img = os.path.join(EXPLAIN_DIR, f"{company}_local_force_plot.png")
    local_decision_img = os.path.join(EXPLAIN_DIR, f"{company}_local_decision_plot.png")

    cols = st.columns(2)
    with cols[0]:
        if os.path.exists(bar_img):
            st.image(bar_img, caption="Global Feature Importance")
        if os.path.exists(local_force_img):
            st.image(local_force_img, caption="Local Force Plot")
    with cols[1]:
        if os.path.exists(summary_img):
            st.image(summary_img, caption="SHAP Summary Plot")
        if os.path.exists(local_decision_img):
            st.image(local_decision_img, caption="Local Decision Plot")

# ==========================
# Top 5 SHAP Features
# ==========================
if show_top5:
    summary_csv = os.path.join(EXPLAIN_DIR, "shap_feature_comparison_summary.csv")
    if os.path.exists(summary_csv):
        df_top5 = pd.read_csv(summary_csv)
        st.subheader("Top-5 SHAP Features (for this company)")
        display_df = df_top5[df_top5['Company'].str.lower() == company.lower()][['Rank','Feature','Mean_SHAP_Value']]
        st.table(display_df)
    else:
        st.info("SHAP comparison summary CSV not found. Run SHAP pipeline to generate it.")

# ==========================
# View CSV Files
# ==========================
if show_csv:
    st.subheader("📂 View CSV Files Generated")
    csv_dir = st.radio("Choose folder:", ["dataset_Used", "Explainability_Results"], horizontal=True)
    folder = DATA_DIR if csv_dir == "dataset_Used" else EXPLAIN_DIR
    csv_files = [f for f in os.listdir(folder) if f.endswith(".csv")]

    if csv_files:
        selected_csv = st.selectbox("Select a CSV file to view:", csv_files)
        file_path = os.path.join(folder, selected_csv)
        df_csv = pd.read_csv(file_path)
        st.write(f"### Preview of {selected_csv}")
        st.dataframe(df_csv.head(50))
    else:
        st.info(f"No CSV files found in {csv_dir}.")

st.write("⚙ You can re-run the pipeline anytime from the sidebar to refresh live data, models, and SHAP visualizations.")

# ==========================
# Chatbot Panel (MarketBot)
# ==========================

st.markdown("---")
st.header("Chat with the MarketBot")

with st.expander("Open Chat Page", expanded=False):
    user_query = st.text_input("Ask MarketBot about stocks, trends, or sentiment:")

    if user_query:
        with st.spinner("Bot is thinking..."):
            response = marketbot_query(user_query)
        st.markdown("**Bot's Response:**")
        st.write(response)
