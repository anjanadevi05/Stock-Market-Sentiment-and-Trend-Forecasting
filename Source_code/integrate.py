import os
import pandas as pd
from joblib import load

# ==========================
# Paths
# ==========================
BASE_DIR = r"D:\minor in ai\Stock_Market_Sentiment_&_Trend_Forecasting"
SOURCE_DIR = os.path.join(BASE_DIR, "Source_code")
DATA_DIR = os.path.join(BASE_DIR, "dataset_Used")
MODEL_DIR = os.path.join(BASE_DIR, "Models")
EXPLAIN_DIR = os.path.join(BASE_DIR, "Explainability_Results")

os.makedirs(EXPLAIN_DIR, exist_ok=True)

# ==========================
# Companies & Features
# ==========================
COMPANIES = ["tesla", "apple", "amazon"]

FEATURES = [
    'Open', 'High', 'Low', 'Volume', 'Daily_Return',
    'MA_3', 'MA_7', 'Volatility_7',
    'sentiment_score_mean', 'Sentiment_MA_3', 'Sentiment_MA_7'
]

# ==========================
# Helper functions
# ==========================

def load_features(company):
    """
    Load the feature CSV for a given company.
    """
    fpath = os.path.join(DATA_DIR, f"{company}_features.csv")
    if not os.path.exists(fpath):
        return None
    df = pd.read_csv(fpath)
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
    return df


def predict_next_7_days(company, df_features):
    """
    Predict the next 7 days of stock prices using saved Random Forest model & scaler.
    """
    scaler_path = os.path.join(MODEL_DIR, f"{company}_scaler.pkl")
    model_path = os.path.join(MODEL_DIR, f"{company}_rf.pkl")
    if not os.path.exists(scaler_path) or not os.path.exists(model_path):
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
        X_df = pd.DataFrame(X, columns=FEATURES)
        X_scaled = scaler.transform(X_df)
        pred = model.predict(X_scaled)[0]
        preds.append(pred)
        next_date = (last_date + pd.Timedelta(days=i)).date()
        dates.append(next_date)

        # Update last_row for next-step prediction
        new_row = last_row.copy()
        new_row.loc[0, 'Open'] = pred
        new_row.loc[0, 'High'] = pred * 1.01
        new_row.loc[0, 'Low'] = pred * 0.99
        new_row.loc[0, 'Close'] = pred
        if 'Volume' in new_row.columns:
            new_row.loc[0, 'Volume'] = last_row['Volume'].iloc[0]
        new_row.loc[0, 'Daily_Return'] = 0.0
        if 'MA_3' in new_row.columns:
            new_row.loc[0, 'MA_3'] = pred
        if 'MA_7' in new_row.columns:
            new_row.loc[0, 'MA_7'] = pred
        if 'Volatility_7' in new_row.columns:
            new_row.loc[0, 'Volatility_7'] = 0.0
        for s in ['sentiment_score_mean', 'Sentiment_MA_3', 'Sentiment_MA_7']:
            if s in new_row.columns:
                new_row.loc[0, s] = last_row[s].iloc[0]

        last_row = new_row

    return pd.DataFrame({"Date": dates, "Predicted_Close": preds})
