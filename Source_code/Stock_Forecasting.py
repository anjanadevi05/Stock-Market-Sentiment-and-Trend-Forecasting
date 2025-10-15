import pandas as pd
import joblib
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

companies = ["tesla", "apple", "amazon"]

# ----------------------------
# Load and Predict Function
# ----------------------------
def load_model_and_predict(company, input_df):
    # Load scaler
    scaler_path = f"Models/{company}_scaler.pkl"
    scaler = joblib.load(scaler_path)

    # Load RF model
    model_path =  f"Models/{company}_rf.pkl"
    model = joblib.load(model_path)

    # Scale input
    features = [
        'Open', 'High', 'Low', 'Volume', 'Daily_Return',
        'MA_3', 'MA_7', 'Volatility_7',
        'sentiment_score_mean', 'Sentiment_MA_3', 'Sentiment_MA_7'
    ]
    X = input_df[features]
    X_scaled = scaler.transform(X)

    # Predict
    predictions = model.predict(X_scaled)
    return predictions

# ----------------------------
# Example usage for all companies
# ----------------------------
for company in companies:
    # Load your merged features dataset
    input_file =f"Dataset_Used/{company}_features.csv"
    df = pd.read_csv(input_file).dropna()

    preds = load_model_and_predict(company, df)
    df['Predicted_Close'] = preds

    # Evaluate locally
    rmse = np.sqrt(mean_squared_error(df['Close'], df['Predicted_Close']))
    r2 = r2_score(df['Close'], df['Predicted_Close'])
    accuracy = max(0, min(100, r2 * 100))

    print(f"\n{company.upper()} Random Forest Predictions")
    print(f"RMSE: {rmse:.2f}, R²: {r2:.3f}, Accuracy: {accuracy:.2f}%")
    print(df[['Date', 'Close', 'Predicted_Close']].tail(5))


#Stock Prediction for Multiple Companies (Next 7 Days)

def load_model_and_predict_next_week(company, df):
    # Paths
    scaler_path = f"Models/{company}_scaler.pkl"
    model_path = f"Models/{company}_rf.pkl"

    # Load scaler and model
    scaler = joblib.load(scaler_path)
    model = joblib.load(model_path)

    features = [
        'Open', 'High', 'Low', 'Volume', 'Daily_Return',
        'MA_3', 'MA_7', 'Volatility_7',
        'sentiment_score_mean', 'Sentiment_MA_3', 'Sentiment_MA_7'
    ]
    
    # Start with last row for iterative prediction
    last_row = df.iloc[-1:].copy()
    predictions = []

    # Get next 7 dates
    last_date = pd.to_datetime(df['Date'].iloc[-1])
    next_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=7)

    for i in range(7):  # Predict next 7 days
        X_scaled = scaler.transform(last_row[features])
        next_close = model.predict(X_scaled)[0]
        predictions.append({'Date': next_dates[i].date(), 'Predicted_Close': next_close})

        # Prepare next row for next iteration
        new_row = last_row.copy()
        new_row['Open'] = next_close
        new_row['High'] = next_close * 1.01
        new_row['Low'] = next_close * 0.99
        new_row['Close'] = next_close
        new_row['Volume'] = last_row['Volume'].values[0]
        new_row['Daily_Return'] = 0
        new_row['MA_3'] = next_close
        new_row['MA_7'] = next_close
        new_row['Volatility_7'] = 0
        new_row['sentiment_score_mean'] = last_row['sentiment_score_mean'].values[0]
        new_row['Sentiment_MA_3'] = last_row['Sentiment_MA_3'].values[0]
        new_row['Sentiment_MA_7'] = last_row['Sentiment_MA_7'].values[0]

        last_row = new_row

    return predictions

# ----------------------------
# Run for all companies
# ----------------------------
for company in companies:
    input_file = f"Dataset_Used/{company}_features.csv"
    df = pd.read_csv(input_file).dropna()
    next_week_preds = load_model_and_predict_next_week(company, df)

    print(f"\n Next 7 days predicted closing prices for {company.upper()}:")
    for pred in next_week_preds:
        print(f"{pred['Date']}: ${pred['Predicted_Close']:.2f}")
