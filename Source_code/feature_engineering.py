import pandas as pd

companies = ["tesla", "apple", "amazon"]

for company in companies:
    # Load merged data
    df = pd.read_csv(f"Dataset_Used/{company}_merged.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Sort by date just to be safe
    df = df.sort_values('Date').reset_index(drop=True)
    
    # ====== 1️⃣ PRICE-BASED FEATURES ======
    
    # Daily return percentage
    df['Daily_Return'] = df['Close'].pct_change() * 100
    
    # 3-day and 7-day moving averages (MA)
    df['MA_3'] = df['Close'].rolling(window=3).mean()
    df['MA_7'] = df['Close'].rolling(window=7).mean()
    
    # 7-day rolling volatility (std dev of returns)
    df['Volatility_7'] = df['Daily_Return'].rolling(window=7).std()
    
    # Lag features (previous day's values)
    df['Lag_1_Close'] = df['Close'].shift(1)
    df['Lag_1_Sentiment'] = df['sentiment_score_mean'].shift(1)
    
    # ====== 2️⃣ SENTIMENT-BASED FEATURES ======
    
    # 3-day and 7-day rolling sentiment averages
    df['Sentiment_MA_3'] = df['sentiment_score_mean'].rolling(window=3).mean()
    df['Sentiment_MA_7'] = df['sentiment_score_mean'].rolling(window=7).mean()
    
    # Sentiment volatility (how varied recent sentiments are)
    df['Sentiment_Volatility_3'] = df['sentiment_score_mean'].rolling(window=3).std()
    
    # ====== 3️⃣ CLEANING ======
    # Drop rows with NaN due to rolling calculations
    df = df.dropna().reset_index(drop=True)
    
    # Save processed file
    output_file = f"Dataset_Used/{company}_features.csv"
    df.to_csv(output_file, index=False)
    print(f"Feature engineered data saved to {output_file}")
