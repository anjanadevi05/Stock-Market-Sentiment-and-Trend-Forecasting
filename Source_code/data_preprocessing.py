import pandas as pd

# List of companies and their corresponding CSV prefixes
companies = {
    "TSLA": "Tesla",
    "AAPL": "Apple",
    "AMZN": "Amazon"
}

combined_data = {}

for ticker, name in companies.items():
    # Load stock prices
    stock_file = f"Dataset_Used/{ticker}_stock_data.csv"
    stock_df = pd.read_csv(stock_file)
    stock_df['Date'] = pd.to_datetime(stock_df['Date']).dt.date
    
    # Load sentiment data
    sentiment_file = f"Dataset_Used/{name.lower()}_sentiment.csv"
    sentiment_df = pd.read_csv(sentiment_file)
    sentiment_df['publishedAt'] = pd.to_datetime(sentiment_df['publishedAt']).dt.date
    
    # Aggregate sentiment per day
    daily_sentiment = sentiment_df.groupby('publishedAt').agg(
        sentiment_score_mean=('sentiment_score', 'mean'),
        positive_count=('sentiment_label', lambda x: (x=="POSITIVE").sum()),
        neutral_count=('sentiment_label', lambda x: (x=="NEUTRAL").sum()),
        negative_count=('sentiment_label', lambda x: (x=="NEGATIVE").sum())
    ).reset_index().rename(columns={'publishedAt':'Date'})
    
    # Merge with stock data on Date
    merged_df = pd.merge(stock_df, daily_sentiment, on='Date', how='left')
    
    # Fill missing sentiment with 0
    merged_df[['sentiment_score_mean','positive_count','neutral_count','negative_count']] = \
        merged_df[['sentiment_score_mean','positive_count','neutral_count','negative_count']].fillna(0)
    
    # Save merged file
    merged_file = f"Dataset_Used/{name.lower()}_merged.csv"
    merged_df.to_csv(merged_file, index=False)
    print(f"{ticker}: Merged data saved to {merged_file}")
    
    combined_data[ticker] = merged_df
