import yfinance as yf
import pandas as pd
import requests
import os
from datetime import datetime, timedelta

def fetch_stock_data(ticker, period="6mo", interval="1d", save_csv=True):
    """
    Fetch historical stock data using yfinance
    :param ticker: stock symbol (e.g., 'AAPL')
    :param period: lookback period (1d, 1mo, 6mo, 1y, 5y)
    :param interval: data interval ('1d', '1h', '5m')
    :param save_csv: if True, save data in /data folder
    :return: pandas DataFrame
    """
    data = yf.Ticker(ticker)
    hist = data.history(period=period, interval=interval)
    hist.reset_index(inplace=True)
    
    if save_csv:
        os.makedirs("../Dataset_Used", exist_ok=True)
        filename = f"../Dataset_Used/{ticker}_stock_data.csv"
        hist.to_csv(filename, index=False)
        print(f"Saved stock data to {filename}")
    
    return hist


def fetch_news(company, from_date=None, to_date=None, api_key="hfk", save_csv=True):
    """
    Fetch financial news using NewsAPI
    :param company: company name or keyword
    :param from_date: YYYY-MM-DD (default: 7 days ago)
    :param to_date: YYYY-MM-DD (default: today)
    :param api_key: NewsAPI key
    :param save_csv: if True, save data in /data
    :return: pandas DataFrame
    """
    if from_date is None:
        from_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
    if to_date is None:
        to_date = datetime.now().strftime("%Y-%m-%d")
    
    url = f"https://newsapi.org/v2/everything?q={company}&from={from_date}&to={to_date}&sortBy=publishedAt&apiKey={api_key}&language=en"
    response = requests.get(url)
    news_data = response.json()
    
    articles = news_data.get("articles", [])
    news_list = []
    for article in articles:
        news_list.append({
            "source": article["source"]["name"],
            "title": article["title"],
            "description": article["description"],
            "url": article["url"],
            "publishedAt": article["publishedAt"]
        })
    
    df = pd.DataFrame(news_list)
    
    if save_csv:
        os.makedirs("../Dataset_Used", exist_ok=True)
        filename = f"../Dataset_Used/{company}_news.csv"
        df.to_csv(filename, index=False)
        print(f"Saved news data to {filename}")
    
    return df

COMPANIES = [
    {"ticker": "TSLA", "name": "Tesla"},
    {"ticker": "AAPL", "name": "Apple"},
    {"ticker": "AMZN", "name": "Amazon"}
]

if __name__ == "__main__":
    NEWS_API_KEY = "784d6d3a6df94b188296d2d686b62ce3"

    for company in COMPANIES:
        print(f"Fetching stock data for {company['name']}...")
        df_stock = fetch_stock_data(company["ticker"], period="6mo")
        
        print(f"Fetching news for {company['name']}...")
        df_news = fetch_news(company["name"], api_key=NEWS_API_KEY)
