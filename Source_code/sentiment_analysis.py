import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from tqdm import tqdm
import os
from huggingface_hub import login

# Login to Hugging Face Hub
login(token="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")

# Model & tokenizer initialization
model_name = "yiyanghkust/finbert-tone"  # FinBERT pretrained for finance
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Create sentiment analysis pipeline
sentiment_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer, device=0)

# Function for sentiment analysis over list of texts
def analyze_sentiment(texts):
    results = []
    for t in tqdm(texts, desc="Processing Sentiment"):
        if pd.isna(t) or t.strip() == "":
            results.append({"label": "NEUTRAL", "score": 0.0})
            continue
        res = sentiment_pipeline(t[:512])[0]  # Truncate texts longer than 512 tokens
        results.append(res)
    return results

# CSV files to process
files = {
    "Tesla": "Dataset_Used/Tesla_news.csv",
    "Apple": "Dataset_Used/Apple_news.csv",
    "Amazon": "Dataset_Used/Amazon_news.csv"
}

# Store DataFrames with sentiments
sentiment_dfs = {}

for company, file in files.items():
    df = pd.read_csv(file)
    print(f"\nProcessing {company} data: {len(df)} rows")

    # Combine title + description to form input text
    df['text'] = df.apply(
        lambda x: str(x['title']) + ". " + str(x['description']) if pd.notna(x['description']) else str(x['title']),
        axis=1
    )

    # Convert 'publishedAt' column to datetime type
    df['publishedAt'] = pd.to_datetime(df['publishedAt'])

    # Run sentiment analysis
    sentiments = analyze_sentiment(df['text'].tolist())

    # Add sentiment results to DataFrame
    df['sentiment_label'] = [s['label'] for s in sentiments]
    df['sentiment_score'] = [s['score'] for s in sentiments]

    # Save updated CSV
    output_file = f"Dataset_Used/{company.lower()}_sentiment.csv"
    df.to_csv(output_file, index=False)
    print(f"Saved sentiment data to {output_file}")

    sentiment_dfs[company] = df
