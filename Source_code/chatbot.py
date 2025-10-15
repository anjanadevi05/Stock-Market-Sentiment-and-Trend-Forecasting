# chatbot.py
import os
import re
import pandas as pd
from integrate import COMPANIES, DATA_DIR, MODEL_DIR, EXPLAIN_DIR, load_features, predict_next_7_days

# ---------------------------
# Query Parsing
# ---------------------------
def parse_query(query: str):
    query = query.lower()
    company = None
    for c in COMPANIES:
        if c.lower() in query:
            company = c.lower()
            break

    # Determine intent
    if any(word in query for word in ["predict", "tomorrow", "next day", "next week"]):
        intent = "prediction"
    elif "sentiment" in query:
        intent = "sentiment"
    elif any(word in query for word in ["shap", "feature", "important"]):
        intent = "shap"
    elif any(word in query for word in ["price movement", "change", "why", "fluctuation"]):
        intent = "daily_change"
    else:
        intent = "general"

    return company, intent

# ---------------------------
# Chatbot Response Logic
# ---------------------------
def marketbot_query(user_query: str):
    company, intent = parse_query(user_query)

    if company is None:
        return "I can provide predictions, sentiment, and feature insights for Tesla, Apple, or Amazon. Please mention a company."

    # Load features
    df = load_features(company)
    if df is None:
        return f"Sorry, I don't have data for {company.title()} right now. Try running the pipeline first."

    # Next 7-day predictions
    pred_df = predict_next_7_days(company, df)

    # ------------------------
    # Handle combined prediction + SHAP for "Explain why" type queries
    # ------------------------
    if "shap" in user_query.lower():
        # Tomorrow price
        tomorrow_price = pred_df['Predicted_Close'].iloc[0]

        # SHAP top 5
        shap_file = os.path.join(EXPLAIN_DIR, "shap_feature_comparison_summary.csv")
        shap_text = ""
        if os.path.exists(shap_file):
            df_shap = pd.read_csv(shap_file)
            comp_shap = df_shap[df_shap['Company'].str.lower() == company.lower()]
            if not comp_shap.empty:
                top_features = comp_shap.head(5)[['Feature', 'Mean_SHAP_Value']]
                shap_text = "\n".join([f"{i+1}. {f.Feature}: {f.Mean_SHAP_Value:.4f}"
                                       for i, f in enumerate(top_features.itertuples())])
        if shap_text:
            return (f"The predicted price of {company.title()} stock tomorrow is **{tomorrow_price:.2f} USD**.\n"
                    f"Top 5 features impacting {company.title()}'s predictions:\n{shap_text}")
        else:
            return f"The predicted price of {company.title()} stock tomorrow is **{tomorrow_price:.2f} USD**. SHAP features not available."

    if "shap" in user_query.lower() and any(word in user_query.lower() for word in ["next week", "week"]):
        # Next week predictions as table
        table_md = pred_df[['Date', 'Predicted_Close']].to_markdown(index=False)

        # SHAP top 5
        shap_file = os.path.join(EXPLAIN_DIR, "shap_feature_comparison_summary.csv")
        if os.path.exists(shap_file):
            df_shap = pd.read_csv(shap_file)
            comp_shap = df_shap[df_shap['Company'].str.lower() == company.lower()]
            if not comp_shap.empty:
                top_features = comp_shap.head(5)[['Feature', 'Mean_SHAP_Value']]
                top_features_str = "\n".join([f"{i+1}. {f.Feature}: {f.Mean_SHAP_Value:.4f}"
                                              for i, f in enumerate(top_features.itertuples())])
                return (f"Next 7-day predicted prices for {company.title()}:\n{table_md}\n\n"
                        f"Top 5 features impacting {company.title()}'s predictions:\n{top_features_str}")
        return f"Next 7-day predicted prices for {company.title()}:\n{table_md}\nSHAP features not available."

    # ------------------------
    # Individual intents
    # ------------------------
    if intent == "prediction":
        if any(word in user_query.lower() for word in ["next week", "week"]):
            table_md = pred_df[['Date', 'Predicted_Close']].to_markdown(index=False)
            return f"Next 7-day predicted prices for {company.title()}:\n{table_md}"
        else:
            tomorrow_price = pred_df['Predicted_Close'].iloc[0]
            return f"The predicted price of {company.title()} stock tomorrow is **{tomorrow_price:.2f} USD**."

    elif intent == "daily_change":
        if 'Close' in df.columns and len(df) >= 2:
            change = df['Close'].iloc[-1] - df['Close'].iloc[-2]
            sentiment = df['sentiment_score_mean'].iloc[-1] if 'sentiment_score_mean' in df.columns else 0
            # Determine sentiment label
            if sentiment > 0.05:
                sentiment_label = "Positive"
            elif sentiment < -0.05:
                sentiment_label = "Negative"
            else:
                sentiment_label = "Neutral"
            explanation = "daily fluctuations"
            if abs(sentiment) > 0.05:
                explanation += f" influenced by recent news sentiment ({sentiment_label})"
            return (f"{company.title()}'s price changed by **{change:.2f} USD** today. "
                    f"Sentiment score: {sentiment:.2f} ({sentiment_label}). Likely cause: {explanation}.")
        else:
            return f"Not enough historical data to determine daily change for {company.title()}."

    elif intent == "sentiment":
        if 'sentiment_score_mean' in df.columns:
            if any(word in user_query.lower() for word in ["today"]):
                sentiment_today = df['sentiment_score_mean'].iloc[-1]
                if sentiment_today > 0.05:
                    label = "Positive"
                elif sentiment_today < -0.05:
                    label = "Negative"
                else:
                    label = "Neutral"
                return f"The sentiment score for {company.title()} today is **{sentiment_today:.2f}** ({label})."
            else:
                avg_sentiment = df['sentiment_score_mean'].iloc[-7:].mean()
                if avg_sentiment > 0.05:
                    label = "Positive"
                elif avg_sentiment < -0.05:
                    label = "Negative"
                else:
                    label = "Neutral"
                return f"The average sentiment for {company.title()} over the last 7 days is **{avg_sentiment:.2f}** ({label})."
        else:
            return f"Sentiment data for {company.title()} is not available."

    elif intent == "shap" and company:
        shap_csv = os.path.join(EXPLAIN_DIR, "shap_feature_comparison_summary.csv")
        if os.path.exists(shap_csv):
            df_top5 = pd.read_csv(shap_csv)
            comp_rows = df_top5[df_top5['Company'].str.lower() == company.lower()]
            if not comp_rows.empty:
                top5 = comp_rows[['Feature', 'Mean_SHAP_Value']].head(5)
                response_lines = [f"Top 5 features impacting {company.title()}'s predictions:"]
                for idx, row in enumerate(top5.itertuples(), 1):
                    response_lines.append(f"{idx}. {row.Feature}: {row.Mean_SHAP_Value:.4f}")
                return "\n".join(response_lines)
            else:
                return f"No SHAP data found for {company.title()}."
        else:
            return "SHAP CSV file not found. Run shap_implementation.py first."

    else:
        return ("Hi! I can help you with Tesla, Apple, and Amazon stock data. "
                "Ask me about price predictions, sentiment trends, daily price changes, or feature importance.")
