# Explainable AI - SHAP for Random Forest Models
import pandas as pd
import numpy as np
import shap
import os
import matplotlib.pyplot as plt
from joblib import load
import webbrowser

# Companies to analyze
companies = ["tesla", "apple", "amazon"]

# Directory paths
base_model_path = "Models"
base_data_path = "Dataset_Used"
output_path = "Explainability_Results"

os.makedirs(output_path, exist_ok=True)

# Features used in training
features = [
    'Open', 'High', 'Low', 'Volume', 'Daily_Return',
    'MA_3', 'MA_7', 'Volatility_7',
    'sentiment_score_mean', 'Sentiment_MA_3', 'Sentiment_MA_7'
]


def explain_with_shap(company):
    print(f"\nRunning SHAP Explainability for {company.upper()}...")

    # Load model and scaler
    model = load(os.path.join(base_model_path, f"{company}_rf.pkl"))
    scaler = load(os.path.join(base_model_path, f"{company}_scaler.pkl"))

    # Load dataset
    df = pd.read_csv(os.path.join(base_data_path, f"{company}_features.csv")).dropna()
    X = df[features]
    y = df["Close"]
    X_scaled = scaler.transform(X)

    # Initialize SHAP Explainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_scaled)

    # ===== GLOBAL EXPLAINABILITY =====
    # Compute mean absolute SHAP value for ranking
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    shap_importance = pd.DataFrame({
        'Feature': features,
        'Mean_SHAP_Value': mean_abs_shap
    }).sort_values(by='Mean_SHAP_Value', ascending=False)

    # Save top 5 for comparison summary later
    shap_importance['Company'] = company.upper()
    top5 = shap_importance.head(5).copy()
    top5['Rank'] = np.arange(1, len(top5) + 1)

    # --- Feature Importance (Bar Plot)
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X, feature_names=features, plot_type="bar", show=False)
    plt.title(f"Feature Importance - {company.upper()}")
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, f"{company}_feature_importance_bar.png"))
    plt.close()

    # --- SHAP Summary Plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X, feature_names=features, show=False)
    plt.title(f"SHAP Summary Plot - {company.upper()}")
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, f"{company}_summary_plot.png"))
    plt.close()

    # ===== LOCAL EXPLAINABILITY =====
    row_index = -1
    shap_value_row = shap_values[row_index]
    feature_values = X.iloc[row_index].copy()

    # Round off moving averages for readability
    for col in ['MA_3', 'MA_7', 'Sentiment_MA_3', 'Sentiment_MA_7']:
        if col in feature_values:
            feature_values[col] = round(feature_values[col], 3)

    # ---- Static Local Force Plot
    shap_contrib = pd.DataFrame({
        "Feature": features,
        "Feature Value": feature_values.values,
        "SHAP Value": shap_value_row
    }).sort_values("SHAP Value", ascending=False)

    plt.figure(figsize=(9, 6))
    colors = shap_contrib["SHAP Value"].apply(lambda x: 'red' if x > 0 else 'blue')
    plt.barh(shap_contrib["Feature"], shap_contrib["SHAP Value"], color=colors)
    plt.xlabel("SHAP Value (Impact on Prediction)")
    plt.title(f"Local Explanation (Feature Push Up/Down) - {company.upper()}")
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, f"{company}_local_force_plot.png"))
    plt.close()

    # ---- Local Decision Plot
    plt.figure(figsize=(10, 5))
    shap.decision_plot(
        explainer.expected_value,
        shap_values[row_index],
        feature_values,
        feature_names=features,
        show=False
    )
    plt.title(f"Local Decision Plot - {company.upper()}")
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, f"{company}_local_decision_plot.png"))
    plt.close()

    print(f"SHAP visualizations saved for {company.upper()} in '{output_path}' folder.")
    return top5


# Run for all companies and create comparison summary CSV
if __name__ == "__main__":
    print("Starting Explainable AI Analysis using SHAP...\n")

    all_top5 = []
    for comp in companies:
        top5_df = explain_with_shap(comp)
        all_top5.append(top5_df)

    comparison_df = pd.concat(all_top5, ignore_index=True)
    comparison_df.to_csv(os.path.join(output_path, "shap_feature_comparison_summary.csv"), index=False)

    print("\nAll explainability visualizations generated successfully!")
    print(f"Outputs saved in: {output_path}")
