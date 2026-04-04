import pandas as pd
import joblib
import shap
import numpy as np
import json
from app.db.connection import engine
import warnings
warnings.filterwarnings("ignore")


# Computing SHAP distribution
def compute_shap_distribution(sample_size=1000):

    query = f"SELECT * FROM borrowers LIMIT {sample_size}"
    df = pd.read_sql(query, engine)

    X = df.drop(columns=["default", "borrower_id"], errors="ignore")

    X = df.copy()

    pipeline = joblib.load("app/models/credit_model.pkl")

    if hasattr(pipeline, "calibrated_classifiers_"):
        pipeline = pipeline.calibrated_classifiers_[0].estimator

    model = pipeline.named_steps["model"]
    preprocessor = pipeline.named_steps["preprocessing"]

    X_transformed = preprocessor.transform(X)
    feature_names = preprocessor.get_feature_names_out()

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_transformed)

    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    shap_df = pd.DataFrame(shap_values, columns=feature_names)

    return shap_df

def map_shap_to_signals(shap_df: pd.DataFrame):
    mapping = {
        "historical_default": "historical_default"
    }

    signal_impacts = {
        "historical_default": []
    }

    for col in shap_df.columns:
        col_lower = col.lower()

        for key in mapping:
            if key in col_lower:
                signal = mapping[key]
                signal_impacts[signal].extend(shap_df[col].values.tolist())

    return signal_impacts

def compute_signal_weights(signal_impacts: dict):

    weights = {}

    for signal, values in signal_impacts.items():
        values = np.array(values)

        if len(values) == 0:
            print(f"[WARNING] No SHAP values found for signal: {signal}")
            weights[signal] = 0.0
            continue

        abs_impacts = np.abs(values)

        weight = np.percentile(abs_impacts, 75)

        weights[signal] = round(float(weight), 4)

    return weights

def normalize_weights(weights: dict, target_max=0.30):

    max_weight = max(weights.values())

    if max_weight == 0:
        return weights

    scaled = {}

    for k, v in weights.items():
        scaled[k] = round((v / max_weight) * target_max, 4)

    return scaled


# Generating and saving weights
def generate_signal_weights():

    shap_df = compute_shap_distribution()
    signal_impacts = map_shap_to_signals(shap_df)

    raw_weights = compute_signal_weights(signal_impacts)
    final_weights = normalize_weights(raw_weights)

    print("\nFinal Model Signal Weights:")
    print(final_weights)

    # ✅ Save to config file (IMPORTANT)
    with open("app/config/model_signal_weights.json", "w") as f:
        json.dump(final_weights, f, indent=4)

    return final_weights


if __name__ == "__main__":
    generate_signal_weights()