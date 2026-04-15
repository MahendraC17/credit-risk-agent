# --------------------------------------------------------------------------------
# SHAP Based Signal Calibration
# Computing feature impact distributions and deriving signal weights
# for use in decision logic
# --------------------------------------------------------------------------------

import pandas as pd
import joblib
import shap
import numpy as np
import json
from app.db.connection import engine
import warnings
warnings.filterwarnings("ignore")


# --------------------------------------------------------------------------------
# SHAP Distribution Computation
# Computing SHAP values across a sample of borrowers
# --------------------------------------------------------------------------------
def compute_shap_distribution(sample_size=1000):

    # Loading sample
    query = f"SELECT * FROM borrowers LIMIT {sample_size}"
    df = pd.read_sql(query, engine)

    # Removing non feature columns
    X = df.drop(columns=["default", "borrower_id"], errors="ignore")

    pipeline = joblib.load("app/models/credit_model.pkl")

    # Handling calibrated model wrappers
    if hasattr(pipeline, "calibrated_classifiers_"):
        pipeline = pipeline.calibrated_classifiers_[0].estimator

    model = pipeline.named_steps["model"]
    preprocessor = pipeline.named_steps["preprocessing"]

    # Applying preprocessing
    X_transformed = preprocessor.transform(X)
    feature_names = preprocessor.get_feature_names_out()

    # Computing SHAP values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_transformed)

    # Handling binary classification output
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    # Creating dataframe for analysis
    shap_df = pd.DataFrame(shap_values, columns=feature_names)

    return shap_df


# --------------------------------------------------------------------------------
# Mapping SHAP Values to Signals
# Grouping feature level impacts into signal level contributions
# --------------------------------------------------------------------------------
def map_shap_to_signals(shap_df: pd.DataFrame):

    # Mapping model features to decision signals
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

                # Collecting all SHAP values for this signal
                signal_impacts[signal].extend(shap_df[col].values.tolist())

    return signal_impacts


# --------------------------------------------------------------------------------
# Signal Weight Computation
# Deriving representative weights from SHAP distributions
# --------------------------------------------------------------------------------
def compute_signal_weights(signal_impacts: dict):

    weights = {}

    for signal, values in signal_impacts.items():
        values = np.array(values)

        if len(values) == 0:
            print(f"[WARNING] No SHAP values found for signal: {signal}")
            weights[signal] = 0.0
            continue

        # Using 75th percentile of absolute impact as weight
        abs_impacts = np.abs(values)
        weight = np.percentile(abs_impacts, 75)

        weights[signal] = round(float(weight), 4)

    return weights


# --------------------------------------------------------------------------------
# Weight Normalization
# Scaling weights to a consistent range for use in signal aggregation
# --------------------------------------------------------------------------------
def normalize_weights(weights: dict, target_max=0.30):

    max_weight = max(weights.values())

    if max_weight == 0:
        return weights

    scaled = {}

    for k, v in weights.items():
        scaled[k] = round((v / max_weight) * target_max, 4)

    return scaled


# --------------------------------------------------------------------------------
# Weight Generation Pipeline
# Running full calibration and saving weights to config file
# --------------------------------------------------------------------------------
def generate_signal_weights():

    shap_df = compute_shap_distribution()
    signal_impacts = map_shap_to_signals(shap_df)

    raw_weights = compute_signal_weights(signal_impacts)
    final_weights = normalize_weights(raw_weights)

    print("\nFinal Model Signal Weights:")
    # print(final_weights)

    # Saving calibrated weights to config file
    with open("app/config/model_signal_weights.json", "w") as f:
        json.dump(final_weights, f, indent=4)

    return final_weights


if __name__ == "__main__":
    generate_signal_weights()