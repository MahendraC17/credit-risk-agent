import joblib
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from app.db.connection import engine

pipeline = joblib.load("app/models/credit_model.pkl")

if hasattr(pipeline, "calibrated_classifiers_"):
    pipeline = pipeline.calibrated_classifiers_[0].estimator

preprocessor = pipeline.named_steps["preprocessing"]

def load_reference_data():
    query = "SELECT * FROM borrowers"
    df = pd.read_sql(query, engine)

    y = df["default"]
    X = df.drop(columns=["default", "borrower_id"], errors="ignore")

    return X, y


X_ref, y_ref = load_reference_data()
X_ref_transformed = preprocessor.transform(X_ref)

scaler = StandardScaler(with_mean=False)
X_ref_scaled = scaler.fit_transform(X_ref_transformed)

knn = NearestNeighbors(n_neighbors=75, metric="cosine")
knn.fit(X_ref_scaled)

def find_similar(applicant_data: dict):

    df = pd.DataFrame([applicant_data])
    df = df.drop(columns=["borrower_id", "default"], errors="ignore")

    X_input = preprocessor.transform(df)
    X_input = scaler.transform(X_input)

    distances, indices = knn.kneighbors(X_input)

    eps = 1e-6
    d = distances[0]

    # inverse distance weights
    weights = 1 / (d + 0.1)

    neighbor_defaults = y_ref.iloc[indices[0]].values

    w_sum = weights.sum() + 1e-8
    w_sq_sum = (weights ** 2).sum() + 1e-8

    # weighted mean
    weighted_mean = (weights * neighbor_defaults).sum() / w_sum
    weighted_var = (weights * (neighbor_defaults - weighted_mean) ** 2).sum() / w_sum

    n_eff = (w_sum ** 2) / w_sq_sum
    
    weighted_std = np.sqrt(weighted_var)

    if weighted_std < 0.01:
        weighted_std = 0.01


    confidence_band = 1.96 * (weighted_std / np.sqrt(n_eff))
    confidence_band = max(confidence_band, 0.02)

    return {
    "mean": round(float(weighted_mean), 4),
    "std": round(float(weighted_std), 4),
    "count": len(indices[0]),
    "effective_n": round(float(n_eff), 2),
    "confidence_band": round(float(confidence_band), 4)
    }

    # neighbor_defaults = y_ref.iloc[indices[0]].values

    # mean = np.mean(neighbor_defaults)
    # std = np.std(neighbor_defaults)
    # count = len(neighbor_defaults)

    # margin = 1.96 * (std / np.sqrt(count)) if count > 0 else 0

    # lower = max(0, mean - margin)
    # upper = min(1, mean + margin)

#     return {
#         "mean": round(float(mean), 4),
#         "std": round(float(std), 4),
#         "count": count,
#         "confidence_band": [round(float(lower), 4), round(float(upper), 4)]
# }