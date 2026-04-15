# --------------------------------------------------------------------------------
# Similarity Validation Layer
# Finding similar historical applicants and estimating default risk using
# distance weighted KNN with uncertainty estimation
# --------------------------------------------------------------------------------

import joblib
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from app.db.connection import engine


# --------------------------------------------------------------------------------
# Loading model pipeline and extracting preprocessing step
# Ensuring using the same feature transformation as the model
# --------------------------------------------------------------------------------
pipeline = joblib.load("app/models/credit_model.pkl")

if hasattr(pipeline, "calibrated_classifiers_"):
    pipeline = pipeline.calibrated_classifiers_[0].estimator

preprocessor = pipeline.named_steps["preprocessing"]


# --------------------------------------------------------------------------------
# Loading reference dataset from database
# Using historical borrowers as comparison base
# --------------------------------------------------------------------------------
def load_reference_data():
    query = "SELECT * FROM borrowers"
    df = pd.read_sql(query, engine)

    y = df["default"]
    X = df.drop(columns=["default", "borrower_id"], errors="ignore")

    return X, y


# Preparing reference dataset once
X_ref, y_ref = load_reference_data()

# Applying same preprocessing as model
X_ref_transformed = preprocessor.transform(X_ref)

# Scaling transformed features for distance computation
scaler = StandardScaler(with_mean=False)
X_ref_scaled = scaler.fit_transform(X_ref_transformed)

# Initializing KNN
knn = NearestNeighbors(n_neighbors=75, metric="cosine")
knn.fit(X_ref_scaled)


# --------------------------------------------------------------------------------
# Similarity Search
# Finding nearest neighbors and estimating risk using weighted statistics
# --------------------------------------------------------------------------------
def find_similar(applicant_data: dict):

    df = pd.DataFrame([applicant_data])
    df = df.drop(columns=["borrower_id", "default"], errors="ignore")

    X_input = preprocessor.transform(df)
    X_input = scaler.transform(X_input)

    distances, indices = knn.kneighbors(X_input)

    d = distances[0]

    # Applying inverse distance weighting
    # That is closer neighbors contribute more
    weights = 1 / (d + 0.1)

    neighbor_defaults = y_ref.iloc[indices[0]].values

    # Computing weighted statistics
    w_sum = weights.sum() + 1e-8
    w_sq_sum = (weights ** 2).sum() + 1e-8

    # Weighted mean default probability
    weighted_mean = (weights * neighbor_defaults).sum() / w_sum

    # Weighted variance and standard deviation
    weighted_var = (weights * (neighbor_defaults - weighted_mean) ** 2).sum() / w_sum
    weighted_std = np.sqrt(weighted_var)

    # Preventing unrealistically low variance
    if weighted_std < 0.01:
        weighted_std = 0.01

    n_eff = (w_sum ** 2) / w_sq_sum

    # Computing confidence band around estimate
    confidence_band = 1.96 * (weighted_std / np.sqrt(n_eff))
    confidence_band = max(confidence_band, 0.02)

    return {
        "mean": round(float(weighted_mean), 4),
        "std": round(float(weighted_std), 4),
        "count": len(indices[0]),
        "effective_n": round(float(n_eff), 2),
        "confidence_band": round(float(confidence_band), 4)
    }