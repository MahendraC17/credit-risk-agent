import joblib
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from app.db.connection import engine

# Load model pipeline
pipeline = joblib.load("app/models/credit_model.pkl")

# unwrap calibrated model
if hasattr(pipeline, "calibrated_classifiers_"):
    pipeline = pipeline.calibrated_classifiers_[0].estimator

preprocessor = pipeline.named_steps["preprocessing"]

# Load dataset once
def load_reference_data():
    query = "SELECT * FROM borrowers"
    df = pd.read_sql(query, engine)

    y = df["default"]
    X = df.drop(columns=["default", "borrower_id"], errors="ignore")

    return X, y


X_ref, y_ref = load_reference_data()
X_ref_transformed = preprocessor.transform(X_ref)

# Fit KNN once
knn = NearestNeighbors(n_neighbors=30, metric="euclidean")
knn.fit(X_ref_transformed)


def find_similar(applicant_data: dict):

    df = pd.DataFrame([applicant_data])
    df = df.drop(columns=["borrower_id", "default"], errors="ignore")

    X_input = preprocessor.transform(df)

    distances, indices = knn.kneighbors(X_input)

    neighbor_defaults = y_ref.iloc[indices[0]]

    default_rate = neighbor_defaults.mean()

    return {
        "similar_default_rate": round(float(default_rate), 4),
        "neighbor_count": len(indices[0])
    }