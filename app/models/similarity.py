import joblib
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
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

knn = NearestNeighbors(n_neighbors=30, metric="euclidean")
knn.fit(X_ref_transformed)


def find_similar(applicant_data: dict):

    df = pd.DataFrame([applicant_data])
    df = df.drop(columns=["borrower_id", "default"], errors="ignore")

    X_input = preprocessor.transform(df)

    distances, indices = knn.kneighbors(X_input)

    neighbor_defaults = y_ref.iloc[indices[0]].values

    mean = np.mean(neighbor_defaults)
    std = np.std(neighbor_defaults)
    count = len(neighbor_defaults)

    margin = 1.96 * (std / np.sqrt(count)) if count > 0 else 0

    lower = max(0, mean - margin)
    upper = min(1, mean + margin)

    return {
        "mean": round(float(mean), 4),
        "std": round(float(std), 4),
        "count": count,
        "confidence_band": [round(float(lower), 4), round(float(upper), 4)]
}