# --------------------------------------------------------------------------------
# Model Training Pipeline
# Training calibrated LightGBM model and deriving risk bands from predictions
# --------------------------------------------------------------------------------

import pandas as pd
import joblib
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from sklearn.calibration import CalibratedClassifierCV
from lightgbm import LGBMClassifier
from app.db.connection import engine
import warnings
warnings.filterwarnings("ignore")


# --------------------------------------------------------------------------------
# Data Loading
# Fetching borrower data from database
# --------------------------------------------------------------------------------
def load_data():
    query = "SELECT * FROM borrowers"
    df = pd.read_sql(query, engine)
    return df


# --------------------------------------------------------------------------------
# Pipeline Construction
# Defining preprocessing + model in a single pipeline
# --------------------------------------------------------------------------------
def build_pipeline(X: pd.DataFrame):

    # Identifying categorical and numerical columns
    categorical_cols = X.select_dtypes(include=["object"]).columns
    numeric_cols = X.select_dtypes(exclude=["object"]).columns

    # Applying one-hot encoding for categorical features
    preprocessor = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ("num", "passthrough", numeric_cols)
    ])

    # The Model
    model = LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        class_weight="balanced"
    )

    # Combining preprocessing and model
    pipeline = Pipeline([
        ("preprocessing", preprocessor),
        ("model", model)
    ])

    return pipeline


# --------------------------------------------------------------------------------
# Training Process
# Splitting data, training calibrated model, evaluating performance,
# and saving model artifact
# --------------------------------------------------------------------------------
def train():

    df = load_data()

    # Target variable
    y = df["default"]

    X = df.drop(columns=["default", "borrower_id"])
    X = X.drop(columns=["loan_grade", "interest_rate", "debt_to_income"], errors="ignore")

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipeline = build_pipeline(X)

    # Applying probability calibration for better probability estimates
    calibrated_model = CalibratedClassifierCV(
        pipeline,
        method="sigmoid",
        cv=3
    )

    calibrated_model.fit(X_train, y_train)

    # Evaluating model performance
    preds = calibrated_model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, preds)

    print(f"AUC Score: {auc:.4f}")

    # Saving trained model
    joblib.dump(calibrated_model, "app/models/credit_model.pkl")
    print("Model saved successfully.")


    # --------------------------------------------------------------------------------
    # Risk Band Calibration
    # Deriving thresholds based on prediction distribution
    # --------------------------------------------------------------------------------
    prob_true, prob_pred = calibration_curve(y_test, preds, n_bins=10)
    # Will be using this in future updates

    percentiles = np.percentile(preds, [50, 75, 85, 90, 95])

    bands = {
        "low": percentiles[1],
        "moderate": percentiles[2],
        "high": percentiles[3],
        "very_high": percentiles[4]
    }

    print("Learned Risk Bands:", bands)


if __name__ == "__main__":
    train()