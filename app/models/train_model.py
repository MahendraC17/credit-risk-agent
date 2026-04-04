import pandas as pd
import joblib
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import numpy as np
from sqlalchemy import create_engine
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


def load_data():
    query = "SELECT * FROM borrowers"
    df = pd.read_sql(query, engine)
    return df


def build_pipeline(X: pd.DataFrame):
    categorical_cols = X.select_dtypes(include=["object"]).columns
    numeric_cols = X.select_dtypes(exclude=["object"]).columns

    preprocessor = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ("num", "passthrough", numeric_cols)
    ])

    model = LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        class_weight="balanced"
    )

    pipeline = Pipeline([
        ("preprocessing", preprocessor),
        ("model", model)
    ])

    return pipeline


def train():
    df = load_data()
    # print(df.columns)

    y = df["default"]

    X = df.drop(columns=["default", "borrower_id"])
    X = X.drop(columns=["loan_grade","interest_rate", "debt_to_income"], errors="ignore")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    pipeline = build_pipeline(X)

    calibrated_model = CalibratedClassifierCV(
        pipeline,
        method="sigmoid", 
        cv=3
    )

    calibrated_model.fit(X_train, y_train)

    preds = calibrated_model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, preds)

    print(f"AUC Score: {auc:.4f}")

    joblib.dump(calibrated_model, "app/models/credit_model.pkl")

    print("Model saved successfully.")

    prob_true, prob_pred = calibration_curve(y_test, preds, n_bins=10)


    percentiles = np.percentile(preds, [50, 75, 85, 90, 95])

    bands = {
        "low": percentiles[1],     
        "moderate": percentiles[2],  
        "high": percentiles[3],       
        "very_high": percentiles[4]  
    }

    print("Learned Risk Bands:", bands)
    
    # plt.plot(prob_pred, prob_true, marker='o')
    # plt.plot([0, 1], [0, 1], linestyle='--')
    # plt.xlabel("Predicted Probability")
    # plt.ylabel("Actual Probability")
    # plt.title("Calibration Curve")
    # plt.show()


if __name__ == "__main__":
    train()