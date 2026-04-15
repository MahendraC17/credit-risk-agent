# --------------------------------------------------------------------------------
# Prediction Layer
# Handles input preparation and model inference for credit risk scoring
# --------------------------------------------------------------------------------

import joblib
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
from app.data_processing.cleaning import clean_credit_data

# Loading trained model once at startup
model = joblib.load("app/models/credit_model.pkl")


# --------------------------------------------------------------------------------
# Input Preparation
# Converting raw applicant data into model ready format
# --------------------------------------------------------------------------------
def prepare_input(applicant_data: dict):

    df = pd.DataFrame([applicant_data])

    df = df.drop(columns=["borrower_id", "default"], errors="ignore")

    # Dropping columns excluded during model training
    # Ensuring consistency between training and inference
    df = df.drop(columns=[
        "loan_grade",
        "interest_rate",
        "debt_to_income"
    ], errors="ignore")

    # Applying the same preprocessing pipeline used during training
    df = clean_credit_data(df)

    return df


# --------------------------------------------------------------------------------
# Risk Prediction
# Running model inference and returns probability of default
# --------------------------------------------------------------------------------
def predict_risk(applicant_data: dict):

    df = prepare_input(applicant_data)

    # Model outputs probability for both classes -> take class 1 (default)
    prob = model.predict_proba(df)[0][1]

    return float(prob)


if __name__ == "__main__":
    from app.db.queries import fetch_multiple_applicants

    # applicants = fetch_multiple_applicants(15)
    # for i, applicant in enumerate(applicants, 1):
    #     risk = predict_risk(applicant)
    #     print(f"Applicant: {i}, Risk: {risk}")