import joblib
import pandas as pd
from app.data_processing.cleaning import clean_credit_data

model = joblib.load("app/models/credit_model.pkl")

def prepare_input(applicant_data: dict) -> pd.DataFrame:
    df = pd.DataFrame([applicant_data])

    df = df.drop(columns=["borrower_id", "default"], errors="ignore")

    df = df.drop(columns=[
        "loan_grade",
        "interest_rate",
        "debt_to_income"
    ], errors="ignore")

    df = clean_credit_data(df)

    return df


def predict_risk(applicant_data: dict) -> float:
    df = prepare_input(applicant_data)
    prob = model.predict_proba(df)[0][1]
    return float(prob)

if __name__ == "__main__":
    from app.db.queries import fetch_multiple_applicants

    applicants = fetch_multiple_applicants(15)

    for i, applicant in enumerate(applicants, 1):
        risk = predict_risk(applicant)
        # result = evaluate_applicant(applicant)
        # print(result)

        # print(f"\nApplicant {i}:")
        # print(applicant)
        # print("Risk score:", round(risk, 4))