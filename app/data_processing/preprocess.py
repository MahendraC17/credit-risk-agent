import pandas as pd
from app.data_processing.cleaning import clean_credit_data


def preprocess_credit_data(df: pd.DataFrame) -> pd.DataFrame:

    # Renaming columns - align with DB schema
    df = df.rename(columns={
        "person_age": "age",
        "person_income": "income",
        "person_home_ownership": "home_ownership",
        "person_emp_length": "employment_length",
        "loan_intent": "loan_purpose",
        "loan_amnt": "loan_amount",
        "loan_status": "default",
        "cb_person_default_on_file": "historical_default",
        "cb_person_cred_hist_length": "credit_history_length"
    })

    # Dropping unwanted columns
    df = df.drop(columns=["loan_grade", "loan_int_rate", "loan_percent_income"], errors="ignore")

    # Handling missing values
    df["employment_length"] = df["employment_length"].fillna(0)
    df["credit_history_length"] = df["credit_history_length"].fillna(0)

    # Normalizing categorical values
    df["historical_default"] = df["historical_default"].fillna("N")

    # Enforcing data types
    df["age"] = df["age"].astype(int)
    df["income"] = df["income"].astype(float)
    df["loan_amount"] = df["loan_amount"].astype(float)
    df["default"] = df["default"].astype(int)

    df["debt_to_income"] = (df["loan_amount"] / df["income"]).round(2)

    df = df[[
        "age",
        "income",
        "home_ownership",
        "employment_length",
        "loan_purpose",
        "loan_amount",
        'debt_to_income',
        "credit_history_length",
        "historical_default",
        "default"
    ]]

    df = clean_credit_data(df)

    return df