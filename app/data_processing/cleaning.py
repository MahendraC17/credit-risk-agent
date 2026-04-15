# --------------------------------------------------------------------------------
# Simple Data cleaning before prreprocessing
# --------------------------------------------------------------------------------

import pandas as pd

def clean_credit_data(df: pd.DataFrame) -> pd.DataFrame:

    df["employment_length"] = df["employment_length"].clip(0, 50)
    df["employment_length"] = df["employment_length"].fillna(0)

    df["credit_history_length"] = df["credit_history_length"].clip(0, 100)
    df["credit_history_length"] = df["credit_history_length"].fillna(0)

    df["income"] = df["income"].clip(lower=1000)

    df["loan_amount"] = df["loan_amount"].clip(lower=500)

    df["historical_default"] = df["historical_default"].fillna("N")

    df["home_ownership"] = df["home_ownership"].str.upper()
    df["loan_purpose"] = df["loan_purpose"].str.upper()

    return df