def build_context(applicant, risk_score, risk_level, drivers):

    income = applicant.get("income", 0)
    loan_amount = applicant.get("loan_amount", 0)
    dti = applicant.get("debt_to_income", 0)

    return {
        "risk_score": risk_score,
        "risk_level": risk_level,

        "income": income,
        "loan_amount": loan_amount,

        "dti": dti,

        "is_high_dti": dti > 0.5,
        "is_moderate_dti": 0.4 < dti <= 0.5,
        "is_low_dti": dti < 0.25,

        "historical_default": applicant.get("historical_default", "N"),
        "credit_history_length": applicant.get("credit_history_length", 0),

        "drivers": drivers
    }