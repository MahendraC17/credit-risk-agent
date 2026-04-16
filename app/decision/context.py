# --------------------------------------------------------------------------------
# Context Builder
# Constructing a singular context bridge combining raw inputs, derived features,
# and model outputs for downstream decision logic and signal extraction
# --------------------------------------------------------------------------------

def build_context(applicant, risk_score, risk_level, drivers):
    from app.config.config_loader import CONFIG
    dti_cfg = CONFIG["dti"]
    
    # Extracting key financial inputs with safe defaults
    income = applicant.get("income", 0)
    loan_amount = applicant.get("loan_amount", 0)
    dti = applicant.get("debt_to_income", 0)

    # Returning a context dictionary used across decision layer
    return {
        "risk_score": risk_score,
        "risk_level": risk_level,

        # Core financial attributes
        "income": income,
        "loan_amount": loan_amount,

        # Debt-to-income ratios
        "dti": dti,

        # Deriving simple rule-based flags for decision logic
        # These act as interpretable features for signal generatio

        "is_high_dti": dti > dti_cfg["high"],
        "is_moderate_dti": dti_cfg["moderate"] < dti <= dti_cfg["high"],
        "is_low_dti": dti < dti_cfg["low"],

        # Behavioral and historical attributes
        "historical_default": applicant.get("historical_default", "N"),
        "credit_history_length": applicant.get("credit_history_length", 0),

        # Top contributing features from SHAP explanation
        "drivers": drivers
    }