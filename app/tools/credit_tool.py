from app.models.predict import predict_risk
from app.models.explain import explain_prediction
from app.decision.context import build_context
from app.decision.signals import extract_signals, aggregate_signals, make_decision, classify_risk_band

def evaluate_applicant(applicant_data: dict) -> dict:

    def normalize_feature_name(feature: str):
        if " = " in feature:
            base, value = feature.split(" = ", 1)
            return base.strip(), value.strip()
        return feature.strip(), None


    def filter_key_drivers(drivers: list, top_n: int = 3):
        IMPORTANT_FEATURES = {
            "debt_to_income",
            "loan_amount",
            "income",
            "credit_history_length",
            "historical_default"
        }

        filtered = []

        for d in drivers:
            base_feature, value = normalize_feature_name(d["feature"])

            if base_feature in IMPORTANT_FEATURES:
                filtered.append({
                    "feature": base_feature,
                    "value": value,  # keep semantic info
                    "impact": d["impact"],
                    "effect": d["effect"]
                })

        filtered = sorted(filtered, key=lambda x: abs(x["impact"]), reverse=True)

        seen = set()
        unique = []

        for d in filtered:
            if d["feature"] not in seen:
                unique.append(d)
                seen.add(d["feature"])

        if len(unique) < 2:
            return drivers[:3]

    risk_score = predict_risk(applicant_data)

    drivers = explain_prediction(applicant_data)
    key_drivers = filter_key_drivers(drivers)

    context = build_context(applicant_data, risk_score, "TEMP", key_drivers)

    signals = extract_signals(context)

    risk_profile = aggregate_signals(signals)
    final_risk = risk_profile["final_risk"]

    risk_level = classify_risk_band(final_risk)

    context = build_context(applicant_data, final_risk, risk_level, key_drivers)

    decision, reason = make_decision(risk_profile, context)

    return {
        "risk_score": round(risk_score, 4),
        "risk_level": risk_level,
        "decision": decision,
        "decision_reason": reason,
        "signals": signals,
        "risk_breakdown": risk_profile,
        "key_drivers": key_drivers,

        "debug": {
            "base_risk": risk_profile["base_risk"],
            "adjustment": risk_profile["adjustment"],
            "final_risk": risk_profile["final_risk"],
            "signals": signals
                }
    }

if __name__ == "__main__":
    from app.db.queries import fetch_applicant

    applicant = fetch_applicant(3)

    result = evaluate_applicant(applicant)

    for i in range(1, 60):
        applicant = fetch_applicant(i)
        result = evaluate_applicant(applicant)

        print(f"\nApplicant {i}:")
        print("Risk Level:", result["risk_level"])
        print("Decision:", result["decision"])
        print("Risk Breakdown:", result["risk_breakdown"])
        print("Signals:", result["signals"])
        print("Debug:", result["debug"])