from app.models.predict import predict_risk
from app.models.explain import explain_prediction
from app.decision.context import build_context
from app.decision.signals import extract_signals, aggregate_signals, make_decision, classify_risk_band
from app.models.similarity import find_similar

def compute_confidence(model_risk, similarity, adjustment, final_risk):
    
    # CONSISTENCY
    gap = abs(model_risk - similarity["mean"])
    consistency_score = max(0, 1 - gap * 2)

    # SIGNAL DOMINANCE
    adjustment_penalty = min(abs(adjustment), 0.5)
    signal_score = 1 - adjustment_penalty

    # DECISION STABILITY
    thresholds = [0.4, 0.65, 0.85] 
    distance_to_boundary = min([abs(final_risk - t) for t in thresholds])

    stability_score = min(distance_to_boundary * 4, 1)  

    # SIMILARITY UNCERTAINTY
    std = similarity["std"]
    uncertainty_penalty = min(std * 1.5, 0.8)
    similarity_score = 1 - uncertainty_penalty

    # FINAL SCORE
    confidence_score = (
        0.5 * consistency_score +    
        0.15 * signal_score +        
        0.15 * stability_score +     
        0.2 * similarity_score
    )

    confidence_score = max(0, min(confidence_score, 1))

    # LEVEL
    if confidence_score > 0.7:
        level = "High"
    elif confidence_score > 0.5:
        level = "Medium"
    else:
        level = "Low"

    # STABILITY LABEL
    if final_risk > 0.85 or final_risk < 0.3:
        stability = "Stable" 

    elif distance_to_boundary < 0.03:
        stability = "Fragile"

    else:
        stability = "Moderate"

    return {
        "score": round(confidence_score, 4),
        "level": level,
        "stability": stability
    }

# def simulate_applicant_change(applicant_data: dict, changes: dict) -> dict:
#     modified = applicant_data.copy()
#     modified.update(changes)

#     return evaluate_applicant(modified)

def simulate_to_threshold(applicant_data: dict, target_risk: float, step: float = 0.05):

    base_loan = applicant_data["loan_amount"]
    current = applicant_data.copy()

    for reduction in range(5, 60, 5):
        new_loan = base_loan * (1 - reduction / 100)

        current["loan_amount"] = new_loan
        result = evaluate_applicant(current)

        if result["risk_breakdown"]["final_risk"] <= target_risk:
            return {
                "reduction_pct": reduction,
                "new_loan": round(new_loan, 2),
                "new_risk": result["risk_breakdown"]["final_risk"],
                "new_decision": result["decision"]
            }

    return None 

def evaluate_applicant(applicant_data: dict) -> dict:

    def filter_key_drivers(drivers: list, top_n: int = 3):

        IMPORTANT_FEATURES = {
            "debt_to_income",
            "loan_amount",
            "income",
            "credit_history_length",
            "historical_default"
        }

        def normalize_feature_name(feature: str):
            if " = " in feature:
                base, value = feature.split(" = ", 1)
                return base.strip(), value.strip()
            return feature.strip(), None

        priority = []
        secondary = []

        for d in drivers:
            base_feature, value = normalize_feature_name(d["feature"])
            base_feature = base_feature.lower()
            
            item = {
                "feature": base_feature if not value else f"{base_feature} = {value}",
                "value": (value if value is not None 
                                else applicant_data.get(base_feature, "N/A")),
                "impact": d["impact"],
                "effect": d["effect"]
            }

            if base_feature in IMPORTANT_FEATURES:
                priority.append(item)
            else:
                secondary.append(item)

        combined = priority + secondary

        combined = sorted(combined, key=lambda x: abs(x["impact"]), reverse=True)

        seen = set()
        unique = []

        for d in combined:
            if d["feature"] not in seen:
                unique.append(d)
                seen.add(d["feature"])

        if len(unique) == 0:
            return drivers[:top_n]

        return unique[:top_n]

    risk_score = predict_risk(applicant_data)

    similarity = find_similar(applicant_data)

    model_risk = risk_score
    similar_risk = similarity["mean"]

    gap = abs(model_risk - similar_risk)

    consistency = {
        "model_risk": round(model_risk, 4),
        "neighbor_risk": round(similar_risk, 4),
        "gap": round(gap, 4),
        "flag": gap > 0.25
    }

    drivers = explain_prediction(applicant_data)

    key_drivers = filter_key_drivers(drivers) or []

    context = build_context(applicant_data, risk_score, "TEMP", key_drivers)

    signals = extract_signals(context)

    risk_profile = aggregate_signals(signals)

    final_risk = risk_profile["final_risk"]

    confidence = compute_confidence(model_risk, similarity, risk_profile["adjustment"], risk_profile["final_risk"])

    risk_level = classify_risk_band(final_risk)

    context = build_context(applicant_data, final_risk, risk_level, key_drivers)

    decision, reason = make_decision(risk_profile, context)

    return {
        "risk_score": round(risk_score, 4),
        "similarity": similarity,
        "risk_level": risk_level,
        "decision": decision,
        "decision_reason": reason,
        "signals": signals,
        "risk_breakdown": risk_profile,
        "key_drivers": key_drivers,
        "consistency_check": consistency,
        "confidence": confidence,

        # "debug": {
        #     "base_risk": risk_profile["base_risk"],
        #     "adjustment": risk_profile["adjustment"],
        #     "final_risk": risk_profile["final_risk"],
        #     "signals": signals
        #         }
    }

if __name__ == "__main__":
    from app.db.queries import fetch_applicant

    # applicant = fetch_applicant(3)

    # result = evaluate_applicant(applicant)

    for i in range(20, 30):
        applicant = fetch_applicant(i)
        result = evaluate_applicant(applicant)

        print(f"\nApplicant {i}:")
        print("Risk Level:", result["risk_level"])
        print("Simmilarity:", result["similarity"])
        print("Decision:", result["decision"])
        print("Risk Breakdown:", result["risk_breakdown"])
        print("Signals:", result["signals"])
        print("Consistenct check:", result["consistency_check"])
        print("Confidence:", result["confidence"])

        # print("Debug:", result["debug"])