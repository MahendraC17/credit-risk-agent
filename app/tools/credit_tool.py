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

def compute_sensitivity(final_risk: float):
    thresholds = [0.4, 0.65, 0.85]

    distances = {t: abs(final_risk - t) for t in thresholds}

    closest_threshold = min(distances, key=distances.get)
    distance = distances[closest_threshold]

    flip_risk = distance < 0.03

    return {
        "distance_to_threshold": round(distance, 4),
        "closest_threshold": closest_threshold,
        "flip_risk": flip_risk
    }

def classify_disagreement(z_gap: float):

    if z_gap > 3:
        return "Severe"
    elif z_gap > 2:
        return "High"
    elif z_gap > 1:
        return "Moderate"
    else:
        return "Low"
    

def compute_tension(signals, model_risk, similarity):
    
    positive = 0
    negative = 0

    for s in signals:
        if s["direction"] == "positive":
            positive += s["strength"]
        elif s["direction"] == "negative":
            negative += s["strength"]

    signal_tension = min(positive, negative)

    external_gap = abs(model_risk - similarity["mean"])

    tension_score = 0.7 * external_gap + 0.3 * signal_tension

    if tension_score > 0.5:
        level = "High"
    elif tension_score > 0.25:
        level = "Medium"
    else:
        level = "Low"

    return {
        "score": round(tension_score, 4),
        "level": level,
        "components": {
            "signal_conflict": round(signal_tension, 4),
            "model_vs_similarity_gap": round(external_gap, 4)
        }
    }

def get_risk_profile(applicant_data: dict):

    risk_score = predict_risk(applicant_data)

    drivers = explain_prediction(applicant_data)
    key_drivers = drivers[:3]

    context = build_context(applicant_data, risk_score, "TEMP", key_drivers)
    signals = extract_signals(context)

    risk_profile = aggregate_signals(signals)
    final_risk = risk_profile["final_risk"]

    risk_level = classify_risk_band(final_risk)

    decision, reason = make_decision(risk_profile, context)

    return {
        "risk_score": risk_score,
        "risk_level": risk_level,
        "decision": decision,
        "decision_reason": reason,
        "risk_breakdown": risk_profile,
        "signals": signals,
        "key_drivers": key_drivers
    }

def get_similarity_analysis(applicant_data: dict):
    return find_similar(applicant_data)

def get_decision_diagnostics(applicant_data: dict):

    risk_score = predict_risk(applicant_data)
    similarity = find_similar(applicant_data)

    gap = abs(risk_score - similarity["mean"])
    std = similarity["std"] + 1e-6
    z_gap = gap / std

    disagreement_level = classify_disagreement(z_gap)
    override_flag = z_gap > 3

    consistency = {
        "model_risk": risk_score,
        "neighbor_risk": similarity["mean"],
        "gap": gap,
        "z_gap": z_gap,
        "flag": z_gap > 1,
        "disagreement_level": disagreement_level,
        "override_flag": override_flag
    }

    # reuse existing functions
    context = build_context(applicant_data, risk_score, "TEMP", [])
    signals = extract_signals(context)
    risk_profile = aggregate_signals(signals)

    final_risk = risk_profile["final_risk"]

    confidence = compute_confidence(
        risk_score,
        similarity,
        risk_profile["adjustment"],
        final_risk
    )

    sensitivity = compute_sensitivity(final_risk)

    tension = compute_tension(signals, risk_score, similarity)

    return {
        "consistency_check": consistency,
        "confidence": confidence,
        "sensitivity": sensitivity,
        "tension": tension
    }

def run_scenario_analysis(applicant_data: dict):
    return simulate_to_threshold(applicant_data, target_risk=0.65)


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

    std = similarity["std"] + 1e-6
    z_gap = gap / std

    disagreement_level = classify_disagreement(z_gap)
    override_flag = z_gap > 3

    consistency = {
    "model_risk": round(model_risk, 4),
    "neighbor_risk": round(similar_risk, 4),
    "gap": round(gap, 4),
    "z_gap": round(z_gap, 4),  
    "flag": z_gap > 1,        
    "disagreement_level": disagreement_level,
    "override_flag": override_flag
    }

    drivers = explain_prediction(applicant_data)

    key_drivers = filter_key_drivers(drivers) or []

    context = build_context(applicant_data, risk_score, "TEMP", key_drivers)

    signals = extract_signals(context)

    risk_profile = aggregate_signals(signals)

    final_risk = risk_profile["final_risk"]

    risk_level = classify_risk_band(final_risk)

    context = build_context(applicant_data, final_risk, risk_level, key_drivers)

    decision, reason = make_decision(risk_profile, context)

    sensitivity = compute_sensitivity(final_risk)

    confidence = compute_confidence(model_risk, similarity, risk_profile["adjustment"], risk_profile["final_risk"])

    context = build_context(applicant_data, final_risk, risk_level, key_drivers)

    tension = compute_tension(signals, model_risk, similarity)

    override = consistency["override_flag"]
    disagreement = consistency["disagreement_level"]
    confidence_level = confidence["level"]

    escalation = None

    if override:
        escalation = "REVIEW_REQUIRED"

    elif confidence_level == "Low":
        escalation = "MANUAL_REVIEW"

    elif sensitivity["flip_risk"]:
        escalation = "BORDERLINE_REVIEW"

    else:
        escalation = "AUTO_DECISION"

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
        "sensitivity": sensitivity,
        "tension": tension,
        "escalation": escalation,

    }

if __name__ == "__main__":
    from app.db.queries import fetch_applicant

    def classify_case(result):
        gap = result["consistency_check"]["gap"]
        flip = result["sensitivity"]["flip_risk"]
        tension = result["tension"]["score"]
        confidence = result["confidence"]["level"]

        if gap > 0.25:
            return "MODEL_DISAGREEMENT"

        if flip:
            return "BOUNDARY_CASE"

        if tension > 0.3:
            return "HIGH_TENSION"

        if confidence == "Low":
            return "LOW_CONFIDENCE"

        return "NORMAL"
    
    interesting_cases = []

    for i in range(1, 200):
        applicant = fetch_applicant(i)
        if not applicant:
            continue

        result = evaluate_applicant(applicant)

        case_type = classify_case(result)

        if case_type != "NORMAL":
            interesting_cases.append((i, case_type, result))

    for i, case_type, result in interesting_cases:

        print("\n" + "="*60)
        print(f"Applicant {i} → {case_type}")
        print("="*60)

        print("Risk:", result["risk_level"], "| Score:", result["risk_score"])
        print("Decision:", result["decision"])

        print("\n--- Sensitivity ---")
        print(result["sensitivity"])

        print("\n--- Tension ---")
        print(result["tension"])

        print("\n--- Consistency ---")
        c = result["consistency_check"]

        print("Model Risk:", c["model_risk"])
        print("Neighbor Risk:", c["neighbor_risk"])
        print("Gap:", c["gap"])
        print("Disagreement Level:", c["disagreement_level"])
        print("Override Flag:", c["override_flag"])

        print("\n--- Confidence ---")
        print(result["confidence"])

    from collections import Counter

    counts = Counter([case_type for _, case_type, _ in interesting_cases])

    print("\n\n===== SUMMARY =====")
    for k, v in counts.items():
        print(f"{k}: {v}")

