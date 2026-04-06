import math
import json

with open("app/config/model_signal_weights.json") as f:
    MODEL_WEIGHTS = json.load(f)

RISK_BANDS = {
    "moderate": 0.4,
    "high": 0.65,
    "very_high": 0.85
}

SIGNAL_CONFIG = {
    "historical_default": {
        "type": "model",
        "weight": MODEL_WEIGHTS["historical_default"]
    },
    "high_dti": {
        "type": "policy",
        "weight": 0.15
    },
    "moderate_dti": {
        "type": "policy",
        "weight": 0.05
    }
}

def classify_risk_band(score: float) -> str:
    if score >= RISK_BANDS["very_high"]:
        return "Very High"
    elif score >= RISK_BANDS["high"]:
        return "High"
    elif score >= RISK_BANDS["moderate"]:
        return "Moderate"
    else:
        return "Low"
    

def extract_signals(context: dict) -> list:
    signals = []

    # BASE MODEL SIGNAL
    signals.append({
        "name": "model_risk",
        "type": "base",
        "strength": context["risk_score"],
        "direction": "negative"
    })

    # POSITIVE FINANCIAL SIGNAL
    if context.get("is_low_dti"):
        signals.append({
            "name": "low_dti",
            "type": "financial",
            "strength": 0.08,
            "direction": "positive"
        })

    # FINANCIAL SIGNALS
    if context["is_high_dti"]:
        signals.append({
            "name": "high_dti",
            "type": "financial",
            "strength": 0.20,
            "direction": "negative"
        })

    elif context["is_moderate_dti"]:
        signals.append({
            "name": "moderate_dti",
            "type": "financial",
            "strength": 0.10,
            "direction": "negative"
        })

    # BEHAVIORAL / PAST DEAFULT SIGNAL
    if context["historical_default"] == "Y":
        signals.append({
            "name": "historical_default",
            "type": "behavioral",
            "strength": SIGNAL_CONFIG["historical_default"]["weight"],
            "direction": "negative"
        })
    
    return signals


def aggregate_signals(signals: list) -> dict:
    base_risk = 0
    adjustment = 0

    for s in signals:
        if s["type"] == "base":
            base_risk = s["strength"]
        else:
            if s["direction"] == "negative":
                adjustment += s["strength"]
            else:
                adjustment -= s["strength"]

    final_risk = 1 - (1 - base_risk) * math.exp(-adjustment)
    return {
        "base_risk": round(base_risk, 4),
        "adjustment": round(adjustment, 4),
        "final_risk": round(final_risk, 4)
    }


def make_decision(risk_profile: dict, context: dict) -> tuple:
    score = risk_profile["final_risk"]

    if score >= 0.88:
        return "Reject", "Extremely high calibrated risk"
    
    elif score >= RISK_BANDS["very_high"]:
        if context["historical_default"] == "Y":
            return "Reject", "Extreme risk with prior default"
        return "Reject or require collateral", "Extreme predicted risk"

    elif score >= RISK_BANDS["high"]:
        if context["is_high_dti"] or context["historical_default"] == "Y":
            return "Reject or require collateral", "Strong negative signals"
        return "Approve with strict conditions", "High risk but no critical triggers"

    elif score >= RISK_BANDS["moderate"]:
        return "Approve with conditions", "Moderate risk"

    else:
        return "Approve", "Low risk"