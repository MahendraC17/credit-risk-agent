# --------------------------------------------------------------------------------
# Decision Logic Layer
# Extracting interpretable signals, aggregating them into adjusted risk,
# and making final underwriting decisions
# --------------------------------------------------------------------------------

import math
import json
from app.config.config_loader import CONFIG

# --------------------------------------------------------------------------------
# Loading model driven signal weights
# --------------------------------------------------------------------------------
with open("app/config/model_signal_weights.json") as f:
    MODEL_WEIGHTS = json.load(f)


# --------------------------------------------------------------------------------
# Risk thresholds and buffer zones
# Used for band classification and decision stability handling
# --------------------------------------------------------------------------------
RISK_BANDS = CONFIG["risk"]["thresholds"]
BUFFER = CONFIG["risk"]["buffer"]


# --------------------------------------------------------------------------------
# Signal configuration
# Defining how different signals contribute to final adjustment
# --------------------------------------------------------------------------------
SIGNAL_CONFIG = {
    "historical_default": {
        "type": "model",
        "weight": MODEL_WEIGHTS["historical_default"]
    },
    "high_dti": {
        "type": "policy",
        "weight": CONFIG["signals"]["high_dti"]
    },
    "moderate_dti": {
        "type": "policy",
        "weight": CONFIG["signals"]["moderate_dti"]
    }
}


# --------------------------------------------------------------------------------
# Risk Band Classification
# Mapping probability score into interpretable risk levels
# --------------------------------------------------------------------------------
def classify_risk_band(score: float):
    if score >= RISK_BANDS["very_high"]:
        return "Very High"
    elif score >= RISK_BANDS["high"]:
        return "High"
    elif score >= RISK_BANDS["moderate"]:
        return "Moderate"
    else:
        return "Low"


# --------------------------------------------------------------------------------
# Probability - Log-Odds Conversion
# Using log-odds space to combine signals additively
# --------------------------------------------------------------------------------
def prob_to_log_odds(p):
    eps = 1e-6
    p = min(max(p, eps), 1 - eps)
    return math.log(p / (1 - p))


def log_odds_to_prob(lo):
    return 1 / (1 + math.exp(-lo))


# --------------------------------------------------------------------------------
# Signal Extraction
# Converting context into structured signals representing risk drivers
# --------------------------------------------------------------------------------
def extract_signals(context: dict):

    signals = []

    # Base model signal
    signals.append({
        "name": "model_risk",
        "type": "base",
        "strength": context["risk_score"],
        "direction": "negative"
    })

    # Positive signal -low financial burden
    if context.get("is_low_dti"):
        signals.append({
            "name": "low_dti",
            "type": "financial",
            "strength": 0.02,
            "direction": "positive"
        })

    # Negative financial signals
    if context["is_high_dti"]:
        signals.append({
            "name": "high_dti",
            "type": "financial",
            "strength": 0.08,
            "direction": "negative"
        })

    elif context["is_moderate_dti"]:
        signals.append({
            "name": "moderate_dti",
            "type": "financial",
            "strength": 0.04,
            "direction": "negative"
        })

    # Behavioral signal -past default
    if context["historical_default"] == "Y":
        signals.append({
            "name": "historical_default",
            "type": "behavioral",
            "strength": SIGNAL_CONFIG["historical_default"]["weight"],
            "direction": "negative"
        })

    return signals


# --------------------------------------------------------------------------------
# Signal Aggregation
# Combining model output with signals in log-odds space to adjust risk
# --------------------------------------------------------------------------------
def aggregate_signals(signals: list) -> dict:

    # Extracting base model probability
    base_risk = next((s["strength"] for s in signals if s["type"] == "base"), 0)

    base_log_odds = prob_to_log_odds(base_risk)

    adjustment = 0

    for s in signals:
        if s["type"] == "base":
            continue

        delta = s["strength"]

        if s["direction"] == "negative":
            adjustment += delta
        else:
            adjustment -= delta

    # ----------------------------------------------------
    # Cap adjustment to prevent signal dominance
    # ----------------------------------------------------
    MAX_ADJUSTMENT = 0.2   # key control knob

    if adjustment > MAX_ADJUSTMENT:
        adjustment = MAX_ADJUSTMENT
    elif adjustment < -MAX_ADJUSTMENT:
        adjustment = -MAX_ADJUSTMENT

    final_log_odds = base_log_odds + adjustment
    final_risk = log_odds_to_prob(final_log_odds)

    return {
        "base_risk": round(base_risk, 4),
        "adjustment": round(adjustment, 4),
        "final_risk": round(final_risk, 4)
    }


# --------------------------------------------------------------------------------
# Decision Engine
# Translating final risk into actionable underwriting decision
# using thresholds and stability buffers
# --------------------------------------------------------------------------------
def make_decision(risk_profile: dict, context: dict):

    score = risk_profile["final_risk"]

    high = RISK_BANDS["high"]
    very_high = RISK_BANDS["very_high"]
    moderate = RISK_BANDS["moderate"]

    # Extreme rejection zone
    if score >= 0.88:
        return "Reject", "Extremely high calibrated risk"

    # Very high risk -stable region
    elif score >= very_high + BUFFER:
        if context["historical_default"] == "Y":
            return "Reject", "Extreme risk with prior default"
        return "Reject or require collateral", "Extreme predicted risk"

    # Very high buffer zone -uncertain region
    elif very_high - BUFFER <= score < very_high + BUFFER:
        return "Reject or require collateral", "Borderline extreme risk (stability buffer)"

    # High risk -stable region
    elif score >= high + BUFFER:
        if context["is_high_dti"] or context["historical_default"] == "Y":
            return "Reject or require collateral", "Strong negative signals"
        return "Approve with strict conditions", "High risk but controlled"

    # High buffer zone
    elif high - BUFFER <= score < high + BUFFER:
        return "Approve with strict conditions", "Borderline high risk (stability buffer)"

    # Moderate risk -stable
    elif score >= moderate + BUFFER:
        return "Approve with conditions", "Moderate risk"

    # Moderate buffer zone
    elif moderate - BUFFER <= score < moderate + BUFFER:
        return "Approve with conditions", "Borderline moderate risk (stability buffer)"

    # Low risk
    else:
        return "Approve", "Low risk"