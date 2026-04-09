import json
from langchain_openai import ChatOpenAI
from app.db.queries import fetch_applicant
from app.tools.credit_tool import evaluate_applicant


llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)


def run_agent(borrower_id: int):

    applicant = fetch_applicant(borrower_id)

    if not applicant:
        raise ValueError("Applicant not found")

    result = evaluate_applicant(applicant)

    agent_input = {
        "decision": {
            "label": result["decision"],
            "reason": result["decision_reason"]
        },

        "risk": {
            "score": result["risk_score"],
            "level": result["risk_level"]
        },

        "drivers": result["key_drivers"],

        "financial": {
            "income": applicant.get("income"),
            "loan_amount": applicant.get("loan_amount"),
            "dti": applicant.get("debt_to_income")
        },

        "behavior": {
            "historical_default": applicant.get("historical_default")
        },

        "validation": {
            "model_risk": result["consistency_check"]["model_risk"],
            "neighbor_risk": result["consistency_check"]["neighbor_risk"],
            "gap": result["consistency_check"]["gap"],
            "disagreement": result["consistency_check"]["flag"]
        },

        "confidence": result["confidence"]
    }

    prompt = f"""
    You are a credit risk analyst.

    STRICT RULES:
    - Use ONLY the provided data
    - DO NOT assume missing values
    - DO NOT restate raw numbers without interpretation
    - If disagreement is present → explicitly explain it
    - If confidence is low → recommend manual review

    You must return a VALID JSON with this structure:
    Return ONLY raw JSON.
    Do NOT wrap in markdown.
    Do NOT add explanations.

    {{
    "summary": "...",
    "risk_factors": ["...", "..."],
    "financial_analysis": "...",
    "behavioral_analysis": "...",
    "validation_analysis": "...",
    "confidence_explanation": "...",
    "improvements": ["...", "..."],
    "final_recommendation": "..."
    }}

    DATA:
    {agent_input}
    """
    

    response = llm.invoke(prompt)

    raw = response.content.strip()

    if raw.startswith("```"):
        raw = raw.replace("```json", "").replace("```", "").strip()

    try:
        explanation = json.loads(raw)

    except Exception as e:
        explanation = {
            "summary": "Unable to parse structured response",
            "risk_factors": [],
            "financial_analysis": raw,
            "behavioral_analysis": "",
            "validation_analysis": "",
            "confidence_explanation": "",
            "improvements": [],
            "final_recommendation": "Manual review required"
        }

    return {
        "structured_output": result,
        "agent_explanation": explanation
    }