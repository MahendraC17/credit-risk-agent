import json
from langchain_openai import ChatOpenAI
from app.db.queries import fetch_applicant
from app.tools.credit_tool import evaluate_applicant, simulate_to_threshold

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

def run_agent(borrower_id: int):

    applicant = fetch_applicant(borrower_id)

    if not applicant:
        raise ValueError("Applicant not found")

    result = evaluate_applicant(applicant)

    decision_prompt = f"""
    You are deciding what additional analysis is needed.

    Available tools:
    - scenario → use when risk is High/Very High or improvement is needed
    - similarity → use when disagreement is True or confidence is Low
    - none → if explanation is straightforward

    Rules:
    - If confidence is Low → use both
    - If disagreement is True → include similarity
    - If risk is High or Very High → include scenario
    - Otherwise → none

    Return ONLY one of:
    ["scenario", "similarity", "both", "none"]

    DATA:
    {{
    "risk_level": "{result["risk_level"]}",
    "confidence": "{result["confidence"]["level"]}",
    "disagreement": {result["consistency_check"]["flag"]}
    }}
    """

    tool_decision = llm.invoke(decision_prompt).content.strip().lower()
    tool_decision = tool_decision.replace('"', '').replace("'", "")

    scenario_results = []

    if tool_decision in ["scenario", "both"]:
        scenario = simulate_to_threshold(applicant, target_risk=0.65)

        if scenario:
            scenario_results.append({
                "goal": "Move to Moderate Risk",
                "required_reduction_pct": scenario["reduction_pct"],
                "new_loan_amount": scenario["new_loan"],
                "new_risk": scenario["new_risk"],
                "new_decision": scenario["new_decision"]
        })
        else:
            scenario_results.append({
                "goal": "Move to Moderate Risk",
                "result": "Not achievable within reasonable limits"
            })

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

    if scenario_results:
        agent_input["scenarios"] = scenario_results

    include_scenario = bool(scenario_results)

    if include_scenario:
        scenario_field = '"scenario_analysis": "...",'
    else:
        scenario_field = ''

    prompt = f"""
    You are a credit risk analyst. Identify the underlying risk theme (e.g., affordability, behavioral risk).

    STRICT RULES:
    - Use ONLY the provided data
    - DO NOT assume missing values
    - DO NOT restate raw numbers without interpretation
    - If disagreement is present → explicitly explain it
    - If confidence is low → recommend manual review

    Allowed improvement actions are LIMITED to:
    - reducing loan amount
    - increasing declared income (without assuming how)

    Do NOT suggest:
    - side jobs
    - employment changes
    - lifestyle changes
    - vague financial advice

    If scenarios are provided:
    - Do NOT include improvement suggestions in final_recommendation
    - Keep final_recommendation focused on the current decision
    - Use scenario_analysis for improvement guidance
    - Explain what change is required to reach a safer risk level
    - Quantify the required adjustment
    - Map the new risk to a risk band
    - Explain how the decision changes
    - Do NOT use vague language

    If no scenarios are provided:
    - Do NOT suggest improvements
    - Do NOT mention hypothetical changes

    Return ONLY valid JSON.
    Do NOT wrap in markdown.
    Do NOT include explanations outside JSON.

    The JSON must follow this structure:

    {{
    "summary": "...",
    "risk_factors": ["...", "..."],
    "financial_analysis": "...",
    "behavioral_analysis": "...",
    "validation_analysis": "...",
    "confidence_explanation": "...",
    {scenario_field}
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
            "final_recommendation": "Manual review required",
            "scenario_analysis": ""
        }

    return {
        "structured_output": result,
        "agent_explanation": explanation
    }