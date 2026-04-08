import json
from langchain_openai import ChatOpenAI
from app.db.queries import fetch_applicant
from app.tools.credit_tool import evaluate_applicant


llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)


def run_agent(borrower_id: int):

    # 1. Fetch applicant
    applicant = fetch_applicant(borrower_id)

    if not applicant:
        raise ValueError("Applicant not found")

    result = evaluate_applicant(applicant)

    filtered_data = {
    "decision": result["decision"],
    "risk_score": result["risk_score"],
    "signals": result["signals"],
    "key_drivers": result["key_drivers"],
    "similarity": result["similarity"],
    "consistency": result["consistency_check"],
    "confidence": result["confidence"]
    }

    safe_data = json.dumps(filtered_data, indent=2, default=str)

    prompt = f"""
    You are a senior credit risk underwriter.

    Your job is to interpret a model-driven decision and explain it like a human expert.

    Do NOT repeat raw values without interpretation.

    Focus on:
    - WHY the applicant is risky
    - WHICH factors matter most
    - WHETHER the decision is reliable

    Use this structure:

    1. Decision Summary  
    - State decision + risk level  
    - Brief justification (1–2 lines)

    2. Primary Risk Drivers  
    - Identify the MOST impactful drivers (not all)  
    - Explain causality (e.g., affordability, behavior)

    3. Financial Assessment  
    - Interpret DTI, income vs loan  
    - Explain financial stress (if present)

    4. Behavioral Risk  
    - Interpret historical default (if present)

    5. Market Comparison  
    - Compare model risk vs similar borrowers  
    - Explain whether model is aligned with reality

    6. Confidence Interpretation  
    - Explain WHY confidence is high/low  
    - Highlight disagreement or instability

    7. Final Underwriting View  
    - Summarize overall risk in plain language

    DATA:
    {safe_data}
    """

    response = llm.invoke(prompt)

    explanation = response.content

    return {
        "structured_output": result,
        "agent_explanation": explanation
    }