from fastapi import APIRouter, HTTPException
from app.db.queries import fetch_applicant
from app.tools.credit_tool import evaluate_applicant
from app.agent.credit_agent import run_agent
import traceback


router = APIRouter()

@router.post("/evaluate/{borrower_id}")
def evaluate_borrower(borrower_id: int):
    applicant = fetch_applicant(borrower_id)

    if not applicant:
        raise HTTPException(status_code=404, detail="Applicant not found router")

    result = evaluate_applicant(applicant)

    return {
        "borrower_id": borrower_id,

        "risk_score": result["risk_score"],
        "risk_level": result["risk_level"],
        "decision": result["decision"],
        "decision_reason": result["decision_reason"],

        "risk_breakdown": result["risk_breakdown"],
        "signals": result["signals"],
        "key_drivers": result["key_drivers"],

        "similarity": result["similarity"],
        "consistency_check": result["consistency_check"],
        "confidence": result["confidence"],

        "sensitivity": result["sensitivity"],
        "tension": result["tension"],
    }


from app.agent.credit_agent import run_agent

@router.post("/analyze/{borrower_id}")
def analyze_borrower(borrower_id: int):

    try:
        result = run_agent(borrower_id)
        return result

    except Exception as e:
        import traceback
        traceback.print_exc()

        raise HTTPException(status_code=500, detail=str(e))