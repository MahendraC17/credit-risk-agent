from fastapi import APIRouter, HTTPException

from app.db.queries import fetch_applicant
from app.tools.credit_tool import evaluate_applicant

router = APIRouter()


@router.post("/evaluate/{borrower_id}")
def evaluate_borrower(borrower_id: int):
    applicant = fetch_applicant(borrower_id)

    if not applicant:
        raise HTTPException(status_code=404, detail="Applicant not found")

    result = evaluate_applicant(applicant)

    return {
        "borrower_id": borrower_id,
        "risk_score": result["risk_score"],
        "risk_level": result["risk_level"],
        "decision": result["decision"],
        "decision_reason": result["decision_reason"],
        "key_drivers": result["key_drivers"],
        "signals": result["signals"]
    }