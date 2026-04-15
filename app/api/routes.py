# --------------------------------------------------------------------------------
# API Layer
# Exposing core system functionality via FastAPI endpoints
# --------------------------------------------------------------------------------

from fastapi import APIRouter, HTTPException
from app.db.queries import fetch_applicant
from app.tools.credit_tool import evaluate_applicant
from app.agent.credit_agent import run_agent

router = APIRouter()


# --------------------------------------------------------------------------------
# Direct Evaluation Endpoint
# Running full rule-based + model pipeline without agent reasoning for testing and manual validation
# --------------------------------------------------------------------------------
@router.post("/evaluate/{borrower_id}")
def evaluate_borrower(borrower_id: int):

    applicant = fetch_applicant(borrower_id)

    if not applicant:
        raise HTTPException(status_code=404, detail="Applicant not found from router")

    # Running core evaluation pipeline
    result = evaluate_applicant(applicant)

    # Returning structured system output
    return {
        "borrower_id": borrower_id,

        # Core decision outputs
        "risk_score": result["risk_score"],
        "risk_level": result["risk_level"],
        "decision": result["decision"],
        "decision_reason": result["decision_reason"],

        # Model + signal breakdown
        "risk_breakdown": result["risk_breakdown"],
        "signals": result["signals"],
        "key_drivers": result["key_drivers"],

        # Peer comparison
        "similarity": result["similarity"],

        # Validation layer
        "consistency_check": result["consistency_check"],

        # Simplified validation summary for quick consumption
        "validation_summary": {
            "disagreement_level": result["consistency_check"]["disagreement_level"],
            "z_gap": result["consistency_check"].get("z_gap"),
            "override": result["consistency_check"]["override_flag"]
        },

        # Diagnostics
        "confidence": result["confidence"],
        "sensitivity": result["sensitivity"],
        "tension": result["tension"],

        # Final routing decision
        "escalation": result["escalation"],
    }


# --------------------------------------------------------------------------------
# Agent Endpoint
# Running full system + LLM explanation layer
# --------------------------------------------------------------------------------
@router.post("/analyze/{borrower_id}")
def analyze_borrower(borrower_id: int):

    try:
        result = run_agent(borrower_id)

        return {
            "structured_output": result["structured_output"],
            "agent_explanation": result["agent_explanation"],

            # Exposing escalation at top level for UI
            "escalation": result["structured_output"]["escalation"]
        }

    except Exception as e:
        import traceback
        traceback.print_exc()

        raise HTTPException(status_code=500, detail=str(e))