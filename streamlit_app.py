# --------------------------------------------------------------------------------
# UI Layer
# Displaying credit decision outputs and agent explanations in a structured format
# --------------------------------------------------------------------------------

import streamlit as st
import requests

st.title("Credit Risk Underwriting System")

# Input
borrower_id = st.number_input("Enter Borrower ID", min_value=1, step=1)


# --------------------------------------------------------------------------------
# Triggering API call
# Fetching analyzed output from backend agent endpoint
# --------------------------------------------------------------------------------
if st.button("Evaluate"):

    response = requests.post(f"http://localhost:8000/analyze/{borrower_id}")

    if response.status_code != 200:
        st.error("Applicant not found from streamlit")
    else:
        api_response = response.json()

        # Splitting structured system output and LLM explanation
        data = api_response["structured_output"]
        explanation = api_response["agent_explanation"]


        # --------------------------------------------------------------------------------
        # Decision Summary
        # --------------------------------------------------------------------------------
        st.header("Decision")
        st.subheader(f"{data['decision']}")
        st.write(data["decision_reason"])


        # --------------------------------------------------------------------------------
        # Risk Overview
        # --------------------------------------------------------------------------------
        st.header("Risk Overview")

        st.metric("Risk Score", round(data["risk_score"], 4))
        st.write("Risk Level:", data["risk_level"])

        rb = data["risk_breakdown"]

        st.write("Base Risk:", rb["base_risk"])
        st.write("Adjustment:", rb["adjustment"])
        st.write("Final Risk:", rb["final_risk"])


        # --------------------------------------------------------------------------------
        # Signals (Rule based + model signals)
        # --------------------------------------------------------------------------------
        st.header("Signals")

        for s in data["signals"]:
            st.write(f"{s['name']} | {s['direction']} | strength: {s['strength']}")


        # --------------------------------------------------------------------------------
        # Key Drivers (Top SHAP features)
        # --------------------------------------------------------------------------------
        st.header("Key Drivers")

        for d in data["key_drivers"]:
            st.write(f"{d['feature']} → {d['effect']} (impact: {d['impact']})")


        # --------------------------------------------------------------------------------
        # Similarity (Peer Comparison)
        # Displaying only if agent decides it is relevant
        # --------------------------------------------------------------------------------
        st.header("Similarity")

        sim = data.get("similarity")

        if sim:
            st.write("Neighbor Mean Default Rate:", sim["mean"])
            st.write("Std Dev:", sim["std"])
            st.write("Sample Size:", sim["count"])
            st.write("Confidence Band:", sim["confidence_band"])
            st.write("Neighbors Used:", sim["count"])
        else:
            st.info("Similarity analysis not required for this case")

        # Showing stability from confidence layer
        st.write("Stability:", data["confidence"]["stability"])


        # --------------------------------------------------------------------------------
        # Consistency Check (Model vs Similarity Validation)
        # --------------------------------------------------------------------------------
        st.header("Consistency Check")

        cc = data["consistency_check"]

        st.write("Model Risk:", cc["model_risk"])
        st.write("Neighbor Risk:", cc["neighbor_risk"])
        st.write("Gap:", cc["gap"])
        st.write("Z-Score Gap:", cc.get("z_gap"))

        # Highlighting disagreement severity
        if cc["override_flag"]:
            st.error("Severe disagreement between model and data")
        elif cc["disagreement_level"] in ["High", "Moderate"]:
            st.warning("Model and similar cases show mismatch")

        st.write("Disagreement Level:", cc["disagreement_level"])
        st.write("Override Flag:", cc["override_flag"])


        # --------------------------------------------------------------------------------
        # Confidence (Decision Reliability)
        # --------------------------------------------------------------------------------
        st.header("Confidence")

        st.metric("Confidence Score", data["confidence"]["score"])
        st.write("Level:", data["confidence"]["level"])

        if data["confidence"]["level"] == "Low":
            st.warning("Low confidence prediction — review recommended")


        # --------------------------------------------------------------------------------
        # Sensitivity (Boundary Analysis)
        # --------------------------------------------------------------------------------
        st.header("Decision Sensitivity")

        sens = data["sensitivity"]

        st.write("Distance to Threshold:", sens["distance_to_threshold"])
        st.write("Closest Threshold:", sens["closest_threshold"])
        st.write("Flip Risk:", sens["flip_risk"])


        # --------------------------------------------------------------------------------
        # Tension (Conflict Measurement)
        # --------------------------------------------------------------------------------
        st.header("Decision Tension")

        tension = data["tension"]

        st.metric("Tension Score", tension["score"])
        st.write("Level:", tension["level"])

        st.write("Signal Conflict:", tension["components"]["signal_conflict"])
        st.write("Model vs Similarity Gap:", tension["components"]["model_vs_similarity_gap"])


        # --------------------------------------------------------------------------------
        # Escalation (Decision Routing)
        # --------------------------------------------------------------------------------
        st.header("Decision Routing")

        escalation = data.get("escalation", "UNKNOWN")

        if escalation == "AUTO_DECISION":
            st.success("Auto Decision — High confidence")
        elif escalation == "MANUAL_REVIEW":
            st.warning("Manual Review Required")
        elif escalation == "BORDERLINE_REVIEW":
            st.warning("Borderline Case — Sensitive Decision")
        elif escalation == "REVIEW_REQUIRED":
            st.error("Model Override — Investigation Needed")
        else:
            st.write(escalation)

        # Reinforcing escalation signals
        if data["confidence"]["level"] == "Low":
            st.warning("Low confidence — system recommends review")

        if escalation != "AUTO_DECISION":
            st.warning(f"Escalation Triggered: {escalation}")


        # --------------------------------------------------------------------------------
        # Agent Output and Explanation
        # --------------------------------------------------------------------------------
        st.header("AI Explanation")

        st.subheader("Summary")
        st.write(explanation["summary"])

        st.subheader("Risk Factors")
        for r in explanation["risk_factors"]:
            st.write("-", r)

        st.subheader("Financial Analysis")
        st.write(explanation["financial_analysis"])

        st.subheader("Behavioral Analysis")
        st.write(explanation["behavioral_analysis"])

        st.subheader("Validation Analysis")
        st.write(explanation["validation_analysis"])

        st.subheader("Confidence Explanation")
        st.write(explanation["confidence_explanation"])

        st.subheader("Final Recommendation")
        st.write(explanation["final_recommendation"])

        # Showing scenario only if present
        if "scenario_analysis" in explanation and explanation["scenario_analysis"]:
            st.subheader("Scenario Analysis")
            st.write(explanation["scenario_analysis"])