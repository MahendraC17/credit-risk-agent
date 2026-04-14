import streamlit as st
import requests

st.title("Credit Risk Underwriting System")

borrower_id = st.number_input("Enter Borrower ID", min_value=1, step=1)

if st.button("Evaluate"):

    response = requests.post(f"http://localhost:8000/analyze/{borrower_id}")

    if response.status_code != 200:
        st.error("Applicant not found stream")
    else:
        api_response = response.json()
        data = api_response["structured_output"]
        explanation = api_response["agent_explanation"]

        st.header("Decision")

        st.subheader(f"{data['decision']}")
        st.write(data["decision_reason"])


        st.header("Risk Overview")

        st.metric("Risk Score", round(data["risk_score"], 4))
        st.write("Risk Level:", data["risk_level"])

        rb = data["risk_breakdown"]

        st.write("Base Risk:", rb["base_risk"])
        st.write("Adjustment:", rb["adjustment"])
        st.write("Final Risk:", rb["final_risk"])


        st.header("Signals")

        for s in data["signals"]:
            st.write(f"{s['name']} | {s['direction']} | strength: {s['strength']}")


        st.header("Key Drivers")

        for d in data["key_drivers"]:
            st.write(f"{d['feature']} → {d['effect']} (impact: {d['impact']})")


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

        st.write("Stability:", data["confidence"]["stability"])

        st.header("Consistency Check")

        cc = data["consistency_check"]

        st.write("Model Risk:", cc["model_risk"])
        st.write("Neighbor Risk:", cc["neighbor_risk"])
        st.write("Gap:", cc["gap"])
        st.write("Z-Score Gap:", cc.get("z_gap"))

        if cc["override_flag"]:
            st.error("Severe disagreement between model and data")
        elif cc["disagreement_level"] in ["High", "Moderate"]:
            st.warning("Model and similar cases show mismatch")

        st.write("Disagreement Level:", cc["disagreement_level"])
        st.write("Override Flag:", cc["override_flag"])

        st.header("Confidence")

        st.metric("Confidence Score", data["confidence"]["score"])
        st.write("Level:", data["confidence"]["level"])
        if data["confidence"]["level"] == "Low":
            st.warning("Low confidence prediction — review recommended")

        st.header("Decision Sensitivity")

        sens = data["sensitivity"]

        st.write("Distance to Threshold:", sens["distance_to_threshold"])
        st.write("Closest Threshold:", sens["closest_threshold"])
        st.write("Flip Risk:", sens["flip_risk"])

        st.header("Decision Tension")

        tension = data["tension"]

        st.metric("Tension Score", tension["score"])
        st.write("Level:", tension["level"])

        st.write("Signal Conflict:", tension["components"]["signal_conflict"])
        st.write("Model vs Similarity Gap:", tension["components"]["model_vs_similarity_gap"])

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

        if data["confidence"]["level"] == "Low":
            st.warning("Low confidence — system recommends review")

        if data.get("escalation") != "AUTO_DECISION":
            st.warning(f"Escalation Triggered: {data.get('escalation')}")

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

        if "scenario_analysis" in explanation and explanation["scenario_analysis"]:
            st.subheader("Scenario Analysis")
            st.write(explanation["scenario_analysis"])