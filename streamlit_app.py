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

        st.write("Default Rate (Neighbors):", data["similarity"]["similar_default_rate"])
        st.write("Neighbors Used:", data["similarity"]["neighbor_count"])


        st.header("Consistency Check")

        cc = data["consistency_check"]

        st.write("Model Risk:", cc["model_risk"])
        st.write("Neighbor Risk:", cc["neighbor_risk"])
        st.write("Gap:", cc["gap"])
        st.write("Flag:", cc["flag"])


        st.header("Confidence")

        st.metric("Confidence Score", data["confidence"]["score"])
        st.write("Level:", data["confidence"]["level"])
        if data["confidence"]["level"] == "Low":
            st.warning("Low confidence prediction — review recommended")

        st.header("AI Explanation")

        st.write(explanation)