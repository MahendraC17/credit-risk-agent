# --------------------------------------------------------------------------------
# UI Layer
# --------------------------------------------------------------------------------

import streamlit as st
import requests
import json
import re
from app.config.config_loader import CONFIG

# --------------------------------------------------------------------------------
# Text cleaning function to return clearner text, solution for when numbers and 
# text came together caused sentences to return without any space in betweent the words
# --------------------------------------------------------------------------------
def clean_text(text: str):
    if not isinstance(text, str):
        return text
    text = text.replace("$", "\\$")

    text = re.sub(r',([^\s])', r', \1', text)

    text = re.sub(r'(\d)([a-zA-Z])', r'\1 \2', text)

    text = re.sub(r'([a-zA-Z])(\d)', r'\1 \2', text)

    return text.strip()

# def format_currency(text: str):
#     return re.sub(r'(\b\d{3,}\b)', r'$\1', text)

st.set_page_config(layout="wide")

st.markdown("""
<style>
.stApp {
    background-image: radial-gradient(circle, rgba(255,255,255,0.12) 1px, transparent 1px);
    background-size: 18px 18px;
    background-color: #0e1117;
}
</style>
""", unsafe_allow_html=True)


st.title("Credit Risk Decision System")
st.caption("System combines model predictions, rule-based signals, and AI-generated reasoning")


# --------------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------------
with st.sidebar:
    st.header("Input")

    borrower_id = st.number_input("Borrower ID", min_value=1, step=1)

    run = st.button("Evaluate")

    # --------------------------------------------------------------------------------
    # Developer Panel
    # --------------------------------------------------------------------------------

    st.markdown("---")
    st.caption("Use Developer Panel below for system insights.")

    with st.expander("Developer Panel"):

        st.markdown("### Risk Thresholds")
        st.json(CONFIG["risk"]["thresholds"])

        st.markdown("### Buffer")
        st.write(CONFIG["risk"]["buffer"])

        st.markdown("### DTI Thresholds")
        st.json(CONFIG["dti"])

        st.markdown("### Confidence Weights")
        st.json(CONFIG["confidence"]["weights"])

        st.markdown("### Similarity Settings")
        st.json(CONFIG["similarity"])

        st.info("These parameters control how strict or lenient the system behaves.")


# --------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------
if run:

    response = requests.post(f"http://localhost:8000/analyze/{borrower_id}")

    if response.status_code != 200:
        st.error("Applicant not found from stream")
        st.stop()

    api_response = response.json()
    data = api_response["structured_output"]
    explanation = api_response["agent_explanation"]

    main_col, dev_col = st.columns([3, 1])

    # --------------------------------------------------------------------------------
    # Decision
    # --------------------------------------------------------------------------------
    col1, col2 = st.columns([2, 3])

    with col1:
        st.markdown("### Recommended Decision      - ")
        st.caption("Recommended action based on risk and system logic.")
    with col2:
        st.markdown(f"### {data['decision']}")
        st.write(data["decision_reason"])
        
    st.markdown("---")
    # --------------------------------------------------------------------------------
    # Risk Assessment
    # --------------------------------------------------------------------------------
    st.subheader("Risk Assessment")

    col1, col2 = st.columns(2)
    with col1:
        st.metric(
            "Model Risk Score",
            round(data["risk_score"], 4),
            help="Base probability of default predicted by the model"
        )

    with col2:
        st.write("Risk Level:", data["risk_level"])

    st.markdown("---")
    # --------------------------------------------------------------------------------
    # Confidence
    # --------------------------------------------------------------------------------
    st.subheader("Decision Confidence & Routing")
    st.caption("Indicates whether the decision can be automated or requires human review")


    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Confidence Score",
            data["confidence"]["score"],
            help="Overall reliability of this decision"
        )
        st.write("Level:", data["confidence"]["level"])

    with col2:
        escalation = data.get("escalation", "UNKNOWN")

        if escalation == "AUTO_DECISION":
            st.success("Auto Decision (High Confidence)")

        elif escalation == "MANUAL_REVIEW":
            st.warning("Review Required (Low Confidence)")

        elif escalation == "BORDERLINE_REVIEW":
            st.warning("Borderline Case (Decision Sensitive)")

        elif escalation == "REVIEW_REQUIRED":
            st.error("Review Required (Model Disagreement)")

    st.markdown("---")
    # --------------------------------------------------------------------------------
    # Key Drivers
    # --------------------------------------------------------------------------------
    st.subheader("Key Drivers")

    for d in data["key_drivers"]:
        st.write(f"{d['feature']} → {d['effect']} (impact: {d['impact']})")

    st.markdown("---")
    # --------------------------------------------------------------------------------
    # AI Explanation
    # --------------------------------------------------------------------------------
    st.subheader("AI Analysis", help="Structured reasoning generated from model outputs and system signals")

    st.write(clean_text(explanation["summary"]))

    with st.expander("Detailed Analysis using AI"):
        st.markdown("**Risk Factors**")
        for r in explanation["risk_factors"]:
            st.write("-", clean_text(r))

        st.markdown("**Financial Analysis**")
        # fa = explanation["financial_analysis"]
        # fa = format_currency(fa)
        # fa = clean_text(fa)

        # st.write(fa)
        st.write(clean_text(explanation["financial_analysis"]))

        st.markdown("**Behavioral Analysis**")
        st.write(clean_text(explanation["behavioral_analysis"]))

        st.markdown("**Validation Analysis**")
        st.write(clean_text(explanation["validation_analysis"]))

        st.markdown("**Confidence Explanation**")
        st.write(clean_text(explanation["confidence_explanation"]))

        scenario_text = explanation.get("scenario_analysis")
        if scenario_text and scenario_text.strip():
            st.markdown("**Scenario Analysis**")
            st.write(clean_text(scenario_text))

        st.markdown("**Final Recommendation**")
        st.write(clean_text(explanation["final_recommendation"]))


    # --------------------------------------------------------------------------------
    # Metrics Overview
    # --------------------------------------------------------------------------------
    st.subheader("Metrics")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Confidence",
            data["confidence"]["score"],
            help="Reliability of decision based on agreement and stability"
        )
        st.write("Level:", data["confidence"]["level"])

    with col2:
        st.metric(
            "Sensitivity",
            data["sensitivity"]["distance_to_threshold"],
            help="Distance from decision boundary"
        )
        st.write("Flip Risk:", data["sensitivity"]["flip_risk"])

    with col3:
        st.metric(
            "Tension",
            data["tension"]["score"],
            help="Conflict between signals and model vs similarity"
        )
        st.write("Level:", data["tension"]["level"])


    # --------------------------------------------------------------------------------
    # Validation & Similarity
    # --------------------------------------------------------------------------------
    with st.expander("Validation & Similarity"):

        sim = data.get("similarity")

        if sim:
            st.write("Neighbor Mean:", sim["mean"])
            st.write("Std Dev:", sim["std"])
            st.write("Sample Size:", sim["count"])
            st.write("Confidence Band:", sim["confidence_band"])

        cc = data["consistency_check"]

        st.markdown("**Consistency Check**")
        st.write("Model Risk:", cc["model_risk"])
        st.write("Neighbor Risk:", cc["neighbor_risk"])
        st.write("Gap:", cc["gap"])
        st.write("Z-Score:", cc.get("z_gap"))

        if cc["override_flag"]:
            st.error("Severe disagreement detected")
        elif cc["disagreement_level"] in ["High", "Moderate"]:
            st.warning("Model and data mismatch")