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

        thresholds = CONFIG["risk"]["thresholds"]

        st.write(
            f"""
        These thresholds define how the final risk score is translated into decision categories.

        - **Moderate Risk ≥ {thresholds['moderate']:.2f}**  
        Borrowers start showing elevated risk. Decisions may include conditions or additional checks.

        - **High Risk ≥ {thresholds['high']:.2f}**  
        Borrowers are significantly risky. Decisions become stricter, often requiring collateral or strong justification.

        - **Very High Risk ≥ {thresholds['very_high']:.2f}**  
        Borrowers are highly likely to default. These cases are typically rejected unless overridden.

        The thresholds are calibrated from system output to ensure each band reflects real-world default behavior.
        """
        )

        st.markdown("---")

        st.markdown("### Buffer")

        st.write(
            f"""
        **Buffer: ±{CONFIG["risk"]["buffer"]:.2f}**

        Defines a margin around thresholds where decisions are treated as unstable.

        If a score falls within this range:
        - the decision is considered sensitive  
        - small changes could flip the outcome  
        - the system may apply stricter rules or escalate the case  
        """
        )

        st.markdown("---")

        st.markdown("### DTI Thresholds")

        dti = CONFIG["dti"]

        st.write(
        f"""
        Debt-to-Income (DTI) thresholds used to trigger risk signals:

        - **Low DTI ≤ {dti['low']}** → considered financially manageable  
        - **Moderate DTI ≥ {dti['moderate']}** → triggers moderate risk signal  
        - **High DTI ≥ {dti['high']}** → triggers strong risk signal  

        These do not directly decide outcomes, but influence the final risk through signals.
        """
        )

        st.markdown("---")

        
        st.markdown("### Confidence Scoring")

        weights = CONFIG["confidence"]["weights"]

        st.write(
            f"""
        Confidence reflects how reliable a decision is based on multiple factors.

        - **Signal Strength (weight {weights.get('signal', 'N/A')})**  
            Measures how strongly risk signals indicate risk.

        - **Similarity Alignment (weight {weights.get('similarity', 'N/A')})**  
            Evaluates how similar historical borrowers behaved compared to the model prediction.

        - **Decision Stability (weight {weights.get('stability', 'N/A')})**  
            Captures how close the score is to a threshold. Closer scores are less stable.

        - **Signal Consistency (weight {weights.get('consistency', 'N/A')})**  
            Checks whether different signals agree with each other. Conflicting signals reduce confidence.


        Lower confidence means:
        - higher uncertainty  
        - higher chance of escalation  
        - less trust in automated decisions  
        """
        )

        st.markdown("---")


        st.markdown("### Similarity Validation")

        similarity = CONFIG["similarity"]

        st.write(
            f"""
        The system compares each applicant to similar historical cases.

        - **Top K neighbors: {similarity.get('k', 'N/A')}**
        - **Distance smoothing: {similarity.get('distance_smoothing', 'N/A')}**
        - **Minimum confidence band: {similarity.get('min_confidence_band', 'N/A')}**

        This acts as a second opinion:
        - validates model predictions  
        - detects unusual or conflicting cases  
        - improves trust in decisions  

        If model and similar cases disagree, confidence is reduced and the case may be escalated.
        """
        )

        st.markdown("---")

        st.info(
            "These parameters define how the system balances prediction, validation, and decision-making."
        )


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