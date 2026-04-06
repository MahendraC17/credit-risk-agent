import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Credit Risk Underwriting", layout="wide")

st.title("Credit Risk Decision System")

st.sidebar.header("Input")

borrower_id = st.sidebar.number_input(
    "Borrower ID",
    min_value=1,
    step=1
)

analyze = st.sidebar.button("Evaluate")

if analyze:
    with st.spinner("Evaluating borrower..."):
        response = requests.post(f"{API_URL}/evaluate/{borrower_id}")

    if response.status_code != 200:
        st.error("Error fetching data")
    else:
        data = response.json()

        col1, col2, col3 = st.columns(3)

        col1.metric("Risk Score", data["risk_score"])
        col2.metric("Risk Level", data["risk_level"])
        col3.metric("Decision", data["decision"])

        st.divider()

        st.subheader("Decision Rationale")
        st.write(data["decision_reason"])

        st.divider()

        st.subheader("Key Risk Drivers")

        for driver in data["key_drivers"]:
            feature = driver["feature"]
            value = driver["value"]
            impact = driver["impact"]

            direction = "↑" if impact > 0 else "↓"

            st.write(
                f"**{feature}** = {value} {direction} "
                f"(impact: {round(impact, 2)})"
            )

        st.divider()

        st.subheader("Risk Signals")

        for signal in data["signals"]:
            st.write(
                f"{signal['name']} → strength: {round(signal['strength'], 3)}"
            )

        st.divider()
        st.subheader("Peer Comparison (Similarity)")

        similarity = data.get("similarity", {})

        default_rate = similarity.get("similar_default_rate", None)
        neighbors = similarity.get("neighbor_count", None)

        if default_rate is not None:
            col1, col2 = st.columns(2)

            col1.metric(
                "Similar Borrower Default Rate",
                f"{round(default_rate * 100, 1)}%"
            )

            col2.metric(
                "Neighbors Analyzed",
                neighbors
            )

            if default_rate > 0.5:
                st.warning("Similar borrowers show elevated default behavior")
            elif default_rate > 0.3:
                st.info("Similar borrowers show moderate default behavior")
            else:
                st.success("Similar borrowers show low default behavior")