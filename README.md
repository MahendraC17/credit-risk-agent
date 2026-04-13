Credit risk decision system that separates prediction, validation, and decision-making into distinct layers.

The model (LightGBM with calibration) produces a probability of default, which serves as the base risk. SHAP is used to understand what influenced the model’s prediction, highlighting key drivers such as income, loan amount, or home ownership. However, SHAP is not used directly for decisions, as it reflects statistical correlations rather than business logic.

To bridge this gap, I introduced a signal layer that converts raw data into a small set of interpretable and stable risk factors, such as debt-to-income ratio (DTI) and historical default behavior. These signals are intentionally limited to financially meaningful features to keep the system explainable and controlled. The model’s base risk is then adjusted using these signals through a non-linear aggregation approach to produce a final risk score.

Beyond prediction, the system includes a validation layer using similarity-based analysis (KNN) to compare each applicant with historical borrowers. This helps estimate a local default rate and detect when the model’s prediction disagrees with real data patterns. Additional layers capture decision stability (sensitivity), conflict between signals and model (tension), and overall reliability (confidence).

A decision engine maps the final risk into actions such as approve, approve with conditions, or reject, using calibrated thresholds and buffer zones. On top of this, an escalation layer identifies cases that require manual review, especially when there is high disagreement, low confidence, or unstable decisions.

This design ensures the system is not just predictive. It separates:

- what the model predicts
- what factors influence the prediction
- how decisions are made
- when decisions should not be trusted

The system is being exposed via FastAPI and visualized through Streamlit, with an agent layer generating structured explanations and guiding further analysis.

This is an evolving project. Current work focuses on building a robust decision pipeline, while future iterations will improve signal calibration, threshold learning, and overall statistical stability using data-driven methods.