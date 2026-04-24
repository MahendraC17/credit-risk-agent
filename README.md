# Credit Decision System with Validation and Uncertainty Awareness

Credit models predict risk, but they do not indicate whether that prediction is reliable, stable, or aligned with real-world outcomes.
This system wraps the model in a decision layer that validates predictions against historical patterns, checks how stable the outcome is, and measures its own confidence. When confidence is low or signals conflict, it escalates to a human rather than forcing a call.
An agentic AI layer handles the cases that need deeper reasoning, surfacing what's driving the decision, flagging disagreements, and suggesting next steps instead of just returning a number or a flag.

---

## Live Application

You can interact with the application  on: https://credit-risk-agent.streamlit.app/

Use any valid borrower/applicant ID from 1 to 32,581 to test the application end-to-end:

- risk prediction  
- decision logic  
- confidence scoring  
- similarity validation  
- AI generated explanation  

The application runs the full pipeline on a cloud-hosted database and reflects the exact system described below.

## Problem

A typical credit model gives a probability of default. That number alone does not answer important questions:

- Why was this decision made?
- Does this prediction agree with real historical patterns?
- How stable is this decision?
- Should this case be reviewed by a human?

Without these, the system lacks reliability and trust.

---

## Solution

This system breaks the decision process into separate steps and handles each one explicitly.

### 1. Prediction  
The model estimates the likelihood of default. This is the starting point, not the final answer.

### 2. Understanding the prediction  
The system identifies the main factors influencing the model, such as income, loan amount, or past behavior. This helps explain what is driving the risk.

### 3. Translating into decision signals  
Raw model outputs are converted into simple, meaningful signals like high debt burden or past defaults. This keeps the system grounded in real financial reasoning.

### 4. Validation using similar cases  
Each applicant is compared with similar historical borrowers. This gives a second view of risk based on real outcomes and helps detect when the model may be wrong.

### 5. Measuring reliability  
The system evaluates:

- how much the model and similar cases agree  
- how close the decision is to important boundaries  
- how much different signals conflict with each other  

This produces a confidence view of the decision.

### 6. Making the decision  
The final risk is mapped into actions such as approve, approve with conditions, or reject.

### 7. Escalation  
If the system detects low confidence, strong disagreement, or unstable decisions, it flags the case for manual review instead of making an automatic decision.

---

## Threshold Calibration and Validation

The system does not rely on fixed, arbitrary thresholds. Instead, thresholds are calibrated using the final system output.

### Why calibration was needed

The system modifies model predictions using signals and adjustments.

### What was done

Thresholds were recalibrated using the **final risk score (post-signal adjustment)**, backed by data driven ensuring alignment between:

- model prediction  
- signal adjustments  
- decision logic  

### How thresholds are derived

Thresholds are computed from the distribution of final risk scores using percentile-based segmentation:

- Moderate → 60th percentile  
- High → 80th percentile  
- Very High → 92nd percentile  

This ensures the system produces meaningful segmentation instead of collapsing into extreme categories.

### Result

The calibrated thresholds create clear separation between risk bands, allowing the system to assign different decision strategies (approve, conditional, reject) in a structured way.

This makes the decision layer consistent with the actual behavior of the system.

---

## Example behavior

The system behaves differently depending on the situation:

- In a clear case, it makes an automatic decision with high confidence  
- If the model disagrees with similar past cases, it highlights the mismatch  
- If the decision is close to a boundary, it marks it as sensitive  
- If confidence is low, it recommends manual review instead of forcing a decision  

---

## How it is used

The system is deployed as an interactive application:

- Users enter a borrower ID  
- The system retrieves data from a cloud-hosted database on supabase 
- The full pipeline is executed in real time  
- Results are displayed with structured outputs and explanations  

Two modes of output are available:

- **Structured decision output**: risk score, decision, confidence, and diagnostics  
- **Agent-based analysis**: a natural language explanation of the decision, including reasoning and validation insights  

This allows users or representatives to both inspect the system behavior and understand the reasoning behind each decision.

---

## System Architecture

<p align="center">
  <img src="https://raw.githubusercontent.com/MahendraC17/assets/main/system_architecture.jpg" width="750"/>
</p>

---

## Current limitations

- Similarity computation currently scans the full dataset and may not scale efficiently to very large datasets   
- The system does not yet learn from past decisions or feedback  

---

## What is next

Future improvements will focus on making the system more robust and data-driven:

- Expanding signal calibration beyond a few features  
- Improving similarity weighting and feature importance  
- Adding tracking and feedback loops for decisions  
- Strengthening consistency between model behavior and decision logic 

---

## Summary

This project is about moving from a model that predicts risk to a system that supports decisions with supoort of generated decisions.

---

## Tech Stack and Setup

### Tech Stack

- Python 3.10+  
- FastAPI (API layer for testing)  
- Streamlit (UI layer)
- Supabase (hosted PostgreSQL database)
- LightGBM (model training)  
- SHAP (model explainability)  
- Scikit-learn (preprocessing, calibration, similarity) 
- LangChain + OpenAI (agent explanation layer)
- For dependencies see requirements.txt

---

## Decision Pipeline

<p align="center">
  <img src="https://raw.githubusercontent.com/MahendraC17/assets/main/decision_pipeline.jpg" width="900"/>
</p>

---