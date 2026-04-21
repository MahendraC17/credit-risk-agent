# Credit Decision System with Validation and Uncertainty Awareness

Credit models predict risk. What they don't do is tell you when that prediction is wrong, unstable, or shouldn't be acted on automatically.
This system wraps the model in a decision layer that validates predictions against historical patterns, checks how stable the outcome is, and measures its own confidence. When confidence is low or signals conflict, it escalates to a human rather than forcing a call.
An agentic AI layer handles the cases that need deeper reasoning, surfacing what's driving the decision, flagging disagreements, and suggesting next steps instead of just returning a number or a flag.

---

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

Thresholds were recalibrated using the **final risk score (post-signal adjustment)**, ensuring alignment between:

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

## What makes this different

It is a system that:

- separates prediction from decision-making  
- checks its own outputs against historical patterns  
- measures how stable and reliable a decision is  
- knows when to step back and ask for human review  

---

## Example behavior

The system behaves differently depending on the situation:

- In a clear case, it makes an automatic decision with high confidence  
- If the model disagrees with similar past cases, it highlights the mismatch  
- If the decision is close to a boundary, it marks it as sensitive  
- If confidence is low, it recommends manual review instead of forcing a decision  

---

## How it is used

The system is exposed through an API and visualized through a simple interface.

There are two ways to interact with it:

- **Direct evaluation**: returns the structured decision and diagnostics  
- **Agent-based analysis**: adds a clear explanation and suggests next steps when needed  

---

## Current limitations

This is an evolving system and not everything is fully data-driven yet.

- Signal strengths are partially calibrated but not fully optimized  
- Similarity logic depends on feature scaling and may not capture all nuances  
- The system does not yet learn from past decisions or feedback  

---

## What is next

Future improvements will focus on making the system more robust and data-driven:

- Learning decision thresholds directly from data  
- Expanding signal calibration beyond a few features  
- Improving similarity weighting and feature importance  
- Adding tracking and feedback loops for decisions  
- Strengthening consistency between model behavior and decision logic 
- Hosting this as an application on streamlit

---

## Summary

This project is about moving from a model that predicts risk to a system that supports decisions.

It not only answers “what is the risk”, but also:

- can we trust this result  
- does it match real-world patterns  
- how stable is this decision  
- should a human take a closer look  

That shift is what makes the system useful in practice.

---

## Tech Stack and Setup

### Tech Stack

- Python 3.10+  
- FastAPI (API layer)  
- Streamlit (UI layer)  
- LightGBM (model training)  
- SHAP (model explainability)  
- Scikit-learn (preprocessing, calibration, similarity) 
- LangChain + OpenAI (agent explanation layer)  

---

### Environment

- Python version: 3.10 or higher recommended  
- A virtual environment is recommended for dependency isolation  

---

### Installation

Install dependencies using:

```bash
pip install -r requirements.txt
```