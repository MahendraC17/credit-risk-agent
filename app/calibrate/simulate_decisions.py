from app.db.queries import fetch_multiple_applicants
from app.tools.credit_tool import evaluate_applicant
from collections import Counter

results = []
decisions = []
risk_levels = []
escalations = []
flip_cases = 0
disagreement_cases = 0

applicants = fetch_multiple_applicants(10000)

for a in applicants:
    r = evaluate_applicant(a)

    decisions.append(r["decision"])
    risk_levels.append(r["risk_level"])
    escalations.append(r["escalation"])

    if r["sensitivity"]["flip_risk"]:
        flip_cases += 1

    if r["consistency_check"]["flag"]:
        disagreement_cases += 1


print("\n=== DECISION DISTRIBUTION ===")
print(Counter(decisions))

print("\n=== RISK LEVEL DISTRIBUTION ===")
print(Counter(risk_levels))

print("\n=== ESCALATION DISTRIBUTION ===")
print(Counter(escalations))

print("\n=== DIAGNOSTICS ===")
print("Flip risk cases:", flip_cases)
print("Disagreement cases:", disagreement_cases)

# from app.db.connection import engine
# import pandas as pd

# df = pd.read_sql("SELECT * FROM borrowers", engine)

# default_rate = df["default"].mean()
# print("Default rate:", round(default_rate, 4))