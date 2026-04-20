from app.db.queries import fetch_multiple_applicants
from app.tools.credit_tool import evaluate_applicant
from collections import defaultdict

# -----------------------------
# CONFIG
# -----------------------------
N = 10000

# -----------------------------
# STORAGE
# -----------------------------
band_counts = defaultdict(int)
band_defaults = defaultdict(int)

# -----------------------------
# FETCH DATA
# -----------------------------
applicants = fetch_multiple_applicants(N)

# -----------------------------
# EVALUATION LOOP
# -----------------------------
for a in applicants:
    result = evaluate_applicant(a)

    risk = result["risk_level"]
    default = a["default"]   # ground truth

    band_counts[risk] += 1
    band_defaults[risk] += default

# -----------------------------
# OUTPUT
# -----------------------------
print("\n=== DEFAULT RATE PER RISK BAND ===\n")

for band in ["Low", "Moderate", "High", "Very High"]:
    count = band_counts[band]

    if count == 0:
        rate = 0
    else:
        rate = band_defaults[band] / count

    print(f"{band:10} | Count: {count:5} | Default Rate: {round(rate, 4)}")