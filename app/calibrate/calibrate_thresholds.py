# --------------------------------------------------------------------------------
# Threshold Calibration
# Computing thresholds from final_risk
# --------------------------------------------------------------------------------

import pandas as pd
import numpy as np
import json
from sqlalchemy import text

from app.db.connection import engine
from app.tools.credit_tool import evaluate_applicant


# --------------------------------------------------------------------------------
# Load full dataset
# --------------------------------------------------------------------------------
def load_data():
    query = text("SELECT * FROM borrowers")
    with engine.connect() as conn:
        return pd.read_sql(query, conn)


# --------------------------------------------------------------------------------
# Computing final_risk for each applicant
# --------------------------------------------------------------------------------
def compute_final_risk(df: pd.DataFrame):

    risks = []

    for _, row in df.iterrows():
        applicant = row.to_dict()

        try:
            result = evaluate_applicant(applicant)
            risks.append(result["risk_breakdown"]["final_risk"])
        except Exception:
            continue

    df = df.iloc[:len(risks)].copy()
    df["final_risk"] = risks

    return df


# --------------------------------------------------------------------------------
# Derive thresholds
# --------------------------------------------------------------------------------
def derive_thresholds(df: pd.DataFrame):

    scores = df["final_risk"].values

    thresholds = {
        "moderate": float(np.percentile(scores, 60)),
        "high": float(np.percentile(scores, 80)),
        "very_high": float(np.percentile(scores, 92))
    }

    return thresholds


# --------------------------------------------------------------------------------
# Saving thresholds into system config
# --------------------------------------------------------------------------------
def save_to_config(thresholds: dict):

    config_path = "app/config/system_config.json"

    with open(config_path, "r") as f:
        config = json.load(f)

    config["risk"]["thresholds"] = {
        "moderate": round(thresholds["moderate"], 2),
        "high": round(thresholds["high"], 2),
        "very_high": round(thresholds["very_high"], 2)
    }

    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)

    print("\nThresholds updated in system_config.json")


def run():

    print("\nLoading data...")
    df = load_data()

    print("Computing final risk...")
    df = compute_final_risk(df)

    print("Deriving thresholds...")
    thresholds = derive_thresholds(df)

    print("\n===== FINAL THRESHOLDS =====")
    print(json.dumps(thresholds, indent=4))

    save_to_config(thresholds)


# --------------------------------------------------------------------------------
# ENTRY
# --------------------------------------------------------------------------------
if __name__ == "__main__":
    run()