# ---------------------------------------
# Extracting the data from postgresql to csv for uploading to supabase,
# redundant necause we are no longer using it
# ---------------------------------------

import pandas as pd
from sqlalchemy import text
from app.db.connection import engine


query = text("SELECT * FROM borrowers")

with engine.connect() as conn:
    df = pd.read_sql(query, conn)

print("Columns:", df.columns.tolist())

int_columns = [
    "income",
    "loan_amount",
    "credit_history_length",
    "employment_length"
]

for col in int_columns:
    if col in df.columns:
        df[col] = df[col].astype(float).astype(int)
    else:
        print(f"[WARN] Column '{col}' not found, skipping...")


output_path = "data/processed/borrowers_clean.csv"
df.to_csv(output_path, index=False)

print(f"Exported and fixed dataset to {output_path}")