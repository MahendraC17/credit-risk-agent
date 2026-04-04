import pandas as pd
from sqlalchemy import text
from app.db.connection import engine
from app.data_processing.preprocess import preprocess_credit_data

def load_data():
    df = pd.read_csv("data/raw/credit_risk_dataset.csv")
    df = preprocess_credit_data(df)

    with engine.connect() as conn:
        conn.execute(text("TRUNCATE TABLE borrowers RESTART IDENTITY"))
        conn.commit()

    df.to_sql("borrowers", engine, if_exists="append", index=False)

    print(f"Loaded {len(df)} rows successfully.")


if __name__ == "__main__":
    load_data()