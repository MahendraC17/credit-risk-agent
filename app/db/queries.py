from sqlalchemy import text
from app.db.connection import engine


def fetch_applicant(applicant_id: int):
    query = text("""SELECT * FROM borrowers WHERE borrower_id = :id""")

    with engine.connect() as conn:
        result = conn.execute(query, {"id": applicant_id})
        row = result.fetchone()

    return dict(row._mapping) if row else None


def fetch_multiple_applicants(limit: int = 5):
    query = text(f"""SELECT * FROM borrowers LIMIT :limit""")

    with engine.connect() as conn:
        result = conn.execute(query, {"limit": limit})
        rows = result.fetchall()

    return [dict(row._mapping) for row in rows]