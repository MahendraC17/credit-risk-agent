from fastapi import FastAPI
from app.api.routes import router

app = FastAPI(
    title="Credit Risk Underwriting API",
    description="Production-style credit decision engine",
    version="1.0"
)

app.include_router(router)