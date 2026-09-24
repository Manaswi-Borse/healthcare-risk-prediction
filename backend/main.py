from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.routes import health
from backend.routes import prediction


app = FastAPI(
    title="Healthcare Disease Risk Prediction API",
    description="Machine learning API for heart disease risk prediction",
    version="1.0.0"
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


app.include_router(health.router)
app.include_router(prediction.router)


@app.get("/")
def root():
    return {
        "message": "Healthcare Disease Risk Prediction API",
        "status": "running"
    }