from fastapi import APIRouter
import pandas as pd

from backend.schemas.prediction import PatientData
from backend.services.predictor import predict_heart_disease


router = APIRouter()


@router.post("/predict")
def predict(patient: PatientData):

    patient_dict = patient.model_dump()

    patient_df = pd.DataFrame([patient_dict])

    prediction, probability = predict_heart_disease(patient_df)

    if prediction == 1:
        risk = "Elevated Risk"
    else:
        risk = "Lower Risk"

    return {
        "prediction": int(prediction),
        "probability": round(float(probability), 4),
        "risk": risk,
        "message": "This is a machine-learning prediction and not a medical diagnosis."
    }