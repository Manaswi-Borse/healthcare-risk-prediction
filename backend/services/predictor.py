from pathlib import Path

import joblib
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_PATH = BASE_DIR / "ml" / "models" / "heart_disease_pipeline.pkl"


def predict_heart_disease(patient_data):
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

    model = joblib.load(MODEL_PATH)

    if isinstance(patient_data, pd.DataFrame):
        patient_df = patient_data
    elif isinstance(patient_data, dict):
        patient_df = pd.DataFrame([patient_data])
    else:
        patient_df = pd.DataFrame(patient_data)

    prediction = model.predict(patient_df)[0]
    probability = model.predict_proba(patient_df)[0][1]

    return int(prediction), float(probability)
