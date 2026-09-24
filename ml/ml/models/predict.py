loaded_model = joblib.load(
    "../ml/models/heart_disease_pipeline.pkl"
)

print("Model loaded successfully!")

# test_prediction = loaded_model.predict(X_test)

# print(test_prediction[:10])

# test_probability = loaded_model.predict_proba(X_test)[:, 1]

# print(test_probability[:10])

import joblib
import pandas as pd


MODEL_PATH = "ml/models/heart_disease_pipeline.pkl"


def predict_heart_disease(patient_data):
    model = joblib.load(MODEL_PATH)

    patient_df = pd.DataFrame([patient_data])

    prediction = model.predict(patient_df)[0]
    probability = model.predict_proba(patient_df)[0][1]

    if probability < 0.30:
        risk = "Low Risk"
    elif probability < 0.70:
        risk = "Moderate Risk"
    else:
        risk = "High Risk"

    return {
        "prediction": int(prediction),
        "probability": float(probability),
        "risk": risk
    }


if __name__ == "__main__":

    sample_patient = {
        "age": 55,
        "sex": 1,
        "cp": 1,
        "trestbps": 140,
        "chol": 250,
        "fbs": 0,
        "restecg": 1,
        "thalach": 150,
        "exang": 0,
        "oldpeak": 1.0,
        "slope": 1,
        "ca": 0,
        "thal": 2
    }

    result = predict_heart_disease(sample_patient)

    print(result)