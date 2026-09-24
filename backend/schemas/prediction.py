from pydantic import BaseModel, ConfigDict


class PatientData(BaseModel):
    model_config = ConfigDict(extra="forbid")

    age: int
    sex: int
    cp: int
    trestbps: int
    chol: int
    fbs: int
    restecg: int
    thalach: int
    exang: int
    oldpeak: float
    slope: int
    ca: int
    thal: int
