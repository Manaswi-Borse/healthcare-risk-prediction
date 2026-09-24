# Intelligent Healthcare Disease Risk Prediction & Clinical Decision Support System

## 📌 Project Overview

This project is a machine-learning-based healthcare application that predicts the risk of heart disease from selected patient health parameters.

The system uses a trained Logistic Regression model and provides predictions through a FastAPI backend and a Streamlit frontend.

> **Disclaimer:** This project is an educational machine-learning prototype and is not a medical diagnosis or a substitute for professional medical advice.

---

## 🎯 Objectives

* Build a supervised machine-learning model for heart disease risk prediction.
* Perform data preprocessing and feature transformation.
* Compare different classification models.
* Develop a REST API using FastAPI.
* Build a simple interactive frontend using Streamlit.
* Integrate the trained ML pipeline with the application.
* Provide model probability along with the prediction result.

---

## 🏗️ System Architecture

```text
                 Patient Input
                      │
                      ▼
              ┌───────────────┐
              │   Streamlit   │
              │   Frontend    │
              └───────┬───────┘
                      │
                      ▼
              ┌───────────────┐
              │    FastAPI    │
              │    Backend    │
              └───────┬───────┘
                      │
                      ▼
              ┌───────────────┐
              │ ML Pipeline   │
              │ Preprocessing │
              │ + Logistic    │
              │ Regression    │
              └───────┬───────┘
                      │
                      ▼
              ┌───────────────┐
              │ Risk Result   │
              │ + Probability │
              └───────────────┘
```

---

## 📊 Dataset

The project uses a publicly available heart disease dataset containing patient-related clinical attributes.

### Features

* Age
* Sex
* Chest pain type
* Resting blood pressure
* Cholesterol
* Fasting blood sugar
* Resting ECG
* Maximum heart rate
* Exercise-induced angina
* ST depression
* Slope
* Number of major vessels
* Thalassemia

### Target

The original target values are converted into a binary classification:

```text
0 → Lower/absence class
1 → Presence/risk class
```

---

## 🔄 Data Preprocessing

The following preprocessing steps are applied:

* Missing-value handling using imputation.
* Standardization of numerical features using `StandardScaler`.
* One-hot encoding of categorical features.
* Train-test split using stratification.
* Target conversion into binary classes.

The preprocessing and model are combined into a single Scikit-learn pipeline.

---

## 🤖 Machine Learning Models

Two classification algorithms were evaluated:

### Logistic Regression

Used as the final model because it achieved higher evaluation metrics on the selected held-out test split.

### Random Forest

Used as a comparison model to evaluate whether a tree-based ensemble provided better performance on the same test split.

---

## 📈 Model Results

### Logistic Regression

| Metric    |  Score |
| --------- | -----: |
| Accuracy  | 88.52% |
| Precision | 83.87% |
| Recall    | 92.86% |
| F1-Score  | 88.14% |
| ROC-AUC   | 96.65% |

### Random Forest

| Metric    |  Score |
| --------- | -----: |
| Accuracy  | 85.25% |
| Precision | 80.65% |
| Recall    | 89.29% |
| F1-Score  | 84.75% |
| ROC-AUC   | 94.32% |

> These results were obtained from the selected held-out test split and should not be interpreted as clinical validation.

---

## 🛠️ Technology Stack

### Programming Language

* Python

### Machine Learning

* Scikit-learn
* Pandas
* NumPy
* Joblib

### Backend

* FastAPI
* Uvicorn
* Pydantic

### Frontend

* Streamlit

### Data Visualization

* Matplotlib
* Seaborn

### Testing

* Pytest
* HTTPX

### Development Tools

* VS Code
* Jupyter Notebook
* Git
* GitHub

---

## 📁 Project Structure

```text
healthcare-risk-prediction/
│
├── backend/
│   ├── main.py
│   ├── routes/
│   │   ├── health.py
│   │   └── prediction.py
│   ├── schemas/
│   │   └── prediction.py
│   └── services/
│       └── predictor.py
│
├── frontend/
│   └── app.py
│
├── ml/
│   ├── train.py
│   ├── predict.py
│   ├── evaluate.py
│   └── models/
│       └── heart_disease_pipeline.pkl
│
├── data/
│   ├── raw/
│   │   └── heart.csv
│   └── processed/
│
├── notebooks/
│   ├── 01_data_understanding.ipynb
│   ├── 02_data_cleaning.ipynb
│   └── 03_eda_preprocessing.ipynb
│
├── tests/
│   ├── test_model.py
│   └── test_api.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 🚀 How to Run the Project

### 1. Clone the repository

```bash
git clone <your-github-repository-url>
cd healthcare-risk-prediction
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

### 3. Activate the environment

#### Windows PowerShell

```powershell
.\venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 5. Start the FastAPI backend

Open Terminal 1:

```bash
python -m uvicorn backend.main:app --reload
```

Backend:

```text
http://127.0.0.1:8000
```

API documentation:

```text
http://127.0.0.1:8000/docs
```

### 6. Start the Streamlit frontend

Open Terminal 2:

```bash
python -m streamlit run frontend/app.py
```

Frontend:

```text
http://localhost:8501
```

---

## 🔌 API Endpoint

### POST `/predict`

The endpoint accepts patient information and returns a machine-learning prediction.

Example request:

```json
{
    "age": 55,
    "sex": 1,
    "cp": 1,
    "trestbps": 140,
    "chol": 240,
    "fbs": 0,
    "restecg": 1,
    "thalach": 150,
    "exang": 0,
    "oldpeak": 1.2,
    "slope": 1,
    "ca": 0,
    "thal": 2
}
```

Example response:

```json
{
    "prediction": 1,
    "probability": 0.82,
    "risk": "Elevated Risk",
    "message": "This is a machine-learning prediction and not a medical diagnosis."
}
```

---

## 🧪 Testing

Run the API tests using:

```bash
python -m pytest
```

The tests verify:

* Health endpoint
* Prediction endpoint
* Response structure
* Model integration

---

## ⚠️ Limitations

* The project uses a publicly available dataset and is not clinically validated.
* Model performance depends on the dataset and selected test split.
* Predictions should not be interpreted as medical diagnoses.
* The application is intended for educational and demonstration purposes.
* Further validation using larger and clinically representative datasets would be required for real-world use.

---

## 🔮 Future Scope

* Add additional disease-risk prediction models.
* Add explainable AI techniques such as SHAP.
* Improve model validation using cross-validation and hyperparameter tuning.
* Add user authentication.
* Add database support.
* Improve frontend visualization.
* Deploy the application using cloud services.
* Explore model monitoring and versioning.

---

## 👩‍💻 Author

**Manaswi Borse**

B.E. Artificial Intelligence & Data Science
