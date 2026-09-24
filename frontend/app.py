import streamlit as st
import requests


API_URL = "http://127.0.0.1:8000/predict"


# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="Heart Risk Prediction",
    page_icon="❤️",
    layout="wide"
)


# ---------------- HEADER ----------------

st.title("❤️ Heart Disease Risk Prediction")

st.write(
    "A machine-learning based application for predicting "
    "heart disease risk from selected patient health parameters."
)

st.info(
    "⚠️ This application is intended for educational and "
    "demonstration purposes. It is not a medical diagnosis."
)


# ---------------- SIDEBAR ----------------

st.sidebar.title("About the Project")

st.sidebar.write(
    """
    **Model:** Logistic Regression

    **Task:** Binary Classification

    **Dataset:** Heart Disease Dataset

    **Backend:** FastAPI

    **Frontend:** Streamlit
    """
)

st.sidebar.markdown("---")

st.sidebar.write(
    "Enter the patient's information and click "
    "**Predict Risk** to obtain the model prediction."
)


# ---------------- INPUT SECTION ----------------

st.header("Patient Information")

col1, col2, col3 = st.columns(3)


with col1:

    st.subheader("Basic Information")

    age = st.number_input(
        "Age",
        min_value=1,
        max_value=120,
        value=55
    )

    sex = st.selectbox(
        "Sex",
        [0, 1],
        format_func=lambda x:
        "Female" if x == 0 else "Male"
    )

    cp = st.selectbox(
        "Chest Pain Type",
        [0, 1, 2, 3]
    )

    trestbps = st.number_input(
        "Resting Blood Pressure",
        min_value=50.0,
        max_value=250.0,
        value=140.0
    )


with col2:

    st.subheader("Medical Measurements")

    chol = st.number_input(
        "Cholesterol",
        min_value=50.0,
        max_value=600.0,
        value=240.0
    )

    fbs = st.selectbox(
        "Fasting Blood Sugar > 120 mg/dl",
        [0, 1],
        format_func=lambda x:
        "No" if x == 0 else "Yes"
    )

    restecg = st.selectbox(
        "Resting ECG",
        [0, 1, 2]
    )

    thalach = st.number_input(
        "Maximum Heart Rate",
        min_value=50.0,
        max_value=250.0,
        value=150.0
    )


with col3:

    st.subheader("Additional Information")

    exang = st.selectbox(
        "Exercise Induced Angina",
        [0, 1],
        format_func=lambda x:
        "No" if x == 0 else "Yes"
    )

    oldpeak = st.number_input(
        "ST Depression (Oldpeak)",
        min_value=0.0,
        max_value=10.0,
        value=1.2
    )

    slope = st.selectbox(
        "Slope",
        [0, 1, 2]
    )

    ca = st.number_input(
        "Number of Major Vessels (CA)",
        min_value=0.0,
        max_value=4.0,
        value=0.0
    )

    thal = st.selectbox(
        "Thalassemia",
        [0, 1, 2, 3]
    )


# ---------------- PREDICTION ----------------

st.markdown("---")

predict_button = st.button(
    "🔍 Predict Heart Disease Risk",
    use_container_width=True
)


if predict_button:

    patient_data = {
        "age": age,
        "sex": sex,
        "cp": cp,
        "trestbps": trestbps,
        "chol": chol,
        "fbs": fbs,
        "restecg": restecg,
        "thalach": thalach,
        "exang": exang,
        "oldpeak": oldpeak,
        "slope": slope,
        "ca": ca,
        "thal": thal
    }

    try:

        with st.spinner("Running prediction..."):

            response = requests.post(
                API_URL,
                json=patient_data,
                timeout=10
            )

        if response.status_code == 200:

            result = response.json()

            st.header("Prediction Result")

            probability = result["probability"] * 100

            if result["prediction"] == 1:

                st.error(
                    "⚠️ Elevated Risk"
                )

            else:

                st.success(
                    "✅ Lower Risk"
                )

            col1, col2 = st.columns(2)

            with col1:

                st.metric(
                    "Model Probability",
                    f"{probability:.2f}%"
                )

            with col2:

                st.metric(
                    "Prediction",
                    result["risk"]
                )

            st.progress(
                min(
                    max(
                        float(result["probability"]),
                        0.0
                    ),
                    1.0
                )
            )

            st.info(
                result["message"]
            )

        else:

            st.error(
                f"Backend returned error "
                f"{response.status_code}"
            )

    except requests.exceptions.ConnectionError:

        st.error(
            "Unable to connect to the FastAPI backend. "
            "Make sure Uvicorn is running."
        )

    except requests.exceptions.Timeout:

        st.error(
            "The prediction request timed out."
        )

    except Exception as e:

        st.error(
            f"Unexpected error: {e}"
        )