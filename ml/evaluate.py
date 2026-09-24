import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
    RocCurveDisplay
)


# -----------------------------
# Load dataset
# -----------------------------

df = pd.read_csv("data/raw/heart.csv")

# Convert target into binary classification
df["target"] = df["target"].apply(
    lambda x: 0 if x == 0 else 1
)


# -----------------------------
# Separate features and target
# -----------------------------

X = df.drop("target", axis=1)
y = df["target"]


# -----------------------------
# Train-test split
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# -----------------------------
# Load saved Logistic Regression
# -----------------------------

model = joblib.load(
    "ml/models/heart_disease_pipeline.pkl"
)


# -----------------------------
# Predictions
# -----------------------------

y_pred = model.predict(X_test)

y_probability = model.predict_proba(X_test)[:, 1]


# -----------------------------
# Metrics
# -----------------------------

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(
    y_test,
    y_probability
)


print("\n===== MODEL EVALUATION =====")

print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")
print(f"ROC-AUC   : {roc_auc:.4f}")


# -----------------------------
# Classification Report
# -----------------------------

print("\n===== CLASSIFICATION REPORT =====")

print(
    classification_report(
        y_test,
        y_pred
    )
)


# -----------------------------
# Confusion Matrix
# -----------------------------

print("\n===== CONFUSION MATRIX =====")

cm = confusion_matrix(
    y_test,
    y_pred
)

print(cm)


ConfusionMatrixDisplay(
    confusion_matrix=cm
).plot()

plt.title(
    "Heart Disease Prediction - Confusion Matrix"
)

plt.show()


# -----------------------------
# ROC Curve
# -----------------------------

RocCurveDisplay.from_predictions(
    y_test,
    y_probability
)

plt.title(
    "Heart Disease Prediction - ROC Curve"
)

plt.show()