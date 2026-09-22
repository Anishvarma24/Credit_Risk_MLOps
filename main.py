import pandas as pd
import joblib
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

file_path = "data/raw/german.data-numeric"

df = pd.read_csv(file_path, sep=r"\s+", header=None)

# Rename target
df.rename(columns={24: "target"}, inplace=True)

# 1 = Good Risk → 0
# 2 = Bad Risk → 1
df["target"] = df["target"].map({1: 0, 2: 1})

# Features and target
X = df.drop("target", axis=1)
y = df["target"]

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# MLflow experiment
mlflow.set_experiment("Credit_Risk_Prediction")

with mlflow.start_run():

    # Model
    model = LogisticRegression(max_iter=1000)

    # Train
    model.fit(X_train, y_train)

    # Prediction
    y_pred = model.predict(X_test)

    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # MLflow logging
    mlflow.log_param("model", "Logistic Regression")
    mlflow.log_param("test_size", 0.2)
    mlflow.log_param("random_state", 42)
    mlflow.log_metric("accuracy", accuracy)

    # Save model in MLflow
    mlflow.sklearn.log_model(model, "credit_risk_model")

    print("Model Accuracy:", accuracy)
    print("MLflow tracking completed!")