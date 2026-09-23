from fastapi import FastAPI, Body
import joblib

app = FastAPI(title="Credit Risk Prediction API")

# Load trained model
model = joblib.load("models/credit_risk_model.pkl")


@app.get("/")
def home():
    return {"message": "Credit Risk Prediction API is running"}


@app.post("/predict")
def predict(features: list[float] = Body(...)):
    prediction = model.predict([features])

    if prediction[0] == 0:
        result = "Good Risk"
    else:
        result = "Bad Risk"

    return {
        "prediction": int(prediction[0]),
        "risk": result
    }