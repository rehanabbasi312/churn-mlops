# main.py

from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd

# Load model
model = joblib.load("models/churn_model.pkl")

# Define input structure
class ChurnInput(BaseModel):
    gender: int
    SeniorCitizen: int
    Partner: int
    Dependents: int
    tenure: int
    PhoneService: int
    MultipleLines: int
    InternetService: int
    OnlineSecurity: int
    OnlineBackup: int
    DeviceProtection: int
    TechSupport: int
    StreamingTV: int
    StreamingMovies: int
    Contract: int
    PaperlessBilling: int
    PaymentMethod: int
    MonthlyCharges: float
    TotalCharges: float

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Churn Prediction API is running."}

@app.post("/predict")
def predict_churn(input: ChurnInput):
    # Convert input to DataFrame
    input_df = pd.DataFrame([input.dict()])
    pred = model.predict(input_df)[0]
    return {"churn_prediction": int(pred)}
