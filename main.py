# main.py

from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
import json

# Load model
model = joblib.load("models/best_model.pkl")

# Load expected columns
with open("models/expected_columns.json", "r") as f:
    expected_cols = json.load(f)

# Define input structure
class ChurnInput(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Churn Prediction API is running."}

@app.post("/predict")
def predict(data: ChurnInput):
    input_df = pd.DataFrame([data.dict()])

    # Convert categorical to dummies
    input_df = pd.get_dummies(input_df)

    # Reindex to match model's input
    input_df = input_df.reindex(columns=expected_cols, fill_value=0)

    prediction = model.predict(input_df)[0]
    return {"churn_prediction": int(prediction)}


# mlflow ui Command
#run mlflowUI -- >  http://127.0.0.1:5000

# RUN FASTAPI
# uvicorn main:app --reload
#test on  http://127.0.0.1:8000/docss
# live test on https://churn-fastapi-app-a3c2cpbhdpgsadcy.canadacentral-01.azurewebsites.net/docs

#docker build -t churn-fastapi-full .

#docker tag churn-fastapi-full rehanacr.azurecr.io/churn-fastapi-full:latest
#docker login rehanacr.azurecr.io
#docker push rehanacr.azurecr.io/churn-fastapi-full:latest