# train.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
import mlflow
import mlflow.sklearn

# Load data
df = pd.read_csv("data/churn.csv")

# Drop useless columns
df.drop(["customerID"], axis=1, inplace=True)

# Convert TotalCharges to numeric
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df.fillna(0, inplace=True)

# Encode categorical variables
for col in df.select_dtypes("object"):
    df[col] = LabelEncoder().fit_transform(df[col])

# Split data
X = df.drop("Churn", axis=1)
y = df["Churn"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Start MLflow experiment
mlflow.set_experiment("Churn-Prediction")

with mlflow.start_run():

    # Train model (try changing model here)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Predict & evaluate
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    # Log params, metrics, and model
    mlflow.log_param("model_type", "RandomForest")
    mlflow.log_param("n_estimators", 100)
    mlflow.log_metric("accuracy", acc)

    # Save model
    joblib.dump(model, "models/churn_model.pkl")

    # Log model in MLflow
    mlflow.sklearn.log_model(model, "churn_model")

    print(f"✅ Accuracy: {acc}")
