import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
import joblib
import os

# Load data
df = pd.read_csv("data/churn.csv")

# Preprocess
df = df.dropna()
label_encoder = LabelEncoder()
df['Churn'] = label_encoder.fit_transform(df['Churn'])
X = df.drop('Churn', axis=1)
y = df['Churn']
X = pd.get_dummies(X)
# Save expected columns for later use in FastAPI
expected_cols = X.columns.tolist()
import json
with open("models/expected_columns.json", "w") as f:
    json.dump(expected_cols, f)


# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Start MLflow experiment
mlflow.set_experiment("Churn-Prediction")

models = {
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "RandomForest": RandomForestClassifier(n_estimators=100)
}

for name, model in models.items():
    with mlflow.start_run(run_name=name):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)

        # Log metrics and params
        mlflow.log_param("model_type", name)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)

        # Save model
        model_path = f"models/{name}.pkl"
        joblib.dump(model, model_path)
        mlflow.sklearn.log_model(model, "model")

        print(f"✅ {name} logged with accuracy: {acc:.4f}")

print("✅ All models logged. Run `mlflow ui` to view.")
