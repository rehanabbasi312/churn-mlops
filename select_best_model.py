import mlflow
from mlflow.tracking import MlflowClient
import joblib
import os

# Define experiment name
EXPERIMENT_NAME = "Churn-Prediction"

# Create client and get experiment
client = MlflowClient()
experiment = client.get_experiment_by_name(EXPERIMENT_NAME)

# Get all runs from experiment
runs = client.search_runs(experiment.experiment_id)

# Filter only valid runs that have 'accuracy' metric
valid_runs = [r for r in runs if "accuracy" in r.data.metrics]

if not valid_runs:
    raise ValueError("❌ No valid MLflow runs with 'accuracy' found.")

# Select the best run based on accuracy
best_run = max(valid_runs, key=lambda run: run.data.metrics["accuracy"])

# Print best run info
print("🏆 Best model found:")
print(f"➡ Model: {best_run.data.params.get('model_type', 'unknown')}")
print(f"➡ Accuracy: {best_run.data.metrics['accuracy']:.4f}")
print(f"➡ Run ID: {best_run.info.run_id}")

# Load model from MLflow run
model_uri = f"runs:/{best_run.info.run_id}/model"
model = mlflow.sklearn.load_model(model_uri)

# Ensure models directory exists
os.makedirs("models", exist_ok=True)

# Save best model as best_model.pkl
joblib.dump(model, "models/best_model.pkl")
print("✅ Best model saved to models/best_model.pkl")
