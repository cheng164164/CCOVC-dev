# scoring/score_registered_model.py
'''
import os
import pandas as pd
import mlflow
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from azureml.core import Workspace, Model
from dotenv import load_dotenv
import azureml.automl.runtime.shared.model_wrappers

# === Load environment variables ===
load_dotenv()

# === Configuration ===
target_column = "CC / OVC"
test_data_path = "cleaned_data KMT 11.17.25_testset.csv"  # Adjust if needed
model_name = "ccovc-automl-model"  # Registered model name in Azure ML
model_version = None  # Set to None to always use latest

# Features used during model training
selected_features = [
    "Legal Entity (Label)",
    "Business Unit (Label)",
    "Employment Type (Label)",
    "Job Classification (Label)"
]

# === Step 1: Connect to Azure ML Workspace ===
print("🔗 Connecting to Azure ML workspace...")

ws = Workspace(
    subscription_id=os.environ["AZURE_SUBSCRIPTION_ID"],
    resource_group=os.environ["AZURE_RESOURCE_GROUP"],
    workspace_name=os.environ["AZURE_WORKSPACE_NAME"]
)

print(f"✅ Connected to workspace: {ws.name}")

# === Step 2: Get registered model ===
print("📦 Fetching registered model...")

if model_version:
    registered_model = Model(ws, name=model_name, version=model_version)
else:
    registered_model = Model(ws, name=model_name)  # latest

# === Step 3: Configure MLflow for AzureML ===
mlflow.set_tracking_uri(ws.get_mlflow_tracking_uri())
mlflow.set_registry_uri(ws.get_mlflow_tracking_uri())

model_uri = f"models:/{registered_model.name}/{registered_model.version}"
print(f"📂 MLflow model URI: {model_uri}")

# === Step 4: Load the model ===
model = mlflow.pyfunc.load_model(model_uri)
print("✅ Model loaded successfully.")

# === Step 5: Load test dataset ===
df = pd.read_csv(test_data_path)

# Check for required columns
missing_cols = [col for col in selected_features + [target_column] if col not in df.columns]
if missing_cols:
    raise ValueError(f"❌ Missing columns in test dataset: {missing_cols}")

X_test = df[selected_features]
y_test = df[target_column]

# === Step 6: Run inference ===
print("🚀 Running inference...")
y_pred = model.predict(X_test)

# === Step 7: Evaluate results ===
acc = accuracy_score(y_test, y_pred)
print(f"\n✅ Accuracy: {acc:.4f}")
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred))
print("\n🧩 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# === Step 8: Save predictions ===
results_df = df.copy()
results_df["Predicted"] = y_pred
results_df["PredictionCorrect"] = results_df[target_column] == y_pred
results_df.to_csv("scoring_results.csv", index=False)

print("\n📁 Predictions saved to: scoring_results.csv")
'''


'''
# scoring/evaluate_local_mlflow_model.py

import os
import pandas as pd
import mlflow.sklearn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Required for Azure AutoML-wrapped models like XGBoostLabelEncoder
import azureml.automl.runtime.shared.model_wrappers

# === Configuration ===
model_dir = "scoring/ccovc_automl_model"  # path to local MLflow model folder
test_data_path = "scoring/cleaned_data KMT 11.17.25_testset.csv"
target_column = "CC / OVC"

selected_features = [
    "Legal Entity (Label)",
    "Business Unit (Label)",
    "Employment Type (Label)",
    "Job Classification (Label)"
]

# === Load model from local MLflow folder ===
print(f"📦 Loading model from: {model_dir}")
model = mlflow.sklearn.load_model(model_dir)
print("✅ Model loaded successfully.")

# === Load and validate test dataset ===
df = pd.read_csv(test_data_path)

missing_cols = [col for col in selected_features + [target_column] if col not in df.columns]
if missing_cols:
    raise ValueError(f"❌ Missing columns in test data: {missing_cols}")

X_test = df[selected_features]
y_test = df[target_column]

# === Run inference ===
print("🚀 Running inference...")
y_pred = model.predict(X_test)

# === Evaluation ===
acc = accuracy_score(y_test, y_pred)
print(f"\n✅ Accuracy: {acc:.4f}")
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred))
print("\n🧩 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# === Save results ===
df["Predicted"] = y_pred
df["PredictionCorrect"] = df[target_column] == y_pred
df.to_csv("scoring/scoring_results.csv", index=False)
print("\n📁 Results saved to: scoring/scoring_results.csv")
'''


import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load model.pkl directly (skips MLflow)
model_path = "scoring/ccovc_automl_model/model.pkl"
model = joblib.load(model_path)  # works for most AutoML outputs

# Load test data
df = pd.read_csv("scoring/cleaned_data KMT 11.17.25_testset.csv")
selected_features = [
    "Legal Entity (Label)",
    "Business Unit (Label)",
    "Employment Type (Label)",
    "Job Classification (Label)"
]
target_column = "CC / OVC"

X_test = df[selected_features]
y_test = df[target_column]

# Inference
y_pred = model.predict(X_test)

# Evaluation
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
