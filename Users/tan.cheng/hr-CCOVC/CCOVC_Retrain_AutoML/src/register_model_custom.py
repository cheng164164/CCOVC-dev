# src/register_model_custom.py
import os
import json
import joblib
from datetime import datetime

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report

from azure.storage.blob import BlobServiceClient
from azureml.core import Run, Model
from sklearn.preprocessing import LabelEncoder


def download_blob_file(connection_string, container_name, blob_name, download_path):
    print(f"📥 Downloading blob '{blob_name}' from container '{container_name}'...")
    blob_service = BlobServiceClient.from_connection_string(connection_string)
    blob_client = blob_service.get_blob_client(container=container_name, blob=blob_name)
    os.makedirs(os.path.dirname(download_path), exist_ok=True)
    with open(download_path, "wb") as f:
        f.write(blob_client.download_blob().readall())
    print(f"✅ Downloaded to: {download_path}")
    return download_path


def upload_blob_file(connection_string, container_name, blob_name, local_file_path):
    print(f"☁ Uploading '{blob_name}' to container '{container_name}'...")
    blob_service = BlobServiceClient.from_connection_string(connection_string)
    blob_client = blob_service.get_blob_client(container=container_name, blob=blob_name)
    with open(local_file_path, "rb") as f:
        blob_client.upload_blob(f, overwrite=True)
    print("✅ Upload complete.")


def main():
    # =========================
    # Configuration
    # =========================
    conn_str = os.environ["BLOB_CONNECTION_STRING"]

    model_container = "model-store"
    model_blob_name = "ccovc-custom-model.pkl"
    model_local_path = "/tmp/ccovc_custom_model.pkl"

    test_container = "test-data"
    test_blob_name = "cleaned_data KMT 11.17.25_testset.csv"
    test_local_path = "/tmp/test_dataset.csv"

    # =========================
    # Download trained model
    # =========================
    print("📥 Downloading model from Blob Storage...")
    download_blob_file(
        connection_string=conn_str,
        container_name=model_container,
        blob_name=model_blob_name,
        download_path=model_local_path
    )

    # =========================
    # Register model
    # =========================
    print("🔐 Using Azure ML run context to register model...")
    run = Run.get_context()
    ws = run.experiment.workspace

    registered_model = Model.register(
        model_path=model_local_path,
        model_name="ccovc-custom-model",
        workspace=ws,
        description="Model registered from Blob after pipeline run",
        tags={"source": "custom-training"},
    )

    print(f"📦 Registered model: {registered_model.name} v{registered_model.version}")

    # =========================
    # Load frozen test dataset
    # =========================
    print("📥 Downloading frozen test dataset...")
    download_blob_file(
        connection_string=conn_str,
        container_name=test_container,
        blob_name=test_blob_name,
        download_path=test_local_path
    )

    print("📄 Loading test dataset...")
    selected_features = [
        'Legal Entity (Label)', 'Business unit (Label)', 'Division (Label)', 
        'Employment Type (Label)', 'Job Classification (Label)'
    ]

    # Apply column selection
    df_test = pd.read_csv(test_local_path)
    df_test = df_test[["CC / OVC"] + selected_features]

    y_true = df_test["CC / OVC"]
    X_test = df_test[selected_features]

    # =========================
    # Load custom model with joblib
    # =========================
    print(f"📂 Loading custom model using joblib from: {model_local_path}")
    model = joblib.load(model_local_path)
    print(f"✅ Model loaded: {type(model)}")

    # =========================
    # Evaluate model
    # =========================
    label_encoder = LabelEncoder()
    y_true_encoded = label_encoder.fit_transform(y_true)
    print("⚙ Generating predictions...")
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_true_encoded, y_pred)
    clf_report = classification_report(y_true_encoded, y_pred, output_dict=True)

    metrics = {
        "model_name": registered_model.name,
        "model_version": registered_model.version,
        "accuracy": accuracy,
        "classification_report": clf_report
    }

    print(f"📊 Accuracy: {accuracy:.4f}")

    # =========================
    # Save & upload metrics
    # =========================
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    metrics_filename = f"custom_training_metrics_{timestamp}.json"
    metrics_local_path = f"/tmp/{metrics_filename}"
    metrics_blob_path = f"model_test_results/{metrics_filename}"

    with open(metrics_local_path, "w") as f:
        json.dump(metrics, f, indent=4)

    upload_blob_file(
        connection_string=conn_str,
        container_name=model_container,
        blob_name=metrics_blob_path,
        local_file_path=metrics_local_path
    )

    print(f"✅ Evaluation metrics uploaded to: {metrics_blob_path}")


if __name__ == "__main__":
    main()
