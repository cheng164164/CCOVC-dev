import argparse
import os
import json
import pandas as pd
import mlflow.pyfunc
from sklearn.metrics import accuracy_score, classification_report
from azure.storage.blob import BlobServiceClient
from azureml.core import Workspace, Model, Run
import mlflow


def download_blob_file(connection_string, container_name, blob_name, download_path):
    print(f"📥 Downloading blob '{blob_name}' from container '{container_name}'...")
    blob_service = BlobServiceClient.from_connection_string(connection_string)
    blob_client = blob_service.get_blob_client(container=container_name, blob=blob_name)
    os.makedirs(os.path.dirname(download_path), exist_ok=True)
    with open(download_path, "wb") as f:
        f.write(blob_client.download_blob().readall())
    print(f"✅ Downloaded test dataset to: {download_path}")
    return download_path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_uri", type=str, required=True,
                        help="Directory containing model_uri.txt from registration step.")
    parser.add_argument("--eval_output", type=str, required=True,
                        help="Folder to save evaluation results.")
    return parser.parse_args()


def main():
    args = parse_args()
    model_name = "ccovc-automl-model"  # Registered model name in Azure ML
    model_version = None  # Set to None to always use latest


    # Read model URI from model_uri.txt
    model_uri_path = os.path.join(args.model_uri, "model_uri.txt")
    if not os.path.exists(model_uri_path):
        raise FileNotFoundError(f"❌ model_uri.txt not found at: {model_uri_path}")

    with open(model_uri_path, "r") as f:
        model_uri = f.read().strip()

    print(f"🔗 Attempting to load MLflow model from URI: {model_uri}")

    # Connect to workspace using environment variables
    run = Run.get_context()
    ws = run.experiment.workspace
    print(f"✅ Connected to workspace: {ws.name}")

    # Get registered model from AzureML
    print("📦 Fetching registered model...")
    if model_version:
        registered_model = Model(ws, name=model_name, version=model_version)
    else:
        registered_model = Model(ws, name=model_name)  # latest version

    # Configure MLflow to use AzureML's tracking URI
    mlflow.set_tracking_uri(ws.get_mlflow_tracking_uri())
    mlflow.set_registry_uri(ws.get_mlflow_tracking_uri())
    print(f"✅ MLflow tracking URI set to: {mlflow.get_tracking_uri()}")

    # Build model URI and load the model ===
    model_uri = f"models:/{registered_model.name}/{registered_model.version}"
    print(f"📂 MLflow model URI: {model_uri}")

    # Try loading the model
    try:
        model = mlflow.pyfunc.load_model(model_uri)
        if model is None:
            raise ValueError("mlflow.pyfunc.load_model() returned None.")
        print(f"✅ Model loaded: {type(model)}")
    except Exception as e:
        print(f"❌ Failed to load MLflow model from: {model_uri}")
        print(f"Error: {e}")
        raise e

    # Download test dataset from blob
    connection_string = os.environ["BLOB_CONNECTION_STRING"]
    container_name = "test-data"
    blob_name = "cleaned_data KMT 11.17.25_testset.csv"
    local_test_path = "/tmp/test_dataset.csv"

    download_blob_file(
        connection_string=connection_string,
        container_name=container_name,
        blob_name=blob_name,
        download_path=local_test_path
    )

    # Load and split dataset
    print("📄 Loading test dataset...")
    df = pd.read_csv(local_test_path)
    y_true = df["CC / OVC"]
    X_test = df.drop(columns=["CC / OVC"])

    # Generate predictions
    print("⚙ Generating predictions...")
    y_pred = model.predict(X_test)

    # Evaluate metrics
    accuracy = accuracy_score(y_true, y_pred)
    clf_report = classification_report(y_true, y_pred, output_dict=True)

    metrics = {
        "accuracy": accuracy,
        "classification_report": clf_report
    }

    # Save evaluation results
    os.makedirs(args.eval_output, exist_ok=True)
    output_file = os.path.join(args.eval_output, "metrics.json")
    with open(output_file, "w") as f:
        json.dump(metrics, f, indent=4)

    print("✅ Evaluation complete.")
    print(f"📊 Accuracy: {accuracy}")
    print(f"📁 metrics.json saved to: {output_file}")

if __name__ == "__main__":
    main()
