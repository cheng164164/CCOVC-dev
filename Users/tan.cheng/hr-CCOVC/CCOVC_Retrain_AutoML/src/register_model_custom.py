# src/register_model.py
import os
from azure.storage.blob import BlobServiceClient
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Model
from azure.identity import DefaultAzureCredential
from azureml.core import Run, Workspace, Model


def main():
    conn_str = os.environ["BLOB_CONNECTION_STRING"]
    container = "model-store"
    blob_name = "ccovc-custom-model.pkl"
    local_path = "/tmp/ccovc_custom_model.pkl"

    # Download model from Blob
    print("📥 Downloading model from Blob Storage...")
    blob_service = BlobServiceClient.from_connection_string(conn_str)
    blob_client = blob_service.get_blob_client(container=container, blob=blob_name)

    with open(local_path, "wb") as f:
        f.write(blob_client.download_blob().readall())
    print(f"✅ Downloaded model to {local_path}")

    print("🔐 Using Azure ML run context to register model...")
    run = Run.get_context()
    ws = run.experiment.workspace  # This gets the current workspace

    registered_model = Model.register(
        model_path=local_path,
        model_name="ccovc-custom-model",
        workspace=ws,
        description="Model registered from Blob after pipeline run",
        tags={"source": "custom-training"},
    )

    print(f"📦 Registered model: {registered_model.name} v{registered_model.version}")

if __name__ == "__main__":
    main()
