import argparse
import os
import pandas as pd
import joblib
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Model
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient
from training import train_and_save_model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cleaned_data_path", type=str)
    args = parser.parse_args()

    # === Step 1: Load cleaned data
    df = pd.read_csv(args.cleaned_data_path)

    # === Step 2: Train model and save locally
    local_model_path = "/tmp/ccovc_model.pkl"
    train_and_save_model(df, local_model_path)
    print(f"✅ Model trained and saved locally to: {local_model_path}")

    # === Step 3: Upload to Blob Storage
    conn_str = os.environ["BLOB_CONNECTION_STRING"]
    container_name = "model-store"
    blob_name = "ccovc_model.pkl"  # You can make this dynamic with timestamp or version

    blob_service = BlobServiceClient.from_connection_string(conn_str)
    blob_client = blob_service.get_blob_client(container=container_name, blob=blob_name)

    with open(local_model_path, "rb") as data:
        blob_client.upload_blob(data, overwrite=True)

    print(f"📤 Model uploaded to Blob Storage at container '{container_name}' with blob name '{blob_name}'")

if __name__ == "__main__":
    main()
