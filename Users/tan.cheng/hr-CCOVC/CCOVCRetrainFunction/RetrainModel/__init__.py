import pandas as pd
from azure.storage.blob import BlobServiceClient
import os
from .preprocessing import preprocess_data
from .training import train_and_save_model


def main(mytimer):
    print("⏰ Starting scheduled retraining...")

    # === Setup Blob Service ===
    blob_conn_str = os.environ["AzureWebJobsStorage"]
    blob_service_client = BlobServiceClient.from_connection_string(blob_conn_str)

    # === Step 1: Load raw training data from 'raw-data' container ===
    raw_container = "raw-data"
    raw_file = "CCOVC_Predictor.csv"
    local_raw_path = "/tmp/raw.csv"

    raw_blob_client = blob_service_client.get_blob_client(container=raw_container, blob=raw_file)
    with open(local_raw_path, "wb") as f:
        f.write(raw_blob_client.download_blob().readall())

    print("📥 Raw dataset downloaded from 'raw-data' container.")

    # === Step 2: Preprocess ===
    df = pd.read_csv(local_raw_path)
    cleaned_df = preprocess_data(df)

    local_clean_path = "/tmp/ccovc_cleaned.csv"
    cleaned_df.to_csv(local_clean_path, index=False)

    # Upload cleaned dataset to 'training-data' container
    training_container = "training-data"
    cleaned_blob_client = blob_service_client.get_blob_client(container=training_container, blob="cleaned_data.csv")
    with open(local_clean_path, "rb") as data:
        cleaned_blob_client.upload_blob(data, overwrite=True)

    print("🧹 Cleaned dataset uploaded to 'training-data' container.")

    # === Step 3: Train model ===
    model_local_path = "/tmp/model.pkl"
    train_and_save_model(cleaned_df, model_local_path)

    # === Step 4: Upload model to 'model-store' container ===
    model_container = "model-store"
    model_blob_client = blob_service_client.get_blob_client(container=model_container, blob="ccovc_model.pkl")
    with open(model_local_path, "rb") as data:
        model_blob_client.upload_blob(data, overwrite=True)

    print("✅ Model retrained and uploaded successfully to 'model-store' container.")