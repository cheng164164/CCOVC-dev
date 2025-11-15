import argparse
import os
import pandas as pd
from azure.storage.blob import BlobServiceClient
from preprocessing import preprocess_data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cleaned_data_path", type=str)
    args = parser.parse_args()

    print("📥 Starting preprocessing step...")

    # === Setup Blob service ===
    blob_conn_str = os.environ["BLOB_CONNECTION_STRING"]
    blob_service_client = BlobServiceClient.from_connection_string(blob_conn_str)

    raw_container = "raw-data"
    raw_file = "CCOVC_Predictor.csv"
    local_raw_path = "/tmp/raw.csv"

    # Download raw data from blob
    raw_blob = blob_service_client.get_blob_client(container=raw_container, blob=raw_file)
    with open(local_raw_path, "wb") as f:
        f.write(raw_blob.download_blob().readall())
    print("✅ Raw data downloaded.")

    # === Preprocess ===
    df = pd.read_csv(local_raw_path)
    cleaned_df = preprocess_data(df)

    # ✅ Save to pipeline output path
    cleaned_df.to_csv(args.cleaned_data_path, index=False)
    print(f"✅ Cleaned data written to: {args.cleaned_data_path}")


    container_name = "training-data"
    blob_name = "cleaned_data.csv"
    blob_path = f"/tmp/cleaned_data.csv"

    # Save a local copy first
    cleaned_df.to_csv(blob_path, index=False)

    # Upload to Blob Storage
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)
    with open(blob_path, "rb") as f:
        blob_client.upload_blob(f, overwrite=True)
    print(f"📤 Cleaned data uploaded to blob: {container_name}/{blob_name}")

if __name__ == "__main__":
    main()
