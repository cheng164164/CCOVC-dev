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

if __name__ == "__main__":
    main()
