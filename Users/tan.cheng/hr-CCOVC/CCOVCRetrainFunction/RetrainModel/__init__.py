import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from category_encoders import TargetEncoder
from sklearn.pipeline import Pipeline
from azure.storage.blob import BlobServiceClient
import os

def main(mytimer):
    print("Starting scheduled retraining...")

    # === Step 1: Load training data from blob ===
    blob_conn_str = os.environ["AzureWebJobsStorage"]
    container_name = "training-data"
    file_name = "ccovc_cleaned.csv"

    blob_service_client = BlobServiceClient.from_connection_string(blob_conn_str)
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=file_name)

    with open("/tmp/train.csv", "wb") as f:
        f.write(blob_client.download_blob().readall())

    df = pd.read_csv("/tmp/train.csv")

    # === Step 2: Prepare features and target ===
    selected_features = [
        'Company (Label)', 'Position Country (Label)', 'Business unit (Label)',
        'Division (Label)', 'Employment Classification (Label)',
        'Employee Type (Label)', 'Employment Type (Label)'
    ]
    target = "CC / OVC"

    X = df[selected_features].fillna("Missing")
    y = df[target]

    # === Step 3: Create and train pipeline ===
    pipeline = Pipeline([
        ("encoder", TargetEncoder(cols=selected_features)),
        ("model", RandomForestClassifier(n_estimators=100, random_state=42))
    ])

    pipeline.fit(X, y)

    # === Step 4: Save model to blob ===
    local_model_path = "/tmp/model.pkl"
    joblib.dump(pipeline, local_model_path)

    model_blob = blob_service_client.get_blob_client(container="model-store", blob="ccovc_model.pkl")
    with open(local_model_path, "rb") as data:
        model_blob.upload_blob(data, overwrite=True)

    print("✅ Model retrained and uploaded successfully.")