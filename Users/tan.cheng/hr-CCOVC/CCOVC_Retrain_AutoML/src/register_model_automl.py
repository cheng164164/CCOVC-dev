'''
## Only save pkl file version - deprecated version
import argparse
from azureml.core import Run, Model
from azure.storage.blob import BlobServiceClient
import os
import glob
import mlflow.pyfunc



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    return parser.parse_args()


def find_pkl_file(model_dir):
    print(f"🔍 Looking for .pkl file inside: {model_dir}")
    pkl_files = glob.glob(os.path.join(model_dir, "**", "*.pkl"), recursive=True)
    if not pkl_files:
        raise FileNotFoundError(f"No .pkl file found in {model_dir}")
    print(f"📄 Found model file: {pkl_files[0]}")
    return pkl_files[0]


def upload_to_blob_storage(connection_string, container_name, blob_name, local_file_path):
    print(f"📤 Uploading model to Blob Storage: {container_name}/{blob_name}")
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    container_client = blob_service_client.get_container_client(container_name)

    # Create container if it doesn't exist
    try:
        container_client.create_container()
    except Exception:
        pass  # Likely already exists

    # Upload model file
    with open(local_file_path, "rb") as data:
        container_client.upload_blob(name=blob_name, data=data, overwrite=True)


def main():
    args = parse_args()
    conn_str = os.environ["BLOB_CONNECTION_STRING"]
    container = "model-store"
    blob_name = "ccovc-automl-model.pkl"

    print(f"📦 Registering MLflow model from path: {args.model_path}")
    
    # Get workspace from run context
    run = Run.get_context()
    ws = run.experiment.workspace

    
    model = Model.register(
        model_path=args.model_path,
        model_name="ccovc-automl-model",
        model_framework=Model.Framework.SCIKITLEARN,
        model_framework_version="1.0",  # Adjust if needed
        workspace=ws,
        description="AutoML-trained model (MLflow)",
        tags={"source": "automl"},
    )

    print(f"✅ Registered AutoML model: {model.name} v{model.version}")

    
    # Locate .pkl file and upload to blob
    model_pkl_path = find_pkl_file(args.model_path)

    upload_to_blob_storage(
        connection_string=conn_str,
        container_name=container,
        blob_name=blob_name,
        local_file_path=model_pkl_path
    )
    

if __name__ == "__main__":
    main()
'''



# src/register_model_automl.py

import argparse
import os
from azureml.core import Run, Model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--model_uri_output", type=str, required=True)
    return parser.parse_args()


def main():
    args = parse_args()

    # Validate model path has MLmodel file (required for MLflow model)
    expected_file = os.path.join(args.model_path, "MLmodel")
    if not os.path.exists(expected_file):
        raise FileNotFoundError(f"❌ 'MLmodel' not found at {expected_file}. Is this an MLflow model folder?")

    print(f"📦 Registering MLflow model from: {args.model_path}")

    # Register model
    run = Run.get_context()
    ws = run.experiment.workspace

    model = Model.register(
        model_path=args.model_path,
        model_name="ccovc-automl-model",
        workspace=ws,
        description="AutoML-trained model (MLflow)",
        tags={"source": "automl"},
    )

    print(f"✅ Registered model: {model.name} v{model.version}")

    model_uri = f"azureml://models/{model.name}/versions/{model.version}"
    print(f"🔗 AzureML model URI: {model_uri}")

    os.makedirs(args.model_uri_output, exist_ok=True)
    output_file = os.path.join(args.model_uri_output, "model_uri.txt")
    with open(output_file, "w") as f:
        f.write(model_uri)

    print(f"📄 Saved model URI to: {output_file}")


if __name__ == "__main__":
    main()
