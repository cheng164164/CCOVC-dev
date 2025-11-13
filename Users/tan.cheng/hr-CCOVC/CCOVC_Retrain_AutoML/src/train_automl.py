# src/train_automl.py
import argparse
import os
import shutil
from azure.ai.ml import MLClient, automl, Input, Model
from azure.identity import DefaultAzureCredential

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cleaned_data_path", type=str)
    parser.add_argument("--model_output_path", type=str)
    args = parser.parse_args()

    # === Initialize ML client
    ml_client = MLClient(
        credential=DefaultAzureCredential(),
        subscription_id=os.getenv("AZURE_SUBSCRIPTION_ID"),
        resource_group=os.getenv("AZURE_RESOURCE_GROUP"),
        workspace=os.getenv("AZURE_WORKSPACE_NAME")
    )

    # === Submit AutoML classification job
    automl_job = automl.classification(
        compute="cpu-cluster",
        experiment_name="ccovc_automl_training",
        training_data=Input(path=args.cleaned_data_path, type="uri_file"),
        target_column_name="CC / OVC",
        primary_metric="accuracy",
        n_cross_validations=5,
        enable_model_explainability=True,
        enable_early_stopping=True
    )

    returned_job = ml_client.jobs.create_or_update(automl_job)
    ml_client.jobs.stream(returned_job.name)

    # === Get best model URI (not the model object yet)
    best_model_output = ml_client.jobs.get(returned_job.name).outputs.get("best_model")
    print(f"🔍 AutoML best model URI: {best_model_output.uri}")

    # === Download model locally
    downloaded_dir = "./tmp_best_model"
    ml_client.jobs.download(name=returned_job.name, output_name="best_model", download_path=downloaded_dir)

    # === Copy model file to output directory
    os.makedirs(args.model_output_path, exist_ok=True)
    model_output_file = os.path.join(args.model_output_path, "ccovc_model.pkl")
    shutil.copy2(os.path.join(downloaded_dir, "model.pkl"), model_output_file)

    print(f"✅ AutoML model copied to: {model_output_file}")

    # === Register model
    registered_model = ml_client.models.create_or_update(
        Model(
            path=args.model_output_path,
            name="ccovc-model",
            description="CCOVC model trained using Azure AutoML",
            type="custom_model",
            tags={"source": "automl", "framework": "AutoML"}
        )
    )

    print(f"📦 Model registered: {registered_model.name} v{registered_model.version}")

if __name__ == "__main__":
    main()

