'''
## AutoML Pipeline for CCOVC Data Processing and Model Training (Using AutoML)
'''

from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient, Input, Output, dsl, command
from azure.ai.ml.automl import classification
from azure.ai.ml.entities import Environment, AmlCompute
from azure.core.exceptions import ResourceNotFoundError
from dotenv import load_dotenv
import os

load_dotenv()

# === Connect to Azure ML workspace ===
ml_client = MLClient(
    DefaultAzureCredential(),
    subscription_id=os.getenv("AZURE_SUBSCRIPTION_ID"),
    resource_group_name=os.getenv("AZURE_RESOURCE_GROUP"),
    workspace_name=os.getenv("AZURE_WORKSPACE_NAME")
)

# === Ensure environment exists ===
env_name = "ccovc-env"
try:
    env = ml_client.environments.get(name=env_name, label="latest")
    print(f"✅ Environment '{env_name}' already exists. Skipping creation.")
except ResourceNotFoundError:
    env = Environment(
        name=env_name,
        description="Environment for CCOVC AutoML pipeline",
        conda_file="src/environment.yml",
        image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest"
    )
    ml_client.environments.create_or_update(env)
    print(f"✅ Environment '{env.name}' registered.")


# === Ensure compute cluster exists ===
try:
    ml_client.compute.get("cpu-cluster")
    print("✅ Compute 'cpu-cluster' already exists.")
except Exception:
    print("🚀 Creating compute cluster...")
    cluster = AmlCompute(
        name="cpu-cluster",
        size="STANDARD_DS3_V2",
        min_instances=0,
        max_instances=2,
        idle_time_before_scale_down=120
    )
    ml_client.begin_create_or_update(cluster).wait()
    print("✅ Compute 'cpu-cluster' created.")


# === Define Preprocess Component ===
preprocess_component = command(
    name="ccovc_preprocess",
    display_name="Preprocess CCOVC HR data",
    code="./src",
    command="python preprocess.py --cleaned_data_path ${{outputs.cleaned_data_path}}",
    environment=env,
    compute="cpu-cluster",
    outputs={
        "cleaned_data_path": Output(type="uri_file", mode="rw_mount")
    },
    environment_variables={
        "BLOB_CONNECTION_STRING": os.getenv("BLOB_CONNECTION_STRING"),
        "AZURE_SUBSCRIPTION_ID": os.getenv("AZURE_SUBSCRIPTION_ID"),
        "AZURE_RESOURCE_GROUP": os.getenv("AZURE_RESOURCE_GROUP"),
        "AZURE_WORKSPACE_NAME": os.getenv("AZURE_WORKSPACE_NAME")
    },
    allow_reuse=False,
)

# === Define Filter Component ===
filter_component = command(
    name="filter_automl_features",
    display_name="Filter features for AutoML",
    code="./src",
    command="python filter_features.py --input_data ${{inputs.input_data}} --output_data ${{outputs.output_data}}",
    environment=env,
    compute="cpu-cluster",
    inputs={"input_data": Input(type="uri_file")},
    outputs={"output_data": Output(type="mltable", mode="rw_mount")},
    environment_variables={
        "BLOB_CONNECTION_STRING": os.getenv("BLOB_CONNECTION_STRING"),
        "AZURE_SUBSCRIPTION_ID": os.getenv("AZURE_SUBSCRIPTION_ID"),
        "AZURE_RESOURCE_GROUP": os.getenv("AZURE_RESOURCE_GROUP"),
        "AZURE_WORKSPACE_NAME": os.getenv("AZURE_WORKSPACE_NAME")
    },
    allow_reuse=False,
)


register_model_component = command(
    name="register_automl_model",
    display_name="Register AutoML model",
    code="./src",
    command="python register_model_automl.py --model_path ${{inputs.model_path}}",
    environment=env,
    compute="cpu-cluster",
    inputs={
        "model_path": Input(type="mlflow_model")
    },
    environment_variables={
        "AZURE_SUBSCRIPTION_ID": os.getenv("AZURE_SUBSCRIPTION_ID"),
        "AZURE_RESOURCE_GROUP": os.getenv("AZURE_RESOURCE_GROUP"),
        "AZURE_WORKSPACE_NAME": os.getenv("AZURE_WORKSPACE_NAME")
    },
    allow_reuse=False,
)


# === Define AutoML Pipeline ===
@dsl.pipeline(name="ccovc_pipeline_automl", compute="cpu-cluster")
def ccovc_pipeline_automl():
    # Step 1: Preprocess raw data (CSV -> cleaned CSV)
    preprocess_step = preprocess_component()

    # Step 2: Filter features, generate MLTable
    filter_step = filter_component(input_data=preprocess_step.outputs.cleaned_data_path)

    # Step 3: Run AutoML classification
    automl_step = classification(
        training_data=filter_step.outputs.output_data,
        target_column_name="CC / OVC",
        compute="cpu-cluster",
        primary_metric="accuracy",
        outputs={"best_model": Output(type="mlflow_model")}
    )
    automl_step.set_limits(max_trials=10, max_concurrent_trials=2)
    automl_step.set_featurization(mode="auto")
    automl_step.set_training(
                            enable_model_explainability=True,
                            allowed_training_algorithms=[
                                "RandomForest",
                                "LogisticRegression",
                                "LightGBM",
                                "XGBoostClassifier",
                                "DecisionTree"
                            ]
                            )

    # Step 4: Register the best model
    register_step = register_model_component(model_path=automl_step.outputs.best_model)

    return {"best_model": automl_step.outputs.best_model}

# === Submit pipeline job ===
pipeline_job = ccovc_pipeline_automl()
ml_client.jobs.create_or_update(pipeline_job, experiment_name="ccovc_automl_pipeline")
print("✅ AutoML pipeline job submitted.")
