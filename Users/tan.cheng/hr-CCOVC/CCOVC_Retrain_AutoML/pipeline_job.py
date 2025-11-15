'''
## Custom Azure ML Pipeline for CCOVC Data Processing and Model Training (Using custom training code)
'''

from azure.ai.ml import MLClient, command, Input, Output, dsl
from azure.ai.ml.automl import classification
from azure.identity import DefaultAzureCredential
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


# === Define Preprocessing Component ===
preprocess_component = command(
    name="ccovc_preprocess",
    display_name="Preprocess CCOVC HR data",
    code="./src",
    command="python preprocess.py --cleaned_data_path ${{outputs.cleaned_data_path}}",
    environment=env,
    environment_variables={
        "BLOB_CONNECTION_STRING": os.getenv("BLOB_CONNECTION_STRING"),
        "AZURE_SUBSCRIPTION_ID": os.getenv("AZURE_SUBSCRIPTION_ID"),
        "AZURE_RESOURCE_GROUP": os.getenv("AZURE_RESOURCE_GROUP"),
        "AZURE_WORKSPACE_NAME": os.getenv("AZURE_WORKSPACE_NAME"),
    },
    compute="cpu-cluster",
    outputs={
        "cleaned_data_path": Output(type="uri_file", mode="rw_mount")
    },
    allow_reuse=False,
)



# === Define Custom Training Component ===
train_component = command(
    name="ccovc_train_custom",
    display_name="Train Random Forest model on CCOVC data",
    code="./src",
    command=(
        "python train.py "
        "--cleaned_data_path ${{inputs.cleaned_data_path}} "
    ),
    environment=env,
    environment_variables={
        "BLOB_CONNECTION_STRING": os.getenv("BLOB_CONNECTION_STRING"),
        "AZURE_SUBSCRIPTION_ID": os.getenv("AZURE_SUBSCRIPTION_ID"),
        "AZURE_RESOURCE_GROUP": os.getenv("AZURE_RESOURCE_GROUP"),
        "AZURE_WORKSPACE_NAME": os.getenv("AZURE_WORKSPACE_NAME"),
    },
    compute="cpu-cluster",
    inputs={
        "cleaned_data_path": Input(type="uri_file")
    },
    outputs={
        "model_output": Output(type="uri_folder", mode="rw_mount")    # The output is not acutally used in train.py but needed for pipeline ordering
    },
    allow_reuse=False,
)


# === Define Register Model Component ===
register_model_custom_component = command(
    name="register_ccovc_model_custom",
    display_name="Register custom CCOVC model (.pkl)",
    code="./src",
    command="python register_model_custom.py --model_path ${{inputs.model_path}}",
    inputs={"model_path": Input(type="uri_folder")},        # The Input (which is output from train_component) is not acutally used in train.py but needed for pipeline ordering
    environment=env,
    environment_variables={
    "BLOB_CONNECTION_STRING": os.getenv("BLOB_CONNECTION_STRING"),
    "AZURE_SUBSCRIPTION_ID": os.getenv("AZURE_SUBSCRIPTION_ID"),
    "AZURE_RESOURCE_GROUP": os.getenv("AZURE_RESOURCE_GROUP"),
    "AZURE_WORKSPACE_NAME": os.getenv("AZURE_WORKSPACE_NAME")
    },
    compute="cpu-cluster",
    allow_reuse=False,
    )


# === Pipeline A: Custom training ===
@dsl.pipeline(name="ccovc_pipeline_custom", compute="cpu-cluster")
def ccovc_pipeline_custom():
    preprocess_step = preprocess_component()
    train_step = train_component(cleaned_data_path=preprocess_step.outputs.cleaned_data_path)
    register_step = register_model_custom_component(model_path=train_step.outputs.model_output)


# === Submit pipeline job ===
pipeline_job = ccovc_pipeline_custom()
print("Submitting job...")
pipeline_job = ml_client.jobs.create_or_update(pipeline_job, experiment_name="ccovc_custom_pipeline")
print(f"🚀 Submitted pipeline job: {pipeline_job.name}")

