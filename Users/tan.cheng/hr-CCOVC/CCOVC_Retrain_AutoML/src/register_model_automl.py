# src/register_model_automl.py
import argparse
from azureml.core import Run, Model

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    return parser.parse_args()

def main():
    args = parse_args()

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

if __name__ == "__main__":
    main()
