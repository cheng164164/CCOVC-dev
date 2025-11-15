# src/train_automl.py

import pandas as pd
import os
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data", type=str, required=True)
    parser.add_argument("--output_model", type=str, required=True)
    args = parser.parse_args()

    print("🚀 AutoML training script started...")
    print(f"📥 Input MLTable path: {args.input_data}")
    print(f"📤 Output model path: {args.output_model}")

    # In pipeline job, Azure ML handles AutoML training step — nothing to do here.
    # This is only a placeholder if needed for future custom logic.
    os.makedirs(args.output_model, exist_ok=True)
    with open(os.path.join(args.output_model, "placeholder.txt"), "w") as f:
        f.write("AutoML training happened in pipeline component.")
    print("✅ Done.")

if __name__ == "__main__":
    main()