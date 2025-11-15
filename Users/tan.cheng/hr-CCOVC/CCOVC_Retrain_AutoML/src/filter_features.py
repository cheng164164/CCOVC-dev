import pandas as pd
import os
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data", type=str, required=True)
    parser.add_argument("--output_data", type=str, required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.input_data)

    selected = [
        'Legal Entity (Label)',
        'Business unit (Label)',
        'Employment Type (Label)',
        'Job Classification (Label)',
        'CC / OVC'
    ]
    df_filtered = df[selected]

    os.makedirs(args.output_data, exist_ok=True)

    # Write CSV
    csv_path = os.path.join(args.output_data, "filtered_data.csv")
    df_filtered.to_csv(csv_path, index=False)

    # Write MLTable (no extension!)
    mltable_path = os.path.join(args.output_data, "MLTable")
    with open(mltable_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(
            "paths:\n"
            "  - file: ./filtered_data.csv\n"
            "transformations:\n"
            "  - read_delimited:\n"
            "      delimiter: ','\n"
            "      encoding: 'utf-8'\n"
        )

    print("✅ Filtered CSV and MLTable written")


if __name__ == "__main__":
    main()
