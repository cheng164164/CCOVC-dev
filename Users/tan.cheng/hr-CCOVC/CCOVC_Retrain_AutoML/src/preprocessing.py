# src/preprocessing.py
import pandas as pd
import numpy as np

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    df_clean = df.astype(object)

    if "CC / OVC (External Code)" in df_clean.columns:
        df_clean = df_clean.dropna(subset=["CC / OVC (External Code)"])
        df_clean = df_clean[df_clean['CC / OVC (External Code)'].isin(['CC', 'OVC'])]    # keep only rows with 'CC' or 'OVC' in the 'CC / OVC (External Code)' column

    if "Employee Status (Label)" in df_clean.columns:
        df_clean = df_clean[df_clean["Employee Status (Label)"] == "Active"]

    if "Employee ID" in df_clean.columns:
        df_clean = df_clean.drop_duplicates(subset=["Employee ID"], keep="last")

    data = df_clean.drop(columns=[
        'Employee ID', 'Employee Status (Label)', 'Position CC / OVC (Picklist Label)',
        'Legal Entity (Legal Entity Code)',
        'Job Classification', 'Cost Center (Cost Center Code)', 'Job Function',
        'Job Function (Job Function Code)'
    ], errors="ignore")

    target = data.pop('CC / OVC (External Code)')
    data.insert(0, 'CC / OVC', target)

    # === Ambiguous row filtering ===
    df_temp = df.copy()
    df_temp = df_temp.drop(columns=['Position CC / OVC (Picklist Label)'], errors="ignore")
    df_temp = df_temp[df_temp['CC / OVC (External Code)'].isin(['CC', 'OVC'])]    # keep only rows with 'CC' or 'OVC' in the 'CC / OVC (External Code)' column
    # df_temp = df_temp[df_temp['Employee Status (Label)'] == 'Active']
    df_temp = df_temp.drop_duplicates(subset=['Employee ID'], keep='last')

    compare_cols = [col for col in df_temp.columns if col not in ["CC / OVC (External Code)", 'Employee ID']]
    df_temp[compare_cols] = df_temp[compare_cols].fillna("__MISSING__")

    conflict_keys = (
        df_temp.groupby(compare_cols)["CC / OVC (External Code)"]
        .nunique()
        .reset_index()
        .query("`CC / OVC (External Code)` > 1")
    )

    conflict_df = df_temp.merge(conflict_keys[compare_cols], on=compare_cols, how="inner")
    data = data.fillna("__MISSING__")
    common_cols = list(set(conflict_df.columns) & set(data.columns))

    final_clean_df = data.merge(conflict_df[common_cols], how='left', on=common_cols, indicator=True)
    final_clean_df = final_clean_df[final_clean_df['_merge'] == 'left_only'].drop(columns=['_merge'])

    final_clean_df = final_clean_df.replace('__MISSING__', np.nan)
    print(f"📊 Cleaned shape: {data.shape} ➡ {final_clean_df.shape}")

    return final_clean_df
