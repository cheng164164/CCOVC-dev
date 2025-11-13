import pandas as pd
import numpy as np


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean raw CCOVC dataset according to EDA notebook logic.
    """

    ## 1. Ensure consistent dtypes
    df_clean = df.astype(object)

    ## 2. Drop rows with critical nulls
    if "CC / OVC (External Code)" in df_clean.columns:
        df_clean = df_clean.dropna(subset=["CC / OVC (External Code)"])

    ## 3. Remove invalid 'DSC' codes
    if "CC / OVC (External Code)" in df_clean.columns:
        df_clean = df_clean[df_clean["CC / OVC (External Code)"] != "DSC"]

    ## 4. Keep only active employees
    if "Employee Status (Label)" in df_clean.columns:
        df_clean = df_clean[df_clean["Employee Status (Label)"] == "Active"]

    ## 5. Drop duplicate Employee IDs (keep last)
    if "Employee ID" in df_clean.columns:
        df_clean = df_clean.drop_duplicates(subset=["Employee ID"], keep="last")

    ## 6. Drop unnecessary columns and reorder so target is first
    data = df_clean.drop(columns=['Employee ID', 'Employee Status (Label)', 'Position CC / OVC (Picklist Label)', 'Legal Entity (Legal Entity Code)', 'Job Classification (externalCode)', \
     'Job Classification', 'Cost Center (Cost Center Code)', 'Job Function', 'Job Function (Job Function Code)'])
    target = data.pop('CC / OVC (External Code)')
    data.insert(0, 'CC / OVC', target)  # move target column to the front of df

    ## 7. Check for ambiguous rows
    # Step 1: Define columns to compare (exclude 'CC / OVC' and 'Class Type')
    df_temp = df.copy()

    df_temp = df_temp.drop(columns=['Position CC / OVC (Picklist Label)'])
    df_temp = df_temp[df_temp['CC / OVC (External Code)'] != 'DSC']    # drop rows with 'DSC' in the 'CC / OVC (External Code)' column
    df_temp = df_temp[df_temp['Employee Status (Label)'] == 'Active']  # drop non active employees
    df_temp = df_temp.drop_duplicates(subset=['Employee ID'], keep='last')

    compare_cols = [col for col in df_temp.columns if col not in ["CC / OVC (External Code)", 'Employee ID']]
    df_temp[compare_cols] = df_temp[compare_cols].fillna("__MISSING__")

    # Step 2: Group by all other columns and count unique CC/OVC values
    duplicate_except_ccovc = (
        df_temp
        .groupby(compare_cols)["CC / OVC (External Code)"]
        .nunique()
        .reset_index()
    )

    # Step 3: Filter groups with more than one unique CC/OVC value (i.e., conflict)
    ambiguous_rows = duplicate_except_ccovc[duplicate_except_ccovc["CC / OVC (External Code)"] > 1]

    # Step 4: Join back to full data to get full rows for review
    conflict_df = df_temp.merge(ambiguous_rows[compare_cols], on=compare_cols, how="inner")

    # Step 5: Sort for clarity
    conflict_df = conflict_df.sort_values(by=[*compare_cols, "CC / OVC (External Code)"]).reset_index(drop=True)

    # Final result
    print('number of ambiguous rows:', conflict_df.shape[0])

    ## 8. Remove ambiguous rows from data
    # Fill missing values in data for accurate comparison
    data = data.fillna("__MISSING__")
    # Determine shared columns between the two dataframes
    common_cols = list(set(conflict_df.columns) & set(data.columns))
    print(common_cols)
    # Drop conflicting rows from data
    final_clean_df = data.merge(conflict_df[common_cols], how='left', on=common_cols, indicator=True)
    final_clean_df = final_clean_df[final_clean_df['_merge'] == 'left_only'].drop(columns=['_merge'])

    print(data.shape, '->' ,final_clean_df.shape)
    final_clean_df = final_clean_df.replace('__MISSING__', np.nan)  # revert back to NaN

    return final_clean_df