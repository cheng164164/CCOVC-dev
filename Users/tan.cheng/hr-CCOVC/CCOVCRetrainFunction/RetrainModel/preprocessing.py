import pandas as pd


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean raw CCOVC dataset according to EDA notebook logic.
    """

    # 1. Ensure consistent dtypes
    df = df.astype(object)

    # 2. Drop rows with critical nulls
    if "CC / OVC (External Code)" in df.columns and "Job Function" in df.columns:
        df = df.dropna(subset=["CC / OVC (External Code)", "Job Function"])

    # 3. Remove invalid 'DSC' codes
    if "CC / OVC (External Code)" in df.columns:
        df = df[df["CC / OVC (External Code)"] != "DSC"]

    # 4. Keep only active employees
    if "Employee Status (Label)" in df.columns:
        df = df[df["Employee Status (Label)"] == "Active"]

    # 5. Drop duplicate Employee IDs (keep last)
    if "Employee ID" in df.columns:
        df = df.drop_duplicates(subset=["Employee ID"], keep="last")

    # 6. Drop unnecessary columns and reorder so target is first
    data = df.drop(columns=['Employee ID', 'Employee Status (Label)', 'Position CC / OVC (Picklist Label)', 'Legal Entity (Legal Entity Code)', 'Job Classification (externalCode)', \
     'Job Classification', 'Cost Center (Cost Center Code)', 'Job Function', 'Job Function (Job Function Code)'])
    target = data.pop('CC / OVC (External Code)')
    data.insert(0, 'CC / OVC', target)  # move target column to the front of df

    return data