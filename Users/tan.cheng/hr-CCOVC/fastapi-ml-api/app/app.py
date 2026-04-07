from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import joblib
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
import sys
from typing import List, Union
from pathlib import Path
import os
from azure.storage.blob import BlobServiceClient
from io import BytesIO
from dotenv import load_dotenv


load_dotenv()

# ==========================
# Define RareCategoryGrouper
# ==========================
class RareCategoryGrouper(BaseEstimator, TransformerMixin):
    def __init__(self, rules: dict = None, new_label="Other"):
        self.rules = rules or {}
        self.new_label = new_label
        self.top_categories_ = {}

    def fit(self, X, y=None):
        X = X.copy()
        for col, top_n in self.rules.items():
            self.top_categories_[col] = X[col].value_counts().nlargest(top_n).index
        return self

    def transform(self, X):
        X = X.copy()
        for col, top_cats in self.top_categories_.items():
            X[col] = X[col].apply(lambda x: x if x in top_cats else self.new_label)
        return X


# === Azure Blob Configuration ===
connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
model_container_name = os.getenv("MODEL_CONTAINER_NAME")
training_data_container_name = os.getenv("TRAINING_DATA_CONTAINER_NAME")
model_blob_name = os.getenv("MODEL_BLOB_NAME")
training_data_blob_name = os.getenv("TRAINING_DATA_BLOB_NAME")

blob_service_client = BlobServiceClient.from_connection_string(connection_string)
model_container_client = blob_service_client.get_container_client(model_container_name)
training_data_container_client = blob_service_client.get_container_client(training_data_container_name)

# === Load model from blob storage ===
model_blob = model_container_client.download_blob(model_blob_name)
model_bytes = BytesIO(model_blob.readall())

sys.modules['__main__'].RareCategoryGrouper = RareCategoryGrouper
model = joblib.load(model_bytes)

# === Load cleaned_data.csv from blob storage ===
csv_blob = training_data_container_client.download_blob(training_data_blob_name)
csv_bytes = BytesIO(csv_blob.readall())
df_cleaned = pd.read_csv(csv_bytes)

# === Feature options for dropdowns ===
def clean_dropdown_options(df: pd.DataFrame, column: str):
    return sorted(
        df[column]
        .dropna()
        .astype(str)
        .str.strip()
        .loc[lambda s: s != ""]
        .unique()
        .tolist()
    )

# === Feature options for dropdowns ===
legal_entity_options = clean_dropdown_options(df_cleaned, "Legal Entity (Label)")
business_unit_options = clean_dropdown_options(df_cleaned, "Business unit (Label)")
employment_type_options = clean_dropdown_options(df_cleaned, "Employment Type (Label)")
division_options = clean_dropdown_options(df_cleaned, "Division (Label)")

# Membership sets (for unseen-value detection)
legal_entity_set = set(legal_entity_options)
business_unit_set = set(business_unit_options)
employment_type_set = set(employment_type_options)
division_set = set(division_options)
job_class_label_set = set(df_cleaned["Job Classification (Label)"].astype(str).unique().tolist())

# --- Job Classification: label + code mapping ---
jc_df = (
    df_cleaned[["Job Classification (Label)", "Job Classification (externalCode)"]]
    .dropna()
    .drop_duplicates()
)

job_classification_options = sorted(
    [
        {
            "label": str(row["Job Classification (Label)"]),
            "code": str(row["Job Classification (externalCode)"]),
            "display": f'{row["Job Classification (Label)"]} ({row["Job Classification (externalCode)"]})',
        }
        for _, row in jc_df.iterrows()
    ],
    key=lambda x: x["label"],
)


app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Class mapping
class_mapping = {0: "CC", 1: "OVC"}

# ## === API health check ===
@app.get("/ping")
def ping():
    return {"status": "OK"}

@app.get("/test", response_class=HTMLResponse)
def test(request: Request):
    return templates.TemplateResponse("form.html", {
        "request": request,
        "legal_entities": legal_entity_options[:5],
        "business_units": business_unit_options[:5],
        "division_labels": division_options[:5],
        "employment_types": employment_type_options[:5],
        "job_classification": job_classification_options[:5],
        "prediction": "TestPrediction",
        "probability": 0.99,
        "Legal_Entity_Label": legal_entity_options[0],
        "Business_Unit_Label": business_unit_options[0],
        "Division_Label": division_options[0],
        "Employment_Type_Label": employment_type_options[0],
        "Job_Classification_Label": job_classification_options[0]["label"]
    })


@app.get("/test_rendering", response_class=HTMLResponse)
def test_html(request: Request):
    return templates.TemplateResponse("form_test.html", {"request": request, "sample": "test"})


## === HTML form route ===
@app.get("/", response_class=HTMLResponse)
def form_page(request: Request):
    return templates.TemplateResponse("form.html", {
    "request": request,
    "legal_entities": legal_entity_options,
    "business_units": business_unit_options,
    "division_labels": division_options,
    "employment_types": employment_type_options,
    "job_classification": job_classification_options
    })


## === Form submission route ===
def confidence_level_from_prob(p: float) -> str:
    if p >= 0.8:
        return "strong"
    elif p >= 0.6:
        return "medium"
    elif p >= 0.5:
        return "low"
    else:
        return "low"

@app.post("/predict_form", response_class=HTMLResponse)
def predict_from_form(
    request: Request,
    Legal_Entity_Label: str = Form(...),
    Business_Unit_Label: str = Form(...),
    Division_Label: str = Form(...),
    Employment_Type_Label: str = Form(...),
    Job_Classification_Label: str = Form(...)
    ):

    df = pd.DataFrame([{
    "Legal Entity (Label)": Legal_Entity_Label,
    "Business unit (Label)": Business_Unit_Label,
    "Division (Label)": Division_Label,
    "Employment Type (Label)": Employment_Type_Label,
    "Job Classification (Label)": Job_Classification_Label
    }])

    pred = model.predict(df)[0]
    probs = model.predict_proba(df)[0]

    prob_cc = float(probs[0])
    prob_ovc = float(probs[1])

    # Use probability of the predicted class
    if pred == 0:  # CC
        prob = prob_cc
    else:  # OVC
        prob = prob_ovc

    confidence_level = confidence_level_from_prob(prob)

    # === Unseen feature detection (field-level flags) ===
    is_new_legal_entity = Legal_Entity_Label not in legal_entity_set
    is_new_business_unit = Business_Unit_Label not in business_unit_set
    is_new_division = Division_Label not in division_set
    is_new_employment_type = Employment_Type_Label not in employment_type_set
    is_new_job_class = Job_Classification_Label not in job_class_label_set

    unseen_fields = []
    if is_new_legal_entity:
        unseen_fields.append("Legal Entity")
    if is_new_business_unit:
        unseen_fields.append("Business Unit")
    if is_new_division:
        unseen_fields.append("Division")
    if is_new_employment_type:
        unseen_fields.append("Employment Type")
    if is_new_job_class:
        unseen_fields.append("Job Classification")

    warning_new_role = ""
    if unseen_fields:
        fields_str = ", ".join(unseen_fields)
        warning_new_role = (
            f"Warning: This is a brand new role and does not exist "
            f"in the current database. Please review and decide the CC/OVC label manually."
        )    

    return templates.TemplateResponse("form.html", {
    "request": request,
    "legal_entities": legal_entity_options,
    "business_units": business_unit_options,
    "division_labels": division_options,
    "employment_types": employment_type_options,
    "job_classification": job_classification_options,
    "prediction": class_mapping[pred],
    "probability": round(float(prob)*100, 2),
    "confidence_level": confidence_level,
    "warning_new_role": warning_new_role,

    # Passed back to retain selected values
    "Legal_Entity_Label": Legal_Entity_Label,
    "Business_Unit_Label": Business_Unit_Label,
    "Division_Label": Division_Label,
    "Employment_Type_Label": Employment_Type_Label,
    "Job_Classification_Label": Job_Classification_Label,

    # flags to tell the template if these are new
    "is_new_legal_entity": is_new_legal_entity,
    "is_new_business_unit": is_new_business_unit,
    "is_new_division": is_new_division,
    "is_new_employment_type": is_new_employment_type,
    "is_new_job_class": is_new_job_class,
    })

## === API prediction endpoint ===
# Define input schema
class InputData(BaseModel):
    Legal_Entity_Label: str
    Business_Unit_Label: str
    Division_Label: str
    Employment_Type_Label: str
    Job_Classification_Label: str

@app.post("/predict")
def predict(input_data: Union[InputData, List[InputData]]):
    if isinstance(input_data, list):
        df = pd.DataFrame([row.dict() for row in input_data])
    else:
        df = pd.DataFrame([input_data.dict()])

    # Rename columns to match training data
    df.columns = ["Legal Entity (Label)", "Business unit (Label)", "Division (Label)", "Employment Type (Label)", "Job Classification (Label)"]

    preds = model.predict(df)
    prob_rows = model.predict_proba(df)

    results = []
    for pred, prob_row in zip(preds, prob_rows):
        prob_cc = float(prob_row[0])
        prob_ovc = float(prob_row[1])

        if pred == 0:  # CC
            prob = prob_cc
        else:  # OVC
            prob = prob_ovc

        results.append({
            "prediction": class_mapping[int(pred)],
            "probability": round(float(prob)*100, 2),
            "confidence_level": confidence_level_from_prob(prob)
        })

    return results if isinstance(input_data, list) else results[0]


@app.post("/predict_file")
def predict_file(file: UploadFile = File(...)):
    df = pd.read_csv(file.file)

    # Rename columns to match training data
    df.columns = ["Legal Entity (Label)", "Business unit (Label)", "Division (Label)", "Employment Type (Label)", "Job Classification (Label)"]

    preds = model.predict(df)
    prob_rows = model.predict_proba(df)

    pred_labels = []
    pred_probs = []
    conf_levels = []

    for pred, prob_row in zip(preds, prob_rows):
        prob_cc = float(prob_row[0])
        prob_ovc = float(prob_row[1])

        if pred == 0:
            prob = prob_cc
        else:
            prob = prob_ovc

        pred_labels.append(class_mapping[int(pred)])
        pred_probs.append(round(float(prob)*100, 2))
        conf_levels.append(confidence_level_from_prob(prob))

    results = pd.DataFrame({
        "prediction": pred_labels,
        "probability": pred_probs,
        "confidence_level": conf_levels
    })

    return results.to_dict(orient="records")


