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

    
# Load model
sys.modules['__main__'].RareCategoryGrouper = RareCategoryGrouper
model = joblib.load("model/Random_Forest_CCOVC_Model.pkl")

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# === Feature options for dropdowns ===
legal_entity_options = ["Joy Global Underground", "Komatsu America Corp.", "Komatsu America Industries"]
division_options = ["Parts", "Selling", "Manufacturing"]
employment_type_options = ["Hourly", "Salaried", "Temporary Contractor"]

# Class mapping
class_mapping = {0: "CC", 1: "OVC"}

# ## === API health check ===
# @app.get("/")
# def read_root():
#     return {"message": "Hello from FastAPI!"}


## === HTML form route ===
@app.get("/", response_class=HTMLResponse)
def form_page(request: Request):
    return templates.TemplateResponse("form.html", {
    "request": request,
    "legal_entities": legal_entity_options,
    "divisions": division_options,
    "employment_types": employment_type_options
    })


## === Form submission route ===
@app.post("/predict_form", response_class=HTMLResponse)
def predict_from_form(
    request: Request,
    Legal_Entity_Label: str = Form(...),
    Division_Label: str = Form(...),
    Employment_Type_Label: str = Form(...)
    ):
    df = pd.DataFrame([{
    "Legal Entity (Label)": Legal_Entity_Label,
    "Division (Label)": Division_Label,
    "Employment Type (Label)": Employment_Type_Label
    }])

    pred = model.predict(df)[0]
    prob = model.predict_proba(df)[0, 1]

    return templates.TemplateResponse("form.html", {
    "request": request,
    "legal_entities": legal_entity_options,
    "divisions": division_options,
    "employment_types": employment_type_options,
    "prediction": class_mapping[pred],
    "probability": round(float(prob), 4),

    # Passed back to retain selected values
    "Legal_Entity_Label": Legal_Entity_Label,
    "Division_Label": Division_Label,
    "Employment_Type_Label": Employment_Type_Label
    })


## === API prediction endpoint ===
# Define input schema
class InputData(BaseModel):
    Legal_Entity_Label: str
    Division_Label: str
    Employment_Type_Label: str

@app.post("/predict")
def predict(input_data: Union[InputData, List[InputData]]):
    if isinstance(input_data, list):
        df = pd.DataFrame([row.dict() for row in input_data])
    else:
        df = pd.DataFrame([input_data.dict()])

    # Rename columns to match training data
    df.columns = ["Legal Entity (Label)", "Division (Label)", "Employment Type (Label)"]

    preds = model.predict(df)
    probs = model.predict_proba(df)[:, 1]

    results = []
    for pred, prob in zip(preds, probs):
        results.append({
            "prediction": class_mapping[int(pred)],
            "probability": round(float(prob), 4)
        })

    return results if isinstance(input_data, list) else results[0]


@app.post("/predict_file")
def predict_file(file: UploadFile = File(...)):
    df = pd.read_csv(file.file)

    # Rename columns to match training data
    df.columns = ["Legal Entity (Label)", "Division (Label)", "Employment Type (Label)"]

    preds = model.predict(df)
    probs = model.predict_proba(df)[:, 1]

    results = pd.DataFrame({
        "prediction": [class_mapping[int(p)] for p in preds],
        "probability": [round(float(prob), 4) for prob in probs]
    })

    return results.to_dict(orient="records")


