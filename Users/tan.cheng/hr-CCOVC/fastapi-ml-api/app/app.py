from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
import sys
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

# Define input schema
class InputData(BaseModel):
    Company_Label: str
    Position_Country_Label: str
    Business_unit_Label: str
    Division_Label: str
    Employment_Classification_Label: str
    Employee_Type_Label: str
    Employment_Type_Label: str

app = FastAPI()

# Class mapping
class_mapping = {0: "CC", 1: "OVC"}

@app.post("/predict")
def predict(input_data: InputData):
    # Convert to DataFrame
    input_df = pd.DataFrame([input_data.dict()])
    # Rename columns to match training data
    input_df.columns = [
        "Company (Label)", "Position Country (Label)",
        "Division (Label)", "Employee Type (Label)", "Employment Type (Label)"
    ]

    # Predict using pipeline
    pred = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]
    
    return {"prediction": class_mapping[int(pred)], "probability": round(prob, 4)}