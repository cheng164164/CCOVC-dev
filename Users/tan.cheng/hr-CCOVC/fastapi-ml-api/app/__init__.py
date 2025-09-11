# app.py
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# Load model
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
    
    return {"prediction": int(pred), "probability": round(prob, 4)}