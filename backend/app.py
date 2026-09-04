from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import os 
app = FastAPI(title="Customer Churn Prediction API")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(os.path.join(BASE_DIR, '..', 'ml', 'model.pkl'))
scaler = joblib.load(os.path.join(BASE_DIR, '..', 'ml', 'scaler.pkl'))
model_columns = joblib.load(os.path.join(BASE_DIR, '..', 'ml', 'model_columns.pkl'))

class CustomerData(BaseModel):
    gender: str
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float

@app.get("/")
def read_root():
    return {"message": "Customer Churn Prediction API is running"}

@app.post("/predict")
def predict_churn(data: CustomerData):
    # Convert incoming data into a DataFrame (same shape as one row)
    input_df = pd.DataFrame([data.dict()])

    binary_map = {'Yes': 1, 'No': 0}
    for col in ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']:
        input_df[col] = input_df[col].map(binary_map)

    input_df['gender'] = input_df['gender'].map({'Male': 1, 'Female': 0})

    multi_cat_cols = [
        'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
        'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
        'Contract', 'PaymentMethod'
    ]
    input_df = pd.get_dummies(input_df, columns=multi_cat_cols)

    input_df = input_df.reindex(columns=model_columns, fill_value=0)

    numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    return {
        "churn_prediction": "Yes" if prediction == 1 else "No",
        "churn_probability": round(float(probability), 4)
    }