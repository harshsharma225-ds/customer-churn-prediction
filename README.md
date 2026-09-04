# Customer Churn Prediction

An end-to-end machine learning project that predicts whether a customer 
will churn (leave a subscription service) based on their account and 
usage data. Built with a full pipeline: data preprocessing, model 
training, a REST API, and an interactive web UI.

## Overview

Customer churn is a costly problem for subscription businesses. This 
project trains a classifier on the Telco Customer Churn dataset to 
predict churn risk, then serves that model through a FastAPI backend 
and a Streamlit frontend so it can be used interactively.

## Tech Stack

- **Data & Modeling:** Python, pandas, scikit-learn, XGBoost
- **Backend API:** FastAPI
- **Frontend:** Streamlit
- **Dataset:** [Telco Customer Churn (Kaggle)](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

## Project Structure
  project/
    ├── data/ # Raw dataset
    ├── notebooks/ # Exploratory data analysis
    ├── ml/ # Training script + saved model artifacts
    ├── backend/ # FastAPI app serving predictions
    ├── frontend/ # Streamlit UI
    └── requirements.txt

## How It Works

1. **Data preprocessing & training** (`ml/train.py`) — cleans the data, 
   encodes categorical features, trains Logistic Regression, Random 
   Forest, and XGBoost models, and saves the best-performing model.
2. **Backend** (`backend/app.py`) — loads the trained model and exposes 
   a `/predict` endpoint that accepts customer data and returns a churn 
   prediction with probability.
3. **Frontend** (`frontend/streamlit_app.py`) — a simple form where a 
   user enters customer details and gets a live prediction from the API.

## Running Locally

**1. Install dependencies:**
```bash
pip install -r requirements.txt
```

**2. Start the backend:**
```bash
cd backend
uvicorn app:app --reload
```

**3. Start the frontend (in a new terminal):**
```bash
cd frontend
streamlit run streamlit_app.py
```

**4. Open the app:**
Visit `http://localhost:8501` in your browser.

## Model Performance

The final model (XGBoost) was chosen based on ROC-AUC score across 
three candidate models (Logistic Regression, Random Forest, XGBoost)
