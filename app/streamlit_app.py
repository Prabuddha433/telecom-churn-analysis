# app/streamlit_app.py
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import streamlit as st # pyright: ignore[reportMissingImports]
import pandas as pd # pyright: ignore[reportMissingModuleSource]
import joblib
import numpy as np
import plotly.express as px
from sklearn.preprocessing import StandardScaler

from src.model import train_and_save


MODEL_PATH = "model/churn_model.pkl"
DATA_PATH = "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"

st.set_page_config(layout="wide", page_title="Telco Churn Dashboard")

@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'].replace(' ', 0))
    df = df.replace({'No internet service': 'No', 'No phone service': 'No'})
    return df


@st.cache_resource
def get_model():
    if not os.path.exists(MODEL_PATH):
        st.warning("Model not found. Training model...")
        train_and_save(
            raw_csv_path=DATA_PATH,
            output_model_path=MODEL_PATH
        )
    return joblib.load(MODEL_PATH)

model_obj = get_model()
model = model_obj["model"]
model_columns = model_obj["columns"]

df = load_data(DATA_PATH)
st.title("Telco Customer Churn - Quick Dashboard")

# top KPIs
churn_rate = (df['Churn'] == 'Yes').mean()
st.metric("Overall Churn Rate", f"{churn_rate:.2%}")

# show some filters
col1, col2 = st.columns(2)
with col1:
    tenure_range = st.slider("Tenure (months)", int(df['tenure'].min()), int(df['tenure'].max()), (1, 72))
with col2:
    monthly_charges_max = st.slider("Max Monthly Charges", int(df['MonthlyCharges'].min()), int(df['MonthlyCharges'].max()), int(df['MonthlyCharges'].max()))

filtered = df[(df['tenure'] >= tenure_range[0]) & (df['tenure'] <= tenure_range[1]) & (df['MonthlyCharges'] <= monthly_charges_max)]

st.subheader("Churn rate by Contract Type")
ct = filtered.groupby('Contract')['Churn'].apply(lambda x: (x=='Yes').mean()).reset_index(name='churn_rate')
fig = px.bar(ct, x='Contract', y='churn_rate', labels={'churn_rate':'Churn Rate'})
st.plotly_chart(fig, use_container_width=True)

st.subheader("Predict churn for a sample customer")
st.write("Fill details for a sample prediction. The model used is a Random Forest trained on the Telco dataset.")

# load model
try:
    model_obj = load_model(MODEL_PATH)
    model = model_obj['model']
    columns = model_obj['columns']
except Exception as e:
    st.error("Model not found. Please run training script to produce model/churn_model.pkl")
    st.stop()

# Simple input form: we will build only a few fields and map them to encoded features
with st.form("predict_form"):
    gender = st.selectbox("Gender", options=['Male','Female'])
    senior = st.selectbox("SeniorCitizen", options=[0,1])
    partner = st.selectbox("Partner", options=['Yes','No'])
    dependents = st.selectbox("Dependents", options=['Yes','No'])
    tenure = st.slider("tenure", 0, 72, 12)
    monthly = st.number_input("MonthlyCharges", min_value=0.0, max_value=1000.0, value=70.0)
    total = st.number_input("TotalCharges", min_value=0.0, max_value=100000.0, value=1000.0)
    contract = st.selectbox("Contract", options=['Month-to-month','One year','Two year'])
    submitted = st.form_submit_button("Predict")

if submitted:
    # Build a single-row dataframe with basic features and one-hot columns like training
    sample = {
        'SeniorCitizen': senior,
        'tenure': tenure,
        'MonthlyCharges': monthly,
        'TotalCharges': total,
        'gender_Male': 1 if gender=='Male' else 0,
        'Partner_Yes': 1 if partner=='Yes' else 0,
        'Dependents_Yes': 1 if dependents=='Yes' else 0,
        'Contract_One year': 1 if contract=='One year' else 0,
        'Contract_Two year': 1 if contract=='Two year' else 0
    }
    # create full columns list and fill 0 for missing
    X = pd.DataFrame([sample])
    for c in columns:
        if c not in X.columns:
            X[c] = 0
    X = X[columns]
    proba = model.predict_proba(X)[:,1][0]
    pred = model.predict(X)[0]
    st.write(f"Predicted probability of churn: **{proba:.2%}**")
    st.write("Model prediction:", "Will churn" if pred==1 else "Will stay")

st.markdown("---")
st.write("Dataset sample:")
st.dataframe(filtered.head())
