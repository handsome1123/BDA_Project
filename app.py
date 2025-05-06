import streamlit as st
import pickle
import pandas as pd

# Load model
model = pickle.load(open('model.pkl', 'rb'))

st.title("Retail Campaign Response Predictor")

# Input fields
age = st.slider("Age", 18, 100)
income = st.number_input("Income", 0, 200000)
marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])

# Convert to dataframe
data = pd.DataFrame([[age, income, marital_status]], columns=["Age", "Income", "Marital_Status"])

# Feature engineering (example only)
data["Marital_Status"] = data["Marital_Status"].map({"Single": 0, "Married": 1, "Divorced": 2})

# Predict
if st.button("Predict"):
    result = model.predict(data)
    st.success(f"Prediction: {'Likely to Respond' if result[0] == 1 else 'Not Likely to Respond'}")
