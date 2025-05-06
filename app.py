import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load model and scaler
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.title("Retail Campaign Response Predictor")

# Input UI for all features
age = st.slider("Age", 18, 100)
income = st.number_input("Income", 0, 200000)
kidhome = st.slider("Number of Kids at Home", 0, 3)
teenhome = st.slider("Number of Teenagers at Home", 0, 3)
recency = st.slider("Last Purchase (Days Ago)", 0, 100)
mnt_wines = st.number_input("Wine Spending", 0, 2000)
mnt_fruits = st.number_input("Fruits Spending", 0, 1000)
mnt_meat = st.number_input("Meat Products Spending", 0, 2000)
mnt_fish = st.number_input("Fish Products Spending", 0, 1000)
mnt_sweets = st.number_input("Sweet Products Spending", 0, 1000)
mnt_gold = st.number_input("Gold Products Spending", 0, 2000)
num_deals = st.slider("Number of Deals Purchases", 0, 10)
num_web = st.slider("Number of Web Purchases", 0, 20)
num_catalog = st.slider("Number of Catalog Purchases", 0, 20)
num_store = st.slider("Number of Store Purchases", 0, 20)
num_web_visits = st.slider("Web Visits Last Month", 0, 20)
accepted_cmp1 = st.checkbox("Accepted Campaign 1")
accepted_cmp2 = st.checkbox("Accepted Campaign 2")
accepted_cmp3 = st.checkbox("Accepted Campaign 3")
accepted_cmp4 = st.checkbox("Accepted Campaign 4")
accepted_cmp5 = st.checkbox("Accepted Campaign 5")

# Create DataFrame with same column order as training
input_data = pd.DataFrame([[
    age, income, kidhome, teenhome, recency, mnt_wines, mnt_fruits, mnt_meat, mnt_fish,
    mnt_sweets, mnt_gold, num_deals, num_web, num_catalog, num_store, num_web_visits,
    int(accepted_cmp1), int(accepted_cmp2), int(accepted_cmp3), int(accepted_cmp4), int(accepted_cmp5)
]], columns=[
    'Age', 'Income', 'Kidhome', 'Teenhome', 'Recency', 'MntWines', 'MntFruits',
    'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds',
    'NumDealsPurchases', 'NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases',
    'NumWebVisitsMonth', 'AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3', 
    'AcceptedCmp4', 'AcceptedCmp5'
])

# Scale the data
input_scaled = scaler.transform(input_data)

# Predict
if st.button("Predict"):
    prediction = model.predict(input_scaled)
    st.success(f"Prediction: {'Will Respond' if prediction[0] == 1 else 'Will Not Respond'}")
