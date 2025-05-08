# app.py
import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load the saved models
kmeans = joblib.load('kmeans_model.pkl')
scaler = joblib.load('scaler.pkl')
pca = joblib.load('pca_model.pkl')

# Load the dataset (use your actual dataset path or URL)
df = pd.read_excel("Online Retail.xlsx")
df.dropna(subset=["CustomerID"], inplace=True)
df = df[df["Quantity"] > 0]
df = df[df["UnitPrice"] > 0]

# Create TotalPrice column
df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]

# Set reference date for Recency calculation
ref_date = df["InvoiceDate"].max()

# RFM Calculation
rfm = df.groupby("CustomerID").agg({
    "InvoiceDate": lambda x: (ref_date - x.max()).days,
    "InvoiceNo": "nunique",
    "TotalPrice": "sum"
}).rename(columns={
    "InvoiceDate": "Recency",
    "InvoiceNo": "Frequency",
    "TotalPrice": "Monetary"
})

# Standardize RFM
rfm_scaled = scaler.transform(rfm)

# PCA for visualization
pca_rfm = pca.transform(rfm_scaled)

# Predict clusters
rfm["Cluster"] = kmeans.predict(rfm_scaled)

# Streamlit Header
st.title("Customer Segmentation with K-Means")

# Display Cluster Summary
st.subheader("Cluster Summary")
st.write(rfm.groupby("Cluster").mean())

# Display Scatter Plot of Clusters
st.subheader("Customer Segments via K-Means")
fig, ax = plt.subplots(figsize=(8, 6))
sns.scatterplot(x=pca_rfm[:, 0], y=pca_rfm[:, 1], hue=rfm["Cluster"], palette="Set2", ax=ax)
ax.set_title("Customer Segments")
ax.set_xlabel("PCA 1")
ax.set_ylabel("PCA 2")
st.pyplot(fig)

