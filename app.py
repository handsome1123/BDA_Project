import streamlit as st
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Load models
kmeans = joblib.load('kmeans_model.pkl')
scaler = joblib.load('scaler.pkl')
pca = joblib.load('pca_model.pkl')

# Load dataset
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
clusters = kmeans.predict(pca_rfm)
rfm["Cluster"] = clusters

# Streamlit header and title
st.title('Customer Segmentation Dashboard')
st.write('This app visualizes customer segmentation based on purchasing behavior.')

# Display cluster summary
st.subheader('Cluster Summary')
summary = rfm.groupby("Cluster").mean()
st.write(summary)

# Plot cluster visualization
st.subheader('Customer Segments Visualization')
plt.figure(figsize=(8, 6))
sns.scatterplot(x=pca_rfm[:, 0], y=pca_rfm[:, 1], hue=clusters, palette="Set2")
plt.title("Customer Segments via K-Means")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
st.pyplot()
