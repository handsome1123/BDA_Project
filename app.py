import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# Load dataset
df = pd.read_excel("Online Retail.xlsx")
df.dropna(subset=["CustomerID"], inplace=True)
df = df[df["Quantity"] > 0]
df = df[df["UnitPrice"] > 0]

# ✅ Create TotalPrice column
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
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm)

# PCA for visualization
pca = PCA(n_components=2)
pca_rfm = pca.fit_transform(rfm_scaled)

# K-Means Clustering
kmeans = KMeans(n_clusters=4, random_state=42)
clusters = kmeans.fit_predict(rfm_scaled)
rfm["Cluster"] = clusters

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

# Save models for later use (optional)
joblib.dump(kmeans, 'kmeans_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(pca, 'pca_model.pkl')

st.write("Models saved as kmeans_model.pkl, scaler.pkl, pca_model.pkl.")
