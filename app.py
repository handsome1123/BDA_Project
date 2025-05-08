import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# Load the saved models
kmeans = joblib.load('kmeans_model.pkl')
scaler = joblib.load('scaler.pkl')
pca = joblib.load('pca_model.pkl')

# Streamlit interface
st.title("Customer Segmentation with K-Means")
st.write("This application uses K-Means clustering to segment customers based on RFM (Recency, Frequency, Monetary) analysis.")

# Upload the dataset
uploaded_file = st.file_uploader("Upload your dataset", type=["xlsx", "csv"])
if uploaded_file is not None:
    # Load and display the dataset
    df = pd.read_excel(uploaded_file) if uploaded_file.name.endswith("xlsx") else pd.read_csv(uploaded_file)
    st.write("Dataset Preview", df.head())
    
    # Clean data (same as in the notebook)
    df.dropna(subset=["CustomerID"], inplace=True)
    df = df[df["Quantity"] > 0]
    df = df[df["UnitPrice"] > 0]
    
    # Create TotalPrice column
    df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]
    
    # Calculate RFM values
    ref_date = df["InvoiceDate"].max()
    rfm = df.groupby("CustomerID").agg({
        "InvoiceDate": lambda x: (ref_date - x.max()).days,
        "InvoiceNo": "nunique",
        "TotalPrice": "sum"
    }).rename(columns={
        "InvoiceDate": "Recency",
        "InvoiceNo": "Frequency",
        "TotalPrice": "Monetary"
    })

    # Standardize RFM values
    rfm_scaled = scaler.transform(rfm)

    # Apply PCA transformation
    pca_rfm = pca.transform(rfm_scaled)

    # Predict clusters using the K-Means model
    clusters = kmeans.predict(pca_rfm)
    rfm["Cluster"] = clusters

    # Display results
    st.write(f"Number of clusters: {len(set(clusters))}")
    st.write(rfm.head())

    # Plot clusters
    st.subheader("Customer Segments Visualization")
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=pca_rfm[:, 0], y=pca_rfm[:, 1], hue=clusters, palette="Set2")
    plt.title("Customer Segments via K-Means")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    st.pyplot()
