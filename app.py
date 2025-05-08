import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# Streamlit page config
st.set_page_config(page_title="Customer Segmentation with K-Means", layout="wide")

# Title
st.title("🔍 Customer Segmentation App with K-Means Clustering")

# Sidebar to upload the dataset
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    # Load dataset
    df = pd.read_csv(uploaded_file)
    
    # Preprocess data: Create TotalPrice column and calculate Recency, Frequency, and Monetary
    df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]
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
    
    # Standardize data
    scaler = joblib.load('scaler.pkl')
    rfm_scaled = scaler.transform(rfm)

    # Apply PCA for 2D visualization
    pca = joblib.load('pca_model.pkl')
    rfm_pca = pca.transform(rfm_scaled)

    # Sidebar to select the number of clusters
    n_clusters = st.sidebar.slider("Select number of clusters (k)", 2, 10, 4)

    # Load the KMeans model and predict clusters
    kmeans = joblib.load('kmeans_model.pkl')
    clusters = kmeans.predict(rfm_pca)
    
    # Add the cluster predictions to the DataFrame
    rfm["Cluster"] = clusters

    # Display the RFM table with clusters
    st.subheader("Customer Segmentation Results")
    st.write(rfm)

    # Plot clusters
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=rfm_pca[:, 0], y=rfm_pca[:, 1], hue=clusters, palette="Set2")
    plt.title(f"Customer Segments with k = {n_clusters}")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    st.pyplot()

    # Display summary of clusters
    st.subheader("Cluster Summary")
    summary = rfm.groupby("Cluster").mean()
    st.write(summary)
