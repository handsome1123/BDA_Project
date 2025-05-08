# app.py
import streamlit as st
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# Load the saved models
kmeans = joblib.load('kmeans_model.pkl')
scaler = joblib.load('scaler.pkl')
pca = joblib.load('pca_model.pkl')

# Set up the Streamlit app interface
st.title("Customer Segmentation with K-Means Clustering")

# Upload CSV file for prediction
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    # Load the uploaded data
    new_data = pd.read_csv(uploaded_file)

    # Preprocess data as required (you may need to modify this depending on the new dataset structure)
    new_data["TotalPrice"] = new_data["Quantity"] * new_data["UnitPrice"]
    
    # Perform RFM calculations for new data
    new_rfm = new_data.groupby("CustomerID").agg({
        "InvoiceDate": lambda x: (new_data["InvoiceDate"].max() - x.max()).days,
        "InvoiceNo": "nunique",
        "TotalPrice": "sum"
    }).rename(columns={
        "InvoiceDate": "Recency",
        "InvoiceNo": "Frequency",
        "TotalPrice": "Monetary"
    })

    # Standardize new data using the scaler
    new_rfm_scaled = scaler.transform(new_rfm)

    # Apply PCA transformation
    new_rfm_pca = pca.transform(new_rfm_scaled)

    # Predict clusters for new data
    new_clusters = kmeans.predict(new_rfm_pca)

    # Display results
    new_rfm["Cluster"] = new_clusters
    st.write(new_rfm)

    # Plot the clusters
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=new_rfm_pca[:, 0], y=new_rfm_pca[:, 1], hue=new_clusters, palette="Set2")
    st.pyplot()

# Display summary
st.subheader("Cluster Summary")
summary = pd.DataFrame(kmeans.cluster_centers_, columns=new_rfm.columns)
st.write(summary)
