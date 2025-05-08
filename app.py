import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

st.set_page_config(page_title="RFM Customer Segmentation", layout="wide")
st.title("Project2_Group-15")

# Load data and models
@st.cache_data
def load_data_and_models():
    rfm = pd.read_csv("rfm_with_clusters.csv", index_col="CustomerID")
    with open("kmeans_model.pkl", "rb") as f:
        kmeans = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return rfm, kmeans, scaler

rfm, kmeans, scaler = load_data_and_models()

# Show RFM + Cluster table
st.subheader("RFM Table with Cluster Labels")
st.dataframe(rfm.reset_index(), use_container_width=True)

# PCA Visualization
st.subheader("Cluster Visualization using PCA")

# Perform PCA again on the RFM features
rfm_features = rfm[["Recency", "Frequency", "Monetary"]]
rfm_scaled = scaler.transform(rfm_features)

pca = PCA(n_components=2)
pca_result = pca.fit_transform(rfm_scaled)

rfm["PCA1"] = pca_result[:, 0]
rfm["PCA2"] = pca_result[:, 1]

fig, ax = plt.subplots(figsize=(8, 6))
sns.scatterplot(data=rfm, x="PCA1", y="PCA2", hue="Cluster", palette="Set2", ax=ax)
ax.set_title("Customer Segments via PCA")
st.pyplot(fig)

# Cluster Summary
st.subheader("Average RFM Values per Cluster")
cluster_summary = rfm.groupby("Cluster")[["Recency", "Frequency", "Monetary"]].mean()
st.dataframe(cluster_summary)
