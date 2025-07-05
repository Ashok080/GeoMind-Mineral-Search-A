import streamlit as st
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

st.set_page_config(page_title="GeoMind Mineral Search AI", layout="wide")
st.title("🧪 GeoMind - Mineral Search AI Dashboard")

uploaded_file = st.file_uploader("📤 Upload a GeoTIFF satellite image", type=["tif"])

if uploaded_file:
    st.success("✅ File uploaded successfully!")
    with rasterio.open(uploaded_file) as src:
        band = src.read(1)
        st.write(f"📏 Raster Dimensions: {band.shape}")
        
        fig, ax = plt.subplots()
        ax.imshow(band, cmap='terrain')
        ax.set_title("📷 Raster Preview")
        st.pyplot(fig)

        flat = band.flatten()
        flat = flat[~np.isnan(flat)].reshape(-1, 1)

        kmeans = KMeans(n_clusters=4)
        kmeans.fit(flat)
        st.write("🔍 KMeans clustering applied.")
        
        fig2, ax2 = plt.subplots()
        ax2.hist(flat, bins=50, color='green')
        ax2.set_title("📊 Value Distribution")
        st.pyplot(fig2)
else:
    st.info("Upload a GeoTIFF (`.tif`) to get started.")
