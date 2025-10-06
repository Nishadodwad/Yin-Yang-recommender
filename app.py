import streamlit as st
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np

# --- Cache model loading ---
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

# --- Cache embeddings creation ---
@st.cache_data
def build_index(df):
    embeddings = model.encode(df['description'].tolist(), convert_to_numpy=True)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index, embeddings

# --- Load data ---
@st.cache_data
def load_data():
    # Replace with your dataset path
    df = pd.read_csv("products.csv")
    return df

# Streamlit UI
st.title("☯️ Yin–Yang Recommendation Engine")
st.write("Find products by energy type and query search.")

# Load model & data
with st.spinner("Loading model..."):
    model = load_model()

df = load_data()
index, embeddings = build_index(df)

# --- Sidebar filter ---
energy_filter = st.selectbox("Select Energy Type:", ["All", "YIN", "YANG", "NEUTRAL"])

# --- Query input ---
query = st.text_input("Enter your product search query:")

# --- Recommendation logic ---
if query:
    query_embedding = model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, 10)

    results = df.iloc[indices[0]]

    if energy_filter != "All":
        results = results[results["energy"].str.upper() == energy_filter]

    st.subheader("Recommended Products:")
    for _, row in results.iterrows():
        st.write(f"**{row['name']}** ({row['energy']}) — {row['description']}")
