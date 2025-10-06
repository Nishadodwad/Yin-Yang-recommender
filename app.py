
# app.py
# Simplified Yin–Yang Recommendation Engine
# Only: query + energy preference (Yin / Neutral / Yang)
# Minimal filters, fast UI, cached model & index

import streamlit as st
import json
import uuid
import random
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from pathlib import Path

st.set_page_config(page_title="Yin–Yang Recommender (Simple)", page_icon="☯️", layout="wide")

# --- Small CSS for nicer cards ---
st.markdown(
    """
    <style>
    .card { background: #fff; border-radius:10px; padding:12px; box-shadow: 0 4px 12px rgba(0,0,0,0.06); margin-bottom:12px; }
    .title { font-size:16px; font-weight:700; }
    .muted { color:#6c6c6c; font-size:13px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------
# Config - adjust as needed
# -------------------------
DATA_PATH = Path("data/data-set (1).json")  # update if your data file has a different name
PLACEHOLDER_IMG = "https://picsum.photos/seed/{}/320/200"

# -------------------------
# Load & enrich dataset
# -------------------------
@st.cache_data(show_spinner=False)
def load_and_enrich_dataset(filepath: Path):
    if not filepath.exists():
        raise FileNotFoundError(f"Dataset not found at: {filepath.resolve()}")
    raw = json.loads(filepath.read_text(encoding="utf8"))
    data = []
    for item in raw:
        it = dict(item)
        it['id'] = str(it.get('id') or uuid.uuid4())
        # title + description
        title_parts = []
        for key in ["Model Number", "Features", "Durability", "Battery Voltage"]:
            if key in it and it[key]:
                title_parts.append(str(it[key]))
        it['title'] = " ".join(title_parts) or f"Product {it['id']}"
        desc_parts = []
        for k, v in it.items():
            if k.lower() != "id":
                desc_parts.append(f"{k}: {v}")
        it['description'] = ". ".join(desc_parts)[:800]
        # sensible defaults
        pol = float(it.get('polarity', random.uniform(-1, 1)))
        it['polarity'] = round(pol, 3)
        it['yin_score'] = round((1 - pol) / 2, 3)
        it['yang_score'] = round((1 + pol) / 2, 3)
        it['elements'] = it.get('elements') or ["water"]
        it['seasonality'] = it.get('seasonality') or ["spring"]
        it['price'] = int(it.get('price', random.randint(500, 5000)))
        it['popularity'] = int(it.get('popularity', random.randint(1, 100)))
        it['image_url'] = it.get('image_url', "")
        data.append(it)
    return data

# -------------------------
# Build model & index once
# -------------------------
@st.cache_resource(show_spinner=False)
def build_model_and_index(data):
    # load model
    model = SentenceTransformer("all-MiniLM-L6-v2")
    # create corpus
    corpus = [f"{d['title']} {d['description']}" for d in data]
    embeddings = model.encode(corpus, normalize_embeddings=True)
    embeddings = np.array(embeddings).astype('float32')
    dim = embeddings.shape[1]
    # Use Inner Product on normalized vectors => cosine similarity
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return model, index, embeddings

# -------------------------
# Search function
# -------------------------
def search(query: str, model, index, data, top_k: int = 5, balance: str = None):
    if not query.strip():
        return []
    q_vec = model.encode([query], normalize_embeddings=True).astype('float32')
    # fetch extra to allow filtering
    D, I = index.search(q_vec, top_k * 3)
    results = []
    seen = set()
    for i in I[0]:
        if i < 0 or i >= len(data):
            continue
        item = data[i]
        if item['id'] in seen:
            continue
        # energy filter
        if balance == "Yin" and item['polarity'] > 0:
            continue
        if balance == "Yang" and item['polarity'] < 0:
            continue
        results.append(item)
        seen.add(item['id'])
        if len(results) >= top_k:
            break
    return results

# -------------------------
# UI - Simple
# -------------------------
st.title("☯️ Yin–Yang Recommendation Engine — Simple")
st.write("Enter a search query and choose an energy preference (Yin / Neutral / Yang).")

# top controls
col1, col2, col3 = st.columns([6, 2, 2])
with col1:
    query = st.text_input("🔍 Search query", value="vacuum cleaner")
with col2:
    energy = st.selectbox("Energy", options=["Neutral", "Yin", "Yang"], index=0)
with col3:
    top_k = st.selectbox("Top K", options=[3, 5, 7, 10], index=1)

# Load data and model (show spinner)
try:
    data = load_and_enrich_dataset(DATA_PATH)
except Exception as e:
    st.error(f"Error loading dataset: {e}")
    st.stop()

with st.spinner("Loading model and building index (cached; first run may take a minute)..."):
    try:
        model, index, embeddings = build_model_and_index(data)
    except Exception as e:
        st.exception(f"Failed to load model/index: {e}")
        st.stop()

# Search action
if st.button("Search"):
    balance = None if energy == "Neutral" else energy
    with st.spinner("Searching for products..."):
        results = search(query, model, index, data, top_k=top_k, balance=balance)
    if not results:
        st.warning("No matching products found. Try a different query or choose 'Neutral'.")
    else:
        st.success(f"Showing {len(results)} result(s) for '{query}' — Energy: {energy}")
        for item in results:
            img = item.get('image_url') or PLACEHOLDER_IMG.format(item['id'][:6])
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.image(img, use_column_width=True)
            st.markdown(f"<div class='title'>{item['title']}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='muted'>{item['description'][:300]}...</div>", unsafe_allow_html=True)
            st.markdown(
                f"- 🧿 Polarity: `{item['polarity']}`  •  🕯️ Yin: {item['yin_score']}  •  ☀️ Yang: {item['yang_score']}  •  💰 ₹{item['price']}"
            )
            st.markdown("</div>", unsafe_allow_html=True)

