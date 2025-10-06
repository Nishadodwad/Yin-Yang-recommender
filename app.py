import streamlit as st
import json
import uuid
import random
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# ------------------------------------------------------
# Page Configuration
# ------------------------------------------------------
st.set_page_config(
    page_title="Yin & Yang Search Engine",
    page_icon="☯️",
    layout="wide",
)

# ------------------------------------------------------
# 1️⃣ Load and Cache Data & Models (Expensive Operations)
# ------------------------------------------------------

# Use Streamlit's caching to load data only once.
@st.cache_data
def load_dataset(path):
    """Loads and preprocesses the dataset from a JSON file."""
    try:
        with open(path, "r", encoding="utf8") as f:
            raw = json.load(f)
        data = []
        for item in raw:
            it = dict(item)
            it["id"] = str(it.get("id") or uuid.uuid4())
            # Build a more descriptive title
            title_parts = [str(it.get(k, "")) for k in ["Model Number", "Type", "Features", "Durability", "Color"] if it.get(k)]
            it["title"] = " ".join(title_parts) or f"Product {it['id']}"
            # Build a comprehensive description
            desc_parts = [f"{k}: {v}" for k, v in it.items() if k.lower() != "id"]
            it["description"] = ". ".join(desc_parts)[:800]
            # Assign polarity for Yin/Yang scoring
            pol = float(it.get("polarity", random.uniform(-1, 1)))
            it["polarity"] = round(pol, 3)
            it["yin_score"] = round((1 - pol) / 2, 3)
            it["yang_score"] = round((1 + pol) / 2, 3)
            it["price"] = int(it.get("price", random.randint(500, 5000)))
            data.append(it)
        return data
    except FileNotFoundError:
        st.error(f"Error: Dataset file not found at {path}. Please make sure 'data-set (1).json' is in the same directory.")
        return []

# Use Streamlit's caching to load the model and build the index only once.
@st.cache_resource
def build_faiss_index(data):
    """Loads the sentence transformer model, encodes the data, and builds a FAISS index."""
    model = SentenceTransformer("all-MiniLM-L6-v2")
    corpus = [f"{d['title']} {d['description']}" for d in data]
    
    with st.spinner('Encoding corpus and building search index... This may take a moment.'):
        embeddings = model.encode(corpus, normalize_embeddings=True)
        embeddings = np.array(embeddings).astype("float32")
        
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)  # Inner product is equivalent to cosine similarity for normalized vectors
        index.add(embeddings)
        
    return model, index

# --- Main App Logic ---
DATA_PATH = "data/data-set (1).json"
data = load_dataset(DATA_PATH)

if data:
    model, index = build_faiss_index(data)
else:
    st.stop()

# ------------------------------------------------------
# 2️⃣ Search Function
# ------------------------------------------------------
def search(query, balance="Neutral", top_k=5):
    """Performs a search on the FAISS index and filters by Yin/Yang balance."""
    if not query:
        return []
        
    q_vec = model.encode([query], normalize_embeddings=True).astype("float32")
    # Search for more items initially to allow for filtering
    distances, indices = index.search(q_vec, top_k * 5)
    
    results = []
    for idx in indices[0]:
        if idx < 0 or idx >= len(data):
            continue
        
        item = data[idx]
        
        # Energy filtering based on polarity
        if balance == "Yin" and item["polarity"] > -0.1:  # Allow slightly neutral items for more results
            continue
        if balance == "Yang" and item["polarity"] < 0.1:   # Allow slightly neutral items for more results
            continue
            
        results.append(item)
        
        if len(results) >= top_k:
            break
            
    return results

# ------------------------------------------------------
# 3️⃣ Streamlit User Interface
# ------------------------------------------------------
st.title("☯️ Yin & Yang E-commerce Search")
st.write("Find products that match not just your query, but also your desired energy.")

# --- Search Inputs ---
col1, col2 = st.columns([3, 1])

with col1:
    search_query = st.text_input(
        "Enter your search query:",
        placeholder="e.g., 'water resistant jacket' or 'powerful vacuum cleaner'"
    )

with col2:
    energy_balance = st.radio(
        "Choose an energy balance:",
        ["Yin", "Yang", "Neutral"],
        horizontal=True,
    )

if st.button("Search", type="primary"):
    search_results = search(search_query, balance=energy_balance, top_k=5)

    st.header(f"Search Results for '{search_query}' — Energy: {energy_balance}")

    if not search_results:
        st.warning("No matching products found. Try a different query or energy balance.")
    else:
        for i, r in enumerate(search_results, 1):
            with st.container(border=True):
                st.subheader(f"{i}. {r['title']}")
                
                res_col1, res_col2, res_col3, res_col4 = st.columns(4)
                res_col1.metric("Polarity", f"{r['polarity']}")
                res_col2.metric("Yin Score", f"{r['yin_score']:.2%}")
                res_col3.metric("Yang Score", f"{r['yang_score']:.2%}")
                res_col4.metric("Price", f"₹{r['price']:,}")

                with st.expander("View Description"):
                    st.write(r['description'])
