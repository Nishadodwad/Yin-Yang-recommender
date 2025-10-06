
import streamlit as st
import json, uuid, random, numpy as np, faiss
from sentence_transformers import SentenceTransformer
import os

# -------------------------------
# 1. Caching setup
# -------------------------------
@st.cache_resource
def load_model():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return model

@st.cache_data
def prepare_data_and_index():
    with open("data/data-set (1).json", "r", encoding="utf-8") as f:
        data = json.load(f)

    for item in data:
        item["id"] = str(item.get("id") or uuid.uuid4())
        title_parts = [str(item.get(k, "")) for k in ["Model Number","Features","Durability","Battery Voltage"] if item.get(k)]
        item["title"] = " ".join(title_parts) or f"Product {item['id']}"
        desc_parts = [f"{k}: {v}" for k, v in item.items() if k.lower() != "id"]
        item["description"] = ". ".join(desc_parts)[:800]
        pol = random.uniform(-1, 1)
        item["polarity"] = round(pol, 3)
        item["yin_score"] = round((1 - pol) / 2, 3)
        item["yang_score"] = round((1 + pol) / 2, 3)
        item["price"] = random.randint(500, 5000)

    model = load_model()
    corpus = [f"{d['title']} {d['description']}" for d in data]
    embeddings = model.encode(corpus, normalize_embeddings=True)
    embeddings = np.array(embeddings).astype("float32")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return data, index

# -------------------------------
# 2. Search function
# -------------------------------
def search(query, energy="Neutral", top_k=5):
    model = load_model()
    data, index = prepare_data_and_index()

    q_vec = model.encode([query], normalize_embeddings=True).astype("float32")
    D, I = index.search(q_vec, top_k * 3)

    results = []
    for idx in I[0]:
        item = data[idx]
        pol = item["polarity"]
        if energy == "Yin" and pol > 0: continue
        if energy == "Yang" and pol < 0: continue
        results.append(item)
        if len(results) >= top_k:
            break
    return results

# -------------------------------
# 3. Streamlit UI
# -------------------------------
st.set_page_config(page_title="Yin–Yang Recommender", page_icon="☯️")
st.title("☯️ Yin–Yang Recommendation Engine")
st.caption("Discover product harmony through Yin and Yang energies.")

query = st.text_input("🔍 Enter a search query:", "")
energy = st.select_slider("⚖️ Energy Preference:", ["Yin", "Neutral", "Yang"], value="Neutral")

if st.button("Search"):
    if not query.strip():
        st.warning("Please enter a query first.")
    else:
        with st.spinner("Finding harmonious matches..."):
            results = search(query, energy)
        if results:
            st.success(f"Top {len(results)} results for '{query}' ({energy} energy):")
            for item in results:
                st.markdown(f"""
                **{item['title']}**
                - 🧿 Polarity: `{item['polarity']}`
                - 💰 Price: ₹{item['price']}
                - ☯️ Yin: {item['yin_score']} | Yang: {item['yang_score']}
                - 📜 {item['description'][:150]}...
                """)
        else:
            st.warning("No matching products found.")
