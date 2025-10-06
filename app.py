import streamlit as st
import json
import uuid
import random
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from pathlib import Path

# -------------------------------
# Utility: CSS + small helpers
# -------------------------------
st.set_page_config(page_title="Yin–Yang Recommendation Engine", page_icon="☯️", layout="wide")

PRIMARY_CSS = """
<style>
/* page background */
.block-container { padding-top: 1rem; padding-bottom: 3rem; }

/* card style */
.card {
  background: #ffffff;
  border-radius: 12px;
  padding: 12px;
  box-shadow: 0 4px 12px rgba(0,0,0,0.06);
  margin-bottom: 12px;
}
.result-title { font-size:18px; font-weight:700; margin-bottom:6px; }
.small-muted { color: #6c6c6c; font-size:13px; }
.key-value { font-weight:600; }
.badge { font-size:12px; padding:4px 8px; border-radius:8px; background:#f1f1f1; }
</style>
"""
st.markdown(PRIMARY_CSS, unsafe_allow_html=True)

# -------------------------------
# 0. Configuration
# -------------------------------
DATA_PATH = Path("data/data-set (1).json")  # <-- adjust if your dataset file name differs
DEFAULT_PLACEHOLDER_IMG = "https://picsum.photos/seed/{}/320/200"

# -------------------------------
# 1. Load & enrich dataset (cached)
# -------------------------------
@st.cache_data(show_spinner=False)
def load_and_enrich_dataset(filepath):
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Dataset not found at {filepath.resolve()}. Please check path.")
    raw = json.loads(filepath.read_text(encoding="utf8"))

    data = []
    for item in raw:
        it = dict(item)  # shallow copy
        # ensure id
        it['id'] = str(it.get('id') or uuid.uuid4())
        # title & description generation
        def generate_text_fields(i):
            title_parts = []
            for key in ["Model Number", "Features", "Durability", "Battery Voltage"]:
                if key in i and i[key]:
                    title_parts.append(str(i[key]))
            title = " ".join(title_parts) or f"Product {i.get('id', '')}"

            desc_parts = []
            for k, v in i.items():
                if k.lower() not in ["id"]:
                    desc_parts.append(f"{k}: {v}")
            description = ". ".join(desc_parts)
            return title.strip(), description.strip()

        title, description = generate_text_fields(it)
        it['title'] = title
        it['description'] = description
        # fill additional fields if missing
        polarity = round(it.get('polarity', random.uniform(-1, 1)), 3)
        it['polarity'] = polarity
        it['yin_score'] = round((1 - polarity) / 2, 3)
        it['yang_score'] = round((1 + polarity) / 2, 3)
        it['elements'] = it.get('elements') or random.choice([["water"], ["fire"], ["earth"], ["wood"], ["metal"]])
        it['seasonality'] = it.get('seasonality') or random.choice([["winter"], ["summer"], ["spring"], ["autumn"]])
        it['compatibility'] = it.get('compatibility', [])
        it['image_url'] = it.get('image_url', "")
        it['price'] = int(it.get('price', random.randint(500, 5000)))
        it['popularity'] = int(it.get('popularity', random.randint(1, 100)))
        it['metadata'] = it.get('metadata', {})
        data.append(it)
    return data

# -------------------------------
# 2. Build or load model + embeddings + faiss index (cached)
# -------------------------------
@st.cache_resource(show_spinner=False)
def build_embedding_index(data):
    # load model (heavy)
    model = SentenceTransformer("all-MiniLM-L6-v2")
    # prepare corpus and compute embeddings
    corpus = [f"{d['title']} {d['description']}" for d in data]
    # encode (normalize so cosine similarity = dot product)
    embeddings = model.encode(corpus, normalize_embeddings=True)
    embeddings = np.array(embeddings).astype('float32')

    dim = embeddings.shape[1]
    # use IndexFlatIP with normalized embeddings -> inner product == cosine similarity
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return model, index, embeddings

# -------------------------------
# 3. Search function (uses FAISS index)
# -------------------------------
def search(query, model, index, data, top_k=5, balance=None, filters=None, embeddings_array=None):
    if query.strip() == "":
        return []

    q_embed = model.encode([query], normalize_embeddings=True).astype('float32')
    # ask for more results to allow filtering to take effect
    D, I = index.search(q_embed, top_k * 5)
    results = []
    seen_ids = set()
    for idx in I[0]:
        if idx < 0 or idx >= len(data):
            continue
        item = data[idx]
        if item['id'] in seen_ids:
            continue
        # apply balance filter
        if balance == "Yin" and item['polarity'] > 0:
            continue
        if balance == "Yang" and item['polarity'] < 0:
            continue
        # apply custom filters
        if filters:
            min_price, max_price = filters.get("price", (None, None))
            if min_price is not None and item['price'] < min_price: continue
            if max_price is not None and item['price'] > max_price: continue
            elems = filters.get("elements")
            if elems and not any(e in item.get('elements', []) for e in elems): continue
            seasons = filters.get("seasonality")
            if seasons and not any(s in item.get('seasonality', []) for s in seasons): continue
            # polarity range
            pol_range = filters.get("polarity_range")
            if pol_range:
                if not (pol_range[0] <= item['polarity'] <= pol_range[1]): continue

        results.append(item)
        seen_ids.add(item['id'])
        if len(results) >= top_k:
            break
    return results

# -------------------------------
# 4. UI: Load dataset + index (show errors, non-blocking)
# -------------------------------
st.title("☯️ Yin–Yang Recommendation Engine")
st.markdown("Explore product harmony through balance of Yin and Yang energies. Use filters on the left and search on the top bar.")

# load data
try:
    data = load_and_enrich_dataset(DATA_PATH)
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()
except Exception as e:
    st.exception(e)
    st.stop()

# build model & index (cached)
with st.spinner("Loading model and building index (cached)... This runs once per session."):
    try:
        model, index, embeddings_arr = build_embedding_index(data)
    except Exception as e:
        st.exception("Failed to build model/index: " + str(e))
        st.stop()

# -------------------------------
# Sidebar: Filters
# -------------------------------
with st.sidebar:
    st.header("Filters")
    # price range
    prices = [d['price'] for d in data]
    pmin, pmax = int(min(prices)), int(max(prices))
    price_range = st.slider("Price range (₹)", pmin, pmax, (pmin, pmax), step=50)

    # elements & seasonality
    all_elements = sorted({el for d in data for el in d.get('elements', [])})
    elem_selected = st.multiselect("Elements", options=all_elements, default=[])

    all_seasons = sorted({s for d in data for s in d.get('seasonality', [])})
    season_selected = st.multiselect("Seasonality", options=all_seasons, default=[])

    # polarity range (Yin <0, Yang >0)
    pol_min, pol_max = st.slider("Polarity range (-1..1)", -1.0, 1.0, (-1.0, 1.0), step=0.01)

    # energy preference quick pick
    yin_yang_pref = st.select_slider("Quick energy preference", options=["Yin", "Neutral", "Yang"], value="Neutral")

    # sorting
    sort_by = st.selectbox("Sort results by", options=["Relevance", "Price: Low→High", "Price: High→Low", "Popularity"], index=0)

    st.markdown("---")
    st.caption("Tip: load may take a moment the first time the model loads.")

# -------------------------------
# Top search area
# -------------------------------
query = st.text_input("🔍 Enter a search query", value="vacuum cleaner")
top_k = st.slider("Top K results", min_value=3, max_value=12, value=6, step=1)

if st.button("Search"):
    # assemble filters
    filters = {
        "price": price_range,
        "elements": elem_selected,
        "seasonality": season_selected,
        "polarity_range": (pol_min, pol_max)
    }
    with st.spinner("Finding harmonious matches..."):
        results = search(query, model, index, data, top_k=top_k, balance=(None if yin_yang_pref=="Neutral" else yin_yang_pref), filters=filters, embeddings_array=embeddings_arr)

    if not results:
        st.warning("No results found with the current filters. Try widening the filters or use a simpler query.")
    else:
        # optional sorting
        if sort_by == "Price: Low→High":
            results.sort(key=lambda x: x['price'])
        elif sort_by == "Price: High→Low":
            results.sort(key=lambda x: -x['price'])
        elif sort_by == "Popularity":
            results.sort(key=lambda x: -x['popularity'])

        st.success(f"Found {len(results)} matching products:")
        # show results in two columns per row
        cols = st.columns((1, 1))
        for i, item in enumerate(results):
            col = cols[i % 2]
            with col:
                img_url = item.get('image_url') or DEFAULT_PLACEHOLDER_IMG.format(item['id'][:6])
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.image(img_url, use_column_width=True, clamp=True)
                st.markdown(f"<div class='result-title'>{item['title']}</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='small-muted'>{item.get('description','')[:220]}...</div>", unsafe_allow_html=True)
                st.markdown("<br/>", unsafe_allow_html=True)
                # small stat row
                left, right = st.columns([2,1])
                with left:
                    st.markdown(f"- 🧿 **Polarity:** `{item['polarity']}`  •  🕯️ **Yin:** {item['yin_score']}  •  ☀️ **Yang:** {item['yang_score']}")
                    st.markdown(f"- 🌊 **Elements:** `{', '.join(item.get('elements',[]))}`  •  🍂 **Season:** `{', '.join(item.get('seasonality',[]))}`")
                    st.markdown(f"- 📜 **ID:** `{item['id']}`")
                with right:
                    st.metric(label="Price", value=f"₹{item['price']}")
                    st.metric(label="Popularity", value=f"{item['popularity']}")
                st.markdown("</div>", unsafe_allow_html=True)

# provide some helpful footer info
st.markdown("---")
st.caption("Engine built with SentenceTransformer (all-MiniLM-L6-v2) + FAISS. Embeddings are normalized so results are cosine-similarity ranked. ")
