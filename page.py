import streamlit as st
import os
import requests
from sentence_transformers import SentenceTransformer
from rag_utils import *

# ----- PAGE CONFIG -----
st.set_page_config(page_title="RAG with Qdrant + Ollama", layout="wide")

# ----- CUSTOM CSS -----
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
        padding: 2rem;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 0.5em 1em;
    }
    .stButton>button:hover {
        background-color: #45a049;
        color: white;
    }
    .uploadedFileName {
        font-weight: bold;
        color: #2c3e50;
    }
    .section-title {
        font-size: 1.5rem;
        margin-top: 2rem;
        font-weight: 600;
        color: #333;
    }
    </style>
""", unsafe_allow_html=True)

# ----- TITLE -----
st.markdown("<h1 style='color: #1f77b4;'> RAG Implementation with Qdrant + Ollama</h1>", unsafe_allow_html=True)

# ----- MODEL AND CLIENT -----
@st.cache_resource
def load_model():
    return SentenceTransformer('paraphrase-MiniLM-L6-v2')

@st.cache_resource
def get_client():
    return client  # Defined in app.py

model = load_model()
qdrant_client = get_client()

# ----- SIDEBAR -----
with st.sidebar:
    st.markdown("### 🛠️ Options")
    st.info("Upload and process a file to start asking questions.")
    if st.session_state.get('processed', False):
        st.success(f"✅ Processed: {st.session_state.get('chunks_count', 0)} chunks")

# ----- FILE UPLOAD -----
st.markdown("### 📤 Upload a file")
file = st.file_uploader("Choose your file", type=["txt", "pdf", "docx", "csv", "html"])

if file:
    st.markdown(f"<p class='uploadedFileName'>✅ Uploaded: {file.name}</p>", unsafe_allow_html=True)

    if st.checkbox("Save to local storage"):
        save_folder = "uploads"
        os.makedirs(save_folder, exist_ok=True)
        save_path = os.path.join(save_folder, file.name)
        with open(save_path, "wb") as f:
            f.write(file.getbuffer())
        st.success(f"Saved to `{save_path}`")

        if st.button(" Process File"):
            try:
                raw_text = read_file(save_path)

                if not raw_text.strip():
                    st.error(" No text content found.")
                else:
                    chunks = chunk_text(raw_text)

                    if not chunks:
                        st.error("No chunks created from text.")
                    else:
                        with st.spinner("Generating embeddings..."):
                            vectors = model.encode(chunks).tolist()

                        with st.spinner("Creating Qdrant collection..."):
                            create_qdrant_collection(qdrant_client, "my_collection", model)

                        with st.spinner("Uploading vectors..."):
                            upload_vectors_to_qdrant(qdrant_client, "my_collection", vectors, chunks)

                        st.success("🎉 File processed and vectors uploaded.")
                        st.session_state.processed = True
                        st.session_state.chunks_count = len(chunks)

            except Exception as e:
                st.error(f"Error: {str(e)}")

# ----- QUERY SECTION -----
if st.session_state.get('processed', False):
    st.markdown("<div class='section-title'> Ask Questions</div>", unsafe_allow_html=True)
    query = st.text_input("Type your question here:", key="query_input")

    if st.button("🔍 Ask"):
        if query.strip():
            try:
                with st.spinner("🔎 Searching in Qdrant..."):
                    query_vector = model.encode(query).tolist()
                    results = search_qdrant(qdrant_client, "my_collection", query_vector)

                if results:
                    retrieved_context = " ".join([r.payload["text"] for r in results])

                    with st.expander("📚 View Retrieved Context"):
                        for i, result in enumerate(results):
                            st.markdown(f"** Chunk {i+1} (Score: {getattr(result, 'score', 'N/A')}):**")
                            st.write(result.payload["text"][:200] + "...")
                            st.markdown("---")

                    ollama_payload = {
                        "model": "llama3",
                        "prompt": f"Context: {retrieved_context}\n\nQuestion: {query}\n\nAnswer:",
                        "stream": False
                    }

                    try:
                        with st.spinner("Getting answer from Ollama..."):
                            response = requests.post(
                                "http://localhost:11434/api/generate",
                                json=ollama_payload,
                                timeout=30
                            )

                        if response.status_code == 200:
                            answer = response.json()["response"].strip()
                            st.subheader("🧠 Answer:")
                            st.write(answer)
                        else:
                            st.error(f"Ollama API returned status code: {response.status_code}")

                    except requests.exceptions.ConnectionError:
                        st.error(" Could not connect to Ollama. Make sure it is running.")
                    except requests.exceptions.Timeout:
                        st.error("Request to Ollama timed out.")
                    except Exception as e:
                        st.error(f"Ollama error: {str(e)}")

                else:
                    st.warning("No relevant results found in Qdrant.")

            except Exception as e:
                st.error(f"❗ Search error: {str(e)}")
        else:
            st.warning("Please enter a question before clicking 'Ask'")