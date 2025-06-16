import streamlit as st
import os
import requests
from sentence_transformers import SentenceTransformer
from rag_utils import *
from prompt import SYSTEM_PROMPT_QA

# ----- PAGE CONFIG -----
st.set_page_config(page_title="RAG Implementation with Qdrant + Ollama", layout="wide")

# ----- LOADING CSS -----
def load_css(file_path):
    with open(file_path, "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("style.css")

# ----- TITLE -----
st.title("RAG Implementation with ")

# ----- LOAD MODEL AND QDRANT CLIENT -----
@st.cache_resource
def load_model():
    return SentenceTransformer('paraphrase-MiniLM-L6-v2')

@st.cache_resource
def get_client():
    return client  # Defined in rag_utils.py

model = load_model()
qdrant_client = get_client()

# ----- SIDEBAR -----
with st.sidebar:
    st.image("chatboticon.png", width=300)
    file = st.file_uploader("Drop your file here:", type=["txt", "pdf", "docx", "csv", "html"])

    if file and not st.session_state.get('processed', False):
        save_folder = "uploads"
        os.makedirs(save_folder, exist_ok=True)
        save_path = os.path.join(save_folder, file.name)
        with open(save_path, "wb") as f:
            f.write(file.getbuffer())

        try:
            raw_text = read_file(save_path)

            if not raw_text.strip():
                st.error("No text content found.")
            else:
                chunks = chunk_text(raw_text)

                if not chunks:
                    st.error("No chunks created from text.")
                else:
                    vectors = model.encode(chunks).tolist()
                    create_qdrant_collection(qdrant_client, "my_collection", model)
                    upload_vectors_to_qdrant(qdrant_client, "my_collection", vectors, chunks)

                    st.success("Your file has been processed")
                    st.session_state.processed = True
                    st.session_state.chunks_count = len(chunks)

        except Exception as e:
            st.error(f"Error: {str(e)}")

# ----- CHAT UI -----
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hi I'm Alice, your document assistant. Upload a document and ask me any question about it!"
        }
    ]

# Render chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask me anything about your file..."):
    if not st.session_state.get("processed", False):
        st.warning("Please upload and process a file first.")
    else:
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            msg_placeholder = st.empty()
            msg_placeholder.markdown("Thinking...")


            try:
                query_vector = model.encode(prompt).tolist()
                results = search_qdrant(qdrant_client, "my_collection", query_vector)

                if results:
                    retrieved_context = " ".join([r.payload["text"] for r in results])

                    ollama_payload = {
                        "model": "llama3",
                        "prompt": f"{SYSTEM_PROMPT_QA}\n\nContext:\n{retrieved_context}\n\nQuestion:\n{prompt}\n\nAnswer:",
                        "stream": False
                    }

                    response = requests.post("http://localhost:11434/api/generate", json=ollama_payload, timeout=300)

                    if response.status_code == 200:
                        answer = response.json()["response"].strip()
                        msg_placeholder.markdown(answer)
                        st.session_state.messages.append({"role": "assistant", "content": answer})
                    else:
                        error_msg = f"Ollama API returned status code: {response.status_code}"
                        msg_placeholder.markdown(error_msg)
                        st.session_state.messages.append({"role": "assistant", "content": error_msg})

                else:
                    no_result_msg = "No relevant results found in Qdrant."
                    msg_placeholder.markdown(no_result_msg)
                    st.session_state.messages.append({"role": "assistant", "content": no_result_msg})

            except Exception as e:
                error_msg = f"Search error: {str(e)}"
                msg_placeholder.markdown(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
