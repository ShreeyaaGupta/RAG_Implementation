import streamlit as st
import os
import requests
from sentence_transformers import SentenceTransformer
from rag_utils import *
from prompt import SYSTEM_PROMPT_QA

# ----- CONFIG -----
st.set_page_config(page_title="RAG with Qdrant + Ollama", layout="wide")

# ----- LOAD MODEL AND QDRANT CLIENT -----
@st.cache_resource
def load_model():
    return SentenceTransformer('paraphrase-MiniLM-L6-v2')

@st.cache_resource
def get_client():
    return client

model = load_model()
qdrant_client = get_client()

# ----- FILE HANDLING -----
def process_with_qdrant(file_path: str):
    raw_text = read_file(file_path)
    chunks = chunk_text(raw_text)
    vectors = model.encode(chunks).tolist()

    create_qdrant_collection(qdrant_client, "my_collection", model)
    upload_vectors_to_qdrant(qdrant_client, "my_collection", vectors, chunks)
    return True

# ----- SIDEBAR -----
with st.sidebar:
    st.image("chatboticon.png", width=300)

    selected_model = st.selectbox('Choose a Llama model', ['Llama3.2-1B','Llama3.2-3B', 'Llama3'], key='selected_model')
    llama_model = {
        'Llama3.2-1B': 'llama3.2:1b',
        'Llama3.2-3B': 'llama3.2:3b',
        'Llama3': 'llama3',
    }[selected_model]

    file = st.file_uploader("Drop your file here:", type=["txt", "pdf", "docx", "csv", "html"], key='file_uploader')

    if file and not st.session_state.get('processed', False):
        save_path = os.path.join("uploads", file.name)
        os.makedirs("uploads", exist_ok=True)
        with open(save_path, "wb") as f:
            f.write(file.getbuffer())

        try:
            with st.spinner("Processing and uploading to Qdrant..."):
                processed = process_with_qdrant(save_path)

                if processed:
                    st.session_state.processed = True
                    st.success("File uploaded and indexed in Qdrant successfully!")

                else:
                    st.error("Failed to process and upload the file.")

            
        except Exception as e:
            st.error(f"Processing Error: {str(e)}")
    st.markdown("""
        <hr>
        <div style="text-align: center; font-size: 12px; color: #666;">
            © 2025, CloudKaptan Consultancy Services Private Limited. All rights reserved.
        </div>
        """, unsafe_allow_html=True)

# ----- CHAT UI -----
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi, I'm Alice. Upload a document and ask me anything about it!"}
    ]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

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
                # Encode query and perform vector search
                query_vector = model.encode([prompt])[0].tolist()
                results = search_qdrant(qdrant_client, "my_collection", query_vector)
                context_chunks = [r.payload["text"] for r in results]
                retrieved_context = "\n\n".join(context_chunks)

                if not retrieved_context:
                    raise ValueError("No context retrieved from Qdrant.")

                # Call Ollama for LLM completion
                ollama_payload = {
                    "model": llama_model,
                    "prompt": f"{SYSTEM_PROMPT_QA}\n\nContext:\n{retrieved_context}\n\nQuestion:\n{prompt}\n\nAnswer:",
                    "stream": False
                }

                response = requests.post("http://localhost:11434/api/generate", json=ollama_payload, timeout=300)

                if response.status_code == 200:
                    answer = response.json()["response"].strip()
                    msg_placeholder.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                else:
                    error_msg = f"Ollama API returned {response.status_code}"
                    msg_placeholder.markdown(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})

            except Exception as e:
                error_msg = f"Error during Qdrant-based QA: {str(e)}"
                msg_placeholder.markdown(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
