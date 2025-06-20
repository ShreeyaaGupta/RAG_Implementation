import streamlit as st
import os
import asyncio
# import aiofiles
import requests
from sentence_transformers import SentenceTransformer
from rag_utils import *
from prompt import SYSTEM_PROMPT_QA

from llama_parse import LlamaParse
from dotenv import load_dotenv
load_dotenv()

# ----- CONFIG -----
st.set_page_config(page_title="RAG with Qdrant + Ollama + LlamaParse", layout="wide")

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


async def process_with_llamaparse(file_path: str) -> str:
    llama_parser = LlamaParse(api_key=os.getenv("LLAMA_PARSE_API_KEY"))
    job = llama_parser.parse(file_path=file_path)
    print("------line36----------", job)
    # Wait for the parsing to complete and fetch the result
    result = job.pages[0]  # This blocks until parsing is complete
    parsed_text = result.text  # Get the full parsed text

    print("------line41----------", parsed_text)
    return parsed_text


# ----- SIDEBAR -----
with st.sidebar:
    st.image("chatboticon.png", width=300)

    selected_model = st.selectbox('Choose a Llama model', ['Llama3.2-1B','Llama3.2-3B', 'Llama3'], key='selected_model')
    llama_model = {
        'Llama3.2-1B': 'llama3.2:1b',
        'Llama3.2-3B': 'llama3.2:3b',
        'Llama3': 'llama3',
    }[selected_model]

    file = st.file_uploader("Drop your file here:", type=["txt", "pdf", "docx", "csv", "html",], key='file_uploader')

    if file and not st.session_state.get('processed', False):
        save_path = os.path.join("uploads", file.name)
        os.makedirs("uploads", exist_ok=True)
        with open(save_path, "wb") as f:
            f.write(file.getbuffer())

        try:
            with st.spinner("Processing with LlamaParse..."):
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

                # Start both but wait only for LlamaParse
                llama_output = loop.run_until_complete(process_with_llamaparse(save_path))
                print("------line79----------"  , llama_output)
                print("-----80-----",process_with_qdrant(save_path) ) # Optional

                if llama_output.strip():
                    st.session_state.processed = True
                    st.session_state.llama_parsed_text = llama_output
                    st.success("File uploaded and processed successfully!")
                else:
                    st.error("LlamaParse returned empty content.")

        except Exception as e:
            st.error(f"Processing Error: {str(e)}")


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
                retrieved_context = st.session_state.get("llama_parsed_text", "")


                if not retrieved_context:
                    raise ValueError("LlamaParse content missing.")

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
                error_msg = f"Error during LlamaParse QA: {str(e)}"
                msg_placeholder.markdown(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
