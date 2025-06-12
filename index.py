import streamlit as st
import os
import requests
from sentence_transformers import SentenceTransformer
from rag_utils import *

st.title("RAG Implementation with Qdrant + Ollama")

# Initialize the model and client outside of conditional blocks
@st.cache_resource
def load_model():
    return SentenceTransformer('paraphrase-MiniLM-L6-v2')

@st.cache_resource
def get_client():
    return client  # This should be defined in your app.py

# Load resources
model = load_model()
qdrant_client = get_client()

with st.sidebar:
    st.write("This is a sidebar")

file = st.file_uploader("Upload a file", type=["txt", "pdf", "docx", "csv", "html"])

if file:
    st.success(f"Uploaded: {file.name}")

    # Save to local folder
    if st.checkbox("Save this file to local storage"):
        save_folder = "uploads"
        os.makedirs(save_folder, exist_ok=True)
        save_path = os.path.join(save_folder, file.name)
        with open(save_path, "wb") as f:
            f.write(file.getbuffer())
        st.success(f"Saved to `{save_path}`")

        if st.button("Process File"):
            try:
                # Step 1: Read, chunk, embed, upload to Qdrant
                raw_text = read_file(save_path)
                
                if not raw_text.strip():
                    st.error("No text content found in the file")
                else:
                    chunks = chunk_text(raw_text)
                    
                    if not chunks:
                        st.error("No chunks created from the text")
                    else:
                        # Generate real text embeddings
                        with st.spinner("Generating embeddings..."):
                            vectors = model.encode(chunks).tolist()

                        # Recreate collection and upload vectors
                        with st.spinner("Setting up Qdrant collection..."):
                            create_qdrant_collection(qdrant_client, collection_name="my_collection")
                        
                        with st.spinner("Uploading vectors..."):
                            upload_vectors_to_qdrant(qdrant_client, "my_collection", vectors, chunks)

                        st.success("File processed and vectors uploaded.")
                        
                        # Store processing state in session
                        st.session_state.processed = True
                        st.session_state.chunks_count = len(chunks)
                        
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")

# Query section - only show if file has been processed
if st.session_state.get('processed', False):
    st.markdown("---")
    st.subheader("Ask Questions")
    
    query = st.text_input("Ask a question:", key="query_input")
    
    if st.button("Ask") and query.strip():
        try:
            with st.spinner("Searching for relevant information..."):
                query_vector = model.encode(query).tolist()
                results = search_qdrant(qdrant_client, "my_collection", query_vector)
                
                print("--res--", results)
            
            if results:
                retrieved_context = " ".join([r.payload["text"] for r in results])
                
                # Show retrieved context (optional)
                with st.expander("View Retrieved Context"):
                    for i, result in enumerate(results):
                        st.write(f"**Chunk {i+1} (Score: {getattr(result, 'score', 'N/A')})**")
                        st.write(result.payload["text"][:200] + "...")
                        st.markdown("---")
                
                # Ollama prompt
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
                        st.subheader("Answer:")
                        st.write(answer)
                    else:
                        st.error(f"Ollama API returned status code: {response.status_code}")
                        
                except requests.exceptions.ConnectionError:
                    st.error("Could not connect to Ollama. Make sure Ollama is running on localhost:11434")
                except requests.exceptions.Timeout:
                    st.error("Request to Ollama timed out. Try again.")
                except Exception as e:
                    st.error(f"Error getting answer from Ollama: {str(e)}")
            else:
                st.warning("No results found in Qdrant.")
                
        except Exception as e:
            st.error(f"Error during search: {str(e)}")
    
    elif query.strip() and not st.button("Ask"):
        st.info("Click 'Ask' to get an answer")

# Show processing status
if st.session_state.get('processed', False):
    st.sidebar.success(f"✅ File processed ({st.session_state.get('chunks_count', 0)} chunks)")
else:
    st.sidebar.info("Upload and process a file to start asking questions")