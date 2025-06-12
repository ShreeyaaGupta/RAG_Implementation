import os
import csv
import docx
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance
from llama_parse import LlamaParse

# Load API keys from .env file
load_dotenv()

# Initialize LlamaParse client
def get_llama_parser():
    return LlamaParse(api_key=os.getenv("LLAMA_PARSE_API_KEY"))

#Initialising Ollama model
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL")
# Initialize Qdrant client
client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
)
# === File Reader ===
def read_file(file_path):
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        print("Parsing PDF with LlamaParse...")
        parser = get_llama_parser()
        documents = parser.load_data(file_path)
        return "\n".join([doc.text for doc in documents])

    elif ext == ".txt":
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()

    elif ext == ".docx":
        doc = docx.Document(file_path)
        return "\n".join([para.text for para in doc.paragraphs])

    elif ext == ".csv":
        text = ""
        with open(file_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                text += " ".join(row) + "\n"
        return text

    elif ext in [".html", ".htm"]:
        with open(file_path, "r", encoding="utf-8") as f:
            soup = BeautifulSoup(f, "html.parser")
            return soup.get_text()

    else:
        raise ValueError(f"Unsupported file type: {ext}")

# === Chunking ===
def chunk_text(text, max_tokens=100):
    words = text.split()
    return [" ".join(words[i:i + max_tokens]) for i in range(0, len(words), max_tokens)]

# === Qdrant Collection Management ===
def create_qdrant_collection(client, collection_name, model, distance=Distance.COSINE):
    size=model.get_sentence_embedding_dimension()

    try:
        existing = client.get_collections().collections
        if any(c.name == collection_name for c in existing):
            print(f"Collection '{collection_name}' already exists.")
            return
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=size, distance=distance),
        )
        print(f"Created collection '{collection_name}' with dimension {size}.")
    except Exception as e:
        print(f"Error creating collection: {e}")

# === Upload Embeddings ===
def upload_vectors_to_qdrant(client, collection_name, vectors, chunks):
    try:
        client.upsert(
            collection_name=collection_name,
            points=[
                {
                    "id": i,
                    "vector": vectors[i],
                    "payload": {"text": chunks[i]},
                }
                for i in range(len(chunks))
            ]
        )
        print("Vectors uploaded successfully.")
    except Exception as e:
        print(f"Error uploading vectors: {e}")

# === Vector Search ===
def search_qdrant(client, collection_name, query_vector, limit=3):
    try:
        results = client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=limit,
            with_vectors=True,
            with_payload=True
        )
        print("Search results:")
        for r in results:
            print(f"- {r.payload['text']} (score: {r.score})")
        return results
    except Exception as e:
        print(f"Error during search: {e}")
        return []

# === Contextual Answering ===
def get_answer_from_ollama(query: str, context: str,) -> str:
    prompt = f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False
    }

    try:
        response = requests.post("http://localhost:11434/api/generate", json=payload)
        response.raise_for_status()
        return response.json()["response"].strip()
    except Exception as e:
        print(f"Error getting answer from Ollama: {e}")
        return "Error: Could not get answer from Ollama."
