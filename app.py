import os
import docx
import csv
from bs4 import BeautifulSoup
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import requests
from llama_parse import LlamaParse

# Initialize LlamaParse client (Set your API key here)
llama_parse = LlamaParse(api_key="llx-NzoqDM23DzsbI6fJYntN2WgT2tXXKdqv5Vd9RFwOQpZL5oQu")


def read_file(file_path):
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        print("Parsing PDF with LlamaParse...")
        documents = llama_parse.load_data(file_path)
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
 
def chunk_text(text, max_tokens=100):
    words = text.split()
    return [" ".join(words[i:i+max_tokens]) for i in range(0, len(words), max_tokens)]

client = QdrantClient(
    url="https://2658912c-32a7-4ba4-b6a1-1fdd6c26c160.europe-west3-0.gcp.cloud.qdrant.io",
    api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.wU-LH_0rgaM_bpzXl2eNJDIX9CNqnyHchLG_nwhQjhc",
)
 
print(client.get_collections())

def recreate_qdrant_collection(client, collection_name="my_collection", size=384, distance=Distance.COSINE):
    try:
        client.delete_collection("my_collection")
        print("Deleted old 'my_collection' successfully.")
    except Exception as e:
        print(f"Error deleting collection: {e}")

            # Recreate with correct size = 384
    try:
        client.create_collection(
        collection_name="my_collection",
        vectors_config=VectorParams(size=384, distance=Distance.COSINE),
        )
        print("Created 'my_collection' with dimension 384.")
    except Exception as e:
        print(f"Error creating collection: {e}")

# file_path = "ml.txt"  # Change to your file path
# raw_text = read_file(file_path)
# chunks = chunk_text(raw_text)

#Generate real text embeddings
# model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
# vectors = model.encode(chunks).tolist()
 
# Upload vectors using upsert instead of upload_collection
def upload_vectors_to_qdrant(client, collection_name, vectors, chunks):
    try:
        client.upsert(
        collection_name="my_collection",
        points=[
            {
                "id": i,
                "vector": vectors[i],
                "payload": {"text": chunks[i]},
            }
            for i in range(len(chunks))
        ]
    )
        print("Vectors uploaded successfully")
    except Exception as e:
        print(f"Error uploading vectors: {e}")

# query = "What are algorithms?"
# query_vector = model.encode(query).tolist()
# print("-----line61----")

results = None  
# Search
def search_qdrant(client, collection_name, query_vector, limit=3):
    try:
        results = client.search(
        collection_name="my_collection",
        query_vector=query_vector,
        limit=3,
        with_vectors=True,
        with_payload=True
        )
        print("Search results:")
        for r in results:
            print(f"- {r.payload['text']} (score: {r.score})")
        return results
    except Exception as e:
        print(f"Error during search: {e}")

# qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

# if not results:
#     print("No results found or query failed. Exiting.")
#     exit()

# retrieved_context = " ".join([r.payload["text"] for r in results])