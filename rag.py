import os
import json
import numpy as np
import faiss
import requests
from pypdf import PdfReader

# --- Load PDF file and extract text with page-level metadata ---
def load_pdf(file_path):
    reader = PdfReader(file_path)
    page_texts = []
    for page_num, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text()
        if page_text:
            page_texts.append({"text": page_text, "page": page_num, "source": os.path.basename(file_path)})
    return page_texts

# --- Load all documents from folder ---
def load_documents(folder="docs"):
    docs = []
    for filename in os.listdir(folder):
        path = os.path.join(folder, filename)
        if filename.endswith((".txt", ".md")):
            with open(path, "r", encoding="utf-8") as f:
                docs.append([{"text": f.read(), "page": None, "source": filename}])
        elif filename.endswith(".pdf"):
            docs.append(load_pdf(path))
    return docs

# --- Split text into overlapping chunks ---
def chunk_text(text, chunk_size=1000, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

import re

def chunk_text_by_sentences(text, max_sentences=5, overlap_sentences=1):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    i = 0
    while i < len(sentences):
        chunk_sentences = sentences[i:i + max_sentences]
        chunks.append(" ".join(chunk_sentences).strip())
        # Move i forward but leave overlap
        i += max_sentences - overlap_sentences
    return chunks

# --- Get embedding for a single text using Ollama ---
def get_embedding(text):
    ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    response = requests.post(
        f"{ollama_host}/api/embeddings",
        json={"model": "nomic-embed-text", "prompt": text}
    )
    response.raise_for_status()
    return np.array(response.json()["embedding"], dtype=np.float32)

# --- Build FAISS index from documents ---
def build_faiss_index(folder="docs", index_path="faiss_index", save=True, chunk_size=500, overlap=50, max_sentences=3, overlap_sentences=1):
    os.makedirs(index_path, exist_ok=True)
    raw_docs = load_documents(folder)
    all_chunks = []
    embeddings_list = []

    for doc_pages in raw_docs:
        for page_info in doc_pages:
            page_text = page_info["text"]
            page_num = page_info["page"]
            source = page_info["source"]
            chunks = chunk_text(page_text, chunk_size=chunk_size, overlap=overlap)
            # chunks = chunk_text_by_sentences(page_text, max_sentences=max_sentences, overlap_sentences=overlap_sentences)
            for chunk in chunks:
                all_chunks.append({"text": chunk, "source": source, "page": page_num})
                embeddings_list.append(get_embedding(chunk))

    embeddings = np.array(embeddings_list, dtype=np.float32)
    faiss.normalize_L2(embeddings)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    # --- Save index, embeddings, and chunks to disk ---
    if save:
        faiss.write_index(index, os.path.join(index_path, "index.faiss"))
        np.save(os.path.join(index_path, "embeddings.npy"), embeddings)
        with open(os.path.join(index_path, "chunks.json"), "w", encoding="utf-8") as f:
            json.dump(all_chunks, f, ensure_ascii=False)

    return all_chunks, index, embeddings

# --- Load FAISS index, embeddings, and chunks from disk ---
def load_faiss_index(index_path="faiss_index"):
    index_file = os.path.join(index_path, "index.faiss")
    chunks_file = os.path.join(index_path, "chunks.json")
    emb_file = os.path.join(index_path, "embeddings.npy")

    if not os.path.exists(index_file) or not os.path.exists(chunks_file) or not os.path.exists(emb_file):
        raise FileNotFoundError("FAISS index not found. Build first.")

    index = faiss.read_index(index_file)
    embeddings = np.load(emb_file)
    with open(chunks_file, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    return chunks, index, embeddings

# --- Search top_k chunks in FAISS index for a query ---
def search(query, chunks, index, top_k=10):
    q_emb = get_embedding(query).reshape(1, -1)
    faiss.normalize_L2(q_emb)
    distances, indices = index.search(q_emb, top_k)
    results = [chunks[i] for i in indices[0]]
    return results