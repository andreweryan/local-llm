import os
import time
import requests
import subprocess
from fastapi import FastAPI
from pydantic import BaseModel
from contextlib import asynccontextmanager

from rag import build_faiss_index, load_faiss_index, search

INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "faiss_index")
DOCS_FOLDER = os.getenv("DOCS_FOLDER", "docs")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3:latest")

chunks = None
faiss_index = None
embeddings = None

def wait_for_ollama(host, timeout=10):
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = requests.get(f"{host}/api/tags", timeout=2)
            if r.status_code == 200:
                return True
        except:
            time.sleep(0.5)
    return False


def ensure_ollama_running():
    if wait_for_ollama(OLLAMA_HOST, timeout=2):
        return

    subprocess.Popen(
        ["ollama", "serve"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

    if not wait_for_ollama(OLLAMA_HOST, timeout=15):
        raise RuntimeError("Ollama failed to start")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global chunks, faiss_index, embeddings

    # Ensure Ollama is running
    ensure_ollama_running()

    # Load or build FAISS index
    if os.path.exists(os.path.join(INDEX_PATH, "index.faiss")):
        print("Loading existing FAISS index...")
        chunks, faiss_index, embeddings = load_faiss_index(INDEX_PATH)
    else:
        print("FAISS index not found. Building from documents...")
        chunks, faiss_index, embeddings = build_faiss_index(DOCS_FOLDER, INDEX_PATH)

    print(f"FAISS index ready with {len(chunks)} chunks.")

    yield

app = FastAPI(lifespan=lifespan)

class Prompt(BaseModel):
    prompt: str

@app.post("/generate")
def generate_text(prompt: Prompt):
    try:
        # Retrieve relevant chunks
        context_chunks = search(prompt.prompt, chunks, faiss_index, top_k=5)

        # Inline citations for accuracy
        context_texts = [
            f"{c['text']} [{c['source']} - page {c['page']}]"
            for c in context_chunks
        ]
        context = "\n\n".join(context_texts)

        # Build prompt
        full_prompt = f"""
You are a helpful assistant. Use ONLY the context below to answer the question.
If the answer is not in the context, say "I don't know."
Cite sources exactly as shown in brackets.

Context:
{context}

Question:
{prompt.prompt}
"""

        # Call Ollama
        response = requests.post(
            f"{OLLAMA_HOST}/api/generate",
            json={
                "model": OLLAMA_MODEL,
                "prompt": full_prompt,
                "stream": False
            },
            timeout=120
        )
        response.raise_for_status()

        data = response.json()
        output = data.get("response", "").strip()

        return {
            "response": output or "(Empty response from model)",
            "context_used": context_chunks
        }

    except requests.RequestException:
        return {
            "error": "Model service unavailable. Ensure Ollama is running."
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=False)