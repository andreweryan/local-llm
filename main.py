import os
import requests
from fastapi import FastAPI
from pydantic import BaseModel
from rag import build_faiss_index, load_faiss_index, search

INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "faiss_index")
DOCS_FOLDER = os.getenv("DOCS_FOLDER", "docs")

if os.path.exists(os.path.join(INDEX_PATH, "index.faiss")):
    print("Loading existing FAISS index...")
    chunks, faiss_index, embeddings = load_faiss_index(INDEX_PATH)
else:
    print("FAISS index not found. Building from documents...")
    chunks, faiss_index, embeddings = build_faiss_index(DOCS_FOLDER, INDEX_PATH)

print(f"FAISS index ready with {len(chunks)} chunks.")

app = FastAPI()

class Prompt(BaseModel):
    prompt: str

@app.post("/generate")
def generate_text(prompt: Prompt):
    try:
        ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        ollama_model = os.getenv("OLLAMA_MODEL", "llama3:latest")

        context_chunks = search(prompt.prompt, chunks, faiss_index)
        context_texts = [c["text"] for c in context_chunks]
        context = "\n\n".join(context_texts)

        full_prompt = f"""
You are a helpful assistant. Use ONLY the context below to answer the question.
If the answer is not in the context, say "I don't know."

Context:
{context}

Question:
{prompt.prompt}
"""

        response = requests.post(
            f"{ollama_host}/api/generate",
            json={"model": ollama_model, "prompt": full_prompt, "stream": False},
            timeout=120
        )
        response.raise_for_status()
        data = response.json()
        output = data.get("response", "").strip()

        return {
            "response": output or "(Empty response from model)",
            "context_used": context_chunks
        }

    except requests.RequestException as e:
        return {"error": f"Ollama request failed: {str(e)}"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=False)