import os
import time
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager

from rag import load_or_build_index, search

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "faiss_index")
DOCS_FOLDER = os.getenv("DOCS_FOLDER", "docs")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3:latest")
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "10"))

SYSTEM_PROMPT = """You are a retrieval assistant. Your only job is to answer questions using the context passages provided by the user. You have no other knowledge.

RULES — follow every rule without exception:
1. Every sentence in your answer MUST end with a citation in this exact format: [filename - page N] for paginated sources, or [filename] for unpaginated sources.
2. If a sentence draws from multiple sources, list all of them: [file1.pdf - page 2][file2.md]
3. Never write phrases like "according to the context", "based on the provided context", "the context states", or any similar meta-reference. Just answer directly and cite.
4. If the answer is not present in the context passages, respond with only this sentence: "I don't know — the provided documents don't cover this."
5. Do not repeat or rephrase the question.
6. Do not add commentary, caveats, or conclusions beyond what the sources state.

EXAMPLE of correct output format:
Question: What projects has Andrew worked on?
Answer: Andrew developed GeoYOLO, a high performance object detection engine for satellite images. [resume_projects.md] He also holds a patent for a target custody platform for modeling navigation trajectories. [resume_projects.md]

EXAMPLE of incorrect output format:
According to the context, Andrew worked on GeoYOLO."""


def wait_for_ollama(host: str, timeout: int = 10) -> bool:
    """
    Ollama connectivity check
    """
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = requests.get(f"{host}/api/tags", timeout=2)
            if r.status_code == 200:
                return True
        except requests.RequestException:
            time.sleep(0.5)
    return False


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not wait_for_ollama(OLLAMA_HOST, timeout=10):
        print(
            f"WARNING: Ollama not reachable at {OLLAMA_HOST}. "
            "The /generate endpoint will fail until it is running.",
            flush=True,
        )
    else:
        print("Ollama is up.", flush=True)

    chunks, index, _ = load_or_build_index(DOCS_FOLDER, INDEX_PATH)
    app.state.chunks = chunks
    app.state.index = index
    print(f"Ready — {len(chunks)} chunks indexed.", flush=True)

    yield


app = FastAPI(lifespan=lifespan)


class PromptRequest(BaseModel):
    """
    Request model
    """

    prompt: str
    top_k: int = RAG_TOP_K


class GenerateResponse(BaseModel):
    """
    Response model
    """

    response: str
    sources: list[dict]


def format_citation(chunk: dict) -> str:
    """
    PDFs have a real page number → [file.pdf - page N]
    Flat files (md, txt) have page=None → [file.md]
    """
    if chunk["page"] is not None:
        return f"[{chunk['source']} - page {chunk['page']}]"
    return f"[{chunk['source']}]"


@app.post("/generate", response_model=GenerateResponse)
def generate(req: PromptRequest):
    chunks = app.state.chunks
    index = app.state.index

    context_chunks = search(req.prompt, chunks, index, top_k=req.top_k)

    if not context_chunks:
        return GenerateResponse(
            response="I don't know — no relevant documents were found for your question.",
            sources=[],
        )

    # Build context string — each passage ends with its citation tag so the
    # model can copy it verbatim into the answer.
    context_parts = []
    for c in context_chunks:
        citation = format_citation(c)
        context_parts.append(f"{c['text']} {citation}")
    context = "\n\n".join(context_parts)

    try:
        response = requests.post(
            f"{OLLAMA_HOST}/api/chat",
            json={
                "model": OLLAMA_MODEL,
                "stream": False,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": f"Context:\n{context}\n\nQuestion: {req.prompt}",
                    },
                ],
            },
            timeout=120,
        )
        response.raise_for_status()
    except requests.RequestException as e:
        raise HTTPException(
            status_code=503,
            detail=f"Ollama unavailable: {e}. Ensure `ollama serve` is running.",
        )

    answer = response.json().get("message", {}).get("content", "").strip()

    return GenerateResponse(
        response=answer or "(Empty response from model)",
        sources=context_chunks,
    )


@app.get("/health")
def health():
    ollama_ok = wait_for_ollama(OLLAMA_HOST, timeout=2)
    return {
        "status": "ok" if ollama_ok else "degraded",
        "ollama": ollama_ok,
        "chunks_loaded": len(app.state.chunks) if hasattr(app.state, "chunks") else 0,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000, reload=False)
