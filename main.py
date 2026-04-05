import os
import time
import requests
import subprocess
from pydantic import BaseModel
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException

from tools.registry import TOOLS
from tools.rag import load_or_build_index

INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "faiss_index")
DOCS_FOLDER = os.getenv("DOCS_FOLDER", "docs")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3:latest")
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "10"))

SYSTEM_PROMPT_TEMPLATE = """
You are a retrieval assistant.

You have access to the following tools:

{tool_list}

Use tools whenever they can help gather information before answering the user.
Just provide the response, do not add additional language such as: According to the tool result...
"""


def build_tool_list():
    """
    Convert the registered tools into a prompt-friendly list.
    """
    lines = []
    for tool in TOOLS.values():
        lines.append(f"{tool.name}: {tool.description}")
    return "\n".join(lines)


def choose_tool(query: str):
    """
    Simple router to decide which tool to use.
    """

    q = query.lower()

    if "distance" in q or "kilometers" in q or "miles" in q:
        # crude detection; you can improve with regex or NLP later
        return "haversine"
    elif "time" in q or "current time" in q:
        return "time"

    # fallback to rag
    return "rag_search"


SYSTEM_PROMPT = SYSTEM_PROMPT_TEMPLATE.format(tool_list=build_tool_list())


def wait_for_ollama(host: str, timeout: int = 10) -> bool:
    """
    Ensure Ollama is running and reachable.
    """

    start = time.time()

    while time.time() - start < timeout:
        try:
            r = requests.get(f"{host}/api/tags", timeout=2)
            if r.status_code == 200:
                return True
        except requests.RequestException:
            pass

        time.sleep(0.5)

    print("Starting Ollama server...")

    subprocess.Popen(
        ["ollama", "serve"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    start = time.time()

    while time.time() - start < timeout:
        try:
            r = requests.get(f"{host}/api/tags", timeout=2)
            if r.status_code == 200:
                return True
        except requests.RequestException:
            pass

        time.sleep(0.5)

    print("Failed to start Ollama within timeout.")
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
    prompt: str
    top_k: int = RAG_TOP_K


class GenerateResponse(BaseModel):
    response: str
    sources: list[dict]


@app.post("/generate", response_model=GenerateResponse)
def generate(req: PromptRequest):

    tool_name = choose_tool(req.prompt)

    tool = TOOLS[tool_name]

    if tool_name == "rag_search":
        context, sources = tool.run(req.prompt, app, top_k=req.top_k)

        if not context:
            return GenerateResponse(
                response="I don't know — no relevant documents were found for your question.",
                sources=[],
            )

        user_message = f"Context:\n{context}\n\nQuestion: {req.prompt}"
    else:
        result, sources = tool.run(req.prompt, app)

        user_message = f"Tool result:\n{result}\n\nQuestion: {req.prompt}"

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
                        "content": user_message,
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
        sources=sources,
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
