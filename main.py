import os
import time
import json
import requests
import subprocess
from pydantic import BaseModel
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException

from tools.registry import TOOLS
from tools.rag import load_or_build_index

DOCS_FOLDER = os.getenv("DOCS_FOLDER", "docs")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3:latest")
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "10"))

ROUTER_PROMPT = """
You are an AI tool router.

Available tools:

rag_search:
Search internal documents for relevant information.

time:
Return the user's current local time.

haversine:
Compute the distance between two coordinates.

Return JSON only.

If a tool is needed:

{
  "tool": "<tool_name>",
  "arguments": { ... }
}

If no tool is needed:

{
  "tool": "none",
  "response": "<answer>"
}
"""


def wait_for_ollama(host: str, timeout: int = 10) -> bool:

    start = time.time()

    while time.time() - start < timeout:
        try:
            r = requests.get(f"{host}/api/tags", timeout=2)
            if r.status_code == 200:
                return True
        except requests.RequestException:
            pass

        time.sleep(0.5)

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

    return False


def call_ollama(messages, format_json=False):

    payload = {
        "model": OLLAMA_MODEL,
        "stream": False,
        "messages": messages,
    }

    if format_json:
        payload["format"] = "json"

    r = requests.post(
        f"{OLLAMA_HOST}/api/chat",
        json=payload,
        timeout=120,
    )

    r.raise_for_status()

    return r.json()["message"]["content"]


@asynccontextmanager
async def lifespan(app: FastAPI):

    if not wait_for_ollama(OLLAMA_HOST, timeout=10):
        print("WARNING: Ollama not reachable.", flush=True)
    else:
        print("Ollama is up.", flush=True)

    chunks, index, _ = load_or_build_index(DOCS_FOLDER)

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

    try:

        router_output = call_ollama(
            [
                {"role": "system", "content": ROUTER_PROMPT},
                {"role": "user", "content": req.prompt},
            ],
            format_json=True,
        )

        decision = json.loads(router_output)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Router failed: {e}")

    tool_name = decision.get("tool")

    # Normal chat
    if tool_name == "none":

        response = decision.get("response", "")

        return GenerateResponse(
            response=response,
            sources=[],
        )

    # Tool execution
    if tool_name not in TOOLS:
        raise HTTPException(status_code=400, detail="Unknown tool requested")

    tool = TOOLS[tool_name]

    try:

        if tool_name == "rag_search":

            context, sources = tool.run(
                req.prompt,
                app,
                top_k=req.top_k,
            )

            user_message = f"Context:\n{context}\n\nQuestion:{req.prompt}"

        else:

            result, sources = tool.run(
                req.prompt,
                app,
                **decision.get("arguments", {}),
            )

            user_message = f"Tool result:\n{result}\n\nQuestion:{req.prompt}"

        answer = call_ollama(
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": user_message},
            ]
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Tool execution failed: {e}")

    return GenerateResponse(
        response=answer.strip(),
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
