import os
import time
import json
import requests
import subprocess
from typing import Optional
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException

from tools.registry import TOOLS
from tools.rag import load_or_build_index
from tools import memory as mem
from tools import logger

DOCS_FOLDER = os.getenv("DOCS_FOLDER", "docs")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3:latest")
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "10"))


ROUTER_PROMPT = """
You are an AI tool router.

Available tools and when to use them:

rag_search:
  Search internal documents for relevant information.
  Use for: explanations, concepts, how-to questions, research topics.

time:
  Return the user's current local time.
  Use ONLY when the user explicitly asks what time it is.
  Never use for general knowledge or explanation questions.

haversine:
  Compute the distance between two geographic coordinates.
  Use ONLY when the user explicitly asks for a distance and provides
  two coordinate pairs. Never use speculatively.

Rules:
- Never call time or haversine unless the user's message makes them
  explicitly and unambiguously necessary.
- Never repeat the same tool twice in a chain.
- If no tool is needed, leave "tools" as an empty array.

Return JSON only — one of these three shapes:

1. No tool needed:
{
  "tools": [],
  "response": "<your answer here>"
}

2. Single tool:
{
  "tools": [
    {"tool": "<tool_name>", "arguments": { ... }}
  ]
}

3. Multiple tools (executed in order; each result may inform the next):
{
  "tools": [
    {"tool": "<tool_name_1>", "arguments": { ... }},
    {"tool": "<tool_name_2>", "arguments": { ... }}
  ]
}

Always use the "tools" array shape. Never wrap the array in anything else.
"""

# Used only when there is conversation history to fold in.
REWRITE_PROMPT = """
You are a query rewriter for a retrieval-augmented search system.

Given a conversation history and the user's latest message, rewrite the message
into a single, self-contained search query that resolves all pronouns, implicit
references, and follow-up shorthand using the conversation context.

Rules:
- Output the rewritten query as plain text only — no preamble, no explanation.
- If the message is already fully self-contained, output it unchanged.
- Never answer the question; only rewrite it.
- Keep the rewrite concise (one or two sentences at most).
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


def call_ollama(messages: list[dict], format_json: bool = False) -> str:
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


def rewrite_query(prompt: str, history: list[dict]) -> str:
    """
    Return a history-aware rewrite of `prompt` suitable for vector search.

    Only fires an LLM call when there is prior history to fold in — if the
    session is fresh the raw prompt is returned immediately, saving a round-trip.

    Falls back to the original prompt on any failure so the pipeline never stalls.
    """
    if not history:
        return prompt

    try:
        # Feed the last 6 messages (3 pairs) to keep the rewrite prompt focused.
        recent = history[-6:]
        rewrite_messages = [
            {"role": "system", "content": REWRITE_PROMPT},
            *recent,
            {"role": "user", "content": prompt},
        ]
        rewritten = call_ollama(rewrite_messages).strip()
        # Reject rewrites that are empty or suspiciously long
        if not rewritten or len(rewritten) > len(prompt) * 4:
            return prompt
        return rewritten
    except Exception as exc:
        print(f"[rewrite] Fell back to raw prompt: {exc}", flush=True)
        return prompt


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
    session_id: Optional[str] = Field(
        default=None,
        description=(
            "Opaque session identifier. Pass the same value across turns "
            "to enable conversation memory and history-aware query rewriting."
        ),
    )


class GenerateResponse(BaseModel):
    response: str
    sources: list[dict]
    session_id: Optional[str] = None
    rewritten_query: Optional[str] = None


def _build_router_messages(prompt: str, history: list[dict]) -> list[dict]:
    """Inject conversation history so the router understands context."""
    messages = [{"role": "system", "content": ROUTER_PROMPT}]
    messages.extend(history)
    messages.append({"role": "user", "content": prompt})
    return messages


def _run_tool_step(
    tool_name: str,
    arguments: dict,
    raw_prompt: str,
    rewritten_query: str,
    app,
) -> tuple[str, list]:
    """Execute one tool step and return (context_fragment, sources)."""
    if tool_name not in TOOLS:
        raise ValueError(f"Unknown tool: {tool_name!r}")

    tool = TOOLS[tool_name]

    if tool_name == "rag_search":
        # Use the history-aware rewrite so FAISS embeds a richer query
        context, sources = tool.run(rewritten_query, app)
        fragment = f"Tool '{tool_name}' result:\n{context}"
    else:
        result, sources = tool.run(raw_prompt, app, **arguments)
        # Discard error results so they don't pollute the final context
        if isinstance(result, str) and result.startswith("Error:"):
            raise ValueError(result)
        fragment = f"Tool '{tool_name}' result:\n{result}"

    return fragment, sources


@app.post("/generate", response_model=GenerateResponse)
def generate(req: PromptRequest):

    t_start = time.monotonic()
    session_id = req.session_id
    if not session_id:
        import uuid

        session_id = str(uuid.uuid4())

    history = mem.get(session_id) if session_id else []

    # -- History-aware query rewrite ------------------------------------------
    # rewritten_query is used only for RAG embedding.
    # raw prompt is preserved for the final answer LLM call.
    rewritten_query = rewrite_query(req.prompt, history)

    # -- Route ----------------------------------------------------------------
    try:
        router_output = call_ollama(
            _build_router_messages(req.prompt, history),
            format_json=True,
        )
        decision = json.loads(router_output)
    except Exception as e:
        _log_error(req, session_id, rewritten_query, t_start, f"Router failed: {e}")
        raise HTTPException(status_code=500, detail=f"Router failed: {e}")

    tool_steps: list[dict] = decision.get("tools", [])

    # -- No tool: direct answer -----------------------------------------------
    if not tool_steps:
        response_text = decision.get("response", "")
        if session_id:
            mem.append(session_id, "user", req.prompt)
            mem.append(session_id, "assistant", response_text)
        logger.log(
            session_id=session_id,
            raw_prompt=req.prompt,
            rewritten_query=rewritten_query,
            router_tools=[],
            sources=[],
            response=response_text,
            latency_ms=(time.monotonic() - t_start) * 1000,
        )
        return GenerateResponse(
            response=response_text,
            sources=[],
            session_id=session_id,
            rewritten_query=rewritten_query,
        )

    # -- Tool chain -----------------------------------------------------------
    all_sources: list[dict] = []
    tool_fragments: list[str] = []
    router_tool_names: list[str] = []

    for step in tool_steps:
        tool_name = step.get("tool")
        arguments = step.get("arguments", {})
        if not tool_name or tool_name not in TOOLS:
            err = f"Unknown tool in chain: {tool_name!r}"
            _log_error(req, session_id, rewritten_query, t_start, err)
            raise HTTPException(status_code=400, detail=err)
        try:
            fragment, sources = _run_tool_step(
                tool_name, arguments, req.prompt, rewritten_query, app
            )
            tool_fragments.append(fragment)
            all_sources.extend(sources)
            router_tool_names.append(tool_name)
        except Exception as e:
            err = f"Tool '{tool_name}' failed: {e}"
            _log_error(req, session_id, rewritten_query, t_start, err)
            raise HTTPException(status_code=500, detail=err)

    # -- Final answer ---------------------------------------------------------
    combined_tool_context = "\n\n".join(tool_fragments)

    final_messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        *history,
        {
            "role": "user",
            "content": (
                f"{combined_tool_context}\n\n"
                f"Using the above tool results, answer the following:\n{req.prompt}"
            ),
        },
    ]

    try:
        answer = call_ollama(final_messages)
    except Exception as e:
        err = f"Final generation failed: {e}"
        _log_error(req, session_id, rewritten_query, t_start, err)
        raise HTTPException(status_code=500, detail=err)

    answer = answer.strip()

    if session_id:
        mem.append(session_id, "user", req.prompt)
        mem.append(session_id, "assistant", answer)

    logger.log(
        session_id=session_id,
        raw_prompt=req.prompt,
        rewritten_query=rewritten_query,
        router_tools=router_tool_names,
        sources=all_sources,
        response=answer,
        latency_ms=(time.monotonic() - t_start) * 1000,
    )

    return GenerateResponse(
        response=answer,
        sources=all_sources,
        session_id=session_id,
        rewritten_query=rewritten_query,
    )


def _log_error(req: PromptRequest, session_id, rewritten_query, t_start, detail):
    logger.log(
        session_id=session_id,
        raw_prompt=req.prompt,
        rewritten_query=rewritten_query,
        router_tools=[],
        sources=[],
        response="",
        latency_ms=(time.monotonic() - t_start) * 1000,
        error=detail,
    )


@app.get("/memory/{session_id}")
def get_memory(session_id: str):
    """Inspect the stored conversation history for a session."""
    return {"session_id": session_id, "history": mem.get(session_id)}


@app.delete("/memory/{session_id}")
def clear_memory(session_id: str):
    """Wipe the conversation history for a session."""
    mem.clear(session_id)
    return {"session_id": session_id, "cleared": True}


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
