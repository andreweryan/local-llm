import os
import time
import json
import requests
import subprocess
from typing import Optional
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException

from tools import logger
from tools import memory as mem
from tools.registry import TOOLS
from tools.rag import load_or_build_index

DOCS_FOLDER = os.getenv("DOCS_FOLDER", "docs")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3:latest")
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "10000"))
INDEX_READY = False

ROUTER_PROMPT = """
You are an AI tool router.

Available tools and when to use them:

rag_search:
  Search internal documents for relevant information.
  Use for: explanations, concepts, how-to questions, research topics, named
  entities, models, papers, or systems you do not recognize with certainty.
  When in doubt, always prefer rag_search over answering directly.

Rules:
- Never repeat the same tool twice in a chain.
- If no tool is needed, leave "tools" as an empty array.
- Valid tool names are ONLY: rag_search, time, haversine.
- Never invent or use any other tool name.

Return JSON only — one of these three shapes:

1. No tool needed:
{"tools": [], "response": ""}

2. Single tool:
{"tools": [{"tool": "<tool_name>", "arguments": {}}]}

3. Multiple tools (executed in order):
{"tools": [{"tool": "<tool_name_1>", "arguments": {}}, {"tool": "<tool_name_2>", "arguments": {}}]}

Always use the "tools" array shape. Never wrap the array in anything else.
"""

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

ANSWER_SYSTEM_PROMPT = """
You are an expert research assistant helping a user conduct deep technical research.
Your goal is to provide thorough, thoughtful answers that combine retrieved document
knowledge with your own expertise.

When answering:
- Write naturally as if the knowledge is your own. Never reference tools, searches,
  retrieval, or context — just answer.
- Always supplement retrieved content with your own knowledge where it adds value.
  If retrieved documents cover specific examples or findings, follow up with broader
  context, related techniques, historical background, or open research questions from
  your own training.
- Structure answers for depth: lead with the core answer, then expand with technical
  detail, implications, and connections to adjacent ideas.
- Be direct. Do not hedge excessively or repeat the question back.
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
    payload = {"model": OLLAMA_MODEL, "stream": False, "messages": messages}
    if format_json:
        payload["format"] = "json"
    r = requests.post(f"{OLLAMA_HOST}/api/chat", json=payload, timeout=120)
    r.raise_for_status()
    return r.json()["message"]["content"]


def rewrite_query(prompt: str, history: list[dict]) -> str:
    if not history:
        return prompt
    try:
        rewritten = call_ollama(
            [
                {"role": "system", "content": REWRITE_PROMPT},
                *history[-10:],
                {"role": "user", "content": prompt},
            ]
        ).strip()
        if not rewritten or len(rewritten) > len(prompt) * 4:
            return prompt
        return rewritten
    except Exception as exc:
        print(f"[rewrite] Fell back to raw prompt: {exc}", flush=True)
        return prompt


@asynccontextmanager
async def lifespan(app: FastAPI):
    global INDEX_READY

    if not wait_for_ollama(OLLAMA_HOST, timeout=10):
        print("WARNING: Ollama not reachable.", flush=True)
    else:
        print("Ollama is up.", flush=True)

    chunks, index, _ = load_or_build_index(DOCS_FOLDER)

    app.state.chunks = chunks
    app.state.index = index

    INDEX_READY = True

    print(f"Ready — {len(chunks)} chunks indexed.", flush=True)
    yield


app = FastAPI(lifespan=lifespan)


class PromptRequest(BaseModel):
    prompt: str
    top_k: int = RAG_TOP_K
    session_id: Optional[str] = Field(
        default=None,
        description="Opaque session identifier for conversation memory.",
    )


class GenerateResponse(BaseModel):
    response: str
    sources: list[dict]
    session_id: Optional[str] = None
    rewritten_query: Optional[str] = None


def _run_tool_step(
    tool_name: str,
    arguments: dict,
    raw_prompt: str,
    rewritten_query: str,
    app,
) -> tuple[str, list]:
    if tool_name not in TOOLS:
        raise ValueError(f"Unknown tool: {tool_name!r}")
    tool = TOOLS[tool_name]
    if tool_name == "rag_search":
        context, sources = tool.run(rewritten_query, app)
        return context, sources
    else:
        result, sources = tool.run(raw_prompt, app, **arguments)
        if isinstance(result, str) and result.startswith("Error:"):
            raise ValueError(result)
        return result, sources


def _commit(
    session_id, prompt, answer, rewritten_query, router_tools, sources, t_start
):
    if session_id:
        mem.append(session_id, "user", prompt)
        mem.append(session_id, "assistant", answer)
    logger.log(
        session_id=session_id,
        raw_prompt=prompt,
        rewritten_query=rewritten_query,
        router_tools=router_tools,
        sources=sources,
        response=answer,
        latency_ms=(time.monotonic() - t_start) * 1000,
    )


@app.post("/generate", response_model=GenerateResponse)
def generate(req: PromptRequest):
    import uuid

    t_start = time.monotonic()
    session_id = req.session_id or str(uuid.uuid4())
    history = mem.get(session_id)
    rewritten_query = rewrite_query(req.prompt, history)

    try:
        router_output = call_ollama(
            [
                {"role": "system", "content": ROUTER_PROMPT},
                *history,
                {"role": "user", "content": req.prompt},
            ],
            format_json=True,
        )
        decision = json.loads(router_output)
    except Exception as e:
        logger.log(
            session_id=session_id,
            raw_prompt=req.prompt,
            rewritten_query=rewritten_query,
            router_tools=[],
            sources=[],
            response="",
            latency_ms=(time.monotonic() - t_start) * 1000,
            error=str(e),
        )
        raise HTTPException(status_code=500, detail=f"Router failed: {e}")

    seen = set()
    tool_steps = []
    for s in decision.get("tools", []):
        tool = s.get("tool")
        if not tool or tool in seen:
            continue
        seen.add(tool)
        tool_steps.append(s)

    if not tool_steps:
        try:
            answer = call_ollama(
                [
                    {"role": "system", "content": ANSWER_SYSTEM_PROMPT},
                    *history,
                    {"role": "user", "content": req.prompt},
                ]
            ).strip()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Generation failed: {e}")
        _commit(session_id, req.prompt, answer, rewritten_query, [], [], t_start)
        return GenerateResponse(
            response=answer,
            sources=[],
            session_id=session_id,
            rewritten_query=rewritten_query,
        )

    all_sources: list[dict] = []
    tool_fragments: list[str] = []
    router_tool_names: list[str] = []

    for step in tool_steps:
        tool_name = step.get("tool")
        if tool_name not in TOOLS:
            # raise HTTPException(status_code=400, detail=f"Unknown tool: {tool_name!r}")
            continue
        try:
            fragment, sources = _run_tool_step(
                tool_name, step.get("arguments", {}), req.prompt, rewritten_query, app
            )
            tool_fragments.append(fragment)
            all_sources.extend(sources)
            router_tool_names.append(tool_name)
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Tool '{tool_name}' failed: {e}"
            )

    combined = "\n\n".join(tool_fragments)

    if all(TOOLS[t].direct_response for t in router_tool_names if t in TOOLS):
        answer = combined.strip()
        _commit(
            session_id,
            req.prompt,
            answer,
            rewritten_query,
            router_tool_names,
            all_sources,
            t_start,
        )
        return GenerateResponse(
            response=answer,
            sources=all_sources,
            session_id=session_id,
            rewritten_query=rewritten_query,
        )

    try:
        answer = call_ollama(
            [
                {"role": "system", "content": ANSWER_SYSTEM_PROMPT},
                *history,
                {
                    "role": "user",
                    "content": (
                        f"{combined}\n\n"
                        f"{req.prompt}\n\n"
                        "Answer directly and naturally. Use the context above as a foundation, "
                        "then expand with your own knowledge and expertise where it adds depth."
                    ),
                },
            ]
        ).strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Final generation failed: {e}")

    _commit(
        session_id,
        req.prompt,
        answer,
        rewritten_query,
        router_tool_names,
        all_sources,
        t_start,
    )
    return GenerateResponse(
        response=answer,
        sources=all_sources,
        session_id=session_id,
        rewritten_query=rewritten_query,
    )


@app.get("/memory/{session_id}")
def get_memory(session_id: str):
    return {"session_id": session_id, "history": mem.get(session_id)}


@app.delete("/memory/{session_id}")
def clear_memory(session_id: str):
    mem.clear(session_id)
    return {"session_id": session_id, "cleared": True}


@app.get("/health")
def health():
    ollama_ok = wait_for_ollama(OLLAMA_HOST, timeout=2)

    return {
        "status": "ok" if (ollama_ok and INDEX_READY) else "starting",
        "ready": INDEX_READY,
        "ollama": ollama_ok,
        "chunks_loaded": len(app.state.chunks) if hasattr(app.state, "chunks") else 0,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000,
        reload=False,
        log_level="warning",
    )
