import json
import os
import threading
from datetime import datetime, timezone

"""
Structured JSONL request logger.

Each request appends one JSON line to LOG_FILE (default: logs/requests.jsonl).
Fields logged per request:

  ts            ISO-8601 UTC timestamp
  session_id    opaque session string, or null
  raw_prompt    exactly what the user sent
  rewritten_query  history-aware rewrite used for RAG (may equal raw_prompt)
  router_tools  list of tool names the router selected
  sources       list of {source, page, rerank_score} for RAG results
  response      final answer text
  latency_ms    wall-clock ms for the whole /generate call
  error         error message string, or null

The file is line-delimited JSON so it is easy to tail, grep, and load into
pandas or any log aggregator.  Rotate it externally (logrotate, a cron job,
etc.) — this module never deletes or truncates the file itself.
"""

LOG_FILE = os.getenv("LOG_FILE", os.path.join("logs", "requests.jsonl"))

_lock = threading.Lock()


def _ensure_dir() -> None:
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)


def log(
    *,
    session_id: str | None,
    raw_prompt: str,
    rewritten_query: str,
    router_tools: list[str],
    sources: list[dict],
    response: str,
    latency_ms: float,
    error: str | None = None,
) -> None:
    """Append one structured line to the log file. Never raises."""
    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "session_id": session_id,
        "raw_prompt": raw_prompt,
        "rewritten_query": rewritten_query,
        "router_tools": router_tools,
        # Keep sources compact — full chunk text is already in the response
        "sources": [
            {
                "source": s.get("source"),
                "page": s.get("page"),
                "rerank_score": s.get("rerank_score"),
                "score": round(s.get("score", 0.0), 4),
            }
            for s in sources
        ],
        "response": response,
        "latency_ms": round(latency_ms, 1),
        "error": error,
    }
    try:
        _ensure_dir()
        line = json.dumps(record, ensure_ascii=False)
        with _lock:
            with open(LOG_FILE, "a", encoding="utf-8") as f:
                f.write(line + "\n")
    except Exception as exc:
        # Logging must never crash the server
        print(f"[logger] Failed to write log entry: {exc}", flush=True)
