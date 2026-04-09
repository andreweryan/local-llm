import os
import json
import logging
import threading
from pathlib import Path
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler

BASE_DIR = Path(__file__).resolve().parent.parent
LOG_DIR = BASE_DIR / "logs"
HISTORY_DIR = BASE_DIR / "history"

LOG_DIR.mkdir(exist_ok=True)
HISTORY_DIR.mkdir(exist_ok=True)

SERVER_LOG = LOG_DIR / "llm-server.log"
HISTORY_LOG = HISTORY_DIR / "history.jsonl"

MODEL = os.getenv("MODEL")

_lock = threading.Lock()


def get_logger(name: str = __name__) -> logging.Logger:
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    logger.propagate = False  # 🔥 prevents duplicates

    formatter = logging.Formatter(
        "%(asctime)s - [%(levelname)8s] - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = RotatingFileHandler(
        SERVER_LOG,
        maxBytes=5_000_000,
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)

    return logger


def log_chat(
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
    """
    Structured JSONL logger of LLM prompt/response.

    Each request appends one JSON line to LOG_FILE (default: history/history.jsonl).
    Fields logged per request:

    ts            ISO-8601 UTC timestamp
    session_id    opaque session string, or null
    raw_prompt    exactly what the user sent
    rewritten_query  history-aware rewrite used for RAG (may equal raw_prompt)
    router_tools  list of tool names the router selected
    sources       list of {source, page, rerank_score} for RAG results
    response      final answer text
    model         model name
    latency_ms    wall-clock ms for the whole /generate call
    error         error message string, or null

    The file is line-delimited JSON so it is easy to tail, grep, and load into
    pandas or any log aggregator.  Rotate it externally (logrotate, a cron job,
    etc.) — this module never deletes or truncates the file itself.
    """

    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "session_id": session_id,
        "raw_prompt": raw_prompt,
        "rewritten_query": rewritten_query,
        "router_tools": router_tools,
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
        "model": MODEL,
        "latency_ms": round(latency_ms, 1),
        "error": error,
    }

    try:
        line = json.dumps(record, ensure_ascii=False)

        with _lock:
            with open(HISTORY_LOG, "a", encoding="utf-8") as f:
                f.write(line + "\n")

    except Exception as exc:
        # fallback to app logger if JSON logging fails
        logger = get_logger("logger")
        logger.error(f"Failed to write chat log: {exc}", exc_info=True)
