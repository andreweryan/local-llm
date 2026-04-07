import os
import json
import threading

"""
Persistent conversation memory, lazily rebuilt from the JSONL request log.

On first access of a session, its history is replayed from the log and
cached in-process. Subsequent reads within the same server lifetime use
the cache. This avoids replaying the full log on startup while still
restoring any session on demand.
"""

MAX_TURNS = int(os.getenv("MEMORY_MAX_TURNS", "5"))
LOG_FILE = os.getenv("LOG_FILE", os.path.join("logs", "requests.jsonl"))

_lock = threading.Lock()

# Loaded sessions cache. None = not yet attempted. [] = loaded but empty.
_store: dict[str, list[dict]] = {}
# Track which session IDs have been loaded from log so we don't re-read
_loaded_sessions: set[str] = set()


def _replay_session(session_id: str) -> list[dict]:
    """
    Scan the log file for all entries belonging to session_id and return
    the reconstructed turn list, capped at MAX_TURNS pairs.
    Errors and empty responses are skipped.
    """
    turns: list[dict] = []

    if not os.path.exists(LOG_FILE):
        return turns

    try:
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if record.get("session_id") != session_id:
                    continue
                if record.get("error"):
                    continue

                raw_prompt = record.get("raw_prompt", "")
                response = record.get("response", "")
                if not raw_prompt or not response:
                    continue

                turns.append({"role": "user", "content": raw_prompt})
                turns.append({"role": "assistant", "content": response})

        # Trim to cap
        max_entries = MAX_TURNS * 2
        if len(turns) > max_entries:
            turns = turns[-max_entries:]

        if turns:
            print(
                f"[memory] Restored {len(turns) // 2} turn(s) for session '{session_id}'.",
                flush=True,
            )
    except Exception as e:
        print(f"[memory] Error replaying session '{session_id}': {e}", flush=True)

    return turns


def _ensure_loaded(session_id: str) -> None:
    """Load session from log into cache if not already done this process lifetime."""
    if session_id not in _loaded_sessions:
        _store[session_id] = _replay_session(session_id)
        _loaded_sessions.add(session_id)


def get(session_id: str) -> list[dict]:
    with _lock:
        _ensure_loaded(session_id)
        return list(_store.get(session_id, []))


def append(session_id: str, role: str, content: str) -> None:
    with _lock:
        _ensure_loaded(session_id)
        _store.setdefault(session_id, []).append({"role": role, "content": content})
        max_entries = MAX_TURNS * 2
        if len(_store[session_id]) > max_entries:
            _store[session_id] = _store[session_id][-max_entries:]


def clear(session_id: str) -> None:
    """
    Clears in-process memory for a session. Marks it as loaded so the
    log is not replayed on next access within this server lifetime.
    Note: does not modify the log file — history will not be restored
    on the next server restart for this session.
    """
    with _lock:
        _store[session_id] = []
        _loaded_sessions.add(session_id)
