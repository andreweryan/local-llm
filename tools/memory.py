from collections import defaultdict

"""
Simple in-process conversation memory.

Each session is a list of {"role": "user"|"assistant", "content": "..."} dicts,
capped at MAX_TURNS pairs to keep context windows manageable.
"""

MAX_TURNS = int(__import__("os").getenv("MEMORY_MAX_TURNS", "10"))

# session_id -> list[dict]
_store: dict[str, list[dict]] = defaultdict(list)


def get(session_id: str) -> list[dict]:
    """Return the full history for a session."""
    return list(_store[session_id])


def append(session_id: str, role: str, content: str) -> None:
    """Add one turn and evict oldest pair if over the cap."""
    _store[session_id].append({"role": role, "content": content})
    # Each pair = 2 entries (user + assistant); trim from the front
    max_entries = MAX_TURNS * 2
    if len(_store[session_id]) > max_entries:
        _store[session_id] = _store[session_id][-max_entries:]


def clear(session_id: str) -> None:
    """Wipe history for a session."""
    _store.pop(session_id, None)
