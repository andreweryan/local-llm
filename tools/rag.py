import os
import re
import json
import logging
import hashlib
import requests
import numpy as np
from tqdm import tqdm
from pypdf import PdfReader
from tools.logger import get_logger
import chromadb
from chromadb.config import Settings

from .base import Tool

logging.getLogger("pypdf").setLevel(logging.ERROR)
logger = get_logger(__name__)
logger.propagate = False

HOST = os.getenv("HOST", "http://localhost:11434")
MODEL = os.getenv("MODEL")

EMBED_MODEL = os.getenv("EMBED_MODEL", "mxbai-embed-large")
EMBED_DIM = os.getenv("EMBED_MODEL_DIMS", "1024")
EMBED_TOKEN_LIMIT = int(os.getenv("EMBED_TOKEN_LIMIT", "460"))

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1000))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 100))
MIN_SCORE = float(os.getenv("RAG_MIN_SCORE", "0.1"))
MAX_PER_SOURCE_PAGE = int(os.getenv("RAG_MAX_PER_SOURCE_PAGE", "3"))
RERANK_TOP_N = int(os.getenv("RAG_RERANK_TOP_N", "5"))

RERANK_PROMPT = """\
Rate how relevant the passage is to the query.
Respond with a single JSON object. No prose, no explanation, no markdown.
The JSON must have exactly one key: "score" with an integer value from 0 to 10.

Examples of valid responses:
{"score": 0}
{"score": 5}
{"score": 10}
"""

_SCORE_RE = re.compile(r'"score"\s*:\s*(\d+)')


def get_collection(index_path: str) -> chromadb.Collection:
    """Return (creating if needed) the persistent Chroma collection."""
    client = chromadb.PersistentClient(
        path=index_path,
        settings=Settings(anonymized_telemetry=False),
    )
    return client.get_or_create_collection(
        name="rag",
        metadata={"hnsw:space": "cosine"},  # cosine ~ your normalised IP
    )


def _extract_score(raw: str) -> float:
    """
    Parse the relevance score from the model's response.

    Tries strict JSON first, then falls back to a regex scan so that
    models which emit prose around the JSON object don't cause a hard failure.
    """
    # 1. Strict parse — works when the model behaves
    try:
        return float(json.loads(raw).get("score", 0))
    except (json.JSONDecodeError, AttributeError):
        pass

    # 2. Extract the first {...} block and try again
    brace_start = raw.find("{")
    brace_end = raw.rfind("}")
    if brace_start != -1 and brace_end != -1:
        try:
            obj = json.loads(raw[brace_start : brace_end + 1])
            return float(obj.get("score", 0))
        except (json.JSONDecodeError, AttributeError):
            pass

    # 3. Regex scan — handles "...score": 7..." anywhere in the string
    m = _SCORE_RE.search(raw)
    if m:
        return float(m.group(1))

    return 0.0


def rerank(query: str, candidates: list[dict], top_n: int = RERANK_TOP_N) -> list[dict]:
    """
    Score each candidate with a lightweight LLM call and return the top_n
    results sorted by that score descending.

    Falls back to the original vector-similarity order if any call fails,
    so a slow/busy Ollama instance degrades gracefully.
    """
    scored = []
    for c in candidates:
        passage = c["text"][:800]  # keep prompts short
        try:
            r = requests.post(
                f"{HOST}/api/chat",
                json={
                    "model": MODEL,
                    "stream": False,
                    "format": "json",
                    "messages": [
                        {"role": "system", "content": RERANK_PROMPT},
                        {
                            "role": "user",
                            "content": (
                                f"Query: {query}\n\n"
                                f"Passage: {passage}\n\n"
                                'Respond with only: {{"score": <integer 0-10>}}'
                            ),
                        },
                    ],
                },
                timeout=30,
            )
            r.raise_for_status()
            raw = r.json()["message"]["content"]
            llm_score = _extract_score(raw)
        except Exception:
            # Fall back to cosine similarity score scaled to 0-10
            llm_score = c.get("score", 0) * 10

        scored.append({**c, "rerank_score": llm_score})

    scored.sort(key=lambda x: x["rerank_score"], reverse=True)
    return scored[:top_n]


def clean_text(text: str) -> str:
    text = text.encode("utf-8", errors="replace").decode("utf-8")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            lines.append("")
            continue
        if len(stripped.split()) < 2:
            continue
        lines.append(stripped)
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


def load_pdf(file_path: str) -> list[dict]:
    reader = PdfReader(file_path)
    pages = []
    for page_num, page in enumerate(reader.pages, start=1):
        text = clean_text(page.extract_text() or "")
        if text:
            pages.append(
                {"text": text, "page": page_num, "source": os.path.basename(file_path)}
            )
    return pages


def load_documents(folder: str = "docs") -> list[list[dict]]:
    docs = []
    for filename in sorted(os.listdir(folder)):
        path = os.path.join(folder, filename)
        if filename.endswith((".txt", ".md")):
            with open(path, "r", encoding="utf-8") as f:
                text = clean_text(f.read())
            if text:
                docs.append([{"text": text, "page": None, "source": filename}])
        elif filename.endswith(".pdf"):
            pages = load_pdf(path)
            if pages:
                docs.append(pages)
    return docs


def _estimate_tokens(text: str) -> int:
    """
    Estimate token count without a tokenizer.
    Scientific text averages ~4 chars/token (vs ~4.5 for plain English) due
    to equations, acronyms, and citations. We use 4.0 as a conservative
    estimate so we stay safely under the embedding model's 512-token limit.
    """
    return max(1, len(text) // 4)


def infer_chunk_params(text: str) -> tuple[int, int]:
    """
    Infer character-based chunk_size and overlap by reasoning about token
    density and content type, then clamping to the embedding model token limit.

    Content classification (in priority order):
      - Structured / tabular: short lines, low chars-per-word
      - Equation-heavy:       high chars-per-word (long tokens: LaTeX, symbols)
      - Column-layout PDF:    medium line length caused by column wrapping
      - Dense technical prose: long lines, moderate chars-per-word
      - Default fallback
    """
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        return CHUNK_SIZE, CHUNK_OVERLAP

    total_chars = sum(len(line) for line in lines)
    avg_line_len = total_chars / len(lines)

    words = text.split()
    word_count = max(len(words), 1)
    chars_per_word = total_chars / word_count

    # -- Content-type classification ------------------------------------------

    # Structured / tabular: very short lines, sparse words
    if avg_line_len < 40 and chars_per_word < 6:
        chunk_chars = 100  # ~200 tokens — keep table rows together
        overlap_chars = 10  # one row of context

    # Equation / citation heavy: long individual tokens inflate chars-per-word
    elif chars_per_word > 8:
        chunk_chars = 1000  # ~300 tokens — equations need surrounding context
        overlap_chars = 100  # larger overlap so equations are not split cold

    # Column-layout PDF: medium lines are an artefact of column wrapping,
    # not short content — treat as normal prose with a moderate window
    elif 45 <= avg_line_len <= 85:
        chunk_chars = 1000  # ~400 tokens
        overlap_chars = 100

    # Dense technical prose: long lines, information-rich sentences
    elif avg_line_len > 85:
        chunk_chars = 1000  # ~450 tokens — near the model limit
        overlap_chars = 100  # ~15% overlap preserves cross-sentence context

    else:
        chunk_chars = CHUNK_SIZE
        overlap_chars = CHUNK_OVERLAP

    # -- Hard clamp to embedding model token limit ----------------------------
    # Convert the token limit back to a character budget and enforce it.
    max_chars = EMBED_TOKEN_LIMIT * 4
    if chunk_chars > max_chars:
        ratio = overlap_chars / chunk_chars
        chunk_chars = max_chars
        overlap_chars = max(20, int(chunk_chars * ratio))

    # overlap must always be strictly less than chunk_size
    overlap_chars = min(overlap_chars, chunk_chars - 20)

    return chunk_chars, overlap_chars


def chunk_by_headings(text: str) -> list[str] | None:
    heading_pattern = re.compile(r"^#{1,3} .+", re.MULTILINE)
    positions = [m.start() for m in heading_pattern.finditer(text)]
    if len(positions) < 2:
        return None
    chunks = []
    for i, pos in enumerate(positions):
        end = positions[i + 1] if i + 1 < len(positions) else len(text)
        chunk = text[pos:end].strip()
        if chunk:
            chunks.append(chunk)
    return chunks


def chunk_text(
    text: str, chunk_size: int | None = None, overlap: int | None = None
) -> tuple[list[str], bool]:
    heading_chunks = chunk_by_headings(text)
    if heading_chunks is not None:
        return heading_chunks, True
    if chunk_size is None or overlap is None:
        chunk_size, overlap = infer_chunk_params(text)
    assert 0 < overlap < chunk_size
    chunks, start, text_len = [], 0, len(text)
    while start < text_len:
        end = min(start + chunk_size, text_len)
        if end < text_len:
            search_start = start + (chunk_size // 2)
            para_pos = text.rfind("\n\n", search_start, end)
            if para_pos != -1:
                end = para_pos + 2
            else:
                boundary = -1
                for punct in (".", "!", "?"):
                    pos = text.rfind(punct, search_start, end)
                    if pos > boundary:
                        boundary = pos
                if boundary != -1:
                    end = boundary + 1
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = max(end - overlap, start + 1)
    return chunks, False


# Patterns that strongly indicate a chunk is from a references section.
# Matched against the chunk text (case-insensitive).
_REFERENCES_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in [
        # Bare DOI lines
        r"^\s*doi:\s*10\.\d{4,}/",
        # ISSN / ISBN lines
        r"\bissn\b.*\d{4}-\d{3}[\dxX]",
        # Classic citation formats: "Author, A. B. (2024)." or "[12] Author"
        r"^\s*\[\d+\]\s+\w+,",
        # "In Proceedings of" / "In IGARSS" / "In NeurIPS" typical conference refs
        r"\bin\s+(?:proceedings of|igarss|neurips|cvpr|iccv|eccv|acl|emnlp|icml)\b",
        # Volume/pages bibliography line: "vol. 13, pp. 1119"
        r"\bvol\.\s*\d+.*\bpp?\.\s*\d+",
        # arXiv reference lines
        r"arxiv(?:\s+preprint)?\s+arxiv:\d{4}\.\d+",
    ]
]

# Heading / section title patterns that mark a references section
_REFERENCES_HEADINGS = re.compile(
    r"^(?:references|bibliography|works cited|citations)\s*$",
    re.IGNORECASE | re.MULTILINE,
)


def is_references_chunk(text: str) -> bool:
    """
    Return True if the chunk looks like it came from a bibliography or
    works-cited section rather than body content.

    Heuristics (any one is sufficient):
    - The chunk starts with a references heading.
    - The majority of lines match citation-style patterns.
    - The chunk is short AND contains a DOI / ISSN / conference-ref pattern.
    """
    stripped = text.strip()

    # 1. Starts with a references section heading
    if _REFERENCES_HEADINGS.search(stripped[:120]):
        return True

    lines = [line for line in stripped.splitlines() if line.strip()]
    if not lines:
        return False

    # 2. Short chunk that matches at least one hard citation pattern
    if len(stripped) < 400:
        for pat in _REFERENCES_PATTERNS:
            if pat.search(stripped):
                return True

    # 3. Most lines look like citation lines (DOI, ISSN, conference ref, etc.)
    hit_lines = sum(
        1 for line in lines if any(pat.search(line) for pat in _REFERENCES_PATTERNS)
    )
    if len(lines) > 0 and hit_lines / len(lines) >= 0.5:
        return True

    return False


def deduplicate(
    chunks: list[dict], max_per_source_page: int = MAX_PER_SOURCE_PAGE
) -> list[dict]:
    """Keep at most max_per_source_page chunks per (source, page) pair."""
    seen: dict[tuple, int] = {}
    result = []
    for c in chunks:
        key = (c["source"], c.get("dedup_key", c["page"]))
        if seen.get(key, 0) < max_per_source_page:
            result.append(c)
            seen[key] = seen.get(key, 0) + 1
    return result


def get_embedding(text: str) -> np.ndarray:
    # Hard truncate to stay within mxbai-embed-large's 512-token window.
    # 4 chars/token estimate; truncate before sending rather than letting
    # Ollama reject it with a 400.
    text = text.strip()
    if not text:
        text = "empty"
    if len(text) > EMBED_TOKEN_LIMIT * 4:
        text = text[: EMBED_TOKEN_LIMIT * 4]

    response = requests.post(
        f"{HOST}/api/embed",
        json={"model": EMBED_MODEL, "input": text},
        timeout=60,
    )

    if response.status_code == 400:
        # Return a zero vector — FAISS will score it at 0 after normalization
        # so it will never surface in search results
        dim = int(EMBED_DIM)  # output dimension
        return np.zeros(dim, dtype=np.float32)
    response.raise_for_status()
    return np.array(response.json()["embeddings"][0], dtype=np.float32)


def checksum_folder(folder: str) -> str:
    h = hashlib.md5()
    for filename in sorted(os.listdir(folder)):
        path = os.path.join(folder, filename)
        if os.path.isfile(path):
            h.update(filename.encode())
            h.update(str(os.path.getmtime(path)).encode())
            h.update(str(os.path.getsize(path)).encode())
    return h.hexdigest()


def read_stored_checksum(index_path: str) -> str | None:
    checksum_file = os.path.join(index_path, "docs_checksum.txt")
    if os.path.exists(checksum_file):
        with open(checksum_file) as f:
            return f.read().strip()
    return None


def write_checksum(index_path: str, checksum: str) -> None:
    with open(os.path.join(index_path, "docs_checksum.txt"), "w") as f:
        f.write(checksum)


# ── index management ──────────────────────────────────────────────────────────


def _chunk_id(source: str, page: int | None, chunk_i: int) -> str:
    """Stable, unique string ID for a chunk. Chroma requires string IDs."""
    return f"{source}::p{page}::c{chunk_i}"


def _collect_chunks(folder: str) -> list[dict]:
    """Load documents, chunk, filter — returns (chunks, embeddings)."""
    raw_docs = load_documents(folder)
    all_chunks, skipped = [], 0

    for doc_pages in raw_docs:
        for page_info in doc_pages:
            page_chunks, from_headings = chunk_text(page_info["text"])
            min_words = 2 if from_headings else 5
            for chunk_i, chunk in enumerate(page_chunks):
                if len(chunk.split()) < min_words:
                    skipped += 1
                    continue
                non_numeric = [
                    w for w in chunk.split() if not re.match(r"^[\d.%,\-]+$", w)
                ]
                if len(non_numeric) / max(len(chunk.split()), 1) < 0.3:
                    skipped += 1
                    continue
                real_page = page_info["page"]
                all_chunks.append(
                    {
                        "text": chunk,
                        "source": page_info["source"],
                        "page": real_page,
                        "dedup_key": real_page if real_page is not None else chunk_i,
                        "from_headings": from_headings,
                        "is_reference": is_references_chunk(chunk),
                    }
                )

    logger.info(f"Chunking complete — {len(all_chunks)} chunks ({skipped} skipped).")
    return all_chunks


def build_index(folder: str = "docs") -> chromadb.Collection:
    index_path = os.path.join(folder, "chroma_index")
    os.makedirs(index_path, exist_ok=True)

    collection = get_collection(index_path)
    all_chunks = _collect_chunks(folder)

    ids, embeddings, metadatas, documents = [], [], [], []

    for chunk_i, c in enumerate(tqdm(all_chunks, desc="Embedding", unit="chunk")):
        vec = get_embedding(c["text"])
        ids.append(_chunk_id(c["source"], c["page"], chunk_i))
        embeddings.append(vec.tolist())
        documents.append(c["text"])
        metadatas.append(
            {
                "source": c["source"],
                "page": c["page"] if c["page"] is not None else -1,
                "dedup_key": str(c["dedup_key"]),
                "from_headings": int(c["from_headings"]),
                "is_reference": int(c["is_reference"]),
            }
        )

    # Upsert in batches — Chroma has a default batch limit of ~41k items
    BATCH = 500
    for start in range(0, len(ids), BATCH):
        sl = slice(start, start + BATCH)
        collection.upsert(
            ids=ids[sl],
            embeddings=embeddings[sl],
            documents=documents[sl],
            metadatas=metadatas[sl],
        )

    write_checksum(index_path, checksum_folder(folder))
    logger.info(f"Chroma index built with {collection.count()} chunks.")
    return collection


def load_index(folder: str = "docs") -> chromadb.Collection:
    index_path = os.path.join(folder, "chroma_index")
    return get_collection(index_path)


def load_or_build_index(folder: str = "docs") -> chromadb.Collection:
    index_path = os.path.join(folder, "chroma_index")
    chroma_db_file = os.path.join(index_path, "chroma.sqlite3")

    if os.path.exists(chroma_db_file):
        current = checksum_folder(folder)
        stored = read_stored_checksum(index_path)
        if current != stored:
            logger.info("Docs folder changed — rebuilding index...")
            return build_index(folder)
        logger.info(f"Loading Chroma index from {index_path}")
        return load_index(folder)

    logger.info("No index found — building...")
    return build_index(folder)


# ── per-document add / remove ─────────────────────────────────────────────────


def add_document(source_path: str, collection: chromadb.Collection) -> int:
    """
    Chunk and embed a single file, upsert into the existing collection.
    Returns the number of chunks added.
    """
    filename = os.path.basename(source_path)
    if source_path.endswith(".pdf"):
        pages = load_pdf(source_path)
    else:
        with open(source_path, "r", encoding="utf-8") as f:
            text = clean_text(f.read())
        pages = [{"text": text, "page": None, "source": filename}]

    ids, embeddings, metadatas, documents = [], [], [], []
    chunk_i = 0

    for page_info in pages:
        page_chunks, from_headings = chunk_text(page_info["text"])
        min_words = 2 if from_headings else 5
        for chunk in page_chunks:
            if len(chunk.split()) < min_words:
                continue
            non_numeric = [w for w in chunk.split() if not re.match(r"^[\d.%,\-]+$", w)]
            if len(non_numeric) / max(len(chunk.split()), 1) < 0.3:
                continue
            vec = get_embedding(chunk)
            ids.append(_chunk_id(filename, page_info["page"], chunk_i))
            embeddings.append(vec.tolist())
            documents.append(chunk)
            metadatas.append(
                {
                    "source": filename,
                    "page": page_info["page"] if page_info["page"] is not None else -1,
                    "dedup_key": str(
                        page_info["page"] if page_info["page"] is not None else chunk_i
                    ),
                    "from_headings": int(from_headings),
                    "is_reference": int(is_references_chunk(chunk)),
                }
            )
            chunk_i += 1

    if ids:
        collection.upsert(
            ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas
        )

    logger.info(f"Added {len(ids)} chunks for '{filename}'.")
    return len(ids)


def remove_document(source: str, collection: chromadb.Collection) -> int:
    """
    Delete all chunks belonging to `source` (basename of the file).
    Returns the number of chunks removed.
    """
    results = collection.get(where={"source": source}, include=[])
    ids = results["ids"]
    if ids:
        collection.delete(ids=ids)
    logger.info(f"Removed {len(ids)} chunks for '{source}'.")
    return len(ids)


# ── search ────────────────────────────────────────────────────────────────────


def search(
    query: str,
    collection: chromadb.Collection,
    top_k: int = 10,
    min_score: float = MIN_SCORE,
) -> list[dict]:
    q_emb = get_embedding(query)
    # Chroma cosine distance: 0 = identical, 2 = opposite.
    # Convert to a similarity in [0, 1]: similarity = 1 - distance/2
    results = collection.query(
        query_embeddings=[q_emb.tolist()],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        score = 1.0 - dist / 2.0
        if score < min_score:
            continue
        chunks.append(
            {
                "text": doc,
                "source": meta["source"],
                "page": meta["page"] if meta["page"] != -1 else None,
                "dedup_key": meta["dedup_key"],
                "from_headings": bool(meta["from_headings"]),
                "is_reference": bool(meta["is_reference"]),
                "score": score,
            }
        )

    chunks.sort(key=lambda x: x["score"], reverse=True)
    return deduplicate(chunks)


class RAGTool(Tool):
    name = "rag_search"
    description = "Search internal documents for relevant context."

    def run(self, query: str, app, **kwargs) -> tuple[str, list]:
        top_k = kwargs.get("top_k", 10)
        rerank_top_n = kwargs.get("rerank_top_n", RERANK_TOP_N)
        collection = app.state.collection  # ← was chunks + index

        candidates = search(query, collection, top_k=top_k)

        body_candidates = [
            c
            for c in candidates
            if not c.get("is_reference", is_references_chunk(c["text"]))
        ]
        if not body_candidates:
            body_candidates = candidates

        reranked = rerank(query, body_candidates, top_n=rerank_top_n)
        results = deduplicate(reranked, max_per_source_page=1)

        context_parts = []
        for c in results:
            citation = (
                f"[{c['source']} - page {c['page']}]"
                if c["page"] is not None
                else f"[{c['source']}]"
            )
            context_parts.append(f"{c['text']} {citation}")

        return "\n\n".join(context_parts), results
