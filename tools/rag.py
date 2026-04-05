import os
import re
import json
import faiss
import hashlib
import requests
import numpy as np
from tqdm import tqdm
from pypdf import PdfReader

from .base import Tool

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1000))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 100))
MIN_SCORE = float(os.getenv("RAG_MIN_SCORE", "0.1"))
MAX_PER_SOURCE_PAGE = int(os.getenv("RAG_MAX_PER_SOURCE_PAGE", "3"))


def clean_text(text: str) -> str:
    """Clean text and remove unsafe Unicode characters."""
    # Replace invalid Unicode characters
    text = text.encode("utf-8", errors="replace").decode("utf-8")

    # Normalize line breaks
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
    """ """
    reader = PdfReader(file_path)
    pages = []
    for page_num, page in enumerate(reader.pages, start=1):
        text = clean_text(page.extract_text() or "")
        if text:
            pages.append(
                {
                    "text": text,
                    "page": page_num,
                    "source": os.path.basename(file_path),
                }
            )
    return pages


def load_documents(folder: str = "docs") -> list[list[dict]]:
    """ """
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


def infer_chunk_params(text: str) -> tuple[int, int]:
    """ """
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        return CHUNK_SIZE, CHUNK_OVERLAP
    avg_line_len = sum(len(line) for line in lines) / len(lines)
    if avg_line_len < 60:
        return 350, 40
    elif avg_line_len < 120:
        return 650, 80
    else:
        return CHUNK_SIZE, CHUNK_OVERLAP


def chunk_by_headings(text: str) -> list[str] | None:
    """
    Split markdown at heading boundaries so each section becomes its own chunk.
    Each chunk includes its heading so the embedding captures both the section
    title and its content. Returns None if fewer than 2 headings are found,
    falling back to character-based chunking.
    """
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
    """
    Returns (chunks, from_headings).
    Tries heading-based splitting first; falls back to adaptive character-based.
    from_headings=True relaxes the minimum word filter in build_faiss_index
    since list items under headings are intentionally short.
    """
    heading_chunks = chunk_by_headings(text)
    if heading_chunks is not None:
        return heading_chunks, True

    if chunk_size is None or overlap is None:
        chunk_size, overlap = infer_chunk_params(text)

    assert 0 < overlap < chunk_size, "overlap must be > 0 and < chunk_size"

    chunks = []
    start = 0
    text_len = len(text)

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


def get_embedding(text: str) -> np.ndarray:
    """ """
    response = requests.post(
        f"{OLLAMA_HOST}/api/embeddings",
        json={"model": EMBED_MODEL, "prompt": text},
        timeout=60,
    )
    response.raise_for_status()
    return np.array(response.json()["embedding"], dtype=np.float32)


def checksum_folder(folder: str) -> str:
    """ """
    h = hashlib.md5()
    for filename in sorted(os.listdir(folder)):
        path = os.path.join(folder, filename)
        if os.path.isfile(path):
            h.update(filename.encode())
            h.update(str(os.path.getmtime(path)).encode())
            h.update(str(os.path.getsize(path)).encode())
    return h.hexdigest()


def read_stored_checksum(index_path: str) -> str | None:
    """ """
    checksum_file = os.path.join(index_path, "docs_checksum.txt")
    if os.path.exists(checksum_file):
        with open(checksum_file) as f:
            return f.read().strip()
    return None


def write_checksum(index_path: str, checksum: str) -> None:
    """ """
    with open(os.path.join(index_path, "docs_checksum.txt"), "w") as f:
        f.write(checksum)


def build_faiss_index(
    folder: str = "docs",
    index_path: str = "faiss_index",
) -> tuple[list[dict], faiss.Index, np.ndarray]:
    """ """
    os.makedirs(index_path, exist_ok=True)
    raw_docs = load_documents(folder)

    # Pass 1: chunk and filter
    all_chunks: list[dict] = []
    skipped = 0

    for doc_pages in raw_docs:
        for page_info in doc_pages:
            page_chunks, from_headings = chunk_text(page_info["text"])

            # Heading-chunked docs (markdown) get a relaxed word minimum since
            # list items like "- Python" or a short patent line are valid content.
            # Character-chunked docs (PDFs, plain text) use a higher threshold
            # to filter out extraction remnants.
            min_words = 2 if from_headings else 5

            for chunk_i, chunk in enumerate(page_chunks):
                if len(chunk.split()) < min_words:
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
                    }
                )

    print(
        f"Chunking complete — {len(all_chunks)} chunks to embed ({skipped} skipped).",
        flush=True,
    )

    # Pass 2: embed
    embeddings_list: list[np.ndarray] = []

    for chunk_dict in tqdm(all_chunks, desc="Embedding chunks", unit="chunk"):
        page_label = (
            f"p{chunk_dict['page']}" if chunk_dict["page"] is not None else "flat"
        )
        # tqdm.set_description(f"{chunk_dict['source']} {page_label}")
        # tqdm.write(f"{chunk_dict['source']} {page_label}")  # prints above the progress bar
        embeddings_list.append(get_embedding(chunk_dict["text"]))

    print("Embedding complete — building FAISS index...", flush=True)

    embeddings = np.array(embeddings_list, dtype=np.float32)
    faiss.normalize_L2(embeddings)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    faiss.write_index(index, os.path.join(index_path, "index.faiss"))
    np.save(os.path.join(index_path, "embeddings.npy"), embeddings)
    with open(os.path.join(index_path, "chunks.json"), "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=4, ensure_ascii=False)
    write_checksum(index_path, checksum_folder(folder))

    print(f"FAISS index built with {len(all_chunks)} chunks.", flush=True)
    return all_chunks, index, embeddings


def load_faiss_index(
    index_path: str = "faiss_index",
) -> tuple[list[dict], faiss.Index, np.ndarray]:
    """ """
    index = faiss.read_index(os.path.join(index_path, "index.faiss"))
    embeddings = np.load(os.path.join(index_path, "embeddings.npy"))
    with open(os.path.join(index_path, "chunks.json"), "r", encoding="utf-8") as f:
        chunks = json.load(f)
    return chunks, index, embeddings


def load_or_build_index(
    folder: str = "docs",
    index_path: str = "faiss_index",
) -> tuple[list[dict], faiss.Index, np.ndarray]:
    """ """
    index_file = os.path.join(index_path, "index.faiss")

    if os.path.exists(index_file):
        current = checksum_folder(folder)
        stored = read_stored_checksum(index_path)
        if current != stored:
            print("Docs folder has changed — rebuilding FAISS index...", flush=True)
            return build_faiss_index(folder, index_path)
        print("Loading existing FAISS index...", flush=True)
        return load_faiss_index(index_path)

    print("No FAISS index found — building from documents...", flush=True)
    return build_faiss_index(folder, index_path)


def deduplicate(
    chunks: list[dict], max_per_source_page: int = MAX_PER_SOURCE_PAGE
) -> list[dict]:
    """ """
    seen: dict[tuple, int] = {}
    result = []
    for c in chunks:
        key = (c["source"], c.get("dedup_key", c["page"]))
        if seen.get(key, 0) < max_per_source_page:
            result.append(c)
            seen[key] = seen.get(key, 0) + 1
    return result


def search(
    query: str,
    chunks: list[dict],
    index: faiss.Index,
    top_k: int = 10,
    min_score: float = MIN_SCORE,
) -> list[dict]:
    """ """
    q_emb = get_embedding(query).reshape(1, -1)
    faiss.normalize_L2(q_emb)
    distances, indices = index.search(q_emb, top_k)

    results = [
        {**chunks[i], "score": float(distances[0][j])}
        for j, i in enumerate(indices[0])
        if distances[0][j] >= min_score
    ]

    results.sort(key=lambda x: x["score"], reverse=True)
    return deduplicate(results)


class RAGTool(Tool):
    name = "rag_search"
    description = "Search internal documents for relevant context."

    def run(self, query: str, app, **kwargs) -> tuple[str, list]:
        top_k = kwargs.get("top_k", 10)
        chunks = app.state.chunks
        index = app.state.index

        results = search(query, chunks, index, top_k=top_k)

        context_parts = []
        for c in results:
            if c["page"] is not None:
                citation = f"[{c['source']} - page {c['page']}]"
            else:
                citation = f"[{c['source']}]"

            context_parts.append(f"{c['text']} {citation}")

        context = "\n\n".join(context_parts)
        return context, results
