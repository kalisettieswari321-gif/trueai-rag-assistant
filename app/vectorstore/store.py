"""
Vector store using FAISS with cosine similarity.
- Loads docs.json on startup
- Chunks documents at ~400 tokens with 50-token overlap
- Generates embeddings via SentenceTransformer
- Normalises vectors for cosine similarity via inner product
- Stores chunk metadata: title, chunk_id, source, text
"""

import json
import os
import re
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from app.utils.logger import get_logger

logger = get_logger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────
CHUNK_SIZE = 400        # approximate tokens per chunk
CHUNK_OVERLAP = 50      # token overlap between consecutive chunks
TOP_K = 3               # number of results to retrieve
SIMILARITY_THRESHOLD = 0.35  # cosine similarity minimum (0-1 scale)
DOCS_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "docs.json")

# ── Model ─────────────────────────────────────────────────────────────────────
_model = SentenceTransformer("all-MiniLM-L6-v2")

# ── State ─────────────────────────────────────────────────────────────────────
_index: faiss.IndexFlatIP | None = None
_chunks: list[dict] = []   # [{text, title, chunk_id, source}]


# ── Chunking ──────────────────────────────────────────────────────────────────

def _tokenise_approx(text: str) -> list[str]:
    """Split text into approximate word-level tokens."""
    return re.findall(r"\S+", text)


def _chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Split text into overlapping chunks of approximately chunk_size tokens.
    Uses a sliding window with the specified overlap.
    """
    tokens = _tokenise_approx(text)
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk_tokens = tokens[start:end]
        chunks.append(" ".join(chunk_tokens))
        if end == len(tokens):
            break
        start += chunk_size - overlap
    return chunks


# ── Indexing ──────────────────────────────────────────────────────────────────

def build_index() -> None:
    """
    Load docs.json, chunk every document, embed chunks, normalise vectors,
    and build a FAISS IndexFlatIP (cosine similarity on unit vectors).
    """
    global _index, _chunks

    docs_path = os.path.abspath(DOCS_PATH)
    logger.info(f"Loading documents from {docs_path}")

    with open(docs_path, "r", encoding="utf-8") as f:
        documents = json.load(f)

    _chunks = []
    for doc in documents:
        title = doc.get("title", "Untitled")
        content = doc.get("content", "")
        text_chunks = _chunk_text(content)
        for i, chunk_text in enumerate(text_chunks):
            _chunks.append({
                "text": chunk_text,
                "title": title,
                "chunk_id": f"{title.lower().replace(' ', '_')}_chunk_{i}",
                "source": title,
            })

    logger.info(f"Total chunks indexed: {len(_chunks)}")

    # Generate embeddings
    texts = [c["text"] for c in _chunks]
    embeddings = _model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
    vectors = np.array(embeddings, dtype="float32")

    # Build FAISS index with inner product (= cosine on unit vectors)
    dimension = vectors.shape[1]
    _index = faiss.IndexFlatIP(dimension)
    _index.add(vectors)

    logger.info(f"FAISS index built | dimension={dimension} | vectors={_index.ntotal}")


# ── Search ─────────────────────────────────────────────────────────────────────

def search(query: str, top_k: int = TOP_K, threshold: float = SIMILARITY_THRESHOLD) -> list[dict]:
    """
    Embed the query, run cosine similarity search, apply threshold filter.

    Returns:
        List of dicts: {text, title, chunk_id, source, score}
        Returns empty list if no chunk exceeds the threshold.
    """
    if _index is None:
        raise RuntimeError("Vector index not built. Call build_index() first.")

    # Embed & normalise query
    query_vec = _model.encode([query], normalize_embeddings=True)
    query_vec = np.array(query_vec, dtype="float32")

    # Search — scores are cosine similarities in [−1, 1]
    scores, indices = _index.search(query_vec, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        cosine_sim = float(score)
        logger.info(
            f"  candidate chunk_id={_chunks[idx]['chunk_id']} | "
            f"cosine_sim={cosine_sim:.4f} | threshold={threshold}"
        )
        if cosine_sim >= threshold:
            results.append({
                "text": _chunks[idx]["text"],
                "title": _chunks[idx]["title"],
                "chunk_id": _chunks[idx]["chunk_id"],
                "source": _chunks[idx]["source"],
                "score": cosine_sim,
            })

    logger.info(
        f"SEARCH | query='{query[:50]}' | candidates={top_k} | "
        f"above_threshold={len(results)}"
    )
    return results
