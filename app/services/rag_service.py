"""
RAG service: orchestrates retrieval → prompt building → LLM generation.
Ensures retrieval always happens before LLM invocation.
"""

from app.vectorstore import store
from app.prompts.templates import build_prompt, FALLBACK_RESPONSE
from app.services import llm_service
from app.utils.logger import get_logger, log_retrieval

logger = get_logger(__name__)


def run_rag(session_id: str, message: str, history: list[dict]) -> dict:
    """
    Full RAG pipeline:
    1. Embed query and retrieve top-K chunks (cosine similarity)
    2. If no chunks pass threshold → return fallback
    3. Build context string from retrieved chunks
    4. Build prompt (context + history + question)
    5. Call LLM
    6. Return structured result

    Args:
        session_id: Unique identifier for the session
        message:    The user's current question
        history:    List of {user, assistant} dicts (last N turns)

    Returns:
        dict with keys: reply, tokensUsed, retrievedChunks, chunks
    """
    logger.info(f"RAG | session={session_id} | query='{message[:60]}'")

    # ── Step 1: Retrieve ──────────────────────────────────────────────────────
    results = store.search(message)
    log_retrieval(logger, message, results)

    # ── Step 2: Fallback if nothing retrieved ─────────────────────────────────
    if not results:
        logger.info(f"RAG | NO RESULTS ABOVE THRESHOLD | session={session_id}")
        return {
            "reply": FALLBACK_RESPONSE,
            "tokensUsed": 0,
            "retrievedChunks": 0,
            "chunks": [],
        }

    # ── Step 3: Build context ─────────────────────────────────────────────────
    context_parts = []
    for r in results:
        context_parts.append(
            f"[Source: {r['title']}]\n{r['text']}"
        )
    retrieved_context = "\n\n".join(context_parts)

    # ── Step 4: Build prompt ──────────────────────────────────────────────────
    prompt = build_prompt(
        retrieved_context=retrieved_context,
        history=history,
        user_question=message,
    )

    # ── Step 5: Call LLM ──────────────────────────────────────────────────────
    reply, tokens_used = llm_service.generate(prompt, session_id=session_id)

    # ── Step 6: Return result ─────────────────────────────────────────────────
    chunk_metadata = [
        {
            "title": r["title"],
            "chunk_id": r["chunk_id"],
            "score": round(r["score"], 4),
        }
        for r in results
    ]

    logger.info(
        f"RAG | DONE | session={session_id} | chunks={len(results)} | tokens={tokens_used}"
    )

    return {
        "reply": reply,
        "tokensUsed": tokens_used,
        "retrievedChunks": len(results),
        "chunks": chunk_metadata,
    }
