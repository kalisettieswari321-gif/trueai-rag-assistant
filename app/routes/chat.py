"""
Chat routes: POST /api/chat
Handles session management, request validation, and RAG invocation.
"""

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from app.models.schemas import ChatRequest, ChatResponse, ErrorResponse
from app.services import rag_service
from app.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/api", tags=["chat"])

# In-memory session store: {session_id: [{user, assistant}, ...]}
# Keeps last 5 message pairs per session
_sessions: dict[str, list[dict]] = {}
MAX_HISTORY = 5


@router.post(
    "/chat",
    response_model=ChatResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Validation error"},
        401: {"model": ErrorResponse, "description": "Invalid API key"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
        500: {"model": ErrorResponse, "description": "Server error"},
    },
    summary="Send a chat message and get a RAG-grounded response",
)
def chat(request: ChatRequest):
    """
    POST /api/chat

    Accepts a sessionId and user message.
    Retrieves relevant knowledge base chunks via cosine similarity,
    builds a grounded prompt, and returns the LLM response.
    """
    session_id = request.sessionId.strip()
    message = request.message.strip()

    if not session_id:
        raise HTTPException(status_code=400, detail="Session ID field is required")
    if not message:
        raise HTTPException(status_code=400, detail="Message field is required")

    logger.info(f"CHAT | session={session_id} | message='{message[:60]}'")

    # Get or initialise session history
    history = _sessions.get(session_id, [])

    # Run RAG pipeline
    result = rag_service.run_rag(
        session_id=session_id,
        message=message,
        history=history,
    )

    # Update session history
    history.append({
        "user": message,
        "assistant": result["reply"],
    })
    _sessions[session_id] = history[-MAX_HISTORY:]

    return ChatResponse(
        reply=result["reply"],
        tokensUsed=result["tokensUsed"],
        retrievedChunks=result["retrievedChunks"],
    )


@router.delete(
    "/session/{session_id}",
    summary="Clear conversation history for a session",
)
async def clear_session(session_id: str):
    """DELETE /api/session/{session_id} — clears stored history."""
    if session_id in _sessions:
        del _sessions[session_id]
    return {"status": "cleared", "sessionId": session_id}
