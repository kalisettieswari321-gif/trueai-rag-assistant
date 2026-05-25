from pydantic import BaseModel, Field
from typing import Optional, List


class ChatRequest(BaseModel):
    sessionId: str = Field(..., min_length=1, description="Unique session identifier")
    message: str = Field(..., min_length=1, max_length=2000, description="User message")

    class Config:
        json_schema_extra = {
            "example": {
                "sessionId": "abc123",
                "message": "How can I reset my password?"
            }
        }


class ChatResponse(BaseModel):
    reply: str
    tokensUsed: int
    retrievedChunks: int


class ErrorResponse(BaseModel):
    error: str


class HealthResponse(BaseModel):
    status: str
