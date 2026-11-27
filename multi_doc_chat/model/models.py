from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field


class PromptType(str, Enum):
    CONTEXTUALIZE_QUESTION = "contextualize_question"
    CONTEXT_QA = "context_qa"


class ChatAnswer(BaseModel):
    answer: str = Field(description="Answer returned by the conversational RAG pipeline")


class ChatMessage(BaseModel):
    role: str = Field(description="'human' or 'ai' / 'assistant'")
    content: str = Field(description="Content of the message")


class ChatRequest(BaseModel):
    session_id: str = Field(default="default", description="Session identifier for multi-user isolation")
    query: str = Field(..., description="User question")
    chat_history: List[ChatMessage] = Field(default=[], description="Previous conversation turns")


class ChatResponse(BaseModel):
    session_id: str
    query: str
    answer: str
    cache_hit: bool = False
    latency_ms: float


class IngestResponse(BaseModel):
    session_id: str
    files: List[str]
    chunks_count: int
    message: str


class HealthResponse(BaseModel):
    status: str
    active_sessions: int
    model_name: str
    embedding_provider: str
    caching_enabled: bool
