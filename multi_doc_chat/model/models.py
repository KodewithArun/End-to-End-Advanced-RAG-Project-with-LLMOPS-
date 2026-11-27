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
    run_id: Optional[str] = None


class FeedbackRequest(BaseModel):
    run_id: str = Field(..., description="LangSmith Run ID")
    score: float = Field(..., ge=0.0, le=1.0, description="Score from 0.0 (negative) to 1.0 (positive)")
    key: str = Field(default="user_satisfaction", description="Feedback metric key")
    comment: Optional[str] = Field(default=None, description="Optional user comment or reason")


class FeedbackResponse(BaseModel):
    success: bool
    feedback_id: Optional[str] = None
    message: str


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
    langsmith_tracing: bool = False
    langsmith_project: Optional[str] = None
    persistence_enabled: bool = True


class AlertsResponse(BaseModel):
    statistics: dict
    recent_alerts: list
