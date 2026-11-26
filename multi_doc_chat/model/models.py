from enum import Enum
from pydantic import BaseModel, Field


class PromptType(str, Enum):
    CONTEXTUALIZE_QUESTION = "contextualize_question"
    CONTEXT_QA = "context_qa"


class ChatAnswer(BaseModel):
    answer: str = Field(description="Answer returned by the conversational RAG pipeline")
