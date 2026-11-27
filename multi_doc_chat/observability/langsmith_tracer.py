import os
import time
from typing import Optional, Dict, Any
from dotenv import load_dotenv
from langsmith import Client, traceable
from multi_doc_chat.logger import GLOBAL_LOGGER as log

load_dotenv()


def get_langsmith_client() -> Optional[Client]:
    """Retrieve an authenticated LangSmith client if configured."""
    api_key = os.getenv("LANGCHAIN_API_KEY")
    if not api_key:
        return None
    try:
        return Client(api_key=api_key)
    except Exception as e:
        log.warning("Could not initialize LangSmith Client", error=str(e))
        return None


def get_langsmith_project() -> str:
    """Return the active LangSmith project name."""
    return os.getenv("LANGCHAIN_PROJECT", "end-to-end-advanced-rag-with-llmops")


def is_tracing_enabled() -> bool:
    """Check whether LangSmith tracing is active."""
    return os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true" and bool(os.getenv("LANGCHAIN_API_KEY"))


def build_run_config(
    session_id: str,
    run_name: str = "ConversationalRAG_Pipeline",
    extra_tags: Optional[list[str]] = None,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Construct RunnableConfig with standardized LLMOps tags and metadata for LangSmith.
    """
    tags = ["rag", "in-memory-vector-store", "fastembed", "groq", "llmops"]
    if extra_tags:
        tags.extend(extra_tags)

    metadata = {
        "session_id": session_id,
        "environment": os.getenv("ENV", "local"),
        "timestamp": time.time(),
        "vector_store": "InMemoryVectorStore",
        "embedding_provider": "fastembed",
        "llm_provider": "groq",
    }
    if extra_metadata:
        metadata.update(extra_metadata)

    return {
        "run_name": run_name,
        "tags": list(set(tags)),
        "metadata": metadata,
    }


@traceable(
    name="SemanticCache_Lookup",
    run_type="retriever",
    tags=["semantic-cache", "cache-hit", "in-memory"],
)
def record_cache_hit_trace(
    query: str,
    matched_query: str,
    similarity_score: float,
    answer: str,
    session_id: str,
) -> Dict[str, Any]:
    """
    Trace semantic cache hits as first-class retriever spans in LangSmith.
    Provides complete visibility into traffic without calling the LLM.
    """
    return {
        "query": query,
        "matched_query": matched_query,
        "similarity_score": round(similarity_score, 4),
        "answer": answer,
        "session_id": session_id,
        "cache_hit": True,
        "cost_usd": 0.0,
    }


def record_feedback(
    run_id: str,
    score: float,
    key: str = "user_satisfaction",
    comment: Optional[str] = None,
) -> Optional[str]:
    """
    Record user feedback or automated evaluation score against a LangSmith run.
    """
    client = get_langsmith_client()
    if not client:
        log.warning("Cannot record feedback; LangSmith client not configured.")
        return None

    try:
        feedback = client.create_feedback(
            run_id=run_id,
            key=key,
            score=score,
            comment=comment,
        )
        log.info("Recorded feedback in LangSmith", run_id=run_id, key=key, score=score)
        return str(feedback.id)
    except Exception as e:
        log.error("Failed to record feedback in LangSmith", error=str(e), run_id=run_id)
        return None
