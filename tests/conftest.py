import os
import io
import pathlib
import sys
import pytest
from fastapi.testclient import TestClient

# Ensure root directory is on sys.path
ROOT_DIR = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from dotenv import load_dotenv
load_dotenv()

# Load test environment variables
os.environ.setdefault("ENV", "local")
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGSMITH_TRACING"] = "false"
if not os.environ.get("GROQ_API_KEY"):
    os.environ["GROQ_API_KEY"] = "gsk_dummy_test_key_for_unit_tests"

import main
from langchain_core.embeddings import Embeddings
from multi_doc_chat.src.document_ingestion.data_ingestion import (
    SESSION_VECTOR_STORES,
    ChatIngestor,
)
from multi_doc_chat.src.document_ingestion.vector_storage import VECTOR_STORE_MANAGER
from multi_doc_chat.observability import ALERT_MANAGER
from multi_doc_chat.src.document_chat.retrieval import (
    ConversationalRAG,
    SemanticCache,
)


def _clear_redis():
    try:
        import redis
        from multi_doc_chat.utils.config_loader import load_config
        cfg_url = load_config().get("distributed_cache", {}).get("redis_url")
        urls = [url for url in [os.getenv("REDIS_URL"), cfg_url, "redis://127.0.0.1:6380/0", "redis://127.0.0.1:6379/0"] if url]
        for url in set(urls):
            try:
                r = redis.from_url(url, socket_timeout=0.5)
                keys = r.keys("rag:*")
                if keys:
                    r.delete(*keys)
            except Exception:
                pass
    except Exception:
        pass


@pytest.fixture(autouse=True)
def reset_session_state():
    """Clear in-memory vector stores, RAG instances, Qdrant collections, and test disk indexes before and after each test."""
    SESSION_VECTOR_STORES.clear()
    main.RAG_INSTANCES.clear()
    VECTOR_STORE_MANAGER.reset_all()
    ALERT_MANAGER.clear_alerts()
    _clear_redis()
    yield
    SESSION_VECTOR_STORES.clear()
    main.RAG_INSTANCES.clear()
    VECTOR_STORE_MANAGER.reset_all()
    ALERT_MANAGER.clear_alerts()
    _clear_redis()
    # Clean up test index files
    index_dir = pathlib.Path("data/indexes")
    if index_dir.exists():
        for f in index_dir.glob("test_*.json"):
            try:
                f.unlink()
            except Exception:
                pass


@pytest.fixture
def client():
    """FastAPI TestClient fixture."""
    return TestClient(main.app)


@pytest.fixture
def sample_text_file(tmp_path: pathlib.Path) -> pathlib.Path:
    """Creates a temporary sample text file for testing ingestion and chat."""
    file_path = tmp_path / "sample_test_doc.txt"
    file_path.write_text(
        "Agentic AI systems are autonomous systems that operate with minimal human intervention. "
        "The core pillars are perception, planning, memory, and action execution. "
        "ReAct stands for Reasoning and Acting, which interleaves reasoning traces with action execution."
    )
    return file_path


@pytest.fixture
def sample_bytes_file():
    """Returns an in-memory BytesIO text file tuple for multipart upload."""
    content = (
        b"Machine Learning Operations (LLMOps) involves managing the lifecycle of large language models, "
        b"including data ingestion, semantic caching, vector retrieval, and continuous evaluation."
    )
    return ("llmops_overview.txt", io.BytesIO(content), "text/plain")


class DummyEmbeddings(Embeddings):
    """Deterministic, zero-latency embeddings for fast unit tests without downloading weights."""
    def embed_query(self, text: str) -> list[float]:
        # Simple hash-based deterministic 16-dim normalized vector
        val = sum(ord(c) for c in text) % 100
        vec = [val / 100.0] * 16
        norm = sum(x ** 2 for x in vec) ** 0.5 or 1.0
        return [x / norm for x in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(t) for t in texts]


class DummyLLM:
    """Mock LLM for isolated unit testing."""
    def __init__(self, fixed_answer: str = "This is a deterministic synthesized response."):
        self.fixed_answer = fixed_answer

    def invoke(self, *args, **kwargs):
        from langchain_core.messages import AIMessage
        return AIMessage(content=self.fixed_answer)

    async def ainvoke(self, *args, **kwargs):
        from langchain_core.messages import AIMessage
        return AIMessage(content=self.fixed_answer)

    async def astream(self, *args, **kwargs):
        from langchain_core.messages import AIMessageChunk
        words = self.fixed_answer.split(" ")
        for w in words:
            yield AIMessageChunk(content=w + " ")


@pytest.fixture
def dummy_embeddings():
    return DummyEmbeddings()


@pytest.fixture
def dummy_llm():
    return DummyLLM()
