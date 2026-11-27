import pytest
import numpy as np

from multi_doc_chat.src.document_chat.retrieval import ConversationalRAG
from multi_doc_chat.utils.semantic_cache import SemanticCache
from multi_doc_chat.src.document_ingestion.data_ingestion import (
    ChatIngestor,
)
from multi_doc_chat.exception.custom_exception import DocumentPortalException


# ======================================================================
# 1. SemanticCache Unit Tests
# ======================================================================

def test_semantic_cache_store_and_hit():
    """Verify storing embeddings and achieving a cache hit on identical or near-identical vectors."""
    cache = SemanticCache(similarity_threshold=0.90)
    assert cache.size() == 0

    vec_a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    cache.store("What is Agentic AI?", vec_a, "Agentic AI refers to autonomous systems.")
    assert cache.size() == 1

    # Exact vector lookup
    result = cache.lookup(vec_a)
    assert result is not None
    answer, score, matched_q = result
    assert answer == "Agentic AI refers to autonomous systems."
    assert score >= 0.999
    assert matched_q == "What is Agentic AI?"


def test_semantic_cache_miss_on_dissimilar_vectors():
    """Verify cache miss when vector cosine similarity is below threshold."""
    cache = SemanticCache(similarity_threshold=0.90)

    vec_a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    cache.store("What is Agentic AI?", vec_a, "Autonomous AI systems.")

    # Orthogonal / dissimilar vector
    vec_b = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    result = cache.lookup(vec_b)
    assert result is None


def test_semantic_cache_max_size_fifo_eviction():
    """Verify FIFO eviction when cache capacity exceeds max_size."""
    cache = SemanticCache(similarity_threshold=0.90, max_size=2)

    cache.store("q1", [1.0, 0.0], "a1")
    cache.store("q2", [0.0, 1.0], "a2")
    assert cache.size() == 2

    # Insert 3rd item -> oldest (q1) should be evicted
    cache.store("q3", [0.5, 0.5], "a3")
    assert cache.size() == 2
    assert cache.cache[0]["query"] == "q2"
    assert cache.cache[1]["query"] == "q3"

    cache.clear()
    assert cache.size() == 0


# ======================================================================
# 2. ConversationalRAG Unit Tests
# ======================================================================

def test_conversational_rag_error_handling_uninitialized():
    """Verify invoking without a loaded retriever raises DocumentPortalException."""
    rag = ConversationalRAG(session_id="uninit_session", enable_cache=True)
    with pytest.raises(DocumentPortalException):
        rag.invoke("Hello, can you answer this?")

    with pytest.raises(DocumentPortalException):
        rag.load_retriever_from_memory(session_id="non_existent_session")


def test_conversational_rag_load_retriever_and_invoke(sample_text_file, tmp_path):
    """Test full RAG pipeline: ingest sample doc, load in-memory retriever, invoke query, and test cache hit."""
    session_id = "test_rag_pipeline_session"
    ingestor = ChatIngestor(temp_base=str(tmp_path), session_id=session_id)
    retriever = ingestor.build_retriever([sample_text_file], chunk_size=300, chunk_overlap=50)

    rag = ConversationalRAG(session_id=session_id, retriever=retriever, enable_cache=True, cache_threshold=0.88)
    assert rag.chain is not None

    # First invoke (LLM generation + cache insertion)
    query = "What are the core pillars of Agentic AI?"
    answer1 = rag.invoke(query)
    assert answer1 is not None and len(answer1) > 0
    assert rag.cache.size() == 1

    # Second invoke with identical query -> instant semantic cache hit
    answer2 = rag.invoke(query)
    assert answer2 == answer1


@pytest.mark.asyncio
async def test_conversational_rag_ainvoke_and_astream(sample_text_file, tmp_path):
    """Verify asynchronous ainvoke and astream with semantic cache integration."""
    session_id = "test_async_rag_session"
    ingestor = ChatIngestor(temp_base=str(tmp_path), session_id=session_id)
    retriever = ingestor.build_retriever([sample_text_file], chunk_size=300, chunk_overlap=50)

    rag = ConversationalRAG(session_id=session_id, retriever=retriever, enable_cache=True, cache_threshold=0.88)

    # Test async invoke
    query = "What does ReAct stand for?"
    answer = await rag.ainvoke(query)
    assert answer is not None
    assert "reason" in answer.lower() or "react" in answer.lower()

    # Test async streaming
    chunks = []
    async for chunk in rag.astream("Summarize the sample document."):
        chunks.append(chunk)

    assert len(chunks) > 0
    full_text = "".join(chunks)
    assert len(full_text) > 0


def test_conversational_rag_backward_compatible_faiss_loader(sample_text_file, tmp_path):
    """Verify load_retriever_from_faiss correctly redirects to in-memory store."""
    session_id = "test_compat_session"
    ingestor = ChatIngestor(temp_base=str(tmp_path), session_id=session_id)
    ingestor.build_retriever([sample_text_file], chunk_size=300, chunk_overlap=50)

    rag = ConversationalRAG(session_id=session_id, enable_cache=False)
    retriever = rag.load_retriever_from_faiss(index_path=f"faiss_index/{session_id}")
    assert retriever is not None
    assert rag.chain is not None
