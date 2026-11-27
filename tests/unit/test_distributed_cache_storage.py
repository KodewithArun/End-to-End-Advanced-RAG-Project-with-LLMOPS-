import pytest
import numpy as np
from langchain_core.documents import Document

from multi_doc_chat.utils.semantic_cache import TieredSemanticCache
from multi_doc_chat.src.document_ingestion.vector_storage import VectorStoreManager
from multi_doc_chat.observability.alerting import AlertManager, AlertType, AlertSeverity


def test_tiered_cache_l1_exact_match():
    """Verify that exact-string queries trigger fast L1 cache hit."""
    cache = TieredSemanticCache(similarity_threshold=0.88)
    session_id = "test_l1_sess"

    cache.store(
        query="What is distributed semantic caching?",
        query_embedding=[1.0, 0.0, 0.0, 0.0],
        answer="Distributed caching shares state across workers.",
        session_id=session_id,
    )

    # Exact query lookup with string provided
    hit = cache.lookup_detailed(
        query_embedding=[1.0, 0.0, 0.0, 0.0],
        query="What is distributed semantic caching?",
        session_id=session_id,
    )
    assert hit is not None
    ans, score, matched_q, tier = hit
    assert ans == "Distributed caching shares state across workers."
    assert score == 1.0
    assert tier == "L1"


def test_tiered_cache_l2_semantic_match():
    """Verify that rephrased queries trigger L2 semantic vector similarity hit."""
    cache = TieredSemanticCache(similarity_threshold=0.85)
    session_id = "test_l2_sess"

    vec_base = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    cache.store(
        query="How does RAG work?",
        query_embedding=vec_base,
        answer="RAG retrieves context from documents and synthesizes answers.",
        session_id=session_id,
    )

    # Semantically close vector (cosine similarity ~0.95) but different query string
    vec_rephrased = np.array([0.95, 0.05, 0.0, 0.0], dtype=np.float32)
    hit = cache.lookup_detailed(
        query_embedding=vec_rephrased,
        query="Explain retrieval augmented generation mechanism",
        session_id=session_id,
    )
    assert hit is not None
    ans, score, matched_q, tier = hit
    assert ans == "RAG retrieves context from documents and synthesizes answers."
    assert score >= 0.85
    assert tier == "L2"


def test_tiered_cache_corpus_invalidation():
    """Verify that re-ingesting documents invalidates prior cached answers for that session."""
    cache = TieredSemanticCache(similarity_threshold=0.85)
    session_id = "test_invalidation_sess"

    vec = [1.0, 0.0, 0.0, 0.0]
    cache.store("Company vacation policy", vec, "15 days PTO per year.", session_id=session_id)

    # Confirm cache hit initially
    hit1 = cache.lookup_detailed(vec, query="Company vacation policy", session_id=session_id)
    assert hit1 is not None

    # Invalidate session (simulates document re-ingestion)
    cache.invalidate_session(session_id)

    # Cache must now report miss
    hit2 = cache.lookup_detailed(vec, query="Company vacation policy", session_id=session_id)
    assert hit2 is None


def test_vector_store_manager_session_isolation(tmp_path):
    """Verify QdrantVectorStore multi-tenant session isolation and deletion."""
    from tests.conftest import DummyEmbeddings
    emb = DummyEmbeddings()
    
    manager = VectorStoreManager(
        backend="qdrant",
        qdrant_url=None,
        local_path=str(tmp_path / "test_qdrant"),
        collection_name="test_multi_tenant_kb",
        embeddings=emb,
    )

    # Ingest docs into two distinct sessions
    docs_s1 = [
        Document(page_content="Session 1 Alpha content", metadata={"source": "a.txt"}),
        Document(page_content="Session 1 Beta content", metadata={"source": "b.txt"}),
    ]
    docs_s2 = [
        Document(page_content="Session 2 Gamma content", metadata={"source": "g.txt"}),
    ]

    manager.add_documents("sess_01", docs_s1)
    manager.add_documents("sess_02", docs_s2)

    assert manager.count_chunks("sess_01") == 2
    assert manager.count_chunks("sess_02") == 1

    # Verify session retrieval isolation
    retriever_s1 = manager.get_retriever("sess_01", k=5)
    results_s1 = retriever_s1.invoke("content")
    for doc in results_s1:
        assert doc.metadata["session_id"] == "sess_01"

    retriever_s2 = manager.get_retriever("sess_02", k=5)
    results_s2 = retriever_s2.invoke("content")
    for doc in results_s2:
        assert doc.metadata["session_id"] == "sess_02"

    # Delete sess_01 and verify
    manager.delete_session("sess_01")
    assert manager.count_chunks("sess_01") == 0
    assert manager.count_chunks("sess_02") == 1
