import pathlib
import pytest
from langchain_core.documents import Document

from multi_doc_chat.src.document_ingestion.data_ingestion import (
    generate_session_id,
    get_session_vector_store,
    delete_persisted_vector_store,
    discover_and_load_all_persisted_sessions,
    ChatIngestor,
    SESSION_VECTOR_STORES,
)
from multi_doc_chat.exception.custom_exception import DocumentPortalException


def test_generate_session_id_format_and_uniqueness():
    """Verify session IDs are unique and follow the session_YYYYMMDD_HHMMSS_<hex> format."""
    id1 = generate_session_id()
    id2 = generate_session_id()

    assert id1 != id2
    assert id1.startswith("session_")
    assert id2.startswith("session_")
    parts = id1.split("_")
    assert len(parts) == 4, f"Expected 4 parts in {id1}, got {len(parts)}"
    assert len(parts[1]) == 8  # YYYYMMDD
    assert len(parts[2]) == 6  # HHMMSS


def test_chat_ingestor_session_directories(tmp_path: pathlib.Path):
    """Verify ChatIngestor creates session-isolated directories under temp_base."""
    custom_sid = "test_custom_session_123"
    ingestor = ChatIngestor(temp_base=str(tmp_path), session_id=custom_sid, use_session_dirs=True)

    assert ingestor.session_id == custom_sid
    assert ingestor.temp_dir.exists()
    assert str(ingestor.temp_dir).endswith(custom_sid)


def test_split_chunks_respects_size_and_overlap(tmp_path: pathlib.Path):
    """Test recursive character text splitting logic."""
    ingestor = ChatIngestor(temp_base=str(tmp_path), session_id="test_split")

    # 2500 character document
    sample_text = ("Artificial Intelligence and Machine Learning are transforming modern software. " * 30)
    docs = [Document(page_content=sample_text, metadata={"source": "ai_overview.txt"})]

    chunk_size = 500
    chunk_overlap = 100
    chunks = ingestor._split(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk.page_content) <= chunk_size + 100  # leeway for splitter boundaries
        assert chunk.metadata["source"] == "ai_overview.txt"


def test_build_retriever_creates_in_memory_vector_store(sample_text_file: pathlib.Path, tmp_path: pathlib.Path):
    """Verify build_retriever indexes chunks into SESSION_VECTOR_STORES and returns a retriever."""
    session_id = "test_build_retriever_session"
    ingestor = ChatIngestor(temp_base=str(tmp_path), session_id=session_id)

    # Ingest the sample text file
    retriever = ingestor.build_retriever([sample_text_file], chunk_size=300, chunk_overlap=50, k=3)
    assert retriever is not None

    # Check that in-memory vector store is registered in session registry
    vs = get_session_vector_store(session_id)
    assert vs is not None
    assert session_id in SESSION_VECTOR_STORES

    # Retrieve relevant documents for a query
    results = retriever.invoke("What are the core pillars of Agentic AI?")
    assert len(results) > 0
    assert any("pillars" in doc.page_content.lower() for doc in results)


def test_build_retriever_adds_to_existing_vector_store(sample_text_file: pathlib.Path, tmp_path: pathlib.Path):
    """Verify ingesting a second document adds documents to the same session vector store."""
    session_id = "test_cumulative_session"
    ingestor = ChatIngestor(temp_base=str(tmp_path), session_id=session_id)

    # Ingest first document
    ingestor.build_retriever([sample_text_file], chunk_size=300, chunk_overlap=50)
    vs = get_session_vector_store(session_id)
    assert vs is not None

    # Create and ingest second document
    doc2 = tmp_path / "second_doc.txt"
    doc2.write_text("Kubernetes and Docker form the foundation of cloud native container orchestration.")
    ingestor.build_retriever([doc2], chunk_size=300, chunk_overlap=50)

    # Search for terms from both documents
    retriever = vs.as_retriever(search_kwargs={"k": 5})
    res_doc1 = retriever.invoke("pillars of Agentic AI")
    res_doc2 = retriever.invoke("container orchestration Kubernetes")

    assert len(res_doc1) > 0
    assert len(res_doc2) > 0


def test_build_retriever_empty_or_invalid_raises_exception(tmp_path: pathlib.Path):
    """Verify that attempting to ingest nonexistent files raises DocumentPortalException."""
    ingestor = ChatIngestor(temp_base=str(tmp_path), session_id="test_error_handling")
    non_existent = tmp_path / "does_not_exist.txt"

    with pytest.raises(DocumentPortalException):
        ingestor.build_retriever([non_existent])


def test_persistent_vector_store_disk_dump_and_reload(sample_text_file: pathlib.Path, tmp_path: pathlib.Path):
    """Verify in-memory vector store can be dumped to disk and restored across simulated process restart."""
    session_id = "test_persistence_lifecycle"
    index_dir = tmp_path / "custom_indexes"

    # 1. Ingest document and persist
    ingestor = ChatIngestor(temp_base=str(tmp_path), session_id=session_id, index_dir=index_dir)
    ingestor.build_retriever([sample_text_file], chunk_size=300, chunk_overlap=50)

    persisted_file = index_dir / f"{session_id}.json"
    assert persisted_file.exists(), "Expected vector store index file to be saved on disk."

    # 2. Simulate server restart: clear in-memory dictionary
    SESSION_VECTOR_STORES.clear()
    assert session_id not in SESSION_VECTOR_STORES

    # 3. Reload from disk using get_session_vector_store
    restored_vs = get_session_vector_store(session_id, embeddings=ingestor.emb, index_dir=index_dir)
    assert restored_vs is not None
    assert session_id in SESSION_VECTOR_STORES

    # 4. Verify search results on restored vector store
    retriever = restored_vs.as_retriever(search_kwargs={"k": 2})
    results = retriever.invoke("pillars of Agentic AI")
    assert len(results) > 0


def test_delete_persisted_vector_store(sample_text_file: pathlib.Path, tmp_path: pathlib.Path):
    """Verify delete_persisted_vector_store removes index file from disk."""
    session_id = "test_delete_persistence"
    index_dir = tmp_path / "indexes_to_clean"

    ingestor = ChatIngestor(temp_base=str(tmp_path), session_id=session_id, index_dir=index_dir)
    ingestor.build_retriever([sample_text_file])
    assert (index_dir / f"{session_id}.json").exists()

    # Delete index file
    deleted = delete_persisted_vector_store(session_id, index_dir=index_dir)
    assert deleted is True
    assert not (index_dir / f"{session_id}.json").exists()


def test_discover_and_load_all_persisted_sessions(sample_text_file: pathlib.Path, tmp_path: pathlib.Path):
    """Verify scanning directory restores multiple persisted sessions."""
    index_dir = tmp_path / "multi_sessions"
    sid1 = "test_sess_01"
    sid2 = "test_sess_02"

    ing1 = ChatIngestor(temp_base=str(tmp_path), session_id=sid1, index_dir=index_dir)
    ing1.build_retriever([sample_text_file])

    ing2 = ChatIngestor(temp_base=str(tmp_path), session_id=sid2, index_dir=index_dir)
    ing2.build_retriever([sample_text_file])

    SESSION_VECTOR_STORES.clear()
    loaded = discover_and_load_all_persisted_sessions(embeddings=ing1.emb, index_dir=index_dir)
    assert sid1 in loaded
    assert sid2 in loaded
    assert sid1 in SESSION_VECTOR_STORES
    assert sid2 in SESSION_VECTOR_STORES
