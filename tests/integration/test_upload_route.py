import io
import pytest


def test_upload_success_single_file(client, sample_bytes_file):
    """Test successful single file upload via POST /api/v1/ingest."""
    filename, file_bytes, content_type = sample_bytes_file
    files = [("files", (filename, file_bytes, content_type))]
    data = {"session_id": "test_upload_session_1", "chunk_size": 500, "chunk_overlap": 100}

    resp = client.post("/api/v1/ingest", files=files, data=data)
    assert resp.status_code == 200

    payload = resp.json()
    assert payload["session_id"] == "test_upload_session_1"
    assert filename in payload["files"]
    assert payload["chunks_count"] > 0
    assert "Successfully indexed" in payload["message"]


def test_upload_success_multiple_files(client):
    """Test multi-document upload in a single request."""
    file1 = ("doc1.txt", io.BytesIO(b"Document one discusses vector databases and indexing strategies."), "text/plain")
    file2 = ("doc2.txt", io.BytesIO(b"Document two explores semantic caching for lowering API latency."), "text/plain")
    files = [("files", file1), ("files", file2)]

    resp = client.post("/api/v1/ingest", files=files, data={"session_id": "multi_doc_session"})
    assert resp.status_code == 200

    payload = resp.json()
    assert payload["session_id"] == "multi_doc_session"
    assert len(payload["files"]) == 2
    assert payload["chunks_count"] >= 2


def test_upload_auto_generates_session_id_when_omitted(client):
    """Test that session_id is automatically generated when not provided."""
    file1 = ("doc_auto.txt", io.BytesIO(b"Automated session generation test content."), "text/plain")
    files = [("files", file1)]

    resp = client.post("/api/v1/ingest", files=files)
    assert resp.status_code == 200

    payload = resp.json()
    assert payload["session_id"].startswith("session_")
    assert payload["chunks_count"] > 0


def test_upload_missing_files_validation_error(client):
    """Test 422 error when files field is missing in request."""
    resp = client.post("/api/v1/ingest", data={"session_id": "missing_files_sess"})
    assert resp.status_code == 422


def test_upload_custom_chunking_parameters(client):
    """Test that custom chunk_size and chunk_overlap are respected."""
    # Long text
    content = b"Knowledge representation and reasoning. " * 50  # ~2000 chars
    files = [("files", ("large_text.txt", io.BytesIO(content), "text/plain"))]
    
    # Small chunks of 200 chars -> should produce at least 10 chunks
    data = {"session_id": "chunk_param_sess", "chunk_size": 200, "chunk_overlap": 50}
    resp = client.post("/api/v1/ingest", files=files, data=data)
    assert resp.status_code == 200
    assert resp.json()["chunks_count"] >= 10
