import io
import json


def test_health_check_endpoint(client):
    """Test GET /api/v1/health returns healthy system status."""
    resp = client.get("/api/v1/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert "model_name" in data
    assert "embedding_provider" in data
    assert data["caching_enabled"] is True


def test_ui_index_html_rendered(client):
    """Test GET / renders the Document Knowledge Workspace HTML page."""
    resp = client.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    assert "Document Knowledge Workspace" in resp.text


def test_chat_unindexed_session_returns_404(client):
    """Test 404 when querying an unindexed session with no active vector stores."""
    body = {
        "session_id": "non_existent_session_id_999",
        "query": "What are the core findings?",
        "chat_history": [],
    }
    resp = client.post("/api/v1/chat", json=body)
    assert resp.status_code == 404
    assert "No documents have been indexed yet" in resp.json()["detail"]


def test_chat_success_and_semantic_cache_hit(client):
    """Ingest document, query via POST /api/v1/chat, and verify semantic cache acceleration on follow-up."""
    session_id = "test_chat_route_session_01"

    # 1. Ingest document
    content = (
        b"Agentic AI systems are composed of four architectural pillars: "
        b"Perception and Goal Formulation, Planning and Reasoning, Memory Systems, and Action Execution with Tools."
    )
    files = [("files", ("agentic_pillars.txt", io.BytesIO(content), "text/plain"))]
    ingest_resp = client.post("/api/v1/ingest", files=files, data={"session_id": session_id})
    assert ingest_resp.status_code == 200

    # 2. First Query -> Miss (LLM generation)
    query_1 = "What are the four architectural pillars of Agentic AI?"
    chat_body_1 = {
        "session_id": session_id,
        "query": query_1,
        "chat_history": [],
    }
    resp_1 = client.post("/api/v1/chat", json=chat_body_1)
    assert resp_1.status_code == 200
    data_1 = resp_1.json()

    assert data_1["session_id"] == session_id
    assert len(data_1["answer"]) > 0
    assert data_1["cache_hit"] is False
    assert data_1["latency_ms"] > 0

    # 3. Second Query (Paraphrased / Semantically Equivalent) -> Semantic Cache Hit (< 25ms)
    query_2 = "What are the four pillars of agentic AI architecture?"
    chat_body_2 = {
        "session_id": session_id,
        "query": query_2,
        "chat_history": [{"role": "human", "content": query_1}, {"role": "ai", "content": data_1["answer"]}],
    }
    resp_2 = client.post("/api/v1/chat", json=chat_body_2)
    assert resp_2.status_code == 200
    data_2 = resp_2.json()

    assert data_2["cache_hit"] is True
    assert data_2["latency_ms"] < 100.0  # Typically 5-15ms
    assert data_2["answer"] == data_1["answer"]


def test_chat_stream_sse_tokens(client):
    """Test real-time token streaming via POST /api/v1/chat/stream."""
    session_id = "test_chat_stream_session"

    # Ingest document
    content = b"ReAct interleaves reasoning and acting to solve multi-step problems systematically."
    files = [("files", ("react_concept.txt", io.BytesIO(content), "text/plain"))]
    client.post("/api/v1/ingest", files=files, data={"session_id": session_id})

    # Stream query
    stream_body = {
        "session_id": session_id,
        "query": "Explain ReAct in one sentence.",
        "chat_history": [],
    }
    resp = client.post("/api/v1/chat/stream", json=stream_body)
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers["content-type"]

    tokens = []
    for line in resp.iter_lines():
        if line.startswith("data: "):
            data = json.loads(line[6:])
            if "token" in data:
                tokens.append(data["token"])

    assert len(tokens) > 0
    full_answer = "".join(tokens)
    assert len(full_answer.strip()) > 0


def test_session_management_list_and_delete(client):
    """Test GET /api/v1/sessions and DELETE /api/v1/sessions/{session_id}."""
    session_id = "test_session_eviction"

    # Ingest document to create session
    files = [("files", ("temp.txt", io.BytesIO(b"Ephemeral document text."), "text/plain"))]
    client.post("/api/v1/ingest", files=files, data={"session_id": session_id})

    # List sessions
    list_resp = client.get("/api/v1/sessions")
    assert list_resp.status_code == 200
    sessions_data = list_resp.json()
    assert any(s["session_id"] == session_id for s in sessions_data["sessions"])

    # Delete session
    del_resp = client.delete(f"/api/v1/sessions/{session_id}")
    assert del_resp.status_code == 200
    assert del_resp.json()["success"] is True

    # Check that session was cleared
    list_resp_after = client.get("/api/v1/sessions")
    assert not any(s["session_id"] == session_id for s in list_resp_after.json()["sessions"])


def test_alerts_endpoints_and_anomaly_statistics(client):
    """Test GET /api/v1/alerts and POST /api/v1/alerts/webhook."""
    # 1. Inspect alerts endpoint
    resp = client.get("/api/v1/alerts")
    assert resp.status_code == 200
    data = resp.json()
    assert "statistics" in data
    assert "recent_alerts" in data
    assert "active_thresholds" in data["statistics"]

    # 2. Test receiving external webhook
    wh_payload = {"event": "langsmith_rule_trigger", "rule_id": "r123", "severity": "WARNING"}
    wh_resp = client.post("/api/v1/alerts/webhook", json=wh_payload)
    assert wh_resp.status_code == 200
    assert wh_resp.json()["status"] == "received"
