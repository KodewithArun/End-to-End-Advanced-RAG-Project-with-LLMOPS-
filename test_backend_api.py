import os
from fastapi.testclient import TestClient
from main import app

os.environ["LANGCHAIN_TRACING_V2"] = "false"

client = TestClient(app)


def test_fastapi_backend():
    print("=" * 70)
    print("  TESTING FASTAPI BACKEND API SUITE")
    print("=" * 70)

    # 1. Health check
    print("\n[Test 1] Testing GET /api/v1/health...")
    res = client.get("/api/v1/health")
    assert res.status_code == 200, f"Health check failed: {res.text}"
    health_data = res.json()
    print("Health Status:", health_data)
    assert health_data["status"] == "healthy"
    print("✅ Health check passed!")

    # 2. Web UI View
    print("\n[Test 2] Testing GET / (HTML UI)...")
    res = client.get("/")
    assert res.status_code == 200
    assert "Document Knowledge Workspace" in res.text
    print("✅ Web UI HTML successfully rendered!")

    # 3. Document Ingestion via Multipart Form Upload
    print("\n[Test 3] Testing POST /api/v1/ingest with 'data/agentic_ai.txt'...")
    session_id = "test_api_session_42"
    with open("data/agentic_ai.txt", "rb") as f:
        files = [("files", ("agentic_ai.txt", f, "text/plain"))]
        data = {"session_id": session_id, "chunk_size": 1000, "chunk_overlap": 200}
        res = client.post("/api/v1/ingest", files=files, data=data)

    assert res.status_code == 200, f"Ingestion failed: {res.text}"
    ingest_data = res.json()
    print("Ingestion Response:", ingest_data)
    assert ingest_data["session_id"] == session_id
    assert ingest_data["chunks_count"] > 0
    print("✅ Document ingestion endpoint passed!")

    # 4. Chat Endpoint (Cache Miss -> Calls LLM)
    print("\n[Test 4] Testing POST /api/v1/chat (Initial Query -> Groq LLM)...")
    chat_payload = {
        "session_id": session_id,
        "query": "What are the four architectural pillars of Agentic AI?",
        "chat_history": [],
    }
    res = client.post("/api/v1/chat", json=chat_payload)
    assert res.status_code == 200, f"Chat failed: {res.text}"
    chat_data = res.json()
    print(f"Answer:\n{chat_data['answer']}")
    print(f"Latency: {chat_data['latency_ms']} ms | Cache Hit: {chat_data['cache_hit']}")
    assert len(chat_data["answer"]) > 20
    print("✅ Chat endpoint passed!")

    # 5. Chat Endpoint (Cache Hit -> Semantic Caching)
    print("\n[Test 5] Testing POST /api/v1/chat (Paraphrased Query -> Semantic Cache Hit)...")
    paraphrased_payload = {
        "session_id": session_id,
        "query": "Can you list the key pillars of agentic AI architecture?",
        "chat_history": [],
    }
    res = client.post("/api/v1/chat", json=paraphrased_payload)
    assert res.status_code == 200
    cache_data = res.json()
    print(f"Cached Answer Preview: {cache_data['answer'][:100]}...")
    print(f"⚡ Latency: {cache_data['latency_ms']} ms | Cache Hit: {cache_data['cache_hit']}")
    assert cache_data["cache_hit"] is True
    assert cache_data["latency_ms"] < 50.0
    print("✅ Semantic Cache Hit verified on FastAPI!")

    # 6. Streaming SSE Endpoint
    print("\n[Test 6] Testing POST /api/v1/chat/stream (Server-Sent Events Token Stream)...")
    stream_payload = {
        "session_id": session_id,
        "query": "What is ReAct in short?",
        "chat_history": [],
    }
    with client.stream("POST", "/api/v1/chat/stream", json=stream_payload) as stream_res:
        assert stream_res.status_code == 200
        events = list(stream_res.iter_lines())
        print(f"Received {len(events)} streaming chunks!")
        assert any("data: " in line for line in events)
    print("✅ Real-time token streaming endpoint passed!")

    # 7. Session Management
    print("\n[Test 7] Testing GET /api/v1/sessions and DELETE /api/v1/sessions/{id}...")
    sessions_res = client.get("/api/v1/sessions")
    assert sessions_res.status_code == 200
    print("Active Sessions:", sessions_res.json())

    del_res = client.delete(f"/api/v1/sessions/{session_id}")
    assert del_res.status_code == 200
    print("Delete Response:", del_res.json())
    print("✅ Session management endpoints passed!")

    print("\n" + "=" * 70)
    print("🎉 ALL FASTAPI BACKEND API TESTS COMPLETED AND VERIFIED!")
    print("=" * 70)


if __name__ == "__main__":
    test_fastapi_backend()
