import os
import sys
import time
import json
import httpx
import redis
from pathlib import Path

BASE_URL = "http://127.0.0.1:8000"
QDRANT_URL = "http://127.0.0.1:6333"
REDIS_URL = "redis://127.0.0.1:6380/0"

def main():
    print("=" * 70)
    print("🚀 LIVE END-TO-END RAG TEST WITH DOCKER COMPOSE (QDRANT & REDIS 8)")
    print("=" * 70)

    # 1. Check Containers
    print("\n[Step 1] Verifying Docker Containers...")
    r = redis.from_url(REDIS_URL, socket_timeout=2.0)
    assert r.ping(), "Redis container ping failed!"
    print(f"  ✅ Redis 8 Container Online on port 6380 (Version: {r.info('server').get('redis_version')})")

    with httpx.Client(timeout=5.0) as client:
        q_resp = client.get(f"{QDRANT_URL}/healthz")
        assert q_resp.status_code == 200
        print(f"  ✅ Qdrant Container Online on port 6333 ({q_resp.text.strip()})")

    # 2. Check FastAPI Health
    print("\n[Step 2] Checking FastAPI Server Health...")
    with httpx.Client(timeout=10.0) as client:
        health_res = client.get(f"{BASE_URL}/api/v1/health")
        assert health_res.status_code == 200, f"Health check failed: {health_res.text}"
        h_data = health_res.json()
        print(f"  ✅ Server Status: {h_data['status']}")
        print(f"  ✅ Active Model: {h_data['model_name']}")
        print(f"  ✅ Embedding Provider: {h_data['embedding_provider']}")

    # 3. Ingest Real Document via API
    session_id = f"docker_live_sess_{int(time.time())}"
    print(f"\n[Step 3] Ingesting Document into Qdrant Container (Session: {session_id})...")
    doc_path = Path("data/agentic_ai.txt")
    if not doc_path.exists():
        doc_path.write_text(
            "Agentic AI systems represent autonomous systems designed to perceive environments, "
            "reason through complex multi-step goals, retain conversational and procedural memory, "
            "and execute concrete actions using external tools."
        )

    with httpx.Client(timeout=30.0) as client:
        with open(doc_path, "rb") as f:
            files = {"files": ("agentic_ai.txt", f, "text/plain")}
            data = {"session_id": session_id, "chunk_size": 500, "chunk_overlap": 100}
            ingest_res = client.post(f"{BASE_URL}/api/v1/ingest", files=files, data=data)
            assert ingest_res.status_code == 200, f"Ingestion failed: {ingest_res.text}"
            ing_data = ingest_res.json()
            print(f"  ✅ Ingest Success: {ing_data['message']}")
            print(f"  ✅ Chunks Indexed: {ing_data['chunks_count']}")

    # 4. Verify Qdrant Points directly via Qdrant REST API
    print("\n[Step 4] Inspecting Qdrant Container Collection & Points...")
    with httpx.Client(timeout=5.0) as client:
        col_res = client.get(f"{QDRANT_URL}/collections/documents_knowledge_base")
        assert col_res.status_code == 200, f"Qdrant collection query failed: {col_res.text}"
        col_info = col_res.json()["result"]
        print(f"  ✅ Qdrant Collection Status: {col_info['status']}")
        print(f"  ✅ Total Vectors Stored in Qdrant: {col_info['points_count']}")
        print(f"  ✅ Vector Size: {col_info['config']['params']['vectors']['size']} dimensions")

    # 5. Execute First Chat Query (Real Groq LLM Generation)
    query_1 = "What are the core capabilities of Agentic AI systems?"
    print(f"\n[Step 5] Query 1 (Real Groq LLM Inference): '{query_1}'...")
    with httpx.Client(timeout=30.0) as client:
        t0 = time.perf_counter()
        chat_res_1 = client.post(
            f"{BASE_URL}/api/v1/chat",
            json={"session_id": session_id, "query": query_1, "chat_history": []},
        )
        t1 = time.perf_counter()
        assert chat_res_1.status_code == 200, f"Chat failed: {chat_res_1.text}"
        c1 = chat_res_1.json()
        print(f"  ✅ Answer: {c1['answer'][:120]}...")
        print(f"  ✅ Latency: {c1['latency_ms']}ms (Clock: {round((t1-t0)*1000, 1)}ms)")
        print(f"  ✅ Cache Hit: {c1['cache_hit']}")
        print(f"  ✅ LangSmith Run ID: {c1['run_id']}")

    # 6. Verify Redis Container Cache Keys
    print("\n[Step 6] Inspecting Redis 8 Container Cache Keys...")
    keys = r.keys("rag:cache:*")
    print(f"  ✅ Active Redis Cache Keys ({len(keys)} total):")
    for k in keys[:5]:
        print(f"     - {k.decode('utf-8') if isinstance(k, bytes) else k}")

    # 7. Execute Second Query (Semantic Similarity Cache Hit)
    query_2 = "Can you describe the capabilities of an agentic AI system?"
    print(f"\n[Step 7] Query 2 (Paraphrased Semantic Cache Hit): '{query_2}'...")
    with httpx.Client(timeout=10.0) as client:
        t0 = time.perf_counter()
        chat_res_2 = client.post(
            f"{BASE_URL}/api/v1/chat",
            json={"session_id": session_id, "query": query_2, "chat_history": []},
        )
        t1 = time.perf_counter()
        assert chat_res_2.status_code == 200
        c2 = chat_res_2.json()
        print(f"  ✅ Answer: {c2['answer'][:120]}...")
        print(f"  ✅ Latency: {c2['latency_ms']}ms (⚡ Cache Accelerated!)")
        print(f"  ✅ Cache Hit Flag: {c2['cache_hit']}")
        assert c2["cache_hit"] is True, "Expected semantic cache hit on query 2"

    # 8. Test Streaming Response
    print(f"\n[Step 8] Testing Live SSE Token Stream...")
    with httpx.Client(timeout=15.0) as client:
        with client.stream(
            "POST",
            f"{BASE_URL}/api/v1/chat/stream",
            json={"session_id": session_id, "query": "Summarize perception in agentic AI", "chat_history": []},
        ) as response:
            assert response.status_code == 200
            token_count = 0
            for line in response.iter_lines():
                if line.startswith("data: "):
                    token_count += 1
            print(f"  ✅ Received {token_count} SSE frames streamed in real time.")

    # 9. Verify Redis Shared Operational Alerts
    print(f"\n[Step 9] Testing Shared Alert Manager in Redis 8 Container...")
    with httpx.Client(timeout=5.0) as client:
        # Trigger test anomaly alert
        t_alert_res = client.post(f"{BASE_URL}/api/v1/alerts/test?alert_type=HIGH_LATENCY")
        assert t_alert_res.status_code == 200
        alerts_res = client.get(f"{BASE_URL}/api/v1/alerts")
        assert alerts_res.status_code == 200
        alerts_data = alerts_res.json()
        print(f"  ✅ Total Alerts in Redis Buffer: {alerts_data['statistics']['total_alerts']}")
        print(f"  ✅ Critical Anomalies: {alerts_data['statistics']['critical_count']}")
        print(f"  ✅ Recent Alert: {alerts_data['recent_alerts'][0]['message']}")

    print("\n" + "=" * 70)
    print("🎉 ALL DOCKER COMPOSE DISTRIBUTED RAG TESTS PASSED SUCCESSFULLY!")
    print("=" * 70)

if __name__ == "__main__":
    main()
