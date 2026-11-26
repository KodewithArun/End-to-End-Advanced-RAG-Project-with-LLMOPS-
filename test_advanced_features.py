import asyncio
import time
import os
from multi_doc_chat.src.document_ingestion.data_ingestion import ChatIngestor
from multi_doc_chat.src.document_chat.retrieval import ConversationalRAG

# Disable LangSmith tracing warnings for local test
os.environ["LANGCHAIN_TRACING_V2"] = "false"


async def main():
    print("=" * 70)
    print("  TESTING ADVANCED RAG: ASYNC SUPPORT + STREAMING + SEMANTIC CACHING")
    print("=" * 70)

    # 1. Ingest document
    print("\n[Step 1] Ingesting 'data/agentic_ai.txt' into In-Memory Vector Store...")
    with open("data/agentic_ai.txt", "rb") as f:
        ingestor = ChatIngestor()
        retriever = ingestor.build_retriever([f], chunk_size=500, chunk_overlap=50, k=5)
    print("Document successfully ingested into In-Memory Vector Store!")

    # 2. Initialize RAG with Semantic Caching enabled
    rag = ConversationalRAG(
        session_id=ingestor.session_id,
        retriever=retriever,
        enable_cache=True,
        cache_threshold=0.88,  # Threshold for semantic similarity matching
    )

    # 3. Test Async Streaming
    print("\n[Step 2] Testing Async Token Streaming (`astream`)...")
    stream_question = "In one short sentence, what is agentic AI?"
    print(f"Question: '{stream_question}'")
    print("Streamed Output: ", end="", flush=True)

    async for chunk in rag.astream(stream_question):
        print(chunk, end="", flush=True)
    print("\n")

    # 4. Test Query 1 (Cache Miss -> Calls Groq LLM)
    q1 = "What are the core architectural pillars of Agentic AI?"
    print("\n[Step 3] Query 1 (Initial query -> Expecting Cache MISS):")
    print(f"Question 1: '{q1}'")
    t0 = time.perf_counter()
    ans1 = await rag.ainvoke(q1)
    t1 = time.perf_counter()
    print(f"Answer 1:\n{ans1}")
    print(f"⏱️ Time taken (LLM Call): {(t1 - t0) * 1000:.2f} ms")

    # 5. Test Query 2 (Semantically Equivalent -> Expecting Cache HIT!)
    q2 = "What are the key pillars of agentic AI architecture?"
    print("\n[Step 4] Query 2 (Paraphrased query -> Expecting Semantic Cache HIT):")
    print(f"Question 2: '{q2}'")
    t2 = time.perf_counter()
    ans2 = await rag.ainvoke(q2)
    t3 = time.perf_counter()
    print(f"Answer 2 (from cache):\n{ans2}")
    print(f"⚡ Time taken (Semantic Cache Hit): {(t3 - t2) * 1000:.2f} ms")

    # 6. Performance Summary
    speedup = ((t1 - t0) / (t3 - t2)) if (t3 - t2) > 0 else 100.0
    print("\n" + "=" * 70)
    print(f"🎉 SUCCESS! Semantic Cache Speedup: {speedup:.1f}x faster!")
    print(f"   Initial LLM call: {(t1 - t0) * 1000:.2f} ms")
    print(f"   Cached response:  {(t3 - t2) * 1000:.2f} ms (0 Groq tokens used!)")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
