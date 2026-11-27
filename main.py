import time
import json
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tracers.context import collect_runs

from multi_doc_chat.src.document_ingestion.data_ingestion import (
    ChatIngestor,
    SESSION_VECTOR_STORES,
    get_session_vector_store,
    delete_persisted_vector_store,
    discover_and_load_all_persisted_sessions,
)
from multi_doc_chat.src.document_chat.retrieval import ConversationalRAG
from multi_doc_chat.model.models import (
    ChatRequest,
    ChatResponse,
    IngestResponse,
    HealthResponse,
    FeedbackRequest,
    FeedbackResponse,
    AlertsResponse,
)
from multi_doc_chat.observability import (
    is_tracing_enabled,
    get_langsmith_project,
    record_feedback,
    ALERT_MANAGER,
)
from multi_doc_chat.exception.custom_exception import DocumentPortalException
from multi_doc_chat.logger import GLOBAL_LOGGER as log
from multi_doc_chat.utils.config_loader import load_config

# Initialize FastAPI App
app = FastAPI(
    title="Document Knowledge Workspace API",
    description="Enterprise-grade Document Intelligence API with In-Memory Vector Indexing, Semantic Retrieval, and Asynchronous Streaming.",
    version="1.0.0",
)

# -------------------------------------------------------------
# Middleware Configuration
# -------------------------------------------------------------

# 1. CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 2. Timing and Observability Middleware
@app.middleware("http")
async def logging_and_timing_middleware(request: Request, call_next):
    start_time = time.perf_counter()
    response = await call_next(request)
    duration_ms = (time.perf_counter() - start_time) * 1000

    log.info(
        "HTTP Request Processed",
        method=request.method,
        path=request.url.path,
        status_code=response.status_code,
        latency_ms=round(duration_ms, 2),
    )
    response.headers["X-Process-Time-Ms"] = str(round(duration_ms, 2))
    return response


# -------------------------------------------------------------
# Static Files & Template Engine
# -------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
static_dir = BASE_DIR / "static"
static_dir.mkdir(exist_ok=True)
templates_dir = BASE_DIR / "templates"
templates_dir.mkdir(exist_ok=True)

app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
templates = Jinja2Templates(directory=str(templates_dir))

# Cache of active ConversationalRAG instances per session
RAG_INSTANCES: dict[str, ConversationalRAG] = {}


def get_or_create_rag(session_id: str) -> ConversationalRAG:
    """Retrieve or create an active RAG instance for the session."""
    if session_id in RAG_INSTANCES:
        return RAG_INSTANCES[session_id]

    vs = get_session_vector_store(session_id)
    if vs is None:
        # Fallback to the most recent active session if available
        if SESSION_VECTOR_STORES:
            last_sid = list(SESSION_VECTOR_STORES.keys())[-1]
            log.info("Session not found; falling back to active session", requested=session_id, fallback=last_sid)
            return get_or_create_rag(last_sid)

        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No documents have been indexed yet. Please upload at least one document (.pdf, .docx, .txt) using the left panel.",
        )

    rag = ConversationalRAG(
        session_id=session_id,
        retriever=vs.as_retriever(search_kwargs={"k": 5}),
        enable_cache=True,
        cache_threshold=0.88,
    )
    RAG_INSTANCES[session_id] = rag
    return rag


# -------------------------------------------------------------
# Exception Handlers
# -------------------------------------------------------------
@app.exception_handler(DocumentPortalException)
async def portal_exception_handler(request: Request, exc: DocumentPortalException):
    log.error("DocumentPortalException handled", error=exc.error_message, file=exc.file_name, line=exc.lineno)
    ALERT_MANAGER.record_pipeline_error(exc.error_message)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"success": False, "error": "Internal RAG Pipeline Error", "detail": exc.error_message},
    )


# -------------------------------------------------------------
# Endpoints
# -------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def index_view(request: Request):
    """Serve the conversational web interface."""
    return templates.TemplateResponse(request=request, name="index.html")


@app.get("/api/v1/health", response_model=HealthResponse)
async def health_check():
    """System health and operational metrics."""
    config = load_config()
    emb_provider = config.get("embedding_model", {}).get("provider", "fastembed")
    llm_provider = config.get("llm", {}).get("groq", {}).get("model_name", "openai/gpt-oss-20b")

    return HealthResponse(
        status="healthy",
        active_sessions=len(SESSION_VECTOR_STORES),
        model_name=llm_provider,
        embedding_provider=emb_provider,
        caching_enabled=True,
        langsmith_tracing=is_tracing_enabled(),
        langsmith_project=get_langsmith_project(),
    )


@app.post("/api/v1/ingest", response_model=IngestResponse)
async def ingest_documents(
    files: List[UploadFile] = File(...),
    session_id: Optional[str] = Form(None),
    chunk_size: int = Form(1000),
    chunk_overlap: int = Form(200),
):
    """
    Upload and index documents (.pdf, .docx, .txt) into an In-Memory Vector Store.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided for upload.")

    try:
        ingestor = ChatIngestor(session_id=session_id)
        # Adapt UploadFiles for ingestion
        retriever = ingestor.build_retriever(
            files,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            k=5,
        )

        sid = ingestor.session_id
        # Invalidate prior cached answers for this session upon new document ingestion
        old_rag = RAG_INSTANCES.get(sid)
        if old_rag and old_rag.cache:
            old_rag.cache.invalidate_session(sid)

        # Invalidate or re-create RAG instance with new retriever
        RAG_INSTANCES[sid] = ConversationalRAG(
            session_id=sid,
            retriever=retriever,
            enable_cache=True,
            cache_threshold=0.88,
        )

        file_names = [f.filename or "unknown" for f in files]
        vs = get_session_vector_store(sid)
        chunk_count = len(vs.store) if vs and hasattr(vs, "store") else len(file_names)

        log.info("Files successfully ingested via API", session_id=sid, files=file_names)

        return IngestResponse(
            session_id=sid,
            files=file_names,
            chunks_count=chunk_count,
            message=f"Successfully indexed {len(file_names)} file(s) into In-Memory Vector Store.",
        )
    except Exception as e:
        log.error("API ingestion failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Standard question-answering with Semantic Caching and LangSmith Trace Tracking.
    Returns answer, latency in ms, cache hit indicator, and LangSmith Run ID.
    """
    start_time = time.perf_counter()
    rag = get_or_create_rag(request.session_id)

    # Convert incoming chat_history into LangChain BaseMessages
    history_messages = []
    for msg in request.chat_history:
        if msg.role.lower() in {"human", "user"}:
            history_messages.append(HumanMessage(content=msg.content))
        else:
            history_messages.append(AIMessage(content=msg.content))

    # Trace run execution
    run_id = None
    with collect_runs() as cb:
        answer = await rag.ainvoke(request.query, chat_history=history_messages)
        if cb.traced_runs:
            run_id = str(cb.traced_runs[0].id)

    duration_ms = (time.perf_counter() - start_time) * 1000
    cache_hit = bool(getattr(rag, "last_cache_hit", {}).get("hit", False)) or (duration_ms < 35.0)

    # Operational Anomaly Check for High Latency
    ALERT_MANAGER.check_latency_anomaly(
        latency_ms=duration_ms,
        session_id=request.session_id,
        run_id=run_id,
        query=request.query,
    )

    return ChatResponse(
        session_id=request.session_id,
        query=request.query,
        answer=answer,
        cache_hit=cache_hit,
        latency_ms=round(duration_ms, 2),
        run_id=run_id,
    )


@app.post("/api/v1/chat/stream")
async def chat_stream(request: ChatRequest):
    """
    Server-Sent Events (SSE) streaming endpoint with LangSmith Tracing.
    Streams answer tokens in real time directly from Groq LLM.
    """
    try:
        rag = get_or_create_rag(request.session_id)
    except HTTPException as e:
        async def err_generator():
            msg = f"⚠️ {e.detail}"
            data = json.dumps({"token": msg, "cached": False, "done": True})
            yield f"data: {data}\n\n"
        return StreamingResponse(err_generator(), media_type="text/event-stream")

    history_messages = []
    for msg in request.chat_history:
        if msg.role.lower() in {"human", "user"}:
            history_messages.append(HumanMessage(content=msg.content))
        else:
            history_messages.append(AIMessage(content=msg.content))

    async def event_generator():
        try:
            # First check semantic cache with tiered lookup
            if rag.enable_cache and rag.cache:
                emb = rag.embeddings.embed_query(request.query)
                cached = rag.cache.lookup_detailed(emb, query=request.query, session_id=request.session_id)
                if cached:
                    ans, score, matched_q, tier = cached
                    data = json.dumps({"token": ans, "cached": True, "tier": tier, "done": True})
                    yield f"data: {data}\n\n"
                    return

            # Stream live tokens from LLM
            collected_tokens = []
            with collect_runs() as cb:
                async for token in rag.astream(request.query, chat_history=history_messages):
                    collected_tokens.append(token)
                    data = json.dumps({"token": token, "cached": False, "done": False})
                    yield f"data: {data}\n\n"
                stream_run_id = str(cb.traced_runs[0].id) if cb.traced_runs else None

            # Store in cache once finished
            full_ans = "".join(collected_tokens)
            if rag.enable_cache and rag.cache and full_ans.strip():
                emb = rag.embeddings.embed_query(request.query)
                rag.cache.store(request.query, emb, full_ans, session_id=request.session_id)

            yield f"data: {json.dumps({'done': True, 'run_id': stream_run_id})}\n\n"
        except Exception as e:
            log.error("Error during streaming response", error=str(e))
            yield f"data: {json.dumps({'token': f'❌ Error: {str(e)}', 'error': str(e), 'done': True})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/api/v1/feedback", response_model=FeedbackResponse)
async def submit_feedback(request: FeedbackRequest):
    """
    Submit user evaluation or feedback score (e.g. thumbs-up/down) to LangSmith.
    Triggers an anomaly alert if score is below the operational threshold.
    """
    feedback_id = record_feedback(
        run_id=request.run_id,
        score=request.score,
        key=request.key,
        comment=request.comment,
    )

    # Anomaly Trigger on Low Feedback Score
    ALERT_MANAGER.check_feedback_anomaly(
        score=request.score,
        run_id=request.run_id,
        comment=request.comment,
    )

    if not feedback_id:
        return FeedbackResponse(
            success=False,
            message="Feedback could not be recorded (LangSmith client unavailable or run not found).",
        )

    return FeedbackResponse(
        success=True,
        feedback_id=feedback_id,
        message="Feedback successfully recorded in LangSmith.",
    )


@app.get("/api/v1/sessions")
async def list_sessions():
    """List all currently active in-memory and persisted sessions."""
    from multi_doc_chat.src.document_ingestion.vector_storage import VECTOR_STORE_MANAGER
    all_sids = set(SESSION_VECTOR_STORES.keys())
    try:
        all_sids.update(VECTOR_STORE_MANAGER.list_sessions())
    except Exception:
        pass

    sessions = []
    for sid in sorted(list(all_sids)):
        rag = RAG_INSTANCES.get(sid)
        sessions.append({
            "session_id": sid,
            "cached_queries": rag.cache.size(sid) if rag and rag.cache else 0,
        })
    return {"sessions": sessions, "total_active": len(sessions)}


@app.delete("/api/v1/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a session, freeing its vector store, cache, and disk persistence."""
    deleted_vs = SESSION_VECTOR_STORES.pop(session_id, None)
    deleted_rag = RAG_INSTANCES.pop(session_id, None)
    if deleted_rag and deleted_rag.cache:
        deleted_rag.cache.invalidate_session(session_id)
    deleted_disk = delete_persisted_vector_store(session_id)

    if not deleted_vs and not deleted_rag and not deleted_disk:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")

    log.info("Session successfully removed from memory and disk", session_id=session_id)
    return {"success": True, "message": f"Session '{session_id}' cleared from memory and disk."}


@app.get("/api/v1/alerts", response_model=AlertsResponse)
async def get_alerts():
    """Operational anomaly statistics and recent alert triggers."""
    return AlertsResponse(
        statistics=ALERT_MANAGER.get_alert_statistics(),
        recent_alerts=ALERT_MANAGER.get_recent_alerts(),
    )


@app.post("/api/v1/alerts/test")
async def trigger_test_alert(alert_type: str = "HIGH_LATENCY"):
    """Trigger a manual test alert for monitoring verification."""
    alert = ALERT_MANAGER.trigger_test_alert(alert_type=alert_type)
    return {"success": True, "alert": alert.model_dump(), "message": "Test anomaly alert generated."}


@app.delete("/api/v1/alerts")
async def clear_alerts():
    """Clear all stored alerts."""
    ALERT_MANAGER.clear_alerts()
    return {"success": True, "message": "Alert buffer cleared."}


@app.post("/api/v1/alerts/webhook")
async def receive_alert_webhook(payload: dict):
    """Ingest external alert webhooks from LangSmith rules or cloud monitoring systems."""
    log.info("External alert webhook received", payload=payload)
    return {"status": "received", "timestamp": time.time()}


# Auto-restore persisted sessions on application startup
try:
    discover_and_load_all_persisted_sessions()
except Exception as _startup_e:
    log.warning("Could not restore persisted sessions on startup", error=str(_startup_e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
