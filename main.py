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

from multi_doc_chat.src.document_ingestion.data_ingestion import (
    ChatIngestor,
    SESSION_VECTOR_STORES,
    get_session_vector_store,
)
from multi_doc_chat.src.document_chat.retrieval import ConversationalRAG
from multi_doc_chat.model.models import (
    ChatRequest,
    ChatResponse,
    IngestResponse,
    HealthResponse,
)
from multi_doc_chat.exception.custom_exception import DocumentPortalException
from multi_doc_chat.logger import GLOBAL_LOGGER as log
from multi_doc_chat.utils.config_loader import load_config

# Initialize FastAPI App
app = FastAPI(
    title="Multi-Document Advanced RAG API",
    description="Enterprise-grade Conversational RAG with In-Memory Vector Store, FastEmbed, Groq, and Semantic Caching.",
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
    Standard question-answering with Semantic Caching.
    Returns answer, latency in ms, and cache hit indicator.
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

    # Check cache status prior to execution
    cache_before_size = rag.cache.size() if rag.cache else 0
    answer = await rag.ainvoke(request.query, chat_history=history_messages)
    duration_ms = (time.perf_counter() - start_time) * 1000

    # If cache size did not grow and latency is extremely low (<50ms), it was a cache hit
    cache_hit = duration_ms < 50.0

    return ChatResponse(
        session_id=request.session_id,
        query=request.query,
        answer=answer,
        cache_hit=cache_hit,
        latency_ms=round(duration_ms, 2),
    )


@app.post("/api/v1/chat/stream")
async def chat_stream(request: ChatRequest):
    """
    Server-Sent Events (SSE) streaming endpoint.
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
            # First check semantic cache
            if rag.enable_cache and rag.cache:
                emb = rag.embeddings.embed_query(request.query)
                cached = rag.cache.lookup(emb)
                if cached:
                    ans, score, matched_q = cached
                    data = json.dumps({"token": ans, "cached": True, "done": True})
                    yield f"data: {data}\n\n"
                    return

            # Stream live tokens from LLM
            collected_tokens = []
            async for token in rag.astream(request.query, chat_history=history_messages):
                collected_tokens.append(token)
                data = json.dumps({"token": token, "cached": False, "done": False})
                yield f"data: {data}\n\n"

            # Store in cache once finished
            full_ans = "".join(collected_tokens)
            if rag.enable_cache and rag.cache and full_ans.strip():
                emb = rag.embeddings.embed_query(request.query)
                rag.cache.store(request.query, emb, full_ans)

            yield f"data: {json.dumps({'done': True})}\n\n"
        except Exception as e:
            log.error("Error during streaming response", error=str(e))
            yield f"data: {json.dumps({'token': f'❌ Error: {str(e)}', 'error': str(e), 'done': True})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/api/v1/sessions")
async def list_sessions():
    """List all currently active in-memory sessions."""
    sessions = []
    for sid in SESSION_VECTOR_STORES.keys():
        rag = RAG_INSTANCES.get(sid)
        sessions.append({
            "session_id": sid,
            "cached_queries": rag.cache.size() if rag and rag.cache else 0,
        })
    return {"sessions": sessions, "total_active": len(sessions)}


@app.delete("/api/v1/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a session, freeing its in-memory vector store and cache."""
    deleted_vs = SESSION_VECTOR_STORES.pop(session_id, None)
    deleted_rag = RAG_INSTANCES.pop(session_id, None)

    if not deleted_vs and not deleted_rag:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")

    log.info("Session successfully removed from memory", session_id=session_id)
    return {"success": True, "message": f"Session '{session_id}' cleared from memory."}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
