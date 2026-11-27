from __future__ import annotations
from pathlib import Path
from typing import Iterable, List, Optional, Dict, Any
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from multi_doc_chat.utils.model_loader import ModelLoader
from multi_doc_chat.logger import GLOBAL_LOGGER as log
from multi_doc_chat.exception.custom_exception import DocumentPortalException
from multi_doc_chat.utils.file_io import save_uploaded_files
from multi_doc_chat.utils.document_ops import load_documents
from datetime import datetime
import uuid
import sys

# In-memory registry to store vector store instances per session
SESSION_VECTOR_STORES: Dict[str, InMemoryVectorStore] = {}
DEFAULT_INDEX_DIR = Path("data/indexes")


def generate_session_id() -> str:
    """Generate a unique session ID with timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = uuid.uuid4().hex[:8]
    return f"session_{timestamp}_{unique_id}"


def get_index_path(session_id: str, index_dir: Optional[Path] = None) -> Path:
    """Get file path for session vector store index."""
    target_dir = Path(index_dir or DEFAULT_INDEX_DIR)
    target_dir.mkdir(parents=True, exist_ok=True)
    return target_dir / f"{session_id}.json"


def persist_session_vector_store(
    session_id: str,
    vector_store: InMemoryVectorStore,
    index_dir: Optional[Path] = None,
) -> Path:
    """Persist an in-memory vector store to disk for cross-restart durability."""
    target_path = get_index_path(session_id, index_dir)
    try:
        vector_store.dump(str(target_path))
        log.info("Vector store persisted to disk", session_id=session_id, path=str(target_path))
        return target_path
    except Exception as e:
        log.error("Failed to persist vector store", error=str(e), session_id=session_id)
        raise DocumentPortalException("Vector store persistence failed", sys)


def load_persisted_vector_store(
    session_id: str,
    embeddings=None,
    index_dir: Optional[Path] = None,
) -> Optional[InMemoryVectorStore]:
    """Load a vector store from disk into the session registry."""
    target_path = get_index_path(session_id, index_dir)
    if not target_path.exists():
        return None

    try:
        if embeddings is None:
            embeddings = ModelLoader().load_embeddings()
        vs = InMemoryVectorStore.load(str(target_path), embeddings)
        SESSION_VECTOR_STORES[session_id] = vs
        log.info("Loaded persisted vector store into memory", session_id=session_id, path=str(target_path))
        return vs
    except Exception as e:
        log.warning("Could not load persisted vector store", error=str(e), session_id=session_id)
        return None


def delete_persisted_vector_store(session_id: str, index_dir: Optional[Path] = None) -> bool:
    """Remove persisted vector store file from disk."""
    target_path = get_index_path(session_id, index_dir)
    if target_path.exists():
        try:
            target_path.unlink()
            log.info("Deleted persisted vector store file", session_id=session_id, path=str(target_path))
            return True
        except Exception as e:
            log.error("Failed to delete vector store file", error=str(e), session_id=session_id)
    return False


def get_session_vector_store(
    session_id: str,
    embeddings=None,
    index_dir: Optional[Path] = None,
) -> Optional[InMemoryVectorStore]:
    """
    Retrieve vector store for a given session.
    First checks in-memory registry, then falls back to persisted disk index.
    """
    if session_id in SESSION_VECTOR_STORES:
        return SESSION_VECTOR_STORES[session_id]
    return load_persisted_vector_store(session_id, embeddings=embeddings, index_dir=index_dir)


def discover_and_load_all_persisted_sessions(
    embeddings=None,
    index_dir: Optional[Path] = None,
) -> List[str]:
    """Scan index directory on startup and restore all persisted sessions."""
    target_dir = Path(index_dir or DEFAULT_INDEX_DIR)
    if not target_dir.exists():
        return []

    loaded_sessions = []
    for file_path in target_dir.glob("*.json"):
        sid = file_path.stem
        if sid not in SESSION_VECTOR_STORES:
            vs = load_persisted_vector_store(sid, embeddings=embeddings, index_dir=target_dir)
            if vs:
                loaded_sessions.append(sid)

    log.info("Restored persisted sessions on startup", count=len(loaded_sessions), sessions=loaded_sessions)
    return loaded_sessions


class ChatIngestor:
    """
    Ingests documents into an InMemoryVectorStore organized by session ID.
    """

    def __init__(
        self,
        temp_base: str = "data",
        faiss_base: str = "faiss_index",
        use_session_dirs: bool = True,
        session_id: Optional[str] = None,
        persist: bool = True,
        index_dir: Optional[Path] = None,
    ):
        try:
            self.model_loader = ModelLoader()
            self.emb = self.model_loader.load_embeddings()

            self.use_session = use_session_dirs
            self.session_id = session_id or generate_session_id()
            self.persist = persist
            self.index_dir = index_dir or DEFAULT_INDEX_DIR

            self.temp_base = Path(temp_base)
            self.temp_base.mkdir(parents=True, exist_ok=True)
            self.temp_dir = self.temp_base / self.session_id if self.use_session else self.temp_base
            self.temp_dir.mkdir(parents=True, exist_ok=True)

            log.info(
                "ChatIngestor initialized (In-Memory with Disk Persistence)",
                session_id=self.session_id,
                temp_dir=str(self.temp_dir),
                persist=self.persist,
            )
        except Exception as e:
            log.error("Failed to initialize ChatIngestor", error=str(e))
            raise DocumentPortalException("Initialization error in ChatIngestor", sys)

    def _split(self, docs: List[Document], chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = splitter.split_documents(docs)
        log.info("Documents split", chunks=len(chunks), chunk_size=chunk_size, overlap=chunk_overlap)
        return chunks

    def build_retriever(
        self,
        uploaded_files: Iterable,
        *,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        k: int = 5,
        search_type: str = "similarity",
        **kwargs: Any,
    ):
        """
        Save files, load documents, split into chunks, and index into InMemoryVectorStore.
        """
        try:
            paths = save_uploaded_files(uploaded_files, self.temp_dir)
            docs = load_documents(paths)
            if not docs:
                raise ValueError("No valid documents loaded from provided files")

            chunks = self._split(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

            # Retrieve existing vector store for this session or create a new one
            if self.session_id in SESSION_VECTOR_STORES:
                vector_store = SESSION_VECTOR_STORES[self.session_id]
                vector_store.add_documents(chunks)
                log.info("Added documents to existing in-memory vector store", count=len(chunks), session_id=self.session_id)
            else:
                vector_store = InMemoryVectorStore.from_documents(chunks, self.emb)
                SESSION_VECTOR_STORES[self.session_id] = vector_store
                log.info("Created new in-memory vector store", count=len(chunks), session_id=self.session_id)

            search_kwargs = {"k": k}
            retriever = vector_store.as_retriever(search_kwargs=search_kwargs)

            # Persist vector store to disk if persistence is enabled
            if self.persist:
                persist_session_vector_store(self.session_id, vector_store, self.index_dir)

            return retriever

        except Exception as e:
            log.error("Failed to build in-memory retriever", error=str(e))
            raise DocumentPortalException("Failed to build retriever", sys)

    # Alias for backward compatibility
    built_retriver = build_retriever
