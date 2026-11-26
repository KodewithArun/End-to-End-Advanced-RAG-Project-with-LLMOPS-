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


def generate_session_id() -> str:
    """Generate a unique session ID with timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = uuid.uuid4().hex[:8]
    return f"session_{timestamp}_{unique_id}"


def get_session_vector_store(session_id: str) -> Optional[InMemoryVectorStore]:
    """Retrieve in-memory vector store for a given session."""
    return SESSION_VECTOR_STORES.get(session_id)


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
    ):
        try:
            self.model_loader = ModelLoader()
            self.emb = self.model_loader.load_embeddings()

            self.use_session = use_session_dirs
            self.session_id = session_id or generate_session_id()

            self.temp_base = Path(temp_base)
            self.temp_base.mkdir(parents=True, exist_ok=True)
            self.temp_dir = self.temp_base / self.session_id if self.use_session else self.temp_base
            self.temp_dir.mkdir(parents=True, exist_ok=True)

            log.info(
                "ChatIngestor initialized (In-Memory)",
                session_id=self.session_id,
                temp_dir=str(self.temp_dir),
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
            return retriever

        except Exception as e:
            log.error("Failed to build in-memory retriever", error=str(e))
            raise DocumentPortalException("Failed to build retriever", sys)

    # Alias for backward compatibility
    built_retriver = build_retriever
