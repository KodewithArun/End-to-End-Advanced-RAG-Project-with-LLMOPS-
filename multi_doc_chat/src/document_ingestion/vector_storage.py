from pathlib import Path
from typing import List, Optional, Dict, Any
from qdrant_client import QdrantClient, models
from langchain_qdrant import QdrantVectorStore
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import InMemoryVectorStore

from multi_doc_chat.logger import GLOBAL_LOGGER as log
from multi_doc_chat.utils.config_loader import load_config
from multi_doc_chat.utils.model_loader import ModelLoader


class SessionVectorStoreProxy:
    """
    Adapter that wraps the multi-tenant QdrantVectorStore to provide a session-scoped
    interface compatible with existing vector store APIs (as_retriever, add_documents, etc.).
    """

    def __init__(self, manager: "VectorStoreManager", session_id: str):
        self.manager = manager
        self.session_id = session_id

    def as_retriever(self, search_kwargs: Optional[Dict[str, Any]] = None, **kwargs):
        merged_kwargs = dict(search_kwargs or {})
        merged_kwargs.update(kwargs)
        k = merged_kwargs.pop("k", 5)
        return self.manager.get_retriever(self.session_id, k=k, search_kwargs=merged_kwargs)

    def add_documents(self, documents: List[Document], **kwargs):
        return self.manager.add_documents(self.session_id, documents)

    def similarity_search(self, query: str, k: int = 5, **kwargs):
        q_filter = self.manager._build_session_filter(self.session_id)
        if self.manager.vector_store is not None:
            return self.manager.vector_store.similarity_search(query, k=k, filter=q_filter, **kwargs)
        return []

    @property
    def store(self):
        """Simulate .store dict for backward-compatibility with chunk counter."""
        count = self.manager.count_chunks(self.session_id)
        return [None] * count


class VectorStoreManager:
    """
    Enterprise-grade Vector Store Manager supporting:
    - Distributed Qdrant with payload-based session partitioning (is_tenant=True)
    - Fallback to local embedded Qdrant (disk or :memory:)
    - Fallback to InMemoryVectorStore if explicitly configured
    - Automatic vector dimension discovery from embedding model
    """

    def __init__(
        self,
        backend: str = "qdrant",
        qdrant_url: Optional[str] = None,
        local_path: str = "data/qdrant",
        collection_name: str = "documents_knowledge_base",
        embeddings: Optional[Embeddings] = None,
    ):
        self.backend = backend.lower()
        self.qdrant_url = qdrant_url
        self.local_path = local_path
        self.collection_name = collection_name
        self.embeddings = embeddings or ModelLoader().load_embeddings()

        self.client: Optional[QdrantClient] = None
        self.vector_store: Optional[QdrantVectorStore] = None
        self.in_memory_stores: Dict[str, InMemoryVectorStore] = {}
        self.vector_dim: Optional[int] = None

        self._initialize_backend()

    @classmethod
    def from_config(cls, embeddings: Optional[Embeddings] = None) -> "VectorStoreManager":
        """Factory method to instantiate VectorStoreManager from config.yaml."""
        try:
            cfg = load_config().get("vector_storage", {})
            import os
            url = os.getenv("QDRANT_URL") or cfg.get("qdrant_url")
            return cls(
                backend=cfg.get("backend", "qdrant"),
                qdrant_url=url,
                local_path=cfg.get("local_path", "data/qdrant"),
                collection_name=cfg.get("collection_name", "documents_knowledge_base"),
                embeddings=embeddings,
            )
        except Exception as e:
            log.warning("Could not read vector_storage config; using defaults", error=str(e))
            return cls(embeddings=embeddings)

    def _determine_vector_dimension(self) -> int:
        """Dynamically detect embedding vector dimension from the loaded model."""
        if self.vector_dim is not None:
            return self.vector_dim
        try:
            sample_vec = self.embeddings.embed_query("dimension_probe")
            self.vector_dim = len(sample_vec)
            log.info("Detected embedding vector dimension", dimension=self.vector_dim)
            return self.vector_dim
        except Exception as e:
            log.warning("Could not probe embedding dimension; defaulting to 384", error=str(e))
            self.vector_dim = 384
            return self.vector_dim

    def _initialize_backend(self) -> None:
        """Initialize Qdrant client or In-Memory storage."""
        if self.backend == "in_memory":
            log.info("VectorStoreManager initialized with in_memory backend")
            return

        # Attempt Qdrant initialization
        try:
            if self.qdrant_url:
                log.info("Connecting to remote Qdrant server", url=self.qdrant_url)
                self.client = QdrantClient(url=self.qdrant_url, timeout=5.0, check_compatibility=False)
                # Test connection
                self.client.get_collections()
            else:
                target_dir = Path(self.local_path)
                target_dir.mkdir(parents=True, exist_ok=True)
                log.info("Initializing local embedded Qdrant", path=str(target_dir))
                self.client = QdrantClient(path=str(target_dir))

            dim = self._determine_vector_dimension()

            # Create collection if it doesn't exist
            if not self.client.collection_exists(self.collection_name):
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE),
                )
                log.info("Created Qdrant collection", collection=self.collection_name, dimension=dim)

            # Ensure payload tenant index on metadata.session_id
            try:
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name="metadata.session_id",
                    field_schema=models.KeywordIndexParams(type="keyword", is_tenant=True),
                )
            except Exception as pe:
                log.debug("Tenant payload index notice", notice=str(pe))

            self.vector_store = QdrantVectorStore(
                client=self.client,
                collection_name=self.collection_name,
                embedding=self.embeddings,
            )
            log.info("QdrantVectorStore initialized successfully", collection=self.collection_name)

        except Exception as e:
            log.warning(
                "Failed to initialize Qdrant backend; falling back to in-memory mode",
                error=str(e),
            )
            self.backend = "in_memory"
            self.client = None
            self.vector_store = None

    def _build_session_filter(self, session_id: str) -> models.Filter:
        """Construct a Qdrant filter condition for the given session_id."""
        return models.Filter(
            must=[
                models.FieldCondition(
                    key="metadata.session_id",
                    match=models.MatchValue(value=session_id),
                )
            ]
        )

    def add_documents(self, session_id: str, documents: List[Document]) -> List[str]:
        """Index document chunks tagged with session_id."""
        if not documents:
            return []

        # Tag every chunk with session_id in its metadata
        for doc in documents:
            doc.metadata = doc.metadata or {}
            doc.metadata["session_id"] = session_id

        if self.backend == "qdrant" and self.vector_store is not None:
            # Check if vector dimension matches current embeddings (handle test fixture switching)
            dim = self._determine_vector_dimension()
            if self.client and not self.client.collection_exists(self.collection_name):
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE),
                )
                self.vector_store = QdrantVectorStore(
                    client=self.client,
                    collection_name=self.collection_name,
                    embedding=self.embeddings,
                )

            ids = self.vector_store.add_documents(documents)
            log.info("Indexed documents in Qdrant", session_id=session_id, count=len(documents))
            return ids
        else:
            # In-memory storage fallback
            if session_id not in self.in_memory_stores:
                self.in_memory_stores[session_id] = InMemoryVectorStore.from_documents(documents, self.embeddings)
            else:
                self.in_memory_stores[session_id].add_documents(documents)
            log.info("Indexed documents in InMemoryVectorStore", session_id=session_id, count=len(documents))
            return [str(i) for i in range(len(documents))]

    def get_retriever(
        self,
        session_id: str,
        k: int = 5,
        search_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """Construct a retriever strictly isolated to the specified session."""
        if self.backend == "qdrant" and self.vector_store is not None:
            q_filter = self._build_session_filter(session_id)
            kwargs = dict(search_kwargs or {})
            kwargs["filter"] = q_filter
            kwargs["k"] = k
            return self.vector_store.as_retriever(search_kwargs=kwargs)
        else:
            vs = self.in_memory_stores.get(session_id)
            if vs is None:
                raise ValueError(f"No in-memory vector store found for session: {session_id}")
            kwargs = dict(search_kwargs or {})
            kwargs["k"] = k
            return vs.as_retriever(search_kwargs=kwargs)

    def count_chunks(self, session_id: str) -> int:
        """Count total indexed chunks for a session."""
        if self.backend == "qdrant" and self.client is not None:
            try:
                res = self.client.count(
                    collection_name=self.collection_name,
                    count_filter=self._build_session_filter(session_id),
                )
                return res.count
            except Exception as e:
                log.warning("Failed to count chunks in Qdrant", error=str(e), session_id=session_id)
                return 0
        else:
            vs = self.in_memory_stores.get(session_id)
            return len(vs.store) if vs and hasattr(vs, "store") else 0

    def delete_session(self, session_id: str) -> bool:
        """Delete all vectors and metadata belonging to a session."""
        deleted = False
        if self.backend == "qdrant" and self.client is not None:
            try:
                self.client.delete(
                    collection_name=self.collection_name,
                    points_selector=models.FilterSelector(
                        filter=self._build_session_filter(session_id)
                    ),
                )
                deleted = True
                log.info("Deleted session vectors from Qdrant", session_id=session_id)
            except Exception as e:
                log.error("Failed to delete session vectors from Qdrant", error=str(e), session_id=session_id)

        if session_id in self.in_memory_stores:
            del self.in_memory_stores[session_id]
            deleted = True

        return deleted

    def list_sessions(self) -> List[str]:
        """List all distinct session IDs across the vector store."""
        sessions = set()
        if self.backend == "qdrant" and self.client is not None:
            try:
                # Scroll points to collect unique session_id tags
                res, _ = self.client.scroll(
                    collection_name=self.collection_name,
                    limit=1000,
                    with_payload=True,
                    with_vectors=False,
                )
                for point in res:
                    if point.payload and "metadata" in point.payload:
                        sid = point.payload["metadata"].get("session_id")
                        if sid:
                            sessions.add(sid)
            except Exception as e:
                log.warning("Failed to scroll sessions in Qdrant", error=str(e))

        sessions.update(self.in_memory_stores.keys())
        return sorted(list(sessions))

    def has_session(self, session_id: str) -> bool:
        """Check if any vectors exist for the given session."""
        return self.count_chunks(session_id) > 0

    def get_session_proxy(self, session_id: str) -> SessionVectorStoreProxy:
        """Return a session-scoped vector store proxy."""
        return SessionVectorStoreProxy(self, session_id)

    def reset_all(self) -> None:
        """Reset and wipe all collections and in-memory caches (for test suites)."""
        self.in_memory_stores.clear()
        if self.backend == "qdrant" and self.client is not None:
            try:
                if self.client.collection_exists(self.collection_name):
                    self.client.delete_collection(self.collection_name)
                    dim = self._determine_vector_dimension()
                    self.client.create_collection(
                        collection_name=self.collection_name,
                        vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE),
                    )
            except Exception as e:
                log.debug("Reset collection notice", notice=str(e))


# Global Singleton VectorStoreManager
VECTOR_STORE_MANAGER = VectorStoreManager.from_config()
