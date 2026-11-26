import sys
import os
from operator import itemgetter
from typing import List, Optional, Dict, Any, AsyncIterator, Iterator

from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import InMemoryVectorStore

from multi_doc_chat.utils.model_loader import ModelLoader
from multi_doc_chat.utils.semantic_cache import SemanticCache
from multi_doc_chat.exception.custom_exception import DocumentPortalException
from multi_doc_chat.logger import GLOBAL_LOGGER as log
from multi_doc_chat.prompts.prompt_library import PROMPT_REGISTRY
from multi_doc_chat.model.models import PromptType, ChatAnswer
from multi_doc_chat.src.document_ingestion.data_ingestion import SESSION_VECTOR_STORES, get_session_vector_store
from pydantic import ValidationError


class ConversationalRAG:
    """
    LCEL-based Conversational RAG with:
    - In-Memory Vector Store
    - Semantic Caching (Cosine similarity over query embeddings)
    - Full Sync & Async Support (invoke, ainvoke, stream, astream)
    """

    def __init__(
        self,
        session_id: Optional[str] = None,
        retriever=None,
        enable_cache: bool = True,
        cache_threshold: float = 0.90,
    ):
        try:
            self.session_id = session_id
            self.enable_cache = enable_cache

            # Load models
            self.model_loader = ModelLoader()
            self.llm = self.model_loader.load_llm()
            self.embeddings = self.model_loader.load_embeddings()

            # Initialize semantic cache
            self.cache = SemanticCache(similarity_threshold=cache_threshold) if enable_cache else None

            # Load prompts
            self.contextualize_prompt: ChatPromptTemplate = PROMPT_REGISTRY[
                PromptType.CONTEXTUALIZE_QUESTION.value
            ]
            self.qa_prompt: ChatPromptTemplate = PROMPT_REGISTRY[
                PromptType.CONTEXT_QA.value
            ]

            self.retriever = retriever
            self.chain = None
            if self.retriever is not None:
                self._build_lcel_chain()

            log.info(
                "ConversationalRAG initialized",
                session_id=self.session_id,
                caching_enabled=self.enable_cache,
            )
        except Exception as e:
            log.error("Failed to initialize ConversationalRAG", error=str(e))
            raise DocumentPortalException("Initialization error in ConversationalRAG", sys)

    # ---------- Public API (Retrievers) ----------

    def load_retriever_from_memory(
        self,
        session_id: Optional[str] = None,
        k: int = 5,
        search_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """Load retriever from in-memory session vector store."""
        try:
            sid = session_id or self.session_id
            if not sid:
                raise ValueError("Session ID must be provided to load in-memory vector store")

            vs = get_session_vector_store(sid)
            if vs is None:
                raise ValueError(f"No in-memory vector store found for session: {sid}")

            kwargs = search_kwargs or {"k": k}
            self.retriever = vs.as_retriever(search_kwargs=kwargs)
            self._build_lcel_chain()

            log.info("In-memory retriever loaded successfully", session_id=sid, k=k)
            return self.retriever

        except Exception as e:
            log.error("Failed to load retriever from memory", error=str(e))
            raise DocumentPortalException("Loading error in ConversationalRAG", sys)

    def load_retriever_from_faiss(
        self,
        index_path: str = "",
        k: int = 5,
        index_name: str = "index",
        search_type: str = "similarity",
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        search_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """Backward-compatible retriever loader redirecting to in-memory store."""
        sid = self.session_id or (os.path.basename(index_path) if index_path else None)
        if sid and sid in SESSION_VECTOR_STORES:
            return self.load_retriever_from_memory(session_id=sid, k=k, search_kwargs=search_kwargs)

        if SESSION_VECTOR_STORES:
            first_sid = next(iter(SESSION_VECTOR_STORES))
            return self.load_retriever_from_memory(session_id=first_sid, k=k, search_kwargs=search_kwargs)

        raise DocumentPortalException(f"No in-memory vector store found for session {sid}", sys)

    # ---------- Synchronous Execution ----------

    def invoke(self, user_input: str, chat_history: Optional[List[BaseMessage]] = None) -> str:
        """Synchronous query invocation with semantic caching."""
        try:
            if self.chain is None:
                raise DocumentPortalException(
                    "RAG chain not initialized. Provide retriever or call load_retriever_from_memory() first.", sys
                )
            chat_history = chat_history or []

            # 1. Semantic Cache check (only for initial questions without chat context dependency)
            query_emb = None
            if self.enable_cache and self.cache is not None:
                query_emb = self.embeddings.embed_query(user_input)
                cache_hit = self.cache.lookup(query_emb)
                if cache_hit:
                    cached_answer, score, matched_q = cache_hit
                    log.info("Returning cached response", similarity=score, matched_query=matched_q)
                    return cached_answer

            # 2. Execute LCEL chain
            payload = {"input": user_input, "chat_history": chat_history}
            answer = self.chain.invoke(payload)
            if not answer:
                return "no answer generated."

            # 3. Validate answer
            validated = ChatAnswer(answer=str(answer))
            final_answer = validated.answer

            # 4. Save to Semantic Cache
            if self.enable_cache and self.cache is not None and query_emb is not None:
                self.cache.store(user_input, query_emb, final_answer)

            return final_answer

        except Exception as e:
            log.error("Failed to invoke ConversationalRAG", error=str(e))
            raise DocumentPortalException("Invocation error in ConversationalRAG", sys)

    def stream(self, user_input: str, chat_history: Optional[List[BaseMessage]] = None) -> Iterator[str]:
        """Synchronously stream answer tokens."""
        if self.chain is None:
            raise DocumentPortalException("RAG chain not initialized", sys)
        chat_history = chat_history or []
        payload = {"input": user_input, "chat_history": chat_history}

        for chunk in self.chain.stream(payload):
            yield chunk

    # ---------- Asynchronous Execution ----------

    async def ainvoke(self, user_input: str, chat_history: Optional[List[BaseMessage]] = None) -> str:
        """Asynchronously invoke the RAG pipeline with semantic caching."""
        try:
            if self.chain is None:
                raise DocumentPortalException("RAG chain not initialized", sys)
            chat_history = chat_history or []

            # 1. Semantic Cache check
            query_emb = None
            if self.enable_cache and self.cache is not None:
                # FastEmbed embedding runs locally
                query_emb = self.embeddings.embed_query(user_input)
                cache_hit = self.cache.lookup(query_emb)
                if cache_hit:
                    cached_answer, score, matched_q = cache_hit
                    log.info("Returning async cached response", similarity=score, matched_query=matched_q)
                    return cached_answer

            # 2. Async chain execution
            payload = {"input": user_input, "chat_history": chat_history}
            answer = await self.chain.ainvoke(payload)
            if not answer:
                return "no answer generated."

            # 3. Validate
            validated = ChatAnswer(answer=str(answer))
            final_answer = validated.answer

            # 4. Save to Semantic Cache
            if self.enable_cache and self.cache is not None and query_emb is not None:
                self.cache.store(user_input, query_emb, final_answer)

            return final_answer

        except Exception as e:
            log.error("Async invoke failed in ConversationalRAG", error=str(e))
            raise DocumentPortalException("Async invocation error in ConversationalRAG", sys)

    async def astream(self, user_input: str, chat_history: Optional[List[BaseMessage]] = None) -> AsyncIterator[str]:
        """Asynchronously stream answer tokens in real time (ideal for FastAPI SSE or WebSockets)."""
        if self.chain is None:
            raise DocumentPortalException("RAG chain not initialized", sys)
        chat_history = chat_history or []
        payload = {"input": user_input, "chat_history": chat_history}

        async for chunk in self.chain.astream(payload):
            yield chunk

    # ---------- Internals ----------

    @staticmethod
    def _format_docs(docs) -> str:
        return "\n\n".join(getattr(d, "page_content", str(d)) for d in docs)

    def _build_lcel_chain(self):
        try:
            if self.retriever is None:
                raise DocumentPortalException("No retriever set before building chain", sys)

            # 1) Contextualize question with chat history
            question_rewriter = (
                {"input": itemgetter("input"), "chat_history": itemgetter("chat_history")}
                | self.contextualize_prompt
                | self.llm
                | StrOutputParser()
            )

            # 2) Retrieve relevant context
            retrieve_docs = question_rewriter | self.retriever | self._format_docs

            # 3) Generate final answer
            self.chain = (
                {
                    "context": retrieve_docs,
                    "input": itemgetter("input"),
                    "chat_history": itemgetter("chat_history"),
                }
                | self.qa_prompt
                | self.llm
                | StrOutputParser()
            )

            log.info("LCEL graph built successfully", session_id=self.session_id)
        except Exception as e:
            log.error("Failed to build LCEL chain", error=str(e), session_id=self.session_id)
            raise DocumentPortalException("Failed to build LCEL chain", sys)
