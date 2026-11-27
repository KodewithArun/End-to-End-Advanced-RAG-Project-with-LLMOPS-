import sys
import os
from operator import itemgetter
from typing import List, Optional, Dict, Any, AsyncIterator, Iterator

from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from multi_doc_chat.utils.model_loader import ModelLoader
from multi_doc_chat.utils.semantic_cache import SemanticCache
from multi_doc_chat.observability import build_run_config, record_cache_hit_trace
from multi_doc_chat.exception.custom_exception import DocumentPortalException
from multi_doc_chat.logger import GLOBAL_LOGGER as log
from multi_doc_chat.prompts.prompt_library import PROMPT_REGISTRY
from multi_doc_chat.model.models import PromptType, ChatAnswer
from multi_doc_chat.src.document_ingestion.data_ingestion import SESSION_VECTOR_STORES, get_session_vector_store


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
            self.cache = SemanticCache.from_config(similarity_threshold=cache_threshold) if enable_cache else None
            self.last_cache_hit = {"hit": False, "tier": None}

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
            sid = self.session_id or "default"
            self.last_cache_hit = {"hit": False, "tier": None}

            # 1. Semantic Cache check with multi-turn context resolution
            query_emb = None
            if self.enable_cache and self.cache is not None:
                # Fast path: check direct user input embedding
                query_emb = self.embeddings.embed_query(user_input)
                cache_hit = self.cache.lookup_detailed(query_emb, query=user_input, session_id=sid)

                # Contextual fallback: if direct missed and conversation history exists, rewrite
                if not cache_hit and chat_history:
                    try:
                        resolved_query = self._rewrite_query(user_input, chat_history)
                        if resolved_query != user_input:
                            resolved_emb = self.embeddings.embed_query(resolved_query)
                            cache_hit = self.cache.lookup_detailed(resolved_emb, query=resolved_query, session_id=sid)
                            if cache_hit:
                                query_emb = resolved_emb
                    except Exception:
                        pass

                if cache_hit:
                    cached_answer, score, matched_q, tier = cache_hit
                    self.last_cache_hit = {"hit": True, "tier": tier, "score": score, "matched_query": matched_q}
                    log.info("Returning cached response", tier=tier, similarity=score, matched_query=matched_q)
                    try:
                        record_cache_hit_trace(
                            query=user_input,
                            matched_query=matched_q,
                            similarity_score=score,
                            answer=cached_answer,
                            session_id=sid,
                        )
                    except Exception as te:
                        log.debug("Cache hit trace recording skipped", error=str(te))
                    return cached_answer

            # 2. Execute LCEL chain with LangSmith Run Config
            payload = {"input": user_input, "chat_history": chat_history}
            run_config = build_run_config(
                session_id=sid,
                run_name="ConversationalRAG_Sync",
                extra_tags=["sync"],
            )
            answer = self.chain.invoke(payload, config=run_config)
            if not answer:
                return "no answer generated."

            # 3. Validate answer
            validated = ChatAnswer(answer=str(answer))
            final_answer = validated.answer

            # 4. Save to Semantic Cache
            if self.enable_cache and self.cache is not None and query_emb is not None:
                self.cache.store(user_input, query_emb, final_answer, session_id=sid)

            return final_answer

        except Exception as e:
            log.error("Failed to invoke ConversationalRAG", error=str(e))
            raise DocumentPortalException("Invocation error in ConversationalRAG", sys)

    def stream(self, user_input: str, chat_history: Optional[List[BaseMessage]] = None) -> Iterator[str]:
        """Synchronously stream answer tokens with LangSmith tracing."""
        if self.chain is None:
            raise DocumentPortalException("RAG chain not initialized", sys)
        chat_history = chat_history or []
        payload = {"input": user_input, "chat_history": chat_history}
        run_config = build_run_config(
            session_id=self.session_id or "default",
            run_name="ConversationalRAG_Stream",
            extra_tags=["stream", "sync"],
        )

        for chunk in self.chain.stream(payload, config=run_config):
            yield chunk

    # ---------- Asynchronous Execution ----------

    async def ainvoke(self, user_input: str, chat_history: Optional[List[BaseMessage]] = None) -> str:
        """Asynchronously invoke the RAG pipeline with semantic caching and LangSmith tracing."""
        try:
            if self.chain is None:
                raise DocumentPortalException("RAG chain not initialized", sys)
            chat_history = chat_history or []
            sid = self.session_id or "default"
            self.last_cache_hit = {"hit": False, "tier": None}

            # 1. Semantic Cache check with multi-turn context resolution
            query_emb = None
            if self.enable_cache and self.cache is not None:
                # Fast path: check direct user input embedding
                query_emb = self.embeddings.embed_query(user_input)
                cache_hit = self.cache.lookup_detailed(query_emb, query=user_input, session_id=sid)

                # Contextual fallback: if direct missed and conversation history exists, rewrite
                if not cache_hit and chat_history:
                    try:
                        resolved_query = await self._arewrite_query(user_input, chat_history)
                        if resolved_query != user_input:
                            resolved_emb = self.embeddings.embed_query(resolved_query)
                            cache_hit = self.cache.lookup_detailed(resolved_emb, query=resolved_query, session_id=sid)
                            if cache_hit:
                                query_emb = resolved_emb
                    except Exception:
                        pass

                if cache_hit:
                    cached_answer, score, matched_q, tier = cache_hit
                    self.last_cache_hit = {"hit": True, "tier": tier, "score": score, "matched_query": matched_q}
                    log.info("Returning async cached response", tier=tier, similarity=score, matched_query=matched_q)
                    try:
                        record_cache_hit_trace(
                            query=user_input,
                            matched_query=matched_q,
                            similarity_score=score,
                            answer=cached_answer,
                            session_id=sid,
                        )
                    except Exception as te:
                        log.debug("Cache hit trace recording skipped", error=str(te))
                    return cached_answer

            # 2. Async chain execution with LangSmith Run Config
            payload = {"input": user_input, "chat_history": chat_history}
            run_config = build_run_config(
                session_id=sid,
                run_name="ConversationalRAG_Async",
                extra_tags=["async"],
            )
            answer = await self.chain.ainvoke(payload, config=run_config)
            if not answer:
                return "no answer generated."

            # 3. Validate
            validated = ChatAnswer(answer=str(answer))
            final_answer = validated.answer

            # 4. Save to Semantic Cache
            if self.enable_cache and self.cache is not None and query_emb is not None:
                self.cache.store(user_input, query_emb, final_answer, session_id=sid)

            return final_answer

        except Exception as e:
            log.error("Async invoke failed in ConversationalRAG", error=str(e))
            raise DocumentPortalException("Async invocation error in ConversationalRAG", sys)

    async def astream(self, user_input: str, chat_history: Optional[List[BaseMessage]] = None) -> AsyncIterator[str]:
        """Asynchronously stream answer tokens in real time with LangSmith tracing."""
        if self.chain is None:
            raise DocumentPortalException("RAG chain not initialized", sys)
        chat_history = chat_history or []
        payload = {"input": user_input, "chat_history": chat_history}
        run_config = build_run_config(
            session_id=self.session_id or "default",
            run_name="ConversationalRAG_AsyncStream",
            extra_tags=["stream", "async"],
        )

        async for chunk in self.chain.astream(payload, config=run_config):
            yield chunk

    # ---------- Internals ----------

    def _rewrite_query(self, user_input: str, chat_history: List[BaseMessage]) -> str:
        if not chat_history:
            return user_input
        payload = {"input": user_input, "chat_history": chat_history}
        rewriter = (
            {"input": itemgetter("input"), "chat_history": itemgetter("chat_history")}
            | self.contextualize_prompt
            | self.llm
            | StrOutputParser()
        )
        return str(rewriter.invoke(payload))

    async def _arewrite_query(self, user_input: str, chat_history: List[BaseMessage]) -> str:
        if not chat_history:
            return user_input
        payload = {"input": user_input, "chat_history": chat_history}
        rewriter = (
            {"input": itemgetter("input"), "chat_history": itemgetter("chat_history")}
            | self.contextualize_prompt
            | self.llm
            | StrOutputParser()
        )
        return str(await rewriter.ainvoke(payload))

    @staticmethod
    def _format_docs(docs) -> str:
        return "\n\n".join(getattr(d, "page_content", str(d)) for d in docs)

    def _build_lcel_chain(self):
        try:
            if self.retriever is None:
                raise DocumentPortalException("No retriever set before building chain", sys)

            # 1) Question rewriter for multi-turn history with explicit span names
            question_rewriter = (
                {"input": itemgetter("input"), "chat_history": itemgetter("chat_history")}
                | self.contextualize_prompt.with_config(run_name="ContextualizePrompt")
                | self.llm.with_config(run_name="HistoryRewriterLLM")
                | StrOutputParser().with_config(run_name="RewriterOutputParser")
            ).with_config(run_name="Question_Rewriter")

            # Route: if chat_history is empty, use user input directly; otherwise rewrite
            def route_query(input_dict: dict) -> str:
                history = input_dict.get("chat_history") or []
                if not history:
                    return input_dict["input"]
                return question_rewriter.invoke(input_dict)

            async def aroute_query(input_dict: dict) -> str:
                history = input_dict.get("chat_history") or []
                if not history:
                    return input_dict["input"]
                return await question_rewriter.ainvoke(input_dict)

            # 2) Retrieve relevant context with explicit retriever span name
            retrieve_docs = (
                route_query
                | self.retriever.with_config(run_name="InMemoryVectorRetriever")
                | self._format_docs
            ).with_config(run_name="Document_Retriever")

            # 3) Generate final answer
            self.chain = (
                {
                    "context": retrieve_docs,
                    "input": itemgetter("input"),
                    "chat_history": itemgetter("chat_history"),
                }
                | self.qa_prompt.with_config(run_name="ContextQAPrompt")
                | self.llm.with_config(run_name="GroqSynthesisLLM")
                | StrOutputParser().with_config(run_name="SynthesisOutputParser")
            ).with_config(run_name="ConversationalRAG_Chain")

            log.info("LCEL graph built successfully", session_id=self.session_id)
        except Exception as e:
            log.error("Failed to build LCEL chain", error=str(e), session_id=self.session_id)
            raise DocumentPortalException("Failed to build LCEL chain", sys)
            raise DocumentPortalException("Failed to build LCEL chain", sys)
