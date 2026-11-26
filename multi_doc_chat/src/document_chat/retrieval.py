import sys
import os
from operator import itemgetter
from typing import List, Optional, Dict, Any

from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import InMemoryVectorStore

from multi_doc_chat.utils.model_loader import ModelLoader
from multi_doc_chat.exception.custom_exception import DocumentPortalException
from multi_doc_chat.logger import GLOBAL_LOGGER as log
from multi_doc_chat.prompts.prompt_library import PROMPT_REGISTRY
from multi_doc_chat.model.models import PromptType, ChatAnswer
from multi_doc_chat.src.document_ingestion.data_ingestion import SESSION_VECTOR_STORES, get_session_vector_store
from pydantic import ValidationError


class ConversationalRAG:
    """
    LCEL-based Conversational RAG with In-Memory Vector Store.

    Usage:
        rag = ConversationalRAG(session_id="abc", retriever=retriever)
        answer = rag.invoke("What is ...?", chat_history=[])
    """

    def __init__(self, session_id: Optional[str] = None, retriever=None):
        try:
            self.session_id = session_id

            # Load LLM and prompts
            self.llm = self._load_llm()
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

            log.info("ConversationalRAG initialized", session_id=self.session_id)
        except Exception as e:
            log.error("Failed to initialize ConversationalRAG", error=str(e))
            raise DocumentPortalException("Initialization error in ConversationalRAG", sys)

    # ---------- Public API ----------

    def load_retriever_from_memory(
        self,
        session_id: Optional[str] = None,
        k: int = 5,
        search_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        Load retriever from in-memory session vector store.
        """
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
        """
        Backward compatible retriever loader that retrieves from in-memory store.
        """
        sid = self.session_id or (os.path.basename(index_path) if index_path else None)
        if sid and sid in SESSION_VECTOR_STORES:
            return self.load_retriever_from_memory(session_id=sid, k=k, search_kwargs=search_kwargs)

        # Fallback to direct memory retrieval if only one session exists
        if SESSION_VECTOR_STORES:
            first_sid = next(iter(SESSION_VECTOR_STORES))
            return self.load_retriever_from_memory(session_id=first_sid, k=k, search_kwargs=search_kwargs)

        raise DocumentPortalException(f"No in-memory vector store found for session {sid}", sys)

    def invoke(self, user_input: str, chat_history: Optional[List[BaseMessage]] = None) -> str:
        """Invoke the LCEL pipeline."""
        try:
            if self.chain is None:
                raise DocumentPortalException(
                    "RAG chain not initialized. Provide retriever or call load_retriever_from_memory() first.", sys
                )
            chat_history = chat_history or []
            payload = {"input": user_input, "chat_history": chat_history}
            answer = self.chain.invoke(payload)
            if not answer:
                log.warning(
                    "No answer generated", user_input=user_input, session_id=self.session_id
                )
                return "no answer generated."

            # Validate answer type and length using Pydantic model
            try:
                validated = ChatAnswer(answer=str(answer))
                answer = validated.answer
            except ValidationError as ve:
                log.error("Invalid chat answer", error=str(ve))
                raise DocumentPortalException("Invalid chat answer", sys)

            log.info(
                "Chain invoked successfully",
                session_id=self.session_id,
                user_input=user_input,
                answer_preview=str(answer)[:150],
            )
            return answer
        except Exception as e:
            log.error("Failed to invoke ConversationalRAG", error=str(e))
            raise DocumentPortalException("Invocation error in ConversationalRAG", sys)

    # ---------- Internals ----------

    def _load_llm(self):
        try:
            llm = ModelLoader().load_llm()
            if not llm:
                raise ValueError("LLM could not be loaded")
            log.info("LLM loaded successfully", session_id=self.session_id)
            return llm
        except Exception as e:
            log.error("Failed to load LLM", error=str(e))
            raise DocumentPortalException("LLM loading error in ConversationalRAG", sys)

    @staticmethod
    def _format_docs(docs) -> str:
        return "\n\n".join(getattr(d, "page_content", str(d)) for d in docs)

    def _build_lcel_chain(self):
        try:
            if self.retriever is None:
                raise DocumentPortalException("No retriever set before building chain", sys)

            # 1) Rewrite user question with chat history context
            question_rewriter = (
                {"input": itemgetter("input"), "chat_history": itemgetter("chat_history")}
                | self.contextualize_prompt
                | self.llm
                | StrOutputParser()
            )

            # 2) Retrieve docs for rewritten question
            retrieve_docs = question_rewriter | self.retriever | self._format_docs

            # 3) Answer using retrieved context + original input + chat history
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
