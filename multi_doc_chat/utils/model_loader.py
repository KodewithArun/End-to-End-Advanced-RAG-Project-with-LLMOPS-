import os
import sys
import json
from typing import List
from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings
from multi_doc_chat.utils.config_loader import load_config
from multi_doc_chat.logger import GLOBAL_LOGGER as log
from multi_doc_chat.exception.custom_exception import DocumentPortalException


class NativeFastEmbedEmbeddings(Embeddings):
    """
    Native LangChain 1.x compatible wrapper for FastEmbed without langchain-community deprecations.
    Uses fastembed.TextEmbedding directly.
    """
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", **kwargs):
        self.model_name = model_name
        from fastembed import TextEmbedding
        self._model = TextEmbedding(model_name=model_name, **kwargs)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [list(vec) for vec in self._model.embed(texts)]

    def embed_query(self, text: str) -> List[float]:
        return list(next(self._model.embed([text])))


class ApiKeyManager:
    """Manages API keys loaded from env or secrets, checking on demand."""

    def __init__(self):
        self.api_keys = {}
        raw = os.getenv("apikeyliveclass")

        if raw:
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, dict):
                    self.api_keys.update(parsed)
                    log.info("Loaded API_KEYS from secrets")
            except Exception as e:
                log.warning("Failed to parse API_KEYS as JSON", error=str(e))

        # Check common individual environment variables
        for key in ["GROQ_API_KEY", "GOOGLE_API_KEY", "OPENAI_API_KEY"]:
            val = os.getenv(key)
            if val:
                self.api_keys[key] = val

    def get(self, key: str, required: bool = True) -> str | None:
        val = self.api_keys.get(key) or os.getenv(key)
        if not val and required:
            log.error(f"Missing required API key: {key}")
            raise DocumentPortalException(f"Missing required API key: {key}", sys)
        return val


class ModelLoader:
    """
    Loads embedding models and LLMs based on config and environment.
    Supports FastEmbed, Groq, Google, and OpenAI.
    """

    def __init__(self):
        if os.getenv("ENV", "local").lower() != "production":
            load_dotenv()
            log.info("Running in LOCAL mode: .env loaded")
        else:
            log.info("Running in PRODUCTION mode")

        self.api_key_mgr = ApiKeyManager()
        self.config = load_config()
        log.info("YAML config loaded", config_keys=list(self.config.keys()))

    def load_embeddings(self):
        """
        Load and return the embedding model based on config.
        Defaults to FastEmbed (runs locally on CPU, ultra fast).
        """
        try:
            emb_config = self.config.get("embedding_model", {})
            provider = emb_config.get("provider", "fastembed").lower()
            model_name = emb_config.get("model_name", "BAAI/bge-small-en-v1.5")

            log.info("Loading embedding model", provider=provider, model=model_name)

            if provider == "fastembed":
                return NativeFastEmbedEmbeddings(model_name=model_name)

            elif provider == "google":
                from langchain_google_genai import GoogleGenerativeAIEmbeddings
                api_key = self.api_key_mgr.get("GOOGLE_API_KEY")
                return GoogleGenerativeAIEmbeddings(model=model_name, google_api_key=api_key)

            elif provider == "openai":
                from langchain_openai import OpenAIEmbeddings
                api_key = self.api_key_mgr.get("OPENAI_API_KEY")
                return OpenAIEmbeddings(model=model_name, api_key=api_key)

            else:
                raise ValueError(f"Unsupported embedding provider: {provider}")

        except Exception as e:
            log.error("Error loading embedding model", error=str(e))
            raise DocumentPortalException("Failed to load embedding model", sys)

    def load_llm(self):
        """
        Load and return the configured LLM model (Groq, Google, OpenAI).
        """
        try:
            llm_block = self.config.get("llm", {})
            provider_key = os.getenv("LLM_PROVIDER", "groq").lower()

            if provider_key not in llm_block:
                # Fallback to first available provider in config
                provider_key = list(llm_block.keys())[0] if llm_block else "groq"

            llm_config = llm_block.get(provider_key, {})
            provider = llm_config.get("provider", provider_key).lower()
            model_name = llm_config.get("model_name", "llama-3.3-70b-versatile")
            temperature = llm_config.get("temperature", 0.0)
            max_tokens = llm_config.get("max_output_tokens", 2048)

            log.info("Loading LLM", provider=provider, model=model_name)

            if provider == "groq":
                from langchain_groq import ChatGroq
                api_key = self.api_key_mgr.get("GROQ_API_KEY")
                return ChatGroq(
                    model=model_name,
                    api_key=api_key,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )

            elif provider == "google":
                from langchain_google_genai import ChatGoogleGenerativeAI
                api_key = self.api_key_mgr.get("GOOGLE_API_KEY")
                return ChatGoogleGenerativeAI(
                    model=model_name,
                    google_api_key=api_key,
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                )

            elif provider == "openai":
                from langchain_openai import ChatOpenAI
                api_key = self.api_key_mgr.get("OPENAI_API_KEY")
                return ChatOpenAI(
                    model=model_name,
                    api_key=api_key,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )

            else:
                raise ValueError(f"Unsupported LLM provider: {provider}")

        except Exception as e:
            log.error("Failed to load LLM", error=str(e))
            raise DocumentPortalException("Failed to load LLM", sys)


if __name__ == "__main__":
    loader = ModelLoader()
    embeddings = loader.load_embeddings()
    print("Embedding model loaded successfully:", embeddings)
    llm = loader.load_llm()
    print("LLM loaded successfully:", llm)
