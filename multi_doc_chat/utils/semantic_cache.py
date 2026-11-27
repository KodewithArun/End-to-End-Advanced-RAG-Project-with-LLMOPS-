import os
import json
import hashlib
from typing import Optional, List, Dict, Any, Tuple, Union
import numpy as np

from multi_doc_chat.logger import GLOBAL_LOGGER as log
from multi_doc_chat.utils.config_loader import load_config


class TieredSemanticCache:
    """
    Production-grade Tiered Distributed Semantic Cache:
    - Tier 1 (L1): Exact-match SHA-256 hash lookup via Redis KV or in-memory dict (~0.5ms).
    - Tier 2 (L2): Vector similarity cosine matching across embeddings (~15-30ms).
    - Session & Corpus Version Isolation: Each session has a corpus_version. When documents
      are updated or re-ingested, incrementing corpus_version invalidates all stale cached entries.
    - Graceful Fallback: Seamlessly falls back to local in-memory caching if Redis is offline.
    """

    def __init__(
        self,
        similarity_threshold: float = 0.88,
        max_size: int = 1000,
        redis_url: Optional[str] = None,
        ttl_seconds: int = 3600,
        enable_l1: bool = True,
        enable_l2: bool = True,
    ):
        self.similarity_threshold = similarity_threshold
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.enable_l1 = enable_l1
        self.enable_l2 = enable_l2

        # In-memory local stores
        self.local_l1: Dict[str, str] = {}  # key -> answer
        self.local_l2: List[Dict[str, Any]] = []  # List of {"session_id": str, "corpus_version": int, "query": str, "embedding": np.ndarray, "answer": str}
        self.session_versions: Dict[str, int] = {}

        # Redis connection
        self.redis_client = None
        self._init_redis(redis_url)

    @classmethod
    def from_config(cls, similarity_threshold: Optional[float] = None) -> "TieredSemanticCache":
        """Factory method to instantiate cache from config.yaml."""
        try:
            cfg = load_config().get("distributed_cache", {})
            redis_url = os.getenv("REDIS_URL") or cfg.get("redis_url")
            thresh = similarity_threshold if similarity_threshold is not None else float(cfg.get("similarity_threshold", 0.88))
            return cls(
                similarity_threshold=thresh,
                redis_url=redis_url,
                ttl_seconds=int(cfg.get("ttl_seconds", 3600)),
                enable_l1=bool(cfg.get("l1_exact_enabled", True)),
                enable_l2=bool(cfg.get("l2_semantic_enabled", True)),
            )
        except Exception as e:
            log.warning("Could not load distributed_cache config; using defaults", error=str(e))
            return cls(similarity_threshold=similarity_threshold or 0.88)

    def _init_redis(self, redis_url: Optional[str] = None) -> None:
        """Attempt connection to Redis instance."""
        target_url = redis_url or os.getenv("REDIS_URL") or "redis://127.0.0.1:6379/0"
        try:
            import redis
            client = redis.from_url(target_url, socket_timeout=1.0, decode_responses=True)
            client.ping()
            self.redis_client = client
            log.info("TieredSemanticCache connected to Redis", url=target_url)
        except Exception as e:
            log.info("Redis unavailable; TieredSemanticCache running in in-memory mode", reason=str(e))
            self.redis_client = None

    @property
    def cache(self):
        """Backward-compatible cache list attribute."""
        return self.local_l2

    def get_corpus_version(self, session_id: str) -> int:
        """Get the current corpus version for a session."""
        if self.redis_client is not None:
            try:
                v = self.redis_client.get(f"rag:session:{session_id}:version")
                if v:
                    return int(v)
                self.redis_client.set(f"rag:session:{session_id}:version", 1)
                return 1
            except Exception:
                pass
        return self.session_versions.get(session_id, 1)

    def invalidate_session(self, session_id: str) -> None:
        """
        Invalidate all cached responses for a session by incrementing its corpus version
        and clearing local in-memory caches.
        """
        new_v = self.get_corpus_version(session_id) + 1
        self.session_versions[session_id] = new_v

        if self.redis_client is not None:
            try:
                self.redis_client.set(f"rag:session:{session_id}:version", new_v)
                self.redis_client.publish(
                    "rag:cache:invalidation",
                    json.dumps({"action": "invalidate", "session_id": session_id, "version": new_v}),
                )
            except Exception as e:
                log.debug("Redis invalidate notice", error=str(e))

        # Evict local L1 and L2 entries for this session
        prefix = f"rag:cache:l1:{session_id}:"
        self.local_l1 = {k: v for k, v in self.local_l1.items() if not k.startswith(prefix)}
        self.local_l2 = [e for e in self.local_l2 if e.get("session_id") != session_id]
        log.info("Cache invalidated for session", session_id=session_id, new_version=new_v)

    @staticmethod
    def _normalize_query(query: str) -> str:
        return " ".join(query.strip().lower().split())

    @staticmethod
    def _cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(vec1, vec2) / (norm1 * norm2))

    def lookup(
        self,
        query_embedding: Union[List[float], np.ndarray],
        query: Optional[str] = None,
        session_id: str = "default",
    ) -> Optional[Tuple[str, float, str]]:
        """
        Backward-compatible lookup returning (cached_answer, score, matched_query) or None.
        """
        res = self.lookup_detailed(query_embedding=query_embedding, query=query, session_id=session_id)
        if res:
            answer, score, matched_q, _ = res
            return (answer, score, matched_q)
        return None

    def lookup_detailed(
        self,
        query_embedding: Union[List[float], np.ndarray],
        query: Optional[str] = None,
        session_id: str = "default",
    ) -> Optional[Tuple[str, float, str, str]]:
        """
        Tiered lookup:
        1. Checks L1 (exact match) if query string is provided.
        2. Checks L2 (semantic cosine similarity) if L1 misses.
        Returns: Tuple of (cached_answer, similarity_score, matched_query, tier) or None
        """
        corpus_version = self.get_corpus_version(session_id)

        # -------------------- Tier 1: Exact Match (L1) --------------------
        if self.enable_l1 and query:
            norm_q = self._normalize_query(query)
            q_hash = hashlib.sha256(norm_q.encode("utf-8")).hexdigest()
            l1_key = f"rag:cache:l1:{session_id}:{corpus_version}:{q_hash}"

            # Check local L1 first
            if l1_key in self.local_l1:
                log.info("Semantic cache L1 EXACT HIT (local)", session_id=session_id)
                return (self.local_l1[l1_key], 1.0, query, "L1")

            # Check Redis L1
            if self.redis_client is not None:
                try:
                    val = self.redis_client.get(l1_key)
                    if val:
                        self.local_l1[l1_key] = val
                        log.info("Semantic cache L1 EXACT HIT (redis)", session_id=session_id)
                        return (val, 1.0, query, "L1")
                except Exception as re:
                    log.debug("Redis L1 lookup notice", error=str(re))

        # -------------------- Tier 2: Vector Semantic Match (L2) --------------------
        if not self.enable_l2 or query_embedding is None:
            return None

        query_vec = np.array(query_embedding, dtype=np.float32)
        best_match = None
        best_score = -1.0

        for entry in self.local_l2:
            # Enforce session and corpus version boundary
            if entry.get("session_id") != session_id or entry.get("corpus_version") != corpus_version:
                continue

            score = self._cosine_similarity(query_vec, entry["embedding"])
            if score > best_score:
                best_score = score
                best_match = entry

        if best_match and best_score >= self.similarity_threshold:
            log.info(
                "Semantic cache L2 SEMANTIC HIT",
                score=round(best_score, 4),
                matched_query=best_match["query"],
                session_id=session_id,
            )
            return (best_match["answer"], best_score, best_match["query"], "L2")

        log.info(
            "Semantic cache MISS",
            highest_score=round(best_score, 4) if best_score > 0 else 0.0,
            session_id=session_id,
        )
        return None

    def store(
        self,
        query: str,
        query_embedding: Union[List[float], np.ndarray],
        answer: str,
        session_id: str = "default",
    ) -> None:
        """Store query, embedding, and answer across L1 and L2 cache tiers."""
        if not query or not answer or answer.strip() == "no answer generated.":
            return

        corpus_version = self.get_corpus_version(session_id)

        # 1. Store in L1
        if self.enable_l1:
            norm_q = self._normalize_query(query)
            q_hash = hashlib.sha256(norm_q.encode("utf-8")).hexdigest()
            l1_key = f"rag:cache:l1:{session_id}:{corpus_version}:{q_hash}"

            self.local_l1[l1_key] = answer
            if self.redis_client is not None:
                try:
                    self.redis_client.set(l1_key, answer, ex=self.ttl_seconds)
                except Exception as re:
                    log.debug("Redis L1 store notice", error=str(re))

        # 2. Store in L2
        if self.enable_l2 and query_embedding is not None:
            if len(self.local_l2) >= self.max_size:
                self.local_l2.pop(0)

            query_vec = np.array(query_embedding, dtype=np.float32)
            self.local_l2.append({
                "session_id": session_id,
                "corpus_version": corpus_version,
                "query": query,
                "embedding": query_vec,
                "answer": answer,
            })

        log.info("Stored query in TieredSemanticCache", session_id=session_id, total_l2=len(self.local_l2))

    def clear(self, session_id: Optional[str] = None) -> None:
        """Clear cache entries for a specific session or globally."""
        if session_id:
            self.invalidate_session(session_id)
        else:
            self.local_l1.clear()
            self.local_l2.clear()
            self.session_versions.clear()
            if self.redis_client is not None:
                try:
                    keys = self.redis_client.keys("rag:cache:*")
                    if keys:
                        self.redis_client.delete(*keys)
                except Exception as re:
                    log.debug("Redis clear notice", error=str(re))
            log.info("TieredSemanticCache completely cleared")

    def size(self, session_id: Optional[str] = None) -> int:
        """Return total active cached entries."""
        if session_id:
            v = self.get_corpus_version(session_id)
            return sum(1 for e in self.local_l2 if e.get("session_id") == session_id and e.get("corpus_version") == v)
        return len(self.local_l2)


# Alias SemanticCache for 100% backward compatibility
SemanticCache = TieredSemanticCache
