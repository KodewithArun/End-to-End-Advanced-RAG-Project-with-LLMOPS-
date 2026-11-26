import numpy as np
from typing import Optional, List, Dict, Any, Tuple
from multi_doc_chat.logger import GLOBAL_LOGGER as log


class SemanticCache:
    """
    In-Memory Semantic Cache using cosine similarity over query embeddings.
    If a query has cosine similarity >= similarity_threshold with a cached query,
    the cached answer is returned immediately (cache hit).
    """

    def __init__(self, similarity_threshold: float = 0.90, max_size: int = 1000):
        self.similarity_threshold = similarity_threshold
        self.max_size = max_size
        self.cache: List[Dict[str, Any]] = []  # List of {"query": str, "embedding": np.ndarray, "answer": str}

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(vec1, vec2) / (norm1 * norm2))

    def lookup(self, query_embedding: List[float] | np.ndarray) -> Optional[Tuple[str, float, str]]:
        """
        Check if query is semantically similar to any cached query.
        Returns: Tuple of (cached_answer, similarity_score, original_cached_query) or None
        """
        if not self.cache:
            return None

        query_vec = np.array(query_embedding, dtype=np.float32)
        best_match = None
        best_score = -1.0

        for entry in self.cache:
            score = self._cosine_similarity(query_vec, entry["embedding"])
            if score > best_score:
                best_score = score
                best_match = entry

        if best_match and best_score >= self.similarity_threshold:
            log.info(
                "Semantic cache HIT",
                score=round(best_score, 4),
                matched_query=best_match["query"],
            )
            return (best_match["answer"], best_score, best_match["query"])

        log.info("Semantic cache MISS", highest_score=round(best_score, 4) if best_score > 0 else 0.0)
        return None

    def store(self, query: str, query_embedding: List[float] | np.ndarray, answer: str) -> None:
        """Store query, its embedding, and answer in the cache."""
        if len(self.cache) >= self.max_size:
            # Evict oldest entry (FIFO)
            self.cache.pop(0)

        query_vec = np.array(query_embedding, dtype=np.float32)
        self.cache.append({
            "query": query,
            "embedding": query_vec,
            "answer": answer,
        })
        log.info("Stored query in semantic cache", query=query, total_cached=len(self.cache))

    def clear(self) -> None:
        """Clear all entries in cache."""
        self.cache.clear()
        log.info("Semantic cache cleared")

    def size(self) -> int:
        return len(self.cache)
