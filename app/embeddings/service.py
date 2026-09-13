from functools import lru_cache
import time
from typing import List, Tuple
from sentence_transformers import SentenceTransformer

from app.core.config import settings


class EmbeddingService:
    """
    Local embedding service using sentence-transformers (default: all-MiniLM-L6-v2).
    Outputs 384-dimensional dense vectors with low CPU latency and zero API cost.
    """

    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or settings.EMBEDDING_MODEL_NAME
        self.dimension = settings.EMBEDDING_DIMENSION
        # Load model into memory
        self.model = SentenceTransformer(self.model_name)

    def embed(self, text: str) -> List[float]:
        """
        Generate a 384-dimensional embedding vector for a single text input.
        """
        vector = self.model.encode(text, convert_to_numpy=True)
        return vector.tolist()

    def embed_with_latency(self, text: str) -> Tuple[List[float], float]:
        """
        Generate an embedding vector and record the generation latency in milliseconds.
        Returns: (embedding_vector, latency_ms)
        """
        start = time.perf_counter()
        vector = self.embed(text)
        latency_ms = (time.perf_counter() - start) * 1000.0
        return vector, round(latency_ms, 2)

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Batch generate embedding vectors for multiple text inputs.
        """
        if not texts:
            return []
        vectors = self.model.encode(texts, convert_to_numpy=True)
        return [v.tolist() for v in vectors]


@lru_cache(maxsize=1)
def get_embedding_service() -> EmbeddingService:
    """
    FastAPI dependency / singleton accessor for EmbeddingService.
    Ensures the 80MB model is loaded only once in memory.
    """
    return EmbeddingService()
