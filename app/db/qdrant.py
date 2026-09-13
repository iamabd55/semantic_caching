from functools import lru_cache
from typing import Any, Dict, List, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from app.core.config import settings


class QdrantService:
    """
    Qdrant vector database service for semantic cache persistence and cosine similarity search.
    Supports both standalone server (Docker / Cloud) and in-memory test instances.
    """

    def __init__(
        self,
        url: Optional[str] = None,
        location: Optional[str] = None,
        timeout: float = 10.0,
    ) -> None:
        if location:
            self.client = QdrantClient(location=location)
        else:
            self.client = QdrantClient(url=url or settings.QDRANT_URL, timeout=timeout)
        self.default_collection = settings.QDRANT_COLLECTION_NAME
        self.dimension = settings.EMBEDDING_DIMENSION

    def ensure_collection(
        self,
        collection_name: Optional[str] = None,
        vector_size: Optional[int] = None,
    ) -> None:
        """
        Check if the collection exists; if not, create it configured with Cosine distance.
        """
        name = collection_name or self.default_collection
        size = vector_size or self.dimension

        if not self.client.collection_exists(name):
            self.client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(size=size, distance=Distance.COSINE),
            )

    def upsert_entry(
        self,
        point_id: str,
        vector: List[float],
        payload: Dict[str, Any],
        collection_name: Optional[str] = None,
    ) -> None:
        """
        Insert or update a dense vector embedding and its associated metadata payload.
        Ensures collection exists prior to insertion.
        """
        name = collection_name or self.default_collection
        self.ensure_collection(name)
        point = PointStruct(id=point_id, vector=vector, payload=payload)
        self.client.upsert(collection_name=name, points=[point])

    def search_nearest(
        self,
        query_vector: List[float],
        limit: int = 1,
        score_threshold: Optional[float] = None,
        collection_name: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search for nearest neighbor vectors using Cosine similarity.
        Returns a list of matching entries: [{'id': str, 'score': float, 'payload': dict}]
        """
        name = collection_name or self.default_collection
        if not self.client.collection_exists(name):
            return []

        response = self.client.query_points(
            collection_name=name,
            query=query_vector,
            limit=limit,
            score_threshold=score_threshold,
        )
        return [
            {
                "id": str(pt.id),
                "score": float(pt.score),
                "payload": pt.payload or {},
            }
            for pt in response.points
        ]

    def count(self, collection_name: Optional[str] = None) -> int:
        """
        Get total point count in the collection.
        """
        name = collection_name or self.default_collection
        if not self.client.collection_exists(name):
            return 0
        return self.client.count(collection_name=name).count

    def delete_collection(self, collection_name: Optional[str] = None) -> None:
        """
        Delete a collection and all its points.
        """
        name = collection_name or self.default_collection
        if self.client.collection_exists(name):
            self.client.delete_collection(collection_name=name)


@lru_cache(maxsize=1)
def get_qdrant_service() -> QdrantService:
    """
    FastAPI dependency and singleton accessor for QdrantService.
    Ensures default collection exists upon first initialization.
    """
    service = QdrantService()
    service.ensure_collection()
    return service
