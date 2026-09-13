import uuid
import pytest
from app.db.qdrant import QdrantService, get_qdrant_service
from app.embeddings.service import get_embedding_service


@pytest.fixture
def qdrant_test_service():
    """
    Isolated Qdrant service fixture using a temporary collection.
    Cleans up after test run.
    """
    service = get_qdrant_service()
    test_collection = f"test_collection_{uuid.uuid4().hex[:8]}"
    service.ensure_collection(collection_name=test_collection)
    yield service, test_collection
    service.delete_collection(collection_name=test_collection)


def test_qdrant_collection_creation_and_count(qdrant_test_service):
    service, collection = qdrant_test_service
    assert service.count(collection_name=collection) == 0


def test_prd_phase5_checkpoint_semantic_search(qdrant_test_service):
    """
    PRD Phase 5 Checkpoint:
    1. Text -> Embedding -> Qdrant
    2. Stored: "What is machine learning?"
    3. Query:  "Can you explain machine learning?"
    4. Top result must receive a high similarity score (>= 0.80).
    5. Unrelated query: "Best recipe for baking chocolate chip cookies"
       must receive a low similarity score (< 0.40).
    """
    service, collection = qdrant_test_service
    embedding_service = get_embedding_service()

    # 1. Embed and Store
    stored_prompt = "What is machine learning?"
    stored_vector = embedding_service.embed(stored_prompt)
    point_id = str(uuid.uuid4())
    payload = {
        "prompt": stored_prompt,
        "response": "Machine learning is a subset of AI focused on learning from data.",
        "model": "dots-studio/dots-3-note-preview:free",
    }
    service.upsert_entry(
        point_id=point_id,
        vector=stored_vector,
        payload=payload,
        collection_name=collection,
    )

    assert service.count(collection_name=collection) == 1

    # 2. Similar Query
    query_similar = "Can you explain machine learning?"
    query_vector_similar = embedding_service.embed(query_similar)
    results_similar = service.search_nearest(
        query_vector=query_vector_similar,
        limit=1,
        collection_name=collection,
    )

    assert len(results_similar) == 1
    top_match = results_similar[0]
    assert top_match["id"] == point_id
    assert top_match["payload"]["prompt"] == stored_prompt
    assert top_match["score"] >= 0.80, f"Expected similarity >= 0.80, got {top_match['score']}"

    # 3. Dissimilar Query
    query_unrelated = "Best recipe for baking chocolate chip cookies"
    query_vector_unrelated = embedding_service.embed(query_unrelated)
    results_unrelated = service.search_nearest(
        query_vector=query_vector_unrelated,
        limit=1,
        collection_name=collection,
    )

    assert len(results_unrelated) == 1
    unrelated_score = results_unrelated[0]["score"]
    assert unrelated_score < 0.40, f"Expected unrelated score < 0.40, got {unrelated_score}"

    # Margin check
    assert top_match["score"] > unrelated_score + 0.50


def test_qdrant_in_memory_mode():
    """
    Test that QdrantService can run with location=':memory:' for standalone/offline tests.
    """
    mem_service = QdrantService(location=":memory:")
    mem_service.ensure_collection("in_memory_test")
    assert mem_service.count("in_memory_test") == 0

    vec = [0.1] * 384
    mem_service.upsert_entry(
        point_id=str(uuid.uuid4()),
        vector=vec,
        payload={"key": "value"},
        collection_name="in_memory_test",
    )
    assert mem_service.count("in_memory_test") == 1

    results = mem_service.search_nearest(query_vector=vec, limit=1, collection_name="in_memory_test")
    assert len(results) == 1
    assert results[0]["score"] >= 0.99
