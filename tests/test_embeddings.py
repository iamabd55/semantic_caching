import math
from app.embeddings.service import EmbeddingService, get_embedding_service


def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    magnitude1 = math.sqrt(sum(a * a for a in vec1))
    magnitude2 = math.sqrt(sum(b * b for b in vec2))
    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0
    return dot_product / (magnitude1 * magnitude2)


def test_embedding_service_dimension():
    service = get_embedding_service()
    vector = service.embed("Explain semantic caching.")

    assert isinstance(vector, list)
    assert len(vector) == 384
    assert all(isinstance(val, float) for val in vector)


def test_embedding_latency_recorded():
    service = get_embedding_service()
    vector, latency_ms = service.embed_with_latency("Testing latency measurement.")

    assert len(vector) == 384
    assert isinstance(latency_ms, float)
    assert latency_ms > 0.0


def test_semantic_similarity_behavior():
    """
    PRD Phase 4 Checkpoint:
    Verify that semantically similar texts yield high cosine similarity,
    while unrelated texts yield significantly lower cosine similarity.
    """
    service = get_embedding_service()

    v_ai_1 = service.embed("How to train a neural network")
    v_ai_2 = service.embed("How to train deep neural networks")
    v_cooking = service.embed("Best recipe for baking chocolate chip cookies")

    sim_related = cosine_similarity(v_ai_1, v_ai_2)
    sim_unrelated = cosine_similarity(v_ai_1, v_cooking)

    # Similar concepts should have high similarity (> 0.80 for all-MiniLM-L6-v2)
    assert sim_related > 0.80, f"Expected high similarity, got {sim_related}"

    # Dissimilar concepts should have low similarity (< 0.20)
    assert sim_unrelated < 0.20, f"Expected low similarity, got {sim_unrelated}"

    # Related similarity must be substantially higher than unrelated
    assert sim_related > sim_unrelated + 0.50


def test_batch_embedding():
    service = get_embedding_service()
    texts = ["First query", "Second query", "Third query"]
    vectors = service.embed_batch(texts)

    assert len(vectors) == 3
    for vec in vectors:
        assert len(vec) == 384
