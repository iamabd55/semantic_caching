import pytest
from app.api.schemas import ChatCompletionRequest, ChatCompletionResponse, ChatMessage
from app.cache.exact_cache import InMemoryExactCache
from app.cache.semantic_cache import SemanticCache
from app.db.qdrant import QdrantService
from app.embeddings.service import get_embedding_service


@pytest.fixture
def test_cache():
    """Provides an isolated in-memory SemanticCache for correctness testing."""
    exact_cache = InMemoryExactCache()
    qdrant_service = QdrantService(location=":memory:")
    embedding_service = get_embedding_service()
    return SemanticCache(
        exact_cache=exact_cache,
        embedding_service=embedding_service,
        qdrant_service=qdrant_service,
        threshold=0.85,
    )


def make_request(prompt: str) -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model="dots-studio/dots-3-note-preview:free",
        messages=[ChatMessage(role="user", content=prompt)],
        temperature=0.0,
    )


def make_response(content: str) -> ChatCompletionResponse:
    return ChatCompletionResponse(
        id="chatcmpl-test-correctness",
        model="dots-studio/dots-3-note-preview:free",
        choices=[
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
    )


@pytest.mark.asyncio
async def test_positive_pairs_true_semantic_hits(test_cache):
    """
    PRD Section 16 & AGENTS.md Section 11:
    Positive pairs with identical semantic meaning must yield a semantic cache HIT.
    """
    positive_pairs = [
        (
            "What is machine learning?",
            "Tell me what is machine learning?",
            "Machine learning is a subset of AI.",
        ),
        (
            "What is the capital of France?",
            "Can you tell me the capital of France?",
            "The capital of France is Paris.",
        ),
        (
            "How do I reverse a string in Python?",
            "In Python, how can I reverse a string?",
            "You can use slicing: s[::-1].",
        ),
    ]

    for original_prompt, rephrased_prompt, canned_answer in positive_pairs:
        # 1. Store original
        req_orig = make_request(original_prompt)
        resp_orig = make_response(canned_answer)
        await test_cache.store(req_orig, resp_orig)

        # 2. Lookup with rephrased prompt
        req_rephrased = make_request(rephrased_prompt)
        result = await test_cache.lookup(req_rephrased)

        assert result.hit is True, f"Expected cache HIT for '{rephrased_prompt}' matching '{original_prompt}'"
        assert result.hit_type == "SEMANTIC"
        assert result.similarity_score >= 0.85
        assert result.response.choices[0].message.content == canned_answer


@pytest.mark.asyncio
async def test_negative_pairs_distinct_intent_yields_miss(test_cache):
    """
    PRD Section 16 & AGENTS.md Section 11:
    Negative pairs share domain vocabulary or topic overlap, but request distinct information.
    The system MUST return a cache MISS to prevent serving an incorrect answer.
    """
    negative_pairs = [
        (
            "What is Python?",
            "How do I install Python on Windows?",
            "Python is a high-level programming language.",
        ),
        (
            "What is the capital of France?",
            "What is the population of France?",
            "Paris is the capital of France.",
        ),
        (
            "How to bake a chocolate cake?",
            "How to bake a vanilla cake?",
            "Recipe for chocolate cake: cocoa powder, flour, sugar...",
        ),
    ]

    for original_prompt, distinct_prompt, canned_answer in negative_pairs:
        # 1. Store original
        req_orig = make_request(original_prompt)
        resp_orig = make_response(canned_answer)
        await test_cache.store(req_orig, resp_orig)

        # 2. Lookup with distinct prompt
        req_distinct = make_request(distinct_prompt)
        result = await test_cache.lookup(req_distinct)

        assert result.hit is False, (
            f"Expected cache MISS for '{distinct_prompt}' when '{original_prompt}' is cached. "
            f"Got similarity score: {result.similarity_score}"
        )
        assert result.similarity_score is not None
        assert result.similarity_score < 0.85


@pytest.mark.asyncio
async def test_borderline_pairs_yields_miss(test_cache):
    """
    PRD Section 16:
    Borderline pairs represent closely related concepts where similarity alone might be ambiguous.
    The cache must reject them at the default production threshold (0.85).
    """
    borderline_pairs = [
        (
            "What is a neural network?",
            "How do neural networks learn?",
            "A neural network is a computational model inspired by biological brains.",
        ),
        (
            "Explain supervised learning",
            "Explain unsupervised learning",
            "Supervised learning trains on labeled input-output pairs.",
        ),
    ]

    for original_prompt, borderline_prompt, canned_answer in borderline_pairs:
        req_orig = make_request(original_prompt)
        resp_orig = make_response(canned_answer)
        await test_cache.store(req_orig, resp_orig)

        req_borderline = make_request(borderline_prompt)
        result = await test_cache.lookup(req_borderline)

        assert result.hit is False, (
            f"Expected MISS for borderline pair '{borderline_prompt}'. "
            f"Got similarity score: {result.similarity_score}"
        )
        assert result.similarity_score < 0.85


@pytest.mark.asyncio
async def test_threshold_sensitivity_demonstration():
    """
    Demonstrates why threshold selection is critical:
    - At threshold=0.75, "How to bake a vanilla cake?" falsely matches "How to bake a chocolate cake?" (score ~0.787).
    - At threshold=0.85, the cache correctly identifies the divergence and yields a MISS.
    """
    embedding_service = get_embedding_service()

    # Cache with overly-permissive threshold (0.75)
    permissive_cache = SemanticCache(
        exact_cache=InMemoryExactCache(),
        embedding_service=embedding_service,
        qdrant_service=QdrantService(location=":memory:"),
        threshold=0.75,
    )

    # Cache with production threshold (0.85)
    strict_cache = SemanticCache(
        exact_cache=InMemoryExactCache(),
        embedding_service=embedding_service,
        qdrant_service=QdrantService(location=":memory:"),
        threshold=0.85,
    )

    orig_req = make_request("How to bake a chocolate cake?")
    orig_resp = make_response("Recipe for chocolate cake: cocoa powder, sugar, flour...")

    await permissive_cache.store(orig_req, orig_resp)
    await strict_cache.store(orig_req, orig_resp)

    divergent_req = make_request("How to bake a vanilla cake?")

    # Permissive threshold falsely hits (score ~0.787 >= 0.75)
    res_permissive = await permissive_cache.lookup(divergent_req)
    assert res_permissive.hit is True
    assert res_permissive.similarity_score > 0.75

    # Strict threshold correctly misses (score ~0.787 < 0.85)
    res_strict = await strict_cache.lookup(divergent_req)
    assert res_strict.hit is False
    assert res_strict.similarity_score < 0.85


def test_polarity_limitation_documented():
    """
    PRD Section 34 / AGENTS.md Section 11:
    Bi-encoders (all-MiniLM-L6-v2) share vocabulary on opposing polarities,
    producing high cosine similarity (~0.94) despite opposite semantic intent.
    This test verifies and documents this known limitation.
    """
    import numpy as np

    embedding_service = get_embedding_service()
    v1 = embedding_service.embed("Why is sleep good for health?")
    v2 = embedding_service.embed("Why is sleep bad for health?")

    cosine_sim = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

    # Documents that bi-encoder alone judges opposing polarity as highly similar (> 0.90)
    assert cosine_sim > 0.90


@pytest.mark.asyncio
async def test_polarity_guard_blocks_antonym_conflict(test_cache):
    """
    Tests that the polarity guard post-filter intercepts our discovered case:
    'Why is sleep good for health?' vs 'Why is sleep bad for health?'.
    Even though cosine similarity > 0.90, the cache returns a MISS.
    """
    req_good = make_request("Why is sleep good for health?")
    resp_good = make_response("Sleep improves memory, cellular repair, and immune function.")
    await test_cache.store(req_good, resp_good)

    req_bad = make_request("Why is sleep bad for health?")
    result = await test_cache.lookup(req_bad)

    # Cosine score is high (~0.94 >= 0.85), but polarity guard MUST block the hit
    assert result.similarity_score > 0.85
    assert result.hit is False
    assert result.response is None


@pytest.mark.asyncio
async def test_polarity_guard_blocks_negation_conflict(test_cache):
    """
    Tests that the polarity guard post-filter intercepts negation conflicts:
    'Is it safe to run migrations on production?' vs
    'Is it not safe to run migrations on production?'.
    """
    req_safe = make_request("Is it safe to run migrations on production?")
    resp_safe = make_response("Running migrations in production requires locks and caution.")
    await test_cache.store(req_safe, resp_safe)

    req_not_safe = make_request("Is it not safe to run migrations on production?")
    result = await test_cache.lookup(req_not_safe)

    # Cosine score is high due to shared tokens, but negation guard MUST block the hit
    assert result.similarity_score > 0.85
    assert result.hit is False
    assert result.response is None

