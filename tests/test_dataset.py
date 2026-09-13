import json
from pathlib import Path
import pytest

DATASET_PATH = Path(__file__).parent.parent / "evaluation" / "dataset" / "cache_eval.json"


def load_dataset():
    assert DATASET_PATH.exists(), f"Evaluation dataset file not found at {DATASET_PATH}"
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def test_dataset_file_exists_and_parses():
    """Validates that cache_eval.json exists and is valid JSON."""
    data = load_dataset()
    assert isinstance(data, list)
    assert len(data) >= 100, f"Expected at least 100 evaluation pairs, got {len(data)}"


def test_dataset_schema_and_types():
    """Validates that each record has all required fields and valid types."""
    data = load_dataset()
    seen_ids = set()

    for item in data:
        # Check required keys
        assert "id" in item, "Missing 'id'"
        assert "query_a" in item, f"Missing 'query_a' in {item.get('id')}"
        assert "query_b" in item, f"Missing 'query_b' in {item.get('id')}"
        assert "expected_match" in item, f"Missing 'expected_match' in {item.get('id')}"
        assert "category" in item, f"Missing 'category' in {item.get('id')}"
        assert "domain" in item, f"Missing 'domain' in {item.get('id')}"

        # Uniqueness
        assert item["id"] not in seen_ids, f"Duplicate ID found: {item['id']}"
        seen_ids.add(item["id"])

        # Type validations
        assert isinstance(item["id"], str) and len(item["id"].strip()) > 0
        assert isinstance(item["query_a"], str) and len(item["query_a"].strip()) > 0
        assert isinstance(item["query_b"], str) and len(item["query_b"].strip()) > 0
        assert isinstance(item["expected_match"], bool), f"expected_match must be boolean in {item['id']}"
        assert isinstance(item["category"], str) and len(item["category"].strip()) > 0
        assert isinstance(item["domain"], str) and len(item["domain"].strip()) > 0


def test_dataset_class_balance_and_polarity_coverage():
    """
    PRD Section 17 & Phase 8 requirements:
    Ensures balanced distribution between positive and negative classes,
    and guarantees deliberate inclusion of negation/polarity pairs for Phase 9 threshold sweeping.
    """
    data = load_dataset()
    positives = [d for d in data if d["expected_match"] is True]
    negatives = [d for d in data if d["expected_match"] is False]

    # Verify both classes are well represented
    assert len(positives) >= 45, f"Expected at least 45 positive pairs, got {len(positives)}"
    assert len(negatives) >= 45, f"Expected at least 45 negative pairs, got {len(negatives)}"

    # Verify deliberate inclusion of polarity and negation conflict pairs
    polarity_and_negation_pairs = [
        d for d in negatives if "polarity" in d["category"] or "negation" in d["category"]
    ]
    assert len(polarity_and_negation_pairs) >= 15, (
        f"Expected at least 15 negation/polarity pairs, got {len(polarity_and_negation_pairs)}"
    )
