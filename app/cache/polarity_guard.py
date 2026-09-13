import re
from typing import Set, Tuple, List

# Explicit set of common negation tokens
NEGATION_TOKENS: Set[str] = {
    "not",
    "no",
    "never",
    "none",
    "neither",
    "nor",
    "cannot",
    "without",
    "hardly",
    "barely",
    "scarcely",
    "isnt",
    "arent",
    "wasnt",
    "werent",
    "havent",
    "hasnt",
    "hadnt",
    "wont",
    "wouldnt",
    "dont",
    "doesnt",
    "didnt",
    "cant",
    "couldnt",
    "shouldnt",
    "isn't",
    "aren't",
    "wasn't",
    "weren't",
    "haven't",
    "hasn't",
    "hadn't",
    "won't",
    "wouldn't",
    "don't",
    "doesn't",
    "didn't",
    "can't",
    "couldn't",
    "shouldn't",
}

# Explicitly-labeled antonym pairs (10-20 domain-relevant pairs)
ANTONYM_PAIRS: List[Tuple[str, str]] = [
    ("good", "bad"),
    ("advantage", "disadvantage"),
    ("advantages", "disadvantages"),
    ("pro", "con"),
    ("pros", "cons"),
    ("benefit", "harm"),
    ("benefits", "harms"),
    ("safe", "dangerous"),
    ("increase", "decrease"),
    ("enable", "disable"),
    ("start", "stop"),
    ("create", "delete"),
    ("pass", "fail"),
    ("true", "false"),
    ("cheap", "expensive"),
    ("fast", "slow"),
    ("high", "low"),
]


def _tokenize(text: str) -> Set[str]:
    """Tokenizes text into lowercase words/contractions."""
    return set(re.findall(r"\b[\w']+\b", text.lower()))


def has_polarity_conflict(text_a: str, text_b: str) -> bool:
    """
    Checks whether two texts exhibit an explicit negation or antonym conflict.
    
    NOTE: This is a targeted post-filtering heuristic for known cases discovered
    during testing (e.g. good vs bad, or negated queries). It is not a comprehensive
    solution for natural language inference (NLI), which requires cross-encoder re-ranking.
    """
    tokens_a = _tokenize(text_a)
    tokens_b = _tokenize(text_b)

    # 1. Negation mismatch: one has negation, the other does not
    has_neg_a = bool(tokens_a & NEGATION_TOKENS)
    has_neg_b = bool(tokens_b & NEGATION_TOKENS)
    if has_neg_a != has_neg_b:
        return True

    # 2. Antonym conflict: one query has word_1, other query has opposing word_2
    for w1, w2 in ANTONYM_PAIRS:
        if (w1 in tokens_a and w2 in tokens_b) or (w2 in tokens_a and w1 in tokens_b):
            # Only conflict if the opposing word isn't present in both (e.g., "compare pros and cons")
            if not (w1 in tokens_a and w2 in tokens_a) and not (w1 in tokens_b and w2 in tokens_b):
                return True

    return False
