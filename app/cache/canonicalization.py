import re
from typing import List
from app.api.schemas import ChatCompletionRequest, ChatMessage


def canonicalize_text(text: str) -> str:
    """
    Normalize text specifically for semantic comparison:
    1. Lowercases text.
    2. Removes simple punctuation.
    3. Normalizes and collapses multiple whitespace/newlines into a single space.
    
    Example:
        'Explain   MACHINE learning!!' -> 'explain machine learning'
    """
    if not text:
        return ""

    # 1. Lowercase
    normalized = text.lower()

    # 2. Simple punctuation normalization (replace punctuation with space)
    normalized = re.sub(r"[^\w\s]", " ", normalized)

    # 3. Collapse multiple whitespaces and strip
    normalized = re.sub(r"\s+", " ", normalized).strip()

    return normalized


def canonicalize_request(request: ChatCompletionRequest) -> str:
    """
    Produce a canonicalized representation of a ChatCompletionRequest
    for semantic embedding and similarity search.
    
    Preserves role context while normalizing the message contents.
    For a single user message, returns the canonicalized text directly.
    For multi-turn conversations, joins role-prefixed canonicalized messages.
    """
    if not request.messages:
        return ""

    if len(request.messages) == 1:
        return canonicalize_text(request.messages[0].content)

    parts: List[str] = []
    for msg in request.messages:
        cleaned_content = canonicalize_text(msg.content)
        parts.append(f"{msg.role}: {cleaned_content}")

    return "\n".join(parts)
