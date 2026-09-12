import hashlib
import json
from app.api.schemas import ChatCompletionRequest


def serialize_request_deterministically(request: ChatCompletionRequest) -> str:
    """
    Deterministically serialize the relevant fields of a ChatCompletionRequest.
    Guarantees stable key ordering and whitespace elimination so equivalent
    requests always generate identical string representations.
    """
    payload = {
        "model": request.model,
        "messages": [
            {"content": msg.content, "role": msg.role}
            for msg in request.messages
        ],
        "temperature": request.temperature,
        "max_tokens": request.max_tokens,
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def compute_request_hash(request: ChatCompletionRequest) -> str:
    """
    Compute the SHA-256 hex digest of a deterministically serialized ChatCompletionRequest.
    Used for the fast-path exact cache lookup.
    """
    serialized = serialize_request_deterministically(request)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()
