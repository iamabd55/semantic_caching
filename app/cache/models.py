from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import uuid
from pydantic import BaseModel, Field

from app.api.schemas import ChatCompletionResponse


class CacheEntry(BaseModel):
    """
    Standard Cache Entry Schema as defined in PRD Section 6 & AGENTS.md Section 9.
    Stores the full context for both exact-match and future semantic caching.
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    original_request_hash: str
    response: ChatCompletionResponse
    model: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # Optional fields for upcoming phases (canonicalization, embeddings, multi-tenancy, TTL)
    normalized_request: Optional[str] = None
    embedding: Optional[List[float]] = None
    model_version: Optional[str] = None
    tenant_id: Optional[str] = None
    expires_at: Optional[datetime] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
