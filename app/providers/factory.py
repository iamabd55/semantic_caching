from typing import Optional

from app.core.config import settings
from app.providers.base import LLMProvider, ProviderException
from app.providers.groq import GroqProvider
from app.providers.openrouter import OpenRouterProvider


def get_provider(provider_name: Optional[str] = None) -> LLMProvider:
    """
    Factory function to instantiate an LLMProvider based on provider name.
    Falls back to settings.DEFAULT_PROVIDER ('openrouter') if not specified.
    """
    name = (provider_name or settings.DEFAULT_PROVIDER).strip().lower()
    if name == "openrouter":
        return OpenRouterProvider()
    elif name == "groq":
        return GroqProvider()
    else:
        raise ProviderException(
            f"Unsupported LLM provider '{name}'. Supported providers are: 'openrouter', 'groq'."
        )
