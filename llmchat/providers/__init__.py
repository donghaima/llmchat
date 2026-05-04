"""Provider registry.

Resolves a model string like ``anthropic:claude-opus-4-7`` to a concrete
provider instance + bare model name. Providers are loaded lazily so missing
optional SDKs don't break the CLI for users who don't need that provider.
"""

from __future__ import annotations

from functools import lru_cache

from .base import ChatMessage, Provider, ProviderError


# Default provider when the user gives a bare model name (no "provider:" prefix).
DEFAULT_PROVIDER = "anthropic"

# Known provider names. The actual classes are imported lazily.
_PROVIDER_NAMES = ("anthropic", "openai", "google", "ollama")


@lru_cache(maxsize=None)
def get_provider(name: str) -> Provider:
    """Lazy-load and cache a provider by name."""
    if name == "anthropic":
        from .anthropic_provider import AnthropicProvider
        return AnthropicProvider()
    if name == "openai":
        from .openai_provider import OpenAIProvider
        return OpenAIProvider()
    if name == "google":
        from .google_provider import GoogleProvider
        return GoogleProvider()
    if name == "ollama":
        from .ollama_provider import OllamaProvider
        return OllamaProvider()
    raise ProviderError(f"Unknown provider: {name!r}. "
                        f"Known: {', '.join(_PROVIDER_NAMES)}")


def parse_model(spec: str) -> tuple[str, str]:
    """Parse "provider:model" or bare "model" into (provider, model)."""
    if ":" in spec:
        provider, model = spec.split(":", 1)
        return provider.strip(), model.strip()
    return DEFAULT_PROVIDER, spec.strip()


def all_provider_names() -> tuple[str, ...]:
    return _PROVIDER_NAMES


__all__ = [
    "ChatMessage",
    "Provider",
    "ProviderError",
    "get_provider",
    "parse_model",
    "all_provider_names",
    "DEFAULT_PROVIDER",
]
