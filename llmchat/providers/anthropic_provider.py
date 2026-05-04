"""Anthropic / Claude provider."""

from __future__ import annotations

import os
from typing import Iterator

from .base import ChatMessage, Provider, ProviderError


# Reasonable defaults for the listing command. The actual API supports
# /v1/models so we'll use that when configured, falling back to this list.
_KNOWN_MODELS = (
    "claude-opus-4-7",
    "claude-opus-4-6",
    "claude-sonnet-4-6",
    "claude-haiku-4-5",
)


class AnthropicProvider(Provider):
    name = "anthropic"

    def __init__(self) -> None:
        self._client = None
        self._import_error: Exception | None = None

    def _ensure_client(self):
        if self._client is not None:
            return self._client
        try:
            from anthropic import Anthropic
        except ImportError as e:
            self._import_error = e
            raise ProviderError(
                "The 'anthropic' package is not installed. "
                "Run: pip install anthropic"
            ) from e
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise ProviderError("ANTHROPIC_API_KEY is not set.")
        self._client = Anthropic()
        return self._client

    def is_configured(self) -> bool:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            return False
        try:
            import anthropic  # noqa: F401
        except ImportError:
            return False
        return True

    def stream(self, model: str, messages: list[ChatMessage],
               system: str | None = None) -> Iterator[str]:
        client = self._ensure_client()

        # Anthropic takes system as a top-level parameter, not a message.
        api_messages = [
            {"role": m.role, "content": m.content}
            for m in messages
            if m.role in ("user", "assistant")
        ]

        kwargs = {
            "model": model,
            "max_tokens": 4096,
            "messages": api_messages,
        }
        if system:
            kwargs["system"] = system

        try:
            with client.messages.stream(**kwargs) as stream:
                for text in stream.text_stream:
                    yield text
        except Exception as e:
            raise ProviderError(f"Anthropic API error: {e}") from e

    def list_models(self) -> list[str]:
        # Try the live API; fall back to the known list.
        try:
            client = self._ensure_client()
            resp = client.models.list(limit=50)
            ids = [m.id for m in resp.data]
            return ids or list(_KNOWN_MODELS)
        except Exception:
            return list(_KNOWN_MODELS)
