"""OpenAI / GPT provider."""

from __future__ import annotations

import os
from typing import Iterator

from .base import ChatMessage, Provider, ProviderError


_KNOWN_MODELS = (
    "gpt-5.5",
    "gpt-5.5-pro",
    "gpt-5.4",
    "gpt-5.4-mini",
    "gpt-5.4-nano",
)


class OpenAIProvider(Provider):
    name = "openai"

    def __init__(self) -> None:
        self._client = None

    def _ensure_client(self):
        if self._client is not None:
            return self._client
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ProviderError(
                "The 'openai' package is not installed. "
                "Run: pip install openai"
            ) from e
        if not os.environ.get("OPENAI_API_KEY"):
            raise ProviderError("OPENAI_API_KEY is not set.")
        self._client = OpenAI()
        return self._client

    def is_configured(self) -> bool:
        if not os.environ.get("OPENAI_API_KEY"):
            return False
        try:
            import openai  # noqa: F401
        except ImportError:
            return False
        return True

    def stream(self, model: str, messages: list[ChatMessage],
               system: str | None = None) -> Iterator[str]:
        client = self._ensure_client()

        # OpenAI takes system as a leading message with role="system".
        api_messages: list[dict] = []
        if system:
            api_messages.append({"role": "system", "content": system})
        for m in messages:
            if m.role in ("user", "assistant"):
                api_messages.append({"role": m.role, "content": m.content})

        try:
            stream = client.chat.completions.create(
                model=model,
                messages=api_messages,
                stream=True,
            )
            for chunk in stream:
                # Each chunk has zero or more choices; we only ever ask for one.
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                text = getattr(delta, "content", None)
                if text:
                    yield text
        except Exception as e:
            raise ProviderError(f"OpenAI API error: {e}") from e

    def list_models(self) -> list[str]:
        try:
            client = self._ensure_client()
            resp = client.models.list()
            # Filter to chat-capable models. OpenAI's listing includes all
            # model types (embeddings, audio, etc.), so we apply a loose filter.
            ids = [
                m.id for m in resp.data
                if m.id.startswith("gpt-") and "audio" not in m.id
                and "image" not in m.id and "realtime" not in m.id
            ]
            return sorted(ids) if ids else list(_KNOWN_MODELS)
        except Exception:
            return list(_KNOWN_MODELS)
