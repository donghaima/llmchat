"""Google / Gemini provider.

Uses the `google-genai` SDK (the current unified SDK that replaces the older
`google-generativeai` package).
"""

from __future__ import annotations

import os
from typing import Iterator

from .base import ChatMessage, Provider, ProviderError


_KNOWN_MODELS = (
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
)


class GoogleProvider(Provider):
    name = "google"

    def __init__(self) -> None:
        self._client = None

    def _ensure_client(self):
        if self._client is not None:
            return self._client
        try:
            from google import genai
        except ImportError as e:
            raise ProviderError(
                "The 'google-genai' package is not installed. "
                "Run: pip install google-genai"
            ) from e
        api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get(
            "GEMINI_API_KEY")
        if not api_key:
            raise ProviderError(
                "GOOGLE_API_KEY (or GEMINI_API_KEY) is not set.")
        self._client = genai.Client(api_key=api_key)
        return self._client

    def is_configured(self) -> bool:
        if not (os.environ.get("GOOGLE_API_KEY")
                or os.environ.get("GEMINI_API_KEY")):
            return False
        try:
            from google import genai  # noqa: F401
        except ImportError:
            return False
        return True

    def stream(self, model: str, messages: list[ChatMessage],
               system: str | None = None) -> Iterator[str]:
        client = self._ensure_client()

        # Gemini wants a `contents` list with role + parts. Roles must be
        # "user" or "model" (not "assistant").
        contents = []
        for m in messages:
            if m.role == "user":
                contents.append({"role": "user", "parts": [{"text": m.content}]})
            elif m.role == "assistant":
                contents.append({"role": "model", "parts": [{"text": m.content}]})

        # System instruction goes through config, not into contents.
        from google.genai import types
        config = None
        if system:
            config = types.GenerateContentConfig(system_instruction=system)

        try:
            stream = client.models.generate_content_stream(
                model=model,
                contents=contents,
                config=config,
            )
            for chunk in stream:
                text = getattr(chunk, "text", None)
                if text:
                    yield text
        except Exception as e:
            raise ProviderError(f"Google API error: {e}") from e

    def list_models(self) -> list[str]:
        try:
            client = self._ensure_client()
            models = client.models.list()
            ids: list[str] = []
            for m in models:
                # Names look like "models/gemini-2.5-pro"; strip the prefix.
                name = getattr(m, "name", "") or ""
                if name.startswith("models/"):
                    name = name[len("models/"):]
                if "gemini" in name and "embedding" not in name:
                    ids.append(name)
            return sorted(set(ids)) if ids else list(_KNOWN_MODELS)
        except Exception:
            return list(_KNOWN_MODELS)
