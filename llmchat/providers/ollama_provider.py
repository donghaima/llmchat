"""Ollama provider for local models.

Implemented against the Ollama HTTP API directly so we don't need a separate
SDK. By default talks to ``http://localhost:11434``; override with
``OLLAMA_HOST``.
"""

from __future__ import annotations

import json
import os
from typing import Iterator
from urllib.error import URLError
from urllib.request import Request, urlopen

from .base import ChatMessage, Provider, ProviderError


def _host() -> str:
    return os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")


class OllamaProvider(Provider):
    name = "ollama"

    def is_configured(self) -> bool:
        # Ollama is "configured" if the daemon answers. We check lazily; for
        # the `models` listing we attempt a connection and report failure
        # there rather than blocking startup with a probe here.
        return True

    def stream(self, model: str, messages: list[ChatMessage],
               system: str | None = None) -> Iterator[str]:
        api_messages: list[dict] = []
        if system:
            api_messages.append({"role": "system", "content": system})
        for m in messages:
            if m.role in ("user", "assistant"):
                api_messages.append({"role": m.role, "content": m.content})

        body = json.dumps({
            "model": model,
            "messages": api_messages,
            "stream": True,
        }).encode("utf-8")

        req = Request(
            f"{_host()}/api/chat",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(req, timeout=600) as resp:
                for raw_line in resp:
                    line = raw_line.decode("utf-8").strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if obj.get("done"):
                        break
                    msg = obj.get("message") or {}
                    text = msg.get("content")
                    if text:
                        yield text
        except URLError as e:
            raise ProviderError(
                f"Cannot reach Ollama at {_host()}. "
                f"Is the daemon running? ({e.reason})"
            ) from e
        except Exception as e:
            raise ProviderError(f"Ollama error: {e}") from e

    def list_models(self) -> list[str]:
        req = Request(f"{_host()}/api/tags", method="GET")
        try:
            with urlopen(req, timeout=5) as resp:
                obj = json.loads(resp.read().decode("utf-8"))
            return sorted(m["name"] for m in obj.get("models", []))
        except Exception:
            return []
