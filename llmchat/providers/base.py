"""Provider abstraction.

A Provider takes a list of `{role, content}` messages plus an optional system
prompt, and yields response text chunks as they're generated. That's the whole
contract — anything more provider-specific stays inside the adapter.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterator


@dataclass
class ChatMessage:
    role: str  # "user" | "assistant" | "system"
    content: str


class ProviderError(Exception):
    """Anything that goes wrong inside a provider call."""


class Provider(ABC):
    name: str  # short identifier, e.g. "anthropic"

    @abstractmethod
    def stream(self, model: str, messages: list[ChatMessage],
               system: str | None = None) -> Iterator[str]:
        """Yield response text chunks. Raises ProviderError on failure."""

    @abstractmethod
    def list_models(self) -> list[str]:
        """Return known model identifiers for this provider.

        May return a hard-coded list if the provider doesn't expose a
        listing API or if listing requires a separate call we want to avoid.
        """

    @abstractmethod
    def is_configured(self) -> bool:
        """True if the provider has the credentials/setup it needs."""
