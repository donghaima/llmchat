"""Provider abstraction.

A Provider takes a list of `{role, content}` messages plus an optional system
prompt and tool list, and yields response chunks as they're generated.

The historical signature ``stream(model, messages, system) -> Iterator[str]``
is preserved — when no tools are passed, the stream yields plain strings,
and the chat loop's old code path keeps working.

When tools ARE passed, providers yield ``StreamEvent`` objects from
``llmchat.mcp.types`` instead of bare strings: ``TextDelta`` for text chunks,
``ToolUseRequest`` for tool invocations, ``TurnEnd`` to signal completion.
The chat loop's tool-calling path consumes these events.

We also accept ``tool_results`` — a list of results from prior tool calls in
the same agentic turn — so the provider can stitch them into the conversation
in whatever format it requires.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterator

from ..mcp.types import StreamEvent, ToolDescriptor, ToolResult


@dataclass
class ChatMessage:
    role: str  # "user" | "assistant" | "system" | "tool"
    content: str
    # When the chat loop runs an agentic iteration, it needs to feed the
    # assistant's prior tool_use AND the resulting tool outputs back into
    # the next API call. We carry these as structured fields on the
    # ChatMessage so providers can translate to their native formats.
    #
    # ``tool_calls``: list of ``ToolCall`` the assistant emitted. Only
    #   meaningful on assistant messages.
    # ``tool_results``: list of ``ToolResult`` for a "tool"-role message.
    #   Only meaningful on tool messages.
    tool_calls: list = None  # type: ignore[assignment]  # list[ToolCall]
    tool_results: list = None  # type: ignore[assignment]  # list[ToolResult]


class ProviderError(Exception):
    """Anything that goes wrong inside a provider call."""


class Provider(ABC):
    name: str
    supports_tools: bool = False
    """Whether this provider's adapter can use MCP tools.

    Set to False for adapters that don't yet implement the tool-calling
    protocol (we use this so the router can avoid sending tool-requiring
    turns to providers that can't handle them)."""

    @abstractmethod
    def stream(self, model: str, messages: list[ChatMessage],
               system: str | None = None,
               tools: list[ToolDescriptor] | None = None,
               ) -> Iterator[str | StreamEvent]:
        """Yield text chunks (or stream events when tools are involved).

        - If ``tools`` is None or empty: yields ``str`` chunks (back-compat).
        - If ``tools`` is non-empty: yields ``StreamEvent`` objects.

        Tool conversation state (the assistant's prior tool_use blocks and
        the tool execution results) is carried inside the ``messages`` list
        as structured ``ChatMessage`` entries with ``tool_calls`` /
        ``tool_results`` populated. The provider translates these into its
        native format."""

    @abstractmethod
    def list_models(self) -> list[str]:
        ...

    @abstractmethod
    def is_configured(self) -> bool:
        ...
