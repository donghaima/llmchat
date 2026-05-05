"""Provider-neutral tool types.

Each provider (Anthropic, OpenAI, Google) has its own tool-call format.
Internally we use these dataclasses; provider adapters translate to/from
their native format.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ToolDescriptor:
    """Metadata about a tool, used to advertise it to a model.

    The fully-qualified name is ``server_name::tool_name`` so two MCP servers
    that both expose a "search" tool don't collide. Providers see this same
    name; we strip the prefix when calling the underlying server.
    """
    fq_name: str         # "<server>::<tool>" — what the model calls
    server: str          # MCP server name (the key in mcp.json)
    tool_name: str       # tool name as the server itself knows it
    description: str
    input_schema: dict   # JSON Schema dict


@dataclass
class ToolCall:
    """A request from the model to invoke a tool."""
    id: str              # provider-supplied id, used to correlate the result
    fq_name: str         # the fully-qualified name we advertised
    arguments: dict[str, Any]


@dataclass
class ToolResult:
    """The data we feed back to the model after running the tool."""
    id: str              # echoes ToolCall.id
    fq_name: str
    content: str         # MCP returns a list of content blocks; we flatten to text
    is_error: bool = False


# ----- streaming events -----------------------------------------------------
# Providers used to yield bare strings. Now they yield events so we can
# distinguish text deltas from tool requests while still streaming.

@dataclass
class TextDelta:
    text: str


@dataclass
class ToolUseRequest:
    calls: list[ToolCall]
    """A complete batch of tool calls the model wants us to run.

    All providers we support emit tool calls atomically (the call id, name,
    and arguments arrive together at the end of a turn). We surface them as
    a single batch so the chat loop can run them in parallel if desired."""


@dataclass
class TurnEnd:
    """The model finished a turn. ``stop_reason`` is provider-specific text
    like "end_turn", "tool_use", or "stop_sequence" — informational only."""
    stop_reason: str | None = None


StreamEvent = TextDelta | ToolUseRequest | TurnEnd
