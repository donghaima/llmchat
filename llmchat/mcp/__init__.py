"""MCP client + tool-calling support.

Exposes a synchronous facade over the async MCP Python SDK so the rest of
llmchat (which is synchronous) can use it without going async-everywhere.

  ToolDescriptor   normalized tool metadata (name, description, schema)
  ToolCall         a request from the model to call a tool
  ToolResult       the response we feed back to the model
  MCPManager       owns the connections, lists tools, executes calls
  ApprovalState    once-per-session approval tracking
"""

from .approval import ApprovalState
from .manager import MCPManager
from .types import ToolCall, ToolDescriptor, ToolResult

__all__ = [
    "ApprovalState",
    "MCPManager",
    "ToolCall",
    "ToolDescriptor",
    "ToolResult",
]
