"""MCPManager: a synchronous facade over the async MCP Python SDK.

The MCP SDK is async (asyncio + anyio). The rest of llmchat is sync. Rather
than rewrite the whole app in async, we run an asyncio event loop in a
background thread; every public method here submits a coroutine to that
loop and blocks for the result.

Lifecycle:

    manager = MCPManager(server_configs)
    manager.start()                  # spins up loop thread, connects servers
    tools = manager.list_tools()     # ToolDescriptor list
    result = manager.call_tool("server::tool", {"x": 1})
    manager.stop()                   # disconnect, join thread

Server config format (a dict, typically loaded from .mcp.json or mcp.json):

    {
      "mcpServers": {
        "filesystem": {
          "command": "npx",
          "args": ["-y", "@modelcontextprotocol/server-filesystem", "/path"]
        },
        "weather": {
          "type": "http",
          "url": "https://example.com/mcp"
        }
      }
    }

Stdio is the default. HTTP/SSE transports are also supported when the
``type`` field is present.

Failure model: if a single server fails to connect or list tools, we log
the error and continue with whatever did work. We do NOT crash the chat
because one server is misconfigured.
"""

from __future__ import annotations

import asyncio
import json
import sys
import threading
from concurrent.futures import Future
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .types import ToolCall, ToolDescriptor, ToolResult

# We import the SDK lazily so 'mcp' becomes an optional dependency: a user
# without any MCP servers configured shouldn't be required to install it.
_mcp_import_error: Exception | None = None


def _try_import_mcp():
    global _mcp_import_error
    try:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client
        return ClientSession, StdioServerParameters, stdio_client
    except ImportError as e:
        _mcp_import_error = e
        return None, None, None


def _try_import_streamable_http():
    try:
        from mcp.client.streamable_http import streamablehttp_client
        return streamablehttp_client
    except ImportError:
        return None


@dataclass
class _ServerConn:
    """Holds the ClientSession for one connected server, plus its tools."""
    name: str
    session: Any  # ClientSession
    tools: list[ToolDescriptor] = field(default_factory=list)
    exit_stack: Any = None  # AsyncExitStack — owns the transport context


class MCPManager:
    """Synchronous facade. Safe to call from the main thread."""

    def __init__(self, server_configs: dict[str, dict]) -> None:
        """``server_configs`` is the ``mcpServers`` mapping from .mcp.json."""
        self._configs = server_configs or {}
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()
        self._connections: dict[str, _ServerConn] = {}
        self._stack: AsyncExitStack | None = None
        self._connect_errors: dict[str, str] = {}

    # ----- lifecycle --------------------------------------------------------

    def start(self) -> None:
        """Spin up the loop thread and connect to every configured server.

        Returns once all connection attempts have settled (success or fail).
        """
        if not self._configs:
            return

        ClientSession, _, _ = _try_import_mcp()
        if ClientSession is None:
            print(f"[llmchat] MCP servers configured but 'mcp' package is "
                  f"not installed. Install with: pip install mcp",
                  file=sys.stderr)
            return

        self._thread = threading.Thread(
            target=self._run_loop, name="llmchat-mcp", daemon=True)
        self._thread.start()
        self._ready.wait(timeout=5.0)

        # Connect to all servers. Errors are recorded but don't abort.
        try:
            self._submit(self._connect_all()).result(timeout=60.0)
        except Exception as e:
            print(f"[llmchat] MCP startup error: {e}", file=sys.stderr)

    def stop(self) -> None:
        """Disconnect from every server and shut the loop down."""
        if self._loop is None:
            return
        try:
            self._submit(self._disconnect_all()).result(timeout=5.0)
        except Exception:
            pass
        self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread:
            self._thread.join(timeout=2.0)
        self._loop = None
        self._thread = None

    # ----- public sync API --------------------------------------------------

    def list_tools(self) -> list[ToolDescriptor]:
        out: list[ToolDescriptor] = []
        for conn in self._connections.values():
            out.extend(conn.tools)
        return out

    def server_status(self) -> dict[str, str]:
        """Per-server status string: 'connected (N tools)' or an error msg.

        Useful for `llmchat plugins list` / status displays."""
        status: dict[str, str] = {}
        for name in self._configs:
            if name in self._connect_errors:
                status[name] = f"error: {self._connect_errors[name]}"
            elif name in self._connections:
                tcount = len(self._connections[name].tools)
                status[name] = f"connected ({tcount} tools)"
            else:
                status[name] = "not connected"
        return status

    def call_tool(self, fq_name: str, arguments: dict) -> ToolResult:
        """Synchronously execute a tool and return its result."""
        if "::" not in fq_name:
            return ToolResult(id="", fq_name=fq_name,
                              content=f"invalid tool name {fq_name!r}",
                              is_error=True)
        server, _, tool_name = fq_name.partition("::")
        conn = self._connections.get(server)
        if conn is None:
            return ToolResult(id="", fq_name=fq_name,
                              content=f"unknown MCP server {server!r}",
                              is_error=True)
        try:
            return self._submit(
                self._call_tool_async(conn, tool_name, fq_name, arguments)
            ).result(timeout=120.0)
        except Exception as e:
            return ToolResult(id="", fq_name=fq_name,
                              content=f"tool call failed: {e}",
                              is_error=True)

    # ----- internals: event loop --------------------------------------------

    def _run_loop(self) -> None:
        loop = asyncio.new_event_loop()
        self._loop = loop
        asyncio.set_event_loop(loop)
        self._ready.set()
        try:
            loop.run_forever()
        finally:
            try:
                pending = asyncio.all_tasks(loop)
                for t in pending:
                    t.cancel()
                loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            except Exception:
                pass
            loop.close()

    def _submit(self, coro) -> Future:
        assert self._loop is not None
        return asyncio.run_coroutine_threadsafe(coro, self._loop)

    # ----- internals: connection management ---------------------------------

    async def _connect_all(self) -> None:
        self._stack = AsyncExitStack()
        await self._stack.__aenter__()
        for name, cfg in self._configs.items():
            try:
                conn = await self._connect_one(name, cfg)
                self._connections[name] = conn
            except Exception as e:
                self._connect_errors[name] = str(e)
                print(f"[llmchat] MCP server {name!r} failed to connect: {e}",
                      file=sys.stderr)

    async def _connect_one(self, name: str, cfg: dict) -> _ServerConn:
        ClientSession, StdioServerParameters, stdio_client = _try_import_mcp()
        assert ClientSession is not None

        transport_type = cfg.get("type", "stdio")
        per_server_stack = AsyncExitStack()
        await per_server_stack.__aenter__()
        # Hand the per-server stack to the parent so the parent stack
        # closes everything in reverse order on shutdown.
        assert self._stack is not None
        await self._stack.enter_async_context(per_server_stack)

        if transport_type == "stdio":
            command = cfg.get("command")
            if not command:
                raise ValueError("stdio server config missing 'command'")
            args = cfg.get("args", []) or []
            env = cfg.get("env") or None
            params = StdioServerParameters(command=command, args=list(args),
                                           env=env)
            read, write = await per_server_stack.enter_async_context(
                stdio_client(params))
        elif transport_type in ("http", "streamable_http"):
            streamablehttp_client = _try_import_streamable_http()
            if streamablehttp_client is None:
                raise RuntimeError(
                    "HTTP MCP transport requires a newer mcp SDK")
            url = cfg.get("url")
            if not url:
                raise ValueError("http server config missing 'url'")
            headers = cfg.get("headers") or {}
            transport = await per_server_stack.enter_async_context(
                streamablehttp_client(url, headers=headers))
            # streamablehttp_client returns (read, write, _session_id_callback)
            read, write = transport[0], transport[1]
        else:
            raise ValueError(f"unsupported MCP transport: {transport_type!r}")

        session = await per_server_stack.enter_async_context(
            ClientSession(read, write))
        await session.initialize()

        # List tools.
        tools_result = await session.list_tools()
        tools: list[ToolDescriptor] = []
        for t in tools_result.tools:
            tools.append(ToolDescriptor(
                fq_name=f"{name}::{t.name}",
                server=name,
                tool_name=t.name,
                description=t.description or "",
                input_schema=_normalize_schema(t.inputSchema),
            ))
        return _ServerConn(name=name, session=session, tools=tools,
                           exit_stack=per_server_stack)

    async def _disconnect_all(self) -> None:
        if self._stack is None:
            return
        try:
            await self._stack.__aexit__(None, None, None)
        finally:
            self._stack = None
            self._connections.clear()

    async def _call_tool_async(self, conn: _ServerConn, tool_name: str,
                               fq_name: str, arguments: dict) -> ToolResult:
        try:
            result = await conn.session.call_tool(tool_name, arguments)
        except Exception as e:
            return ToolResult(id="", fq_name=fq_name,
                              content=f"tool call raised: {e}",
                              is_error=True)
        # MCP returns a CallToolResult with a content list of blocks. Each
        # block can be text, image, or resource. We flatten to plain text;
        # images and resources we describe by metadata since we don't have
        # a multimodal pipeline (yet).
        chunks: list[str] = []
        for block in (result.content or []):
            block_type = getattr(block, "type", None)
            if block_type == "text":
                chunks.append(getattr(block, "text", "") or "")
            elif block_type == "image":
                chunks.append("[image returned by tool — not displayed]")
            elif block_type == "resource":
                ref = getattr(block, "resource", None)
                uri = getattr(ref, "uri", "") if ref else ""
                chunks.append(f"[resource: {uri}]")
            else:
                # Unknown block type; serialize what we can.
                try:
                    chunks.append(json.dumps(block.model_dump(), default=str))
                except Exception:
                    chunks.append(str(block))
        is_error = bool(getattr(result, "isError", False))
        return ToolResult(id="", fq_name=fq_name,
                          content="\n".join(chunks).strip() or "(no output)",
                          is_error=is_error)


def _normalize_schema(schema: Any) -> dict:
    """Coerce an MCP tool schema into a plain dict.

    The SDK may return a Pydantic model or a dict; we normalize so the
    provider adapters always see a dict.
    """
    if schema is None:
        return {"type": "object", "properties": {}}
    if isinstance(schema, dict):
        return schema
    # Pydantic v2 model.
    if hasattr(schema, "model_dump"):
        return schema.model_dump()
    # Last resort.
    try:
        return dict(schema)
    except Exception:
        return {"type": "object", "properties": {}}


# ----- config loading -------------------------------------------------------

def load_mcp_configs(paths: list[Path]) -> dict[str, dict]:
    """Load and merge MCP server configs from one or more JSON files.

    Conflicting names are resolved last-wins, so a project-level config can
    override a user-level one. Missing files are silently skipped.
    """
    merged: dict[str, dict] = {}
    for p in paths:
        if not p.is_file():
            continue
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[llmchat] failed to parse {p}: {e}", file=sys.stderr)
            continue
        servers = data.get("mcpServers") or data.get("mcp_servers") or {}
        if not isinstance(servers, dict):
            print(f"[llmchat] {p}: 'mcpServers' must be a dict", file=sys.stderr)
            continue
        for name, cfg in servers.items():
            if isinstance(cfg, dict):
                merged[name] = cfg
    return merged
