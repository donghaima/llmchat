"""Anthropic / Claude provider with tool-use support.

Tool conversation flow (Anthropic protocol):

  1. We send messages + tool definitions.
  2. Model streams text deltas, then emits one or more tool_use blocks at
     the end. ``stop_reason`` is "tool_use" when the model wants to call
     tools, "end_turn" when it's done.
  3. Caller (the chat loop) executes the tools.
  4. Caller calls us again with the updated messages: the assistant's
     previous turn (with its tool_use blocks) plus a synthetic user message
     containing tool_result blocks for each call.
  5. Repeat until stop_reason="end_turn".

We translate our internal ``ChatMessage`` (with optional ``tool_calls`` and
``tool_results``) into Anthropic's wire format on each request.
"""

from __future__ import annotations

import os
from typing import Iterator

from ..mcp.types import (
    StreamEvent, TextDelta, ToolCall, ToolDescriptor,
    ToolUseRequest, TurnEnd,
)
from .base import ChatMessage, Provider, ProviderError, RateLimitError


_KNOWN_MODELS = (
    "claude-opus-4-7",
    "claude-opus-4-6",
    "claude-sonnet-4-6",
    "claude-haiku-4-5",
)


class AnthropicProvider(Provider):
    name = "anthropic"
    supports_tools = True

    def __init__(self) -> None:
        self._client = None

    def _ensure_client(self):
        if self._client is not None:
            return self._client
        try:
            from anthropic import Anthropic
        except ImportError as e:
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

    # ----- entry point ------------------------------------------------------

    def stream(self, model, messages, system=None, tools=None):
        if not tools:
            yield from self._stream_text_only(model, messages, system)
            return
        yield from self._stream_with_tools(model, messages, system, tools)

    # ----- text-only path ---------------------------------------------------

    def _stream_text_only(self, model, messages, system) -> Iterator[str]:
        client = self._ensure_client()
        api_messages = [
            {"role": m.role, "content": m.content}
            for m in messages if m.role in ("user", "assistant")
        ]
        kwargs = {"model": model, "max_tokens": 4096,
                  "messages": api_messages}
        if system:
            kwargs["system"] = system
        try:
            with client.messages.stream(**kwargs) as stream:
                for text in stream.text_stream:
                    yield text
        except Exception as e:
            if getattr(e, "status_code", None) == 429:
                raise RateLimitError(f"Anthropic rate limit: {e}") from e
            raise ProviderError(f"Anthropic API error: {e}") from e

    # ----- tool-calling path ------------------------------------------------

    def _stream_with_tools(self, model: str,
                           messages: list[ChatMessage],
                           system: str | None,
                           tools: list[ToolDescriptor]
                           ) -> Iterator[StreamEvent]:
        client = self._ensure_client()
        api_messages = self._messages_to_anthropic(messages)
        api_tools = [self._tool_to_anthropic(t) for t in tools]
        kwargs = {
            "model": model,
            "max_tokens": 4096,
            "messages": api_messages,
            "tools": api_tools,
        }
        if system:
            kwargs["system"] = system

        try:
            with client.messages.stream(**kwargs) as stream:
                for event in stream:
                    et = getattr(event, "type", None)
                    if et == "content_block_delta":
                        delta = getattr(event, "delta", None)
                        if delta and getattr(delta, "type", None) == "text_delta":
                            yield TextDelta(text=delta.text or "")
                    elif et == "message_stop":
                        break
                final = stream.get_final_message()
                tool_uses: list[ToolCall] = []
                for block in final.content:
                    if getattr(block, "type", None) == "tool_use":
                        tool_uses.append(ToolCall(
                            id=block.id,
                            fq_name=block.name,
                            arguments=dict(block.input or {}),
                        ))
                if tool_uses:
                    yield ToolUseRequest(calls=tool_uses)
                yield TurnEnd(stop_reason=getattr(final, "stop_reason", None))
        except Exception as e:
            if getattr(e, "status_code", None) == 429:
                raise RateLimitError(f"Anthropic rate limit: {e}") from e
            raise ProviderError(f"Anthropic API error: {e}") from e

    # ----- message translation ---------------------------------------------

    def _messages_to_anthropic(self, messages: list[ChatMessage]) -> list[dict]:
        """Translate internal messages into Anthropic's wire format.

        Special handling:
          - assistant messages with ``tool_calls`` produce a ``content`` list
            mixing text + tool_use blocks.
          - tool-role messages (``role == "tool"``) produce a synthetic user
            message with tool_result blocks.
        """
        out: list[dict] = []
        for m in messages:
            if m.role == "user":
                out.append({"role": "user", "content": m.content})
            elif m.role == "assistant":
                if m.tool_calls:
                    blocks: list[dict] = []
                    if m.content:
                        blocks.append({"type": "text", "text": m.content})
                    for call in m.tool_calls:
                        blocks.append({
                            "type": "tool_use",
                            "id": call.id,
                            "name": call.fq_name,
                            "input": call.arguments or {},
                        })
                    out.append({"role": "assistant", "content": blocks})
                else:
                    out.append({"role": "assistant", "content": m.content})
            elif m.role == "tool":
                # The chat loop wraps a batch of tool results in a single
                # ChatMessage(role="tool"). Convert to a user message with
                # tool_result blocks.
                if not m.tool_results:
                    continue
                blocks = [
                    {
                        "type": "tool_result",
                        "tool_use_id": r.id,
                        "content": r.content,
                        "is_error": r.is_error,
                    }
                    for r in m.tool_results
                ]
                out.append({"role": "user", "content": blocks})
            # system role is handled via the top-level kwarg, not in messages.
        return out

    def _tool_to_anthropic(self, t: ToolDescriptor) -> dict:
        return {
            "name": t.fq_name,
            "description": t.description,
            "input_schema": t.input_schema or
                            {"type": "object", "properties": {}},
        }

    # ----- model listing ----------------------------------------------------

    def list_models(self) -> list[str]:
        try:
            client = self._ensure_client()
            resp = client.models.list(limit=50)
            ids = [m.id for m in resp.data]
            return ids or list(_KNOWN_MODELS)
        except Exception:
            return list(_KNOWN_MODELS)
