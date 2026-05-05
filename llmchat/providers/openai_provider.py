"""OpenAI / GPT provider with tool-use support.

OpenAI's chat completions API uses ``tools`` (function specs) and emits
``tool_calls`` on assistant messages. Subsequent messages with ``role:
tool`` and ``tool_call_id`` carry the tool results back. Streaming
deltas can include both ``content`` and partial ``tool_calls``; we
accumulate the tool call name + arguments JSON across deltas.
"""

from __future__ import annotations

import json
import os
from typing import Iterator

from ..mcp.types import (
    StreamEvent, TextDelta, ToolCall, ToolDescriptor,
    ToolUseRequest, TurnEnd,
)
from .base import ChatMessage, Provider, ProviderError, RateLimitError


_KNOWN_MODELS = (
    "gpt-5.5",
    "gpt-5.5-pro",
    "gpt-5.4",
    "gpt-5.4-mini",
    "gpt-5.4-nano",
)


class OpenAIProvider(Provider):
    name = "openai"
    supports_tools = True

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

    # ----- entry point ------------------------------------------------------

    def stream(self, model, messages, system=None, tools=None):
        if not tools:
            yield from self._stream_text_only(model, messages, system)
            return
        yield from self._stream_with_tools(model, messages, system, tools)

    # ----- text-only path ---------------------------------------------------

    def _stream_text_only(self, model, messages, system) -> Iterator[str]:
        client = self._ensure_client()
        api_messages = []
        if system:
            api_messages.append({"role": "system", "content": system})
        for m in messages:
            if m.role in ("user", "assistant"):
                api_messages.append({"role": m.role, "content": m.content})
        try:
            stream = client.chat.completions.create(
                model=model, messages=api_messages, stream=True)
            for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                text = getattr(delta, "content", None)
                if text:
                    yield text
        except Exception as e:
            if getattr(e, "status_code", None) == 429:
                raise RateLimitError(f"OpenAI rate limit: {e}") from e
            raise ProviderError(f"OpenAI API error: {e}") from e

    # ----- tool-calling path ------------------------------------------------

    def _stream_with_tools(self, model, messages, system, tools
                           ) -> Iterator[StreamEvent]:
        client = self._ensure_client()
        api_messages = self._messages_to_openai(messages, system)
        api_tools = [self._tool_to_openai(t) for t in tools]

        try:
            stream = client.chat.completions.create(
                model=model,
                messages=api_messages,
                tools=api_tools,
                stream=True,
            )
        except Exception as e:
            if getattr(e, "status_code", None) == 429:
                raise RateLimitError(f"OpenAI rate limit: {e}") from e
            raise ProviderError(f"OpenAI API error: {e}") from e

        # Accumulate tool_calls across delta chunks. The stream sends
        # incremental updates: first a chunk with the tool name, then
        # chunks with progressively-more-complete arguments JSON.
        # Each tool call is identified by an ``index`` field.
        tool_calls_acc: dict[int, dict] = {}
        finish_reason: str | None = None

        try:
            for chunk in stream:
                if not chunk.choices:
                    continue
                choice = chunk.choices[0]
                delta = choice.delta
                if getattr(delta, "content", None):
                    yield TextDelta(text=delta.content)
                tc_deltas = getattr(delta, "tool_calls", None) or []
                for tc in tc_deltas:
                    idx = getattr(tc, "index", 0)
                    acc = tool_calls_acc.setdefault(
                        idx, {"id": "", "name": "", "arguments": ""})
                    if getattr(tc, "id", None):
                        acc["id"] = tc.id
                    fn = getattr(tc, "function", None)
                    if fn:
                        if getattr(fn, "name", None):
                            acc["name"] = fn.name
                        if getattr(fn, "arguments", None):
                            acc["arguments"] += fn.arguments
                if choice.finish_reason:
                    finish_reason = choice.finish_reason
        except Exception as e:
            if getattr(e, "status_code", None) == 429:
                raise RateLimitError(f"OpenAI rate limit: {e}") from e
            raise ProviderError(f"OpenAI streaming error: {e}") from e

        if tool_calls_acc:
            calls: list[ToolCall] = []
            for idx in sorted(tool_calls_acc.keys()):
                acc = tool_calls_acc[idx]
                try:
                    arguments = json.loads(acc["arguments"]) if acc["arguments"] else {}
                except json.JSONDecodeError:
                    # Some models occasionally emit malformed JSON deltas;
                    # we surface the raw string in a single-key dict so the
                    # downstream tool can still attempt to handle it.
                    arguments = {"_raw": acc["arguments"]}
                calls.append(ToolCall(
                    id=acc["id"], fq_name=acc["name"], arguments=arguments))
            yield ToolUseRequest(calls=calls)
        yield TurnEnd(stop_reason=finish_reason)

    # ----- message translation ---------------------------------------------

    def _messages_to_openai(self, messages: list[ChatMessage],
                             system: str | None) -> list[dict]:
        out: list[dict] = []
        if system:
            out.append({"role": "system", "content": system})
        for m in messages:
            if m.role == "user":
                out.append({"role": "user", "content": m.content})
            elif m.role == "assistant":
                if m.tool_calls:
                    out.append({
                        "role": "assistant",
                        "content": m.content or None,
                        "tool_calls": [
                            {
                                "id": call.id,
                                "type": "function",
                                "function": {
                                    "name": call.fq_name,
                                    "arguments": json.dumps(call.arguments or {}),
                                },
                            }
                            for call in m.tool_calls
                        ],
                    })
                else:
                    out.append({"role": "assistant", "content": m.content})
            elif m.role == "tool":
                # OpenAI uses one tool message per result.
                if not m.tool_results:
                    continue
                for r in m.tool_results:
                    out.append({
                        "role": "tool",
                        "tool_call_id": r.id,
                        "content": r.content,
                    })
        return out

    def _tool_to_openai(self, t: ToolDescriptor) -> dict:
        return {
            "type": "function",
            "function": {
                "name": t.fq_name,
                "description": t.description,
                "parameters": t.input_schema or
                              {"type": "object", "properties": {}},
            },
        }

    # ----- model listing ----------------------------------------------------

    def list_models(self) -> list[str]:
        try:
            client = self._ensure_client()
            resp = client.models.list()
            ids = [
                m.id for m in resp.data
                if m.id.startswith("gpt-") and "audio" not in m.id
                and "image" not in m.id and "realtime" not in m.id
            ]
            return sorted(ids) if ids else list(_KNOWN_MODELS)
        except Exception:
            return list(_KNOWN_MODELS)
