"""Google / Gemini provider with tool-use support.

Gemini's tool calling uses ``Tool`` objects with ``function_declarations``;
the model emits ``functionCall`` parts that we translate to ``ToolCall``.
Tool results go back as ``functionResponse`` parts on a model-role message.
A few important details:

  - Roles are "user" and "model" (not "assistant").
  - One model turn can include multiple ``functionCall`` parts (parallel calls).
  - Function call/response IDs are not part of the protocol — Gemini uses
    the function NAME to correlate. We ignore call IDs on the way in and
    use ``fq_name`` to match results.
"""

from __future__ import annotations

import json
import os
from typing import Iterator

from ..mcp.types import (
    StreamEvent, TextDelta, ToolCall, ToolDescriptor,
    ToolUseRequest, TurnEnd,
)
from .base import ChatMessage, Provider, ProviderError


_KNOWN_MODELS = (
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
)


class GoogleProvider(Provider):
    name = "google"
    supports_tools = True

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

    # ----- entry point ------------------------------------------------------

    def stream(self, model, messages, system=None, tools=None):
        if not tools:
            yield from self._stream_text_only(model, messages, system)
            return
        yield from self._stream_with_tools(model, messages, system, tools)

    # ----- text-only path ---------------------------------------------------

    def _stream_text_only(self, model, messages, system) -> Iterator[str]:
        client = self._ensure_client()
        contents = self._messages_to_gemini_contents(messages)
        from google.genai import types
        config = None
        if system:
            config = types.GenerateContentConfig(system_instruction=system)
        try:
            stream = client.models.generate_content_stream(
                model=model, contents=contents, config=config)
            for chunk in stream:
                text = getattr(chunk, "text", None)
                if text:
                    yield text
        except Exception as e:
            raise ProviderError(f"Google API error: {e}") from e

    # ----- tool-calling path ------------------------------------------------

    def _stream_with_tools(self, model, messages, system, tools
                           ) -> Iterator[StreamEvent]:
        client = self._ensure_client()
        from google.genai import types

        contents = self._messages_to_gemini_contents(messages)
        gemini_tool = types.Tool(function_declarations=[
            self._tool_to_gemini(t) for t in tools
        ])
        config = types.GenerateContentConfig(
            tools=[gemini_tool],
            system_instruction=system,
        )

        finish_reason: str | None = None
        captured_calls: list[ToolCall] = []
        try:
            stream = client.models.generate_content_stream(
                model=model, contents=contents, config=config)
            for chunk in stream:
                # text deltas
                text = getattr(chunk, "text", None)
                if text:
                    yield TextDelta(text=text)
                # function calls — these may appear at any chunk
                for cand in getattr(chunk, "candidates", []) or []:
                    content = getattr(cand, "content", None)
                    parts = getattr(content, "parts", None) or [] if content else []
                    for part in parts:
                        fc = getattr(part, "function_call", None)
                        if fc and getattr(fc, "name", None):
                            args = dict(getattr(fc, "args", {}) or {})
                            captured_calls.append(ToolCall(
                                id=getattr(fc, "id", "") or fc.name,
                                fq_name=fc.name,
                                arguments=args,
                            ))
                    fr = getattr(cand, "finish_reason", None)
                    if fr is not None:
                        finish_reason = str(fr)
        except Exception as e:
            raise ProviderError(f"Google API error: {e}") from e

        if captured_calls:
            yield ToolUseRequest(calls=captured_calls)
        yield TurnEnd(stop_reason=finish_reason)

    # ----- message translation ---------------------------------------------

    def _messages_to_gemini_contents(self, messages):
        from google.genai import types
        contents = []
        for m in messages:
            if m.role == "user":
                contents.append({"role": "user",
                                 "parts": [{"text": m.content}]})
            elif m.role == "assistant":
                parts: list[dict] = []
                if m.content:
                    parts.append({"text": m.content})
                for call in (m.tool_calls or []):
                    parts.append({
                        "function_call": {
                            "name": call.fq_name,
                            "args": call.arguments or {},
                        }
                    })
                contents.append({"role": "model", "parts": parts})
            elif m.role == "tool":
                # Gemini wants function responses on a "user"-role message
                # with functionResponse parts. (The SDK accepts either user
                # or function role on the way back; user is widely accepted.)
                if not m.tool_results:
                    continue
                parts = [
                    {
                        "function_response": {
                            "name": r.fq_name,
                            "response": {"content": r.content,
                                         "is_error": r.is_error},
                        }
                    }
                    for r in m.tool_results
                ]
                contents.append({"role": "user", "parts": parts})
        return contents

    def _tool_to_gemini(self, t: ToolDescriptor):
        # Gemini's FunctionDeclaration accepts a ``parameters`` dict in the
        # same shape as JSON Schema, with one wrinkle: Gemini doesn't
        # accept some advanced JSON Schema fields. We pass through the
        # schema as-is and let the SDK reject anything it can't handle —
        # which beats trying to second-guess the schema at translation time.
        from google.genai import types
        return types.FunctionDeclaration(
            name=t.fq_name,
            description=t.description,
            parameters=t.input_schema or {"type": "OBJECT", "properties": {}},
        )

    # ----- model listing ----------------------------------------------------

    def list_models(self) -> list[str]:
        try:
            client = self._ensure_client()
            models = client.models.list()
            ids: list[str] = []
            for m in models:
                name = getattr(m, "name", "") or ""
                if name.startswith("models/"):
                    name = name[len("models/"):]
                if "gemini" in name and "embedding" not in name:
                    ids.append(name)
            return sorted(set(ids)) if ids else list(_KNOWN_MODELS)
        except Exception:
            return list(_KNOWN_MODELS)
