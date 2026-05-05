"""Hooks: scripts run at chat lifecycle events.

We support a small, deliberately limited set of events:

  pre_user_message    Before the user's text is sent to the model. Hooks
                      can transform the text by writing to stdout (used as
                      the new user message) or block it by exiting non-zero.
  post_assistant_message  After the model's response. Hooks see the response
                      on stdin; their stdout is informational only.
  pre_tool_call       Before an MCP tool call. Hooks see a JSON blob with
                      tool name + args on stdin. Non-zero exit blocks the call.
  post_tool_call      After a tool result. Informational only.

Hooks are declared in a `hooks.toml` in any extension root, like:

    [[hooks]]
    event = "pre_user_message"
    command = ["./scripts/redact-secrets.sh"]
    timeout_seconds = 5

The command is run as a subprocess. We don't sandbox; the user owns the
filesystem these scripts come from. Hooks should be small and fast.

Hooks are mostly out of scope for the initial version — this module
provides the registry and dispatch, and we wire pre_user_message and
post_assistant_message into the chat loop. Tool hooks will be wired in
when the MCP loop is added.
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Iterable

from .discovery import ExtensionRoots


class HookEvent(str, Enum):
    PRE_USER_MESSAGE = "pre_user_message"
    POST_ASSISTANT_MESSAGE = "post_assistant_message"
    PRE_TOOL_CALL = "pre_tool_call"
    POST_TOOL_CALL = "post_tool_call"


@dataclass
class Hook:
    event: HookEvent
    command: list[str]
    timeout_seconds: float = 10.0
    cwd: Path | None = None  # the root the hook was loaded from
    source_path: Path | None = None


@dataclass
class HookResult:
    """Outcome of running a hook on some payload."""
    blocked: bool = False
    transformed_payload: str | None = None
    stdout: str = ""
    stderr: str = ""


@dataclass
class HookRegistry:
    by_event: dict[HookEvent, list[Hook]] = field(default_factory=dict)

    def for_event(self, event: HookEvent) -> list[Hook]:
        return self.by_event.get(event, [])

    def all(self) -> list[Hook]:
        out: list[Hook] = []
        for hooks in self.by_event.values():
            out.extend(hooks)
        return out

    def dispatch(self, event: HookEvent, payload: str) -> HookResult:
        """Run all hooks for an event in order. First one to block wins.

        For events that allow transformation (pre_*), the *first hook's*
        non-empty stdout is used as the transformed payload. We don't
        chain transformations; that's confusing and error-prone.
        """
        result = HookResult()
        current_payload = payload
        for hook in self.for_event(event):
            outcome = _run_hook(hook, current_payload)
            result.stdout += outcome.stdout
            result.stderr += outcome.stderr
            if outcome.blocked:
                result.blocked = True
                return result
            if (event in (HookEvent.PRE_USER_MESSAGE,
                          HookEvent.PRE_TOOL_CALL)
                    and outcome.transformed_payload
                    and result.transformed_payload is None):
                result.transformed_payload = outcome.transformed_payload
                current_payload = outcome.transformed_payload
        return result


def _run_hook(hook: Hook, payload: str) -> HookResult:
    """Run a single hook. Block on non-zero exit; transform on stdout."""
    try:
        proc = subprocess.run(
            hook.command,
            input=payload,
            capture_output=True,
            text=True,
            timeout=hook.timeout_seconds,
            cwd=str(hook.cwd) if hook.cwd else None,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return HookResult(blocked=True,
                          stderr=f"hook timed out: {' '.join(hook.command)}\n")
    except FileNotFoundError as e:
        # Don't abort the whole chat over a misconfigured hook; just warn.
        return HookResult(stderr=f"hook command not found: {e}\n")
    except Exception as e:
        return HookResult(stderr=f"hook failed to run: {e}\n")

    if proc.returncode != 0:
        return HookResult(blocked=True,
                          stdout=proc.stdout,
                          stderr=proc.stderr or
                          f"hook exited {proc.returncode}\n")
    transformed = proc.stdout.rstrip("\n") if proc.stdout else None
    return HookResult(transformed_payload=transformed,
                      stdout=proc.stdout, stderr=proc.stderr)


# ----- loader ---------------------------------------------------------------

def load_hooks(roots: ExtensionRoots) -> HookRegistry:
    """Read `hooks.toml` from each root, in order."""
    registry = HookRegistry()
    if sys.version_info >= (3, 11):
        import tomllib as _toml  # type: ignore[import-not-found]
    else:
        try:
            import tomli as _toml  # type: ignore[import-not-found]
        except ImportError:
            return registry  # Without TOML support, hooks are simply skipped.

    for root in roots.existing():
        hooks_file = root / "hooks.toml"
        if not hooks_file.is_file():
            continue
        try:
            with open(hooks_file, "rb") as f:
                data = _toml.load(f)
        except Exception as e:
            print(f"[llmchat] failed to parse {hooks_file}: {e}",
                  file=sys.stderr)
            continue
        for entry in data.get("hooks", []):
            try:
                hook = _build_hook(entry, root, hooks_file)
            except Exception as e:
                print(f"[llmchat] bad hook in {hooks_file}: {e}",
                      file=sys.stderr)
                continue
            registry.by_event.setdefault(hook.event, []).append(hook)
    return registry


def _build_hook(entry: dict, root: Path, source: Path) -> Hook:
    event_name = entry.get("event")
    if not event_name:
        raise ValueError("hook is missing 'event'")
    try:
        event = HookEvent(event_name)
    except ValueError:
        raise ValueError(f"unknown event {event_name!r}")
    cmd = entry.get("command")
    if not cmd:
        raise ValueError("hook is missing 'command'")
    if isinstance(cmd, str):
        cmd = [cmd]
    if not isinstance(cmd, list) or not all(isinstance(x, str) for x in cmd):
        raise ValueError("'command' must be a string or list of strings")
    return Hook(
        event=event,
        command=list(cmd),
        timeout_seconds=float(entry.get("timeout_seconds", 10.0)),
        cwd=root,
        source_path=source,
    )
