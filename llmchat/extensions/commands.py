"""Slash commands: prompt templates triggered by /name.

A slash command is a single ``.md`` file in a ``commands/`` directory.
Optional frontmatter; the body is the prompt template. ``$ARGS`` in the body
is replaced with the text the user typed after ``/name``.

Example: ``commands/pr.md``::

    ---
    description: Open a pull request for the current branch.
    ---
    Run `git diff main` and summarize the changes. Then propose a PR title
    and description, including: $ARGS

User invocation:

    /pr fix the auth race condition

Becomes a regular user message with $ARGS replaced. The model sees the
expanded prompt as if the user had typed it.

We deliberately do NOT make slash commands themselves invoke tools or run
scripts. That's MCP territory. If a user wants a command that runs `git`,
they should use an MCP server that exposes a `git` tool, then write a
slash command that asks the model to use it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from .discovery import ExtensionRoots
from .skills import _parse_frontmatter


@dataclass
class SlashCommand:
    name: str          # without the leading slash
    description: str
    template: str
    source_path: Path | None = None

    def expand(self, args: str) -> str:
        """Substitute $ARGS in the template with the user-supplied text."""
        return self.template.replace("$ARGS", args.strip())


def load_commands(roots: ExtensionRoots) -> dict[str, SlashCommand]:
    """Walk roots and collect commands. Project roots override user roots."""
    commands: dict[str, SlashCommand] = {}
    for root in roots.existing():
        cmd_dir = root / "commands"
        if not cmd_dir.is_dir():
            continue
        for entry in sorted(cmd_dir.iterdir()):
            if not entry.is_file() or entry.suffix.lower() != ".md":
                continue
            try:
                cmd = _load_command_file(entry)
            except Exception as e:
                import sys
                print(f"[llmchat] failed to load command {entry}: {e}",
                      file=sys.stderr)
                continue
            commands[cmd.name] = cmd
    return commands


_NAME_SAFE = re.compile(r"^[A-Za-z0-9_-]+$")


def _load_command_file(path: Path) -> SlashCommand:
    text = path.read_text(encoding="utf-8")
    meta, body = _parse_frontmatter(text)
    name = meta.get("name") or path.stem
    name = str(name)
    if not _NAME_SAFE.match(name):
        raise ValueError(
            f"Command name {name!r} from {path} contains illegal characters; "
            f"use only letters, digits, dashes, and underscores")
    return SlashCommand(
        name=name,
        description=str(meta.get("description", "")),
        template=body.strip(),
        source_path=path,
    )
