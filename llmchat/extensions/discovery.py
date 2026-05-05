"""Find extension roots on disk.

We look in four places, in this order of precedence (later wins):

  1. ~/.claude/      — Claude Code user-level
  2. ~/.llmchat/     — llmchat user-level
  3. ./.claude/      — Claude Code project-level
  4. ./.llmchat/     — llmchat project-level

Project roots override user roots so a project can pin its own version of a
skill or command. Within each root we look for these subdirs:

  skills/    — one folder per skill, each with SKILL.md
  commands/  — one .md file per slash command
  hooks/     — one .json or .toml file per hook (or hooks.toml, plural)
  plugins/   — one folder per plugin, each with a manifest

A root may also contain an `mcp.json` (or `.mcp.json`, Claude convention)
declaring MCP servers; that's loaded by the MCP package, not here.

A root that doesn't exist or is empty is silently skipped — no errors.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ExtensionRoots:
    """The roots we'll search, in order. Later entries override earlier ones."""
    paths: list[Path] = field(default_factory=list)

    def existing(self) -> list[Path]:
        """Only the roots that actually exist on disk."""
        return [p for p in self.paths if p.is_dir()]


def discover_roots(project_dir: Path | None = None,
                   home_dir: Path | None = None) -> ExtensionRoots:
    """Build the canonical search order for extension roots.

    `project_dir` defaults to the current working directory. `home_dir`
    defaults to `~`. Both are exposed for testing.
    """
    project = project_dir or Path.cwd()
    home = home_dir or Path.home()
    paths = [
        home / ".claude",
        home / ".llmchat",
        project / ".claude",
        project / ".llmchat",
    ]
    # Don't include the same path twice if project == home, which would be
    # weird but we shouldn't crash on it.
    seen: set[Path] = set()
    deduped = []
    for p in paths:
        rp = p.resolve() if p.exists() else p
        if rp in seen:
            continue
        seen.add(rp)
        deduped.append(p)
    return ExtensionRoots(paths=deduped)
