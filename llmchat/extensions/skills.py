"""Skills: markdown files that get conditionally injected into the system prompt.

A skill is a directory with a `SKILL.md` file. The frontmatter (YAML between
``---`` markers at the top of the file) declares metadata; the body is the
instruction text.

Frontmatter fields (compatible with Claude Code's format):

    ---
    name: code-reviewer
    description: Reviews Python code for style and bugs.
    triggers:                   # any of these phrases activates the skill
      - "review this code"
      - "/review"
    auto_invoke: true           # default true — skill activates on trigger match
    file_patterns:              # optional — activate when these files referenced
      - "*.py"
    ---
    You are a senior Python reviewer. When asked to review code, look for:
    - ...

When does a skill activate?
  1. Always-on skills (``auto_invoke: true`` and no triggers/patterns) inject
     into every system prompt.
  2. Trigger skills inject when any of their trigger phrases appears in the
     user message.
  3. Pattern skills inject when the user's message references a matching file
     path (rough heuristic — not bulletproof, but useful).

Skills loaded from project roots override skills with the same name from
user roots.

We deliberately do NOT execute scripts inside skill folders. Some of
Claude Code's skills bundle helper scripts that the model can call via tools;
that crosses into MCP territory and we'd rather have one mechanism for tool
execution than two.
"""

from __future__ import annotations

import fnmatch
import re
from dataclasses import dataclass, field
from pathlib import Path

from .discovery import ExtensionRoots


@dataclass
class Skill:
    name: str
    description: str
    body: str
    triggers: list[str] = field(default_factory=list)
    file_patterns: list[str] = field(default_factory=list)
    auto_invoke: bool = True
    source_path: Path | None = None  # for debugging / `route show` style output

    def is_always_on(self) -> bool:
        """True if this skill should be injected unconditionally."""
        return (self.auto_invoke
                and not self.triggers
                and not self.file_patterns)

    def matches(self, user_text: str) -> bool:
        """Should this skill activate for this user message?"""
        if not self.auto_invoke:
            return False
        if self.is_always_on():
            return True
        text_lower = user_text.lower()
        for trigger in self.triggers:
            if trigger.lower() in text_lower:
                return True
        # Pattern check: extract candidate paths and glob against them.
        for token in _extract_pathlike_tokens(user_text):
            for pattern in self.file_patterns:
                if fnmatch.fnmatch(token, pattern):
                    return True
        return False


# Loose: matches `foo.py`, `src/x.go`, `path/to/file.md`, etc.
_PATH_TOKEN_RE = re.compile(r"\b[\w./-]+\.[a-zA-Z0-9]{1,8}\b")


def _extract_pathlike_tokens(text: str) -> list[str]:
    return _PATH_TOKEN_RE.findall(text)


# ----- frontmatter parser ---------------------------------------------------

# We use a tiny YAML subset parser here rather than pulling PyYAML as a
# dependency. We support strings, booleans, and lists of strings — the only
# things skill frontmatter actually uses in practice. If a skill needs richer
# metadata, the frontmatter can still be parsed manually by the user.

_FRONTMATTER_RE = re.compile(
    r"\A---\s*\n(.*?)\n---\s*\n(.*)\Z", re.DOTALL)


def _parse_frontmatter(text: str) -> tuple[dict, str]:
    """Return (metadata, body). If no frontmatter, metadata is empty."""
    m = _FRONTMATTER_RE.match(text)
    if not m:
        return {}, text
    raw, body = m.group(1), m.group(2)
    return _parse_simple_yaml(raw), body


def _parse_simple_yaml(text: str) -> dict:
    """Parse the tiny YAML subset that skill frontmatter actually uses.

    Supports:
      key: value             (string)
      key: true / false      (bool)
      key:                   (followed by `- item` lines, a list)
        - item

    This is intentionally minimal. If users need full YAML, they can install
    PyYAML and we can swap implementations later.
    """
    result: dict = {}
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            i += 1
            continue
        if ":" not in line:
            i += 1
            continue
        key, _, value = line.partition(":")
        key = key.strip()
        value = value.strip()
        if value:
            # Inline scalar.
            result[key] = _coerce_scalar(value)
            i += 1
            continue
        # Empty value -> may be a list on subsequent lines.
        items: list[str] = []
        j = i + 1
        while j < len(lines):
            nxt = lines[j]
            nstripped = nxt.strip()
            if not nstripped:
                j += 1
                continue
            if nstripped.startswith("- "):
                items.append(_strip_quotes(nstripped[2:].strip()))
                j += 1
                continue
            break
        result[key] = items
        i = j
    return result


def _coerce_scalar(s: str):
    s = _strip_quotes(s)
    low = s.lower()
    if low in ("true", "yes"):
        return True
    if low in ("false", "no"):
        return False
    return s


def _strip_quotes(s: str) -> str:
    if (len(s) >= 2 and s[0] == s[-1] and s[0] in ("'", '"')):
        return s[1:-1]
    return s


# ----- registry -------------------------------------------------------------

@dataclass
class SkillRegistry:
    skills: dict[str, Skill] = field(default_factory=dict)

    def all(self) -> list[Skill]:
        return list(self.skills.values())

    def select_for(self, user_text: str) -> list[Skill]:
        """Skills that should activate for this turn."""
        return [s for s in self.skills.values() if s.matches(user_text)]

    def render_system_prompt_addition(
            self, active: list[Skill]) -> str:
        """Combine active skills into a block to append to the system prompt.

        Each skill is wrapped in a clearly-delimited section so the model
        can tell them apart. We include the description as a header.
        """
        if not active:
            return ""
        parts = ["", "# Active skills"]
        for skill in active:
            parts.append("")
            parts.append(f"## {skill.name}")
            if skill.description:
                parts.append(f"_{skill.description.strip()}_")
                parts.append("")
            parts.append(skill.body.strip())
        return "\n".join(parts)


def load_skills(roots: ExtensionRoots) -> SkillRegistry:
    """Walk the roots and load every SKILL.md found.

    Later roots override earlier ones (project beats user) when names collide.
    """
    registry = SkillRegistry()
    for root in roots.existing():
        skills_dir = root / "skills"
        if not skills_dir.is_dir():
            continue
        for entry in sorted(skills_dir.iterdir()):
            if not entry.is_dir():
                continue
            skill_md = entry / "SKILL.md"
            if not skill_md.is_file():
                continue
            try:
                skill = _load_skill_file(skill_md)
            except Exception as e:
                # Don't let one broken skill break startup; just warn.
                import sys
                print(f"[llmchat] failed to load skill {skill_md}: {e}",
                      file=sys.stderr)
                continue
            registry.skills[skill.name] = skill
    return registry


def _load_skill_file(path: Path) -> Skill:
    text = path.read_text(encoding="utf-8")
    meta, body = _parse_frontmatter(text)
    name = meta.get("name") or path.parent.name
    return Skill(
        name=str(name),
        description=str(meta.get("description", "")),
        body=body.strip(),
        triggers=_as_list(meta.get("triggers")),
        file_patterns=_as_list(meta.get("file_patterns")),
        auto_invoke=bool(meta.get("auto_invoke", True)),
        source_path=path,
    )


def _as_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value]
    return [str(value)]
