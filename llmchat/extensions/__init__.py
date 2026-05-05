"""Extensions: skills, slash commands, hooks, plugins.

These are non-MCP customizations — file-based, no network, no tool execution.
The MCP support lives in `llmchat.mcp` because it has fundamentally different
lifecycle requirements (long-lived processes, async I/O, tool-call loops).

A "plugin" here is just a directory that bundles skills, commands, and/or
hook scripts together. Plugins can also declare MCP server configs (see
`plugins.py`), which the MCP manager picks up.

Discovery walks two roots:
  ~/.llmchat/ and ./.llmchat/  — our native format
  ~/.claude/ and ./.claude/    — Claude Code compatibility
The two formats are nearly identical; differences are isolated in the
parsers below.
"""

from .commands import SlashCommand, load_commands
from .discovery import ExtensionRoots, discover_roots
from .hooks import HookEvent, HookRegistry, load_hooks
from .plugins import Plugin, load_plugins
from .skills import Skill, SkillRegistry, load_skills

__all__ = [
    "ExtensionRoots", "discover_roots",
    "Skill", "SkillRegistry", "load_skills",
    "SlashCommand", "load_commands",
    "HookEvent", "HookRegistry", "load_hooks",
    "Plugin", "load_plugins",
]
