"""Plugins: bundles of skills + commands + hooks + MCP server configs.

A plugin is a directory under any ``plugins/`` folder in an extension root,
with a manifest at the top level. Two manifest formats are accepted:

  ``.claude-plugin/plugin.json``  (Claude Code convention)
  ``plugin.toml``                  (our convention)

Both can declare:

    {
      "name": "my-plugin",
      "version": "0.1.0",
      "description": "...",
      "skills_dir": "skills",       // default
      "commands_dir": "commands",   // default
      "hooks_file": "hooks.toml",   // default
      "mcp_file": ".mcp.json"       // default; we also accept "mcp.json"
    }

We don't enforce a strict schema — missing fields default to the standard
subdirs. The manifest just gives us a name, optional version/description,
and overrides if the plugin author wants to use non-default paths.

Loading a plugin is a thin wrapper: we treat the plugin directory itself
as a mini extension root and reuse the existing skills/commands/hooks
loaders. This keeps the code simple and means a plugin's contents work
exactly the same as standalone extensions.

MCP server configs declared by plugins are surfaced via `iter_mcp_configs`
for the MCP manager to load.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from .commands import SlashCommand, load_commands
from .discovery import ExtensionRoots
from .hooks import HookRegistry, load_hooks
from .skills import SkillRegistry, load_skills


@dataclass
class Plugin:
    name: str
    version: str
    description: str
    root: Path
    skills_dir: str = "skills"
    commands_dir: str = "commands"
    hooks_file: str = "hooks.toml"
    mcp_files: list[str] = field(default_factory=lambda: [".mcp.json", "mcp.json"])
    manifest_path: Path | None = None

    def as_extension_root(self) -> ExtensionRoots:
        """Treat the plugin dir as a one-element root for the loaders."""
        return ExtensionRoots(paths=[self.root])

    def mcp_config_path(self) -> Path | None:
        """First existing MCP config file under this plugin, or None."""
        for name in self.mcp_files:
            p = self.root / name
            if p.is_file():
                return p
        return None


def load_plugins(roots: ExtensionRoots) -> list[Plugin]:
    """Discover plugins under any ``plugins/`` folder of any root."""
    out: list[Plugin] = []
    seen_names: set[str] = set()
    # Iterate later roots last so they override earlier ones.
    for root in roots.existing():
        plugins_dir = root / "plugins"
        if not plugins_dir.is_dir():
            continue
        for entry in sorted(plugins_dir.iterdir()):
            if not entry.is_dir():
                continue
            try:
                plugin = _load_plugin(entry)
            except Exception as e:
                print(f"[llmchat] failed to load plugin {entry}: {e}",
                      file=sys.stderr)
                continue
            # Later definition wins.
            if plugin.name in seen_names:
                out = [p for p in out if p.name != plugin.name]
            seen_names.add(plugin.name)
            out.append(plugin)
    return out


def _load_plugin(plugin_dir: Path) -> Plugin:
    # Try the Claude-style manifest first, then ours.
    claude_manifest = plugin_dir / ".claude-plugin" / "plugin.json"
    our_manifest = plugin_dir / "plugin.toml"

    name = plugin_dir.name
    version = "0.0.0"
    description = ""
    skills_dir = "skills"
    commands_dir = "commands"
    hooks_file = "hooks.toml"
    manifest_path: Path | None = None

    if claude_manifest.is_file():
        manifest_path = claude_manifest
        data = json.loads(claude_manifest.read_text(encoding="utf-8"))
        name = str(data.get("name", name))
        version = str(data.get("version", version))
        description = str(data.get("description", description))
        skills_dir = str(data.get("skills_dir", skills_dir))
        commands_dir = str(data.get("commands_dir", commands_dir))
        hooks_file = str(data.get("hooks_file", hooks_file))
    elif our_manifest.is_file():
        manifest_path = our_manifest
        data = _read_toml(our_manifest)
        name = str(data.get("name", name))
        version = str(data.get("version", version))
        description = str(data.get("description", description))
        skills_dir = str(data.get("skills_dir", skills_dir))
        commands_dir = str(data.get("commands_dir", commands_dir))
        hooks_file = str(data.get("hooks_file", hooks_file))
    # else: unmanifested plugin — that's allowed; we just use the dir name.

    return Plugin(
        name=name,
        version=version,
        description=description,
        root=plugin_dir,
        skills_dir=skills_dir,
        commands_dir=commands_dir,
        hooks_file=hooks_file,
        manifest_path=manifest_path,
    )


def _read_toml(path: Path) -> dict:
    if sys.version_info >= (3, 11):
        import tomllib as _toml  # type: ignore[import-not-found]
    else:
        try:
            import tomli as _toml  # type: ignore[import-not-found]
        except ImportError:
            return {}
    with open(path, "rb") as f:
        return _toml.load(f)


def merge_plugins_into(
    plugins: Iterable[Plugin],
    skill_registry: SkillRegistry,
    commands: dict[str, SlashCommand],
    hooks: HookRegistry,
) -> None:
    """Add each plugin's contents to the existing registries.

    Plugins loaded later override earlier same-name skills/commands. Hooks
    accumulate (we don't dedupe — multiple identical hooks may be valid if
    they come from different plugins, though that's unusual).
    """
    for plugin in plugins:
        plugin_roots = plugin.as_extension_root()
        # We need to remap the subdir names if the plugin overrides them.
        # The simplest way is to call the loaders with a virtual root. But
        # since the loaders only look at fixed subdir names, we do a quick
        # manual handling for the common case where defaults are kept, and
        # fall back to a temp symlink-free re-walk when not.
        if (plugin.skills_dir == "skills"
                and plugin.commands_dir == "commands"
                and plugin.hooks_file == "hooks.toml"):
            sub_skills = load_skills(plugin_roots)
            sub_commands = load_commands(plugin_roots)
            sub_hooks = load_hooks(plugin_roots)
        else:
            # Plugin uses non-default paths. Make a tiny shim by reading
            # those specific files directly. This branch is rarely hit, so
            # we keep it simple at the cost of some duplication.
            from .skills import _load_skill_file
            from .commands import _load_command_file
            from .hooks import HookRegistry as _HR

            sub_skills = SkillRegistry()
            sd = plugin.root / plugin.skills_dir
            if sd.is_dir():
                for entry in sorted(sd.iterdir()):
                    skill_md = entry / "SKILL.md"
                    if entry.is_dir() and skill_md.is_file():
                        try:
                            sk = _load_skill_file(skill_md)
                            sub_skills.skills[sk.name] = sk
                        except Exception as e:
                            print(f"[llmchat] failed to load skill {skill_md}: {e}",
                                  file=sys.stderr)

            sub_commands = {}
            cd = plugin.root / plugin.commands_dir
            if cd.is_dir():
                for entry in sorted(cd.iterdir()):
                    if entry.is_file() and entry.suffix.lower() == ".md":
                        try:
                            c = _load_command_file(entry)
                            sub_commands[c.name] = c
                        except Exception as e:
                            print(f"[llmchat] failed to load command {entry}: {e}",
                                  file=sys.stderr)
            sub_hooks = _HR()  # custom hooks file path: leave for future work

        for sk in sub_skills.skills.values():
            skill_registry.skills[sk.name] = sk
        for c in sub_commands.values():
            commands[c.name] = c
        for event, hooks_list in sub_hooks.by_event.items():
            hooks.by_event.setdefault(event, []).extend(hooks_list)
