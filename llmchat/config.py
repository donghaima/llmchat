"""Routing config file loader.

The user can drop a ``routing.toml`` next to the SQLite DB to customize the
router. If absent, we use sensible defaults.

Example file:

    default_tier = "cheap"
    enabled = true
    explain = true

    [tiers.local]
    model = "ollama:llama3.2"
    max_input_chars = 4000
    description = "Local model"

    [tiers.cheap]
    model = "anthropic:claude-haiku-4-5"
    max_input_chars = 20000

    [tiers.flagship]
    model = "anthropic:claude-opus-4-7"
    max_input_chars = 200000
"""

from __future__ import annotations

import sys
from pathlib import Path

from .routing import RoutingConfig, Tier, default_config


# Python 3.10 doesn't have tomllib in the stdlib; fall back to a third-party
# 'tomli' if available, otherwise tell the user.
if sys.version_info >= (3, 11):
    import tomllib as _toml  # type: ignore[import-not-found]
else:  # pragma: no cover
    try:
        import tomli as _toml  # type: ignore[import-not-found]
    except ImportError:
        _toml = None  # type: ignore[assignment]


def config_path_for(db_path: Path) -> Path:
    """Where the routing config lives, relative to the DB."""
    return db_path.parent / "routing.toml"


def load_routing_config(db_path: Path) -> RoutingConfig:
    """Load routing config from disk, merging with defaults.

    Anything missing from the file inherits from defaults — so a user can
    override just the flagship model and leave the rest alone.
    """
    cfg = default_config()
    path = config_path_for(db_path)
    if not path.exists():
        return cfg

    if _toml is None:
        # Older Python without tomli installed: warn but continue with defaults.
        print(f"[llmchat] routing.toml found at {path} but tomllib/tomli is "
              f"unavailable on this Python; using defaults.", file=sys.stderr)
        return cfg

    with open(path, "rb") as f:
        data = _toml.load(f)

    # Top-level fields.
    if "default_tier" in data:
        cfg.default_tier = str(data["default_tier"])
    if "enabled" in data:
        cfg.enabled = bool(data["enabled"])
    if "explain" in data:
        cfg.explain = bool(data["explain"])

    # Per-tier overrides.
    tiers_data = data.get("tiers", {})
    for tier_name in ("local", "cheap", "flagship"):
        section = tiers_data.get(tier_name)
        if not section:
            continue
        existing = cfg.tiers.get(tier_name)
        cfg.tiers[tier_name] = Tier(
            name=tier_name,
            model=section.get("model",
                              existing.model if existing else None),
            max_input_chars=int(section.get(
                "max_input_chars",
                existing.max_input_chars if existing else 20_000)),
            description=section.get(
                "description",
                existing.description if existing else ""),
            supports_tools=bool(section.get(
                "supports_tools",
                existing.supports_tools if existing else True)),
        )

    return cfg


def write_default_config(db_path: Path) -> Path:
    """Write a starter routing.toml the user can edit. Returns the path."""
    path = config_path_for(db_path)
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_DEFAULT_TOML)
    return path


_DEFAULT_TOML = """\
# llmchat routing config
#
# The router picks a tier (local/cheap/flagship) for each turn based on
# heuristics, then resolves to the model configured below. Any tier can be
# omitted — if you don't have an Ollama instance, just remove [tiers.local]
# and the router will skip that tier.
#
# Set `enabled = false` to disable routing entirely (manual /model only).
# Set `explain = false` to suppress the per-turn routing decision message.

default_tier = "cheap"
enabled = true
explain = true

[tiers.local]
model = "ollama:llama3.2"
max_input_chars = 4000
supports_tools = false
description = "Local Ollama model for quick factual lookups"

[tiers.cheap]
model = "anthropic:claude-haiku-4-5"
max_input_chars = 20000
supports_tools = true
description = "Fast, inexpensive paid model for everyday tasks"

[tiers.flagship]
model = "anthropic:claude-opus-4-7"
max_input_chars = 200000
supports_tools = true
description = "Most capable model — for hard reasoning, long context, complex code"
"""
