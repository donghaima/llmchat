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

    [semantic]
    enabled = true
"""

from __future__ import annotations

import sys
from pathlib import Path

from .routing import RoutingConfig, SemanticConfig, Tier, default_config


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

    # Optional [semantic] section — the System One judge. Absent means the
    # defaults in SemanticConfig, which leave it switched off.
    sem_data = data.get("semantic")
    if sem_data:
        base = cfg.semantic
        cfg.semantic = SemanticConfig(
            enabled=bool(sem_data.get("enabled", base.enabled)),
            model=sem_data.get("model", base.model),
            timeout_seconds=float(sem_data.get(
                "timeout_seconds", base.timeout_seconds)),
            local_at=float(sem_data.get("local_at", base.local_at)),
            flagship_at=float(sem_data.get("flagship_at", base.flagship_at)),
            min_confidence=float(sem_data.get(
                "min_confidence", base.min_confidence)),
            context_threshold=float(sem_data.get(
                "context_threshold", base.context_threshold)),
            code_threshold=float(sem_data.get(
                "code_threshold", base.code_threshold)),
            trust_tool_judgment=bool(sem_data.get(
                "trust_tool_judgment", base.trust_tool_judgment)),
            no_tools_below=float(sem_data.get(
                "no_tools_below", base.no_tools_below)),
        )

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

# ----------------------------------------------------------------------------
# Optional: semantic routing via a TypeSafe System One model.
#
# The heuristics above read regexes and lengths, so they miss anything
# phrased outside their vocabulary — "why does this hang sometimes" has no
# reasoning word in it and goes local. Switching this on asks a System One
# model (Jev) four typed questions per turn instead: how much capability the
# request needs, whether it depends on earlier turns, whether it needs real
# reasoning about code, and whether it needs a tool.
#
# It is not a chat completion. Output tokens are free and input is $0.042
# per million, so a turn costs roughly $0.00006 — about 15,000 routing
# decisions per dollar, against one wrongly-flagship turn costing more than
# all of them together.
#
# Requires:  pip install 'llmchat[typesafe]'
#            export TYPESAFE_API_KEY=...     (console.typesafe.ai)
#
# If the key is missing, the SDK isn't installed, or the call times out,
# routing silently falls back to the heuristics above.

[semantic]
enabled = false

# Which Jev build to use. Omit for the SDK default (jev-latest).
# model = "jev-latest"

# Give up and use the heuristics rather than keep the user waiting.
timeout_seconds = 3.0

# Difficulty comes back as a position from 0 to 2 on this scale:
#   0  a fact, a definition, a translation, a rephrasing
#   1  a routine task with a clear procedure and no judgment calls
#   2  several dependent steps, weighing alternatives, or working out why
# These two numbers cut that line into tiers; between them is "cheap".
local_at = 0.6
flagship_at = 1.5

# When the model splits its probability across levels, escalate one tier
# instead of trusting the midpoint — a needless flagship turn costs cents,
# a botched local one costs the user another round trip.
min_confidence = 0.5

# A local pick gets lifted to cheap when either of these is exceeded;
# small local models lose the thread and reason poorly about program behavior.
context_threshold = 0.7
code_threshold = 0.7

# Having any MCP server configured normally excludes tiers that can't call
# tools, for every turn. Switch this on to let the local tier back in when
# the judge is near-certain this particular turn needs no tool. The risk is
# a turn that wanted a tool landing on a model that hasn't got any.
trust_tool_judgment = false
no_tools_below = 0.10
"""
