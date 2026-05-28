"""Model pricing and a token-count estimator.

Cost estimation keys on the full ``provider:model`` string, NOT on a routing
tier — a tier's model can change, and ``/model`` / ``/compare`` pin arbitrary
models that don't belong to any tier. Prices are USD per one million tokens.

The numbers below are approximate and meant to be edited as published prices
change; an unknown model returns ``None`` from :func:`cost_for` so the UI can
show token counts without inventing a dollar figure. Local models (``ollama:``)
are always free.
"""

from __future__ import annotations

from dataclasses import dataclass

from .mcp.types import Usage


@dataclass(frozen=True)
class ModelPrice:
    input_per_mtok: float   # USD per 1M input tokens
    output_per_mtok: float  # USD per 1M output tokens


# Keyed by the exact "provider:model" string the rest of the app uses.
_PRICES: dict[str, ModelPrice] = {
    # Anthropic
    "anthropic:claude-opus-4-7": ModelPrice(15.0, 75.0),
    "anthropic:claude-opus-4-6": ModelPrice(15.0, 75.0),
    "anthropic:claude-sonnet-4-6": ModelPrice(3.0, 15.0),
    "anthropic:claude-haiku-4-5": ModelPrice(1.0, 5.0),
    # OpenAI
    "openai:gpt-5.5": ModelPrice(10.0, 30.0),
    "openai:gpt-5.5-pro": ModelPrice(20.0, 80.0),
    "openai:gpt-5.4": ModelPrice(5.0, 15.0),
    "openai:gpt-5.4-mini": ModelPrice(0.6, 2.4),
    "openai:gpt-5.4-nano": ModelPrice(0.15, 0.6),
    # Google
    "google:gemini-2.5-pro": ModelPrice(1.25, 10.0),
    "google:gemini-2.5-flash": ModelPrice(0.30, 2.50),
    "google:gemini-2.5-flash-lite": ModelPrice(0.10, 0.40),
}

_FREE = ModelPrice(0.0, 0.0)


def _lookup(model: str) -> ModelPrice | None:
    if model in _PRICES:
        return _PRICES[model]
    if model.startswith("ollama:"):
        return _FREE
    return None


def cost_for(model: str, usage: Usage) -> float | None:
    """USD cost of one invocation, or ``None`` if the model's price is unknown."""
    price = _lookup(model)
    if price is None:
        return None
    return (usage.input_tokens / 1_000_000 * price.input_per_mtok
            + usage.output_tokens / 1_000_000 * price.output_per_mtok)


def estimate_tokens(text: str) -> int:
    """Rough token estimate used only when a provider reports no real counts.

    The ~4-chars-per-token rule is crude but good enough to give the user a
    ballpark when exact counts are unavailable."""
    if not text:
        return 0
    return max(1, len(text) // 4)
