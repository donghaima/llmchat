"""Heuristic router.

Scores each conversation turn against a handful of signals and picks the
cheapest model tier that can plausibly handle it. The whole thing runs in
microseconds — no LLM call required.

Why heuristic-first, not a classifier?
    A classifier would itself be an LLM call on every turn, which is
    exactly the token tax we're trying to avoid. The 80/20 case is easy
    to detect with regex and length checks; the ambiguous 20% goes to
    the configured default tier, which is a deliberate choice.

Tiers (by convention):
    "local"     — runs on your machine, free, smaller. Good for short
                  factual Q&A, simple rephrasing, basic completion.
    "cheap"     — paid but inexpensive (e.g. Haiku, gpt-5.4-mini, Flash).
                  Good for everyday tasks, summarization, simple code.
    "flagship"  — most capable (Opus, gpt-5.5, Pro). For long context,
                  hard reasoning, complex code, agentic work.

Each model in the config has a tier; the router picks a tier and resolves
to the configured model for that tier.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable

from .providers import ChatMessage


# ----- signals --------------------------------------------------------------

# Words that strongly suggest the user wants careful reasoning. These bias
# toward flagship even on short prompts.
_HARD_REASONING_WORDS = re.compile(
    r"\b("
    r"analy[sz]e|reason|prove|derive|optimi[sz]e|debug|refactor|architect|"
    r"design (?:a |an |the )?system|trade-?offs?|compare and contrast|"
    r"step.by.step|think (?:carefully|hard|step)|chain.of.thought|"
    r"plan (?:out|the)|strategi[sz]e|evaluate|critique"
    r")\b",
    re.IGNORECASE,
)

# Words that suggest a quick lookup / chat / rephrase. Bias toward cheap/local.
_EASY_WORDS = re.compile(
    r"\b("
    r"what is|what's|who is|who's|when (?:is|was|did)|where (?:is|was)|"
    r"define|definition of|spell|abbreviation|acronym|"
    r"translate|rephrase|reword|shorter|tldr|tl;dr|summary|summarize|"
    r"capitalize|lowercase|uppercase"
    r")\b",
    re.IGNORECASE,
)

# Code-block detection: triple-backtick fences or significant indented blocks.
_CODE_FENCE = re.compile(r"```")
_LONG_CODE_HINT = re.compile(r"(?m)^( {4,}|\t)\S")

# Rough "this looks like code that needs real understanding" heuristic.
_CODE_COMPLEXITY_HINTS = re.compile(
    r"\b(class |def |function |async |await|trait |impl |interface |"
    r"struct |enum |template|generic |concurrenc|thread|mutex|"
    r"goroutine|channel|monad)\b"
)


@dataclass
class Tier:
    """Configuration for a single model tier."""
    name: str                  # "local" | "cheap" | "flagship"
    model: str | None          # full "provider:model" string, or None if absent
    max_input_chars: int       # rough upper bound; above this, escalate
    description: str = ""
    supports_tools: bool = True  # set False for Ollama-style local models
    """Does this tier's model support tool calls?

    The router uses this when ``need_tools`` is set on the route call: if
    the prompt requires tools (because MCP servers are configured and the
    model is expected to be able to call them), the router skips tiers
    where ``supports_tools`` is False, even if they'd otherwise be picked."""


@dataclass
class RoutingConfig:
    tiers: dict[str, Tier]
    default_tier: str = "cheap"      # tier used when signals are mixed
    enabled: bool = True             # global on/off switch
    explain: bool = True             # print routing decisions to the user

    def available_tiers(self) -> list[str]:
        """Tiers that have a configured model, in escalation order."""
        order = ["local", "cheap", "flagship"]
        return [t for t in order if self.tiers.get(t) and self.tiers[t].model]


@dataclass
class RoutingDecision:
    tier: str
    model: str
    reasons: list[str] = field(default_factory=list)

    def explanation(self) -> str:
        if not self.reasons:
            return f"{self.tier} (default)"
        return f"{self.tier} ({'; '.join(self.reasons)})"


# ----- the router --------------------------------------------------------------

class Router:
    """Picks a tier for a given turn based on the prompt + conversation state.

    Routing logic, in order of precedence:
      1. Long context  (history big or single prompt huge)  -> flagship
      2. Hard reasoning words / complex code present        -> flagship
      3. Many turns deep into a complex thread               -> at least cheap
      4. Easy words + short prompt + short history          -> local (or cheap)
      5. Otherwise -> default tier
    Any pick is then clamped to the available tiers (we never route to a tier
    the user hasn't configured).
    """

    # Thresholds. Exposed as instance attrs so a future config can override.
    SHORT_PROMPT_CHARS = 200
    LONG_PROMPT_CHARS = 4000
    LONG_HISTORY_CHARS = 8000
    SHORT_HISTORY_TURNS = 4
    MANY_TURNS = 10

    def __init__(self, config: RoutingConfig) -> None:
        self.config = config

    def route(self, new_prompt: str,
              history: Iterable[ChatMessage],
              need_tools: bool = False) -> RoutingDecision:
        history_list = list(history)
        history_chars = sum(len(m.content) for m in history_list)
        prompt_chars = len(new_prompt)
        turn_count = sum(1 for m in history_list if m.role == "user")

        reasons: list[str] = []
        chosen_tier: str | None = None

        # 1. Long context — must escalate.
        if (prompt_chars >= self.LONG_PROMPT_CHARS
                or history_chars >= self.LONG_HISTORY_CHARS):
            chosen_tier = "flagship"
            if prompt_chars >= self.LONG_PROMPT_CHARS:
                reasons.append(f"long prompt ({prompt_chars} chars)")
            if history_chars >= self.LONG_HISTORY_CHARS:
                reasons.append(f"long history ({history_chars} chars)")

        # 2. Hard reasoning / complex code signals.
        elif _HARD_REASONING_WORDS.search(new_prompt):
            chosen_tier = "flagship"
            reasons.append("reasoning task")
        elif _CODE_FENCE.search(new_prompt) and _CODE_COMPLEXITY_HINTS.search(
                new_prompt):
            chosen_tier = "flagship"
            reasons.append("non-trivial code")
        elif _CODE_FENCE.search(new_prompt) and prompt_chars > 800:
            chosen_tier = "flagship"
            reasons.append("substantial code block")

        # 3. Deep into a complex conversation: don't downgrade.
        elif turn_count >= self.MANY_TURNS:
            chosen_tier = "cheap" if self.config.default_tier == "local" else self.config.default_tier
            reasons.append(f"deep conversation ({turn_count} turns)")

        # 4. Easy quick lookups.
        elif (_EASY_WORDS.search(new_prompt)
              and prompt_chars <= self.SHORT_PROMPT_CHARS
              and turn_count <= self.SHORT_HISTORY_TURNS):
            chosen_tier = "local"
            reasons.append("short factual question")

        # 5. Default for everything else.
        if chosen_tier is None:
            chosen_tier = self.config.default_tier

        # Clamp to what's actually available, escalating if our pick is missing.
        chosen_tier, model = self._resolve(chosen_tier, reasons,
                                            need_tools=need_tools)
        return RoutingDecision(tier=chosen_tier, model=model, reasons=reasons)

    def _resolve(self, requested: str, reasons: list[str],
                 need_tools: bool = False) -> tuple[str, str]:
        """Map a requested tier to the closest available tier + concrete model.

        When ``need_tools`` is set, we filter out tiers whose configured
        models don't support tool calling.
        """
        tiers = self.config.tiers
        order = ["local", "cheap", "flagship"]
        avail = self.config.available_tiers()
        if need_tools:
            avail = [t for t in avail
                     if tiers[t].supports_tools]
            if not avail:
                raise ValueError(
                    "Tools are required but no configured tier has "
                    "supports_tools=True. Update routing.toml.")
        if not avail:
            raise ValueError(
                "No tiers have a configured model. "
                "Set up at least one tier in routing config.")

        if requested in avail:
            return requested, tiers[requested].model  # type: ignore[return-value]

        idx = order.index(requested)
        for t in order[idx + 1:]:
            if t in avail:
                reasons.append(f"{requested} unavailable, escalated to {t}")
                return t, tiers[t].model  # type: ignore[return-value]
        for t in reversed(order[:idx]):
            if t in avail:
                reasons.append(f"{requested} unavailable, fell back to {t}")
                return t, tiers[t].model  # type: ignore[return-value]
        raise RuntimeError("no available tier could be resolved")


# ----- default config -------------------------------------------------------

def default_config() -> RoutingConfig:
    """A reasonable starting config. Users will override via TOML."""
    return RoutingConfig(
        tiers={
            "local": Tier(
                name="local",
                model="ollama:llama3.2",
                max_input_chars=4_000,
                description="Local model for quick lookups.",
                supports_tools=False),
            "cheap": Tier(
                name="cheap",
                model="anthropic:claude-haiku-4-5",
                max_input_chars=20_000,
                description="Fast paid model for everyday tasks."),
            "flagship": Tier(
                name="flagship",
                model="anthropic:claude-opus-4-7",
                max_input_chars=200_000,
                description="Most capable model for hard problems."),
        },
        default_tier="cheap",
        enabled=True,
        explain=True,
    )
