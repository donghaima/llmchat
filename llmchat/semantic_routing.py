"""TypeSafe-backed semantic signals for the router.

The heuristic router in :mod:`llmchat.routing` reads regexes and lengths.
That gets the obvious cases right for free, but it is blind to anything
phrased outside its vocabulary: "why does this hang sometimes" carries no
reasoning word and thirty characters, so the heuristic sends it to the
local tier. Conversely "summarize this" matches ``_EASY_WORDS`` whether
the thing to summarize is a tweet or a distributed-systems postmortem.

This module asks a System One model (TypeSafe's Jev) for the judgments the
regexes cannot make and returns them as typed numbers the router composes.

Why this doesn't contradict the "no classifier" rule in routing.py
------------------------------------------------------------------
That rule assumes a classifier is a chat completion: prose out, output
tokens billed, seconds of latency. A System One call is none of those. It
returns a typed answer, output tokens are free, and at $0.042 per million
input tokens a routing judgment on a normal turn costs about $0.00006 —
roughly 15,000 decisions per dollar, and a fraction of what a single turn
misrouted to flagship costs. The token tax the heuristic was avoiding is
real; this just isn't it.

Design rules followed here
--------------------------
* **Deterministic rules stay in code.** Long-context escalation is a
  character count and tier availability is a config lookup; neither is
  asked of the model. Those checks also run *first* in the router, so the
  most expensive prompts never become the most expensive judgments.
* **Every failure path returns ``None``.** A missing SDK, absent API key,
  timeout, rate limit, or malformed response all fall back to the
  heuristic router. Routing must never be able to break the chat.
* **One request per turn.** The questions are independent, so they are
  asked together and answered in parallel.
* **Raw judgments stay reusable.** This module returns signals; the policy
  that turns them into a tier lives in :mod:`llmchat.routing`, where it can
  be re-tuned without re-running inference.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Iterable

from .providers import ChatMessage


# How much conversation to show the judge. The router escalates on long
# history before we ever get here, so this only needs enough to tell a
# follow-up ("and the other one?") from a fresh question.
_MAX_RECENT_TURNS = 4
_MAX_TURN_CHARS = 400

# Bounded latency matters more than a retried answer: a slow route is worse
# than a heuristic one, because the user is waiting on it either way.
_DEFAULT_TIMEOUT_SECONDS = 3.0

# Jev's published input price, USD per million tokens. Output is free.
JEV_INPUT_PER_MTOK = 0.042


@dataclass
class SemanticSignals:
    """What the judge saw in one turn.

    ``difficulty`` is a position on the level number line described by
    :data:`_DIFFICULTY_LEVELS` (0.0 = trivial lookup, 2.0 = hard reasoning);
    it can land between levels when the model is genuinely torn, which is
    what ``difficulty_confidence`` reports. The rest are probabilities of
    yes, where 0.5 means "as likely as not" rather than "medium".
    """
    difficulty: float
    difficulty_confidence: float
    needs_context: float
    program_reasoning: float
    needs_tools: float
    input_tokens: int | None = None
    # Keyed by level number (the SDK parses these as ints, not the
    # strings the JSON shows on the wire).
    probabilities: dict[int, float] = field(default_factory=dict)

    @property
    def cost_usd(self) -> float | None:
        if self.input_tokens is None:
            return None
        return self.input_tokens / 1_000_000 * JEV_INPUT_PER_MTOK

    def summary(self) -> str:
        """One-line form for ``/route show`` and ``llmchat route test``."""
        return (f"difficulty {self.difficulty:.2f} "
                f"(conf {self.difficulty_confidence:.2f}), "
                f"context {self.needs_context:.2f}, "
                f"code {self.program_reasoning:.2f}, "
                f"tools {self.needs_tools:.2f}")


# Score levels describe concrete situations, not degrees of a word like
# "hard" — the model matches text against each level independently, so
# "Moderately complex" would give it nothing to match on. Ordered lowest
# capability required to highest; the router maps bands of the result onto
# its own tiers, so these stay written in terms of the work, not the tiers.
_DIFFICULTY_LEVELS = [
    "Answerable directly from general knowledge in a sentence or two: a "
    "definition, a date, a translation, a unit conversion, or restating "
    "something in different words.",

    "A routine task with a clear procedure and no real judgment calls: "
    "summarizing a passage, drafting a short message, reformatting data, "
    "explaining a familiar concept, or a small self-contained code edit.",

    "Needs several dependent steps, weighing alternatives against each "
    "other, working out why something behaves the way it does, or "
    "designing something where the approach itself is in question.",
]


def _build_questions() -> dict[str, Any]:
    """The four judgments, built fresh so the SDK import stays lazy."""
    from typesafe_sdk import Noul, NoulCriteria, Score

    return {
        "difficulty": Score(
            instructions="Judge what `prompt` asks of the assistant "
                         "answering it. Judge only the work the request "
                         "calls for, not how politely or briefly it is "
                         "phrased — a short question can ask for hard work.",
            criteria=_DIFFICULTY_LEVELS,
        ),
        "needs_context": Noul(
            instructions="Does answering `prompt` require information from "
                         "`recent_turns`?",
            criteria=NoulCriteria(
                true="It refers back to something earlier — a pronoun with "
                     "no antecedent in `prompt`, a follow-up correction, or "
                     "a request to continue or revise previous work.",
                false="It stands on its own and would mean the same thing "
                      "as the first message of a new conversation.",
            ),
        ),
        "program_reasoning": Noul(
            instructions="Does answering `prompt` require reasoning about "
                         "how code actually behaves?",
            criteria=NoulCriteria(
                true="Debugging from symptoms, predicting the effect of a "
                     "change, reasoning about control flow, state, "
                     "concurrency, or performance, or designing how "
                     "components should fit together.",
                false="No code is involved, or the code work is mechanical: "
                      "formatting, renaming, recalling syntax, or a short "
                      "self-contained snippet with an obvious shape.",
            ),
        ),
        "needs_tools": Noul(
            instructions="Does answering `prompt` require information the "
                         "assistant cannot supply from its own knowledge "
                         "and `recent_turns` alone?",
            criteria=NoulCriteria(
                true="It asks about the current state of a file, a "
                     "repository, a database, a running system, or anything "
                     "else that must be read, run, or fetched to answer.",
                false="It can be answered from general knowledge, from text "
                      "the user has already provided, or by producing new "
                      "text.",
            ),
        ),
    }


def _build_state(new_prompt: str, history: list[ChatMessage],
                 turn_count: int) -> dict[str, Any]:
    """Named fields, not one blob: the questions reference them by path.

    Only the tail of the conversation goes in. The router has already
    escalated on long history by this point, so more would cost tokens
    without changing any of the four answers.
    """
    recent = []
    for msg in history[-_MAX_RECENT_TURNS:]:
        text = msg.content
        if len(text) > _MAX_TURN_CHARS:
            text = text[:_MAX_TURN_CHARS] + "…"
        recent.append({"role": msg.role, "text": text})
    return {
        "prompt": new_prompt,
        "recent_turns": recent,
        "turn_number": turn_count,
    }


class SemanticJudge:
    """Sync wrapper around one System One request per turn.

    Construction never raises and never touches the network. If the SDK
    isn't installed or no API key is set, :meth:`available` is False and
    :meth:`judge` returns ``None`` forever, which the router reads as "use
    the heuristics".
    """

    def __init__(self, *, api_key: str | None = None,
                 model: str | None = None,
                 timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS) -> None:
        self.model = model
        self.timeout_seconds = timeout_seconds
        self._api_key = api_key or os.environ.get("TYPESAFE_API_KEY")
        self._client: Any = None
        self._unavailable_reason: str | None = None
        self.last_error: str | None = None

        if not self._api_key:
            self._unavailable_reason = (
                "TYPESAFE_API_KEY is not set (get one at console.typesafe.ai)")
            return
        try:
            import typesafe_sdk  # noqa: F401
        except ImportError:
            self._unavailable_reason = (
                "typesafe-sdk is not installed "
                "(pip install 'llmchat[typesafe]')")

    def available(self) -> bool:
        return self._unavailable_reason is None

    def unavailable_reason(self) -> str | None:
        return self._unavailable_reason

    def _ensure_client(self) -> Any:
        if self._client is None:
            from typesafe_sdk import RetryPolicy, TypeSafeClient
            self._client = TypeSafeClient(
                api_key=self._api_key,
                model=self.model,
                # One retry, not the default two: past a couple of seconds
                # the heuristic answer is the better trade.
                retry=RetryPolicy(max_retries=1,
                                  timeout=self.timeout_seconds * 2),
                timeout=self.timeout_seconds,
            )
        return self._client

    def judge(self, new_prompt: str,
              history: Iterable[ChatMessage]) -> SemanticSignals | None:
        """Ask the four questions. Returns ``None`` on any failure."""
        if not self.available():
            return None

        history_list = list(history)
        turn_count = sum(1 for m in history_list if m.role == "user")
        state = _build_state(new_prompt, history_list, turn_count)

        try:
            from typesafe_sdk import TypeSafeError
        except ImportError:  # pragma: no cover - guarded in __init__
            self._unavailable_reason = "typesafe-sdk is not installed"
            return None

        try:
            response = self._ensure_client().system_one(
                state, _build_questions())
        except TypeSafeError as e:
            # Auth and connection problems will recur on every turn; stop
            # paying the timeout for them and say so once.
            self.last_error = f"{type(e).__name__}: {e}"
            if type(e).__name__ in ("TypeSafeAuthenticationError",
                                    "TypeSafePermissionDeniedError"):
                self._unavailable_reason = f"TypeSafe rejected the API key ({e})"
            return None
        except Exception as e:  # pragma: no cover - defensive
            self.last_error = f"{type(e).__name__}: {e}"
            return None

        try:
            answers = response.answers
            difficulty = answers["difficulty"]
            usage = getattr(response, "usage", None)
            return SemanticSignals(
                difficulty=float(difficulty.score),
                difficulty_confidence=float(difficulty.confidence),
                needs_context=float(answers["needs_context"].noul),
                program_reasoning=float(answers["program_reasoning"].noul),
                needs_tools=float(answers["needs_tools"].noul),
                input_tokens=(getattr(usage, "input_tokens", None)
                              if usage else None),
                probabilities=dict(getattr(difficulty, "probabilities", {})
                                   or {}),
            )
        except (KeyError, AttributeError, TypeError, ValueError) as e:
            self.last_error = f"unexpected response shape: {e}"
            return None


def make_judge(semantic_config: Any) -> SemanticJudge | None:
    """Build a judge from a :class:`~llmchat.routing.SemanticConfig`.

    Returns ``None`` when semantic routing is switched off, so the router
    keeps its heuristic path without even constructing a client. When it is
    on but unusable, a judge still comes back — callers can read
    :meth:`SemanticJudge.unavailable_reason` to tell the user why their
    setting isn't taking effect instead of leaving them guessing.
    """
    if not getattr(semantic_config, "enabled", False):
        return None
    return SemanticJudge(
        model=getattr(semantic_config, "model", None),
        timeout_seconds=getattr(semantic_config, "timeout_seconds",
                                _DEFAULT_TIMEOUT_SECONDS),
    )
