"""Smoke tests for storage, CLI, chat loop, and router.

Run with: python tests/test_smoke.py
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

_tmpdir = tempfile.mkdtemp(prefix="llmchat-test-")
os.environ["LLMCHAT_DB"] = str(Path(_tmpdir) / "test.db")


from llmchat.providers import ChatMessage  # noqa: E402
from llmchat.providers.base import Provider  # noqa: E402
from llmchat.routing import (  # noqa: E402
    Router, RoutingConfig, Tier, default_config,
)
from llmchat.storage import Store  # noqa: E402


# ----- a fake provider so we don't hit any real API -------------------------

class FakeProvider(Provider):
    name = "fake"

    def is_configured(self) -> bool:
        return True

    def stream(self, model, messages, system=None):
        last = next((m.content for m in reversed(messages) if m.role == "user"),
                    "(nothing)")
        for part in (f"[{model}] ", "you said: ", last):
            yield part

    def list_models(self):
        return ["fake-small", "fake-large"]


import llmchat.providers as registry  # noqa: E402

_original_get = registry.get_provider
_fake_instance = FakeProvider()


def _patched_get(name: str):
    if name == "fake":
        return _fake_instance
    return _original_get(name)


registry.get_provider = _patched_get
import llmchat.chat as chat_module  # noqa: E402

chat_module.get_provider = _patched_get


def _fake_routing_config() -> RoutingConfig:
    """Routing config that uses our fake provider for all tiers."""
    return RoutingConfig(
        tiers={
            "local": Tier("local", "fake:fake-tiny", 4_000),
            "cheap": Tier("cheap", "fake:fake-small", 20_000),
            "flagship": Tier("flagship", "fake:fake-large", 200_000),
        },
        default_tier="cheap",
        enabled=True,
        explain=False,  # quieter test output
    )


# ----- storage tests --------------------------------------------------------

def test_storage_roundtrip():
    store = Store()
    sess = store.create_session(model="fake:fake-small", title="hello")
    assert sess.id
    store.add_message(sess.id, "user", "hi", model="fake:fake-small")
    store.add_message(sess.id, "assistant", "hello!", model="fake:fake-small")
    msgs = store.get_messages(sess.id)
    assert len(msgs) == 2
    found = store.get_session(sess.id[:6])
    assert found and found.id == sess.id
    print("  storage roundtrip OK")


# ----- chat loop tests ------------------------------------------------------

def test_chat_loop_uses_router_by_default():
    from llmchat.chat import ChatLoop
    from rich.console import Console

    store = Store()
    sess = store.create_session(model="fake:fake-small", title="t")
    cfg = _fake_routing_config()
    loop = ChatLoop(store=store, session=sess, routing_config=cfg,
                    console=Console(file=open(os.devnull, "w")))

    # Short factual question should route to local.
    loop._handle_user_turn("what is the capital of France?")
    msgs = store.get_messages(sess.id)
    assert msgs[-1].model == "fake:fake-tiny", \
        f"expected local tier, got {msgs[-1].model}"
    print("  chat loop routes short factual to local OK")


def test_chat_loop_routes_hard_to_flagship():
    from llmchat.chat import ChatLoop
    from rich.console import Console

    store = Store()
    sess = store.create_session(model="fake:fake-small", title="t")
    cfg = _fake_routing_config()
    loop = ChatLoop(store=store, session=sess, routing_config=cfg,
                    console=Console(file=open(os.devnull, "w")))

    loop._handle_user_turn(
        "Please analyze the trade-offs between these database designs "
        "step by step and recommend the optimal architecture.")
    msgs = store.get_messages(sess.id)
    assert msgs[-1].model == "fake:fake-large", \
        f"expected flagship tier, got {msgs[-1].model}"
    print("  chat loop routes reasoning task to flagship OK")


def test_pinned_model_overrides_router():
    from llmchat.chat import ChatLoop
    from rich.console import Console

    store = Store()
    sess = store.create_session(model="fake:fake-small", title="t")
    cfg = _fake_routing_config()
    loop = ChatLoop(store=store, session=sess, routing_config=cfg,
                    console=Console(file=open(os.devnull, "w")))

    # Without a pin, a short factual question goes to local.
    loop._handle_user_turn("what is 2 + 2?")
    msgs = store.get_messages(sess.id)
    assert msgs[-1].model == "fake:fake-tiny"

    # Pin to flagship, then ask the same kind of trivial question.
    loop._handle_command("/model fake:fake-large")
    loop._handle_user_turn("what is 3 + 3?")
    msgs = store.get_messages(sess.id)
    assert msgs[-1].model == "fake:fake-large", \
        f"pin should override router, got {msgs[-1].model}"

    # /auto re-engages the router.
    loop._handle_command("/auto")
    loop._handle_user_turn("what is 4 + 4?")
    msgs = store.get_messages(sess.id)
    assert msgs[-1].model == "fake:fake-tiny", \
        f"/auto should re-engage router, got {msgs[-1].model}"
    print("  /model pin and /auto override OK")


def test_resume_picks_up_history():
    from llmchat.chat import ChatLoop
    from rich.console import Console

    store = Store()
    sess = store.create_session(model="fake:fake-small", title="t")
    cfg = _fake_routing_config()
    devnull = Console(file=open(os.devnull, "w"))
    loop1 = ChatLoop(store=store, session=sess, routing_config=cfg,
                     console=devnull)
    loop1._handle_user_turn("first")
    loop1._handle_user_turn("second")

    sess2 = store.get_session(sess.id)
    loop2 = ChatLoop(store=store, session=sess2, routing_config=cfg,
                     console=devnull)
    loop2._handle_user_turn("third")

    user_msgs = [m.content for m in store.get_messages(sess.id) if m.role == "user"]
    assert user_msgs == ["first", "second", "third"]
    print("  resume preserves history OK")


# ----- router unit tests ----------------------------------------------------

def test_router_short_factual_to_local():
    cfg = _fake_routing_config()
    router = Router(cfg)
    d = router.route("what is the capital of France?", [])
    assert d.tier == "local", f"expected local, got {d.tier}"
    assert "factual" in " ".join(d.reasons).lower()
    print("  router: short factual -> local OK")


def test_router_reasoning_words_to_flagship():
    cfg = _fake_routing_config()
    router = Router(cfg)
    cases = [
        "Please analyze this carefully.",
        "Debug the following code and explain.",
        "Compare and contrast these two architectures.",
        "Refactor this module to use async.",
    ]
    for prompt in cases:
        d = router.route(prompt, [])
        assert d.tier == "flagship", \
            f"prompt {prompt!r} routed to {d.tier}, expected flagship"
    print("  router: reasoning words -> flagship OK")


def test_router_long_prompt_to_flagship():
    cfg = _fake_routing_config()
    router = Router(cfg)
    long_prompt = "x " * 3000  # 6000 chars
    d = router.route(long_prompt, [])
    assert d.tier == "flagship"
    assert any("long" in r for r in d.reasons)
    print("  router: long prompt -> flagship OK")


def test_router_long_history_to_flagship():
    cfg = _fake_routing_config()
    router = Router(cfg)
    history = [ChatMessage(role="user", content="x " * 2500),
               ChatMessage(role="assistant", content="y " * 2500)]
    d = router.route("ok thanks", history)
    assert d.tier == "flagship"
    print("  router: long history -> flagship OK")


def test_router_default_for_ambiguous():
    cfg = _fake_routing_config()
    router = Router(cfg)
    d = router.route("Tell me about React hooks.", [])
    # No strong signal either way; falls to default tier (cheap).
    assert d.tier == "cheap"
    print("  router: ambiguous -> default tier (cheap) OK")


def test_router_escalates_when_tier_missing():
    """If 'local' isn't configured, the router should escalate, not crash."""
    cfg = RoutingConfig(
        tiers={
            "local": Tier("local", None, 4_000),  # not configured
            "cheap": Tier("cheap", "fake:fake-small", 20_000),
            "flagship": Tier("flagship", "fake:fake-large", 200_000),
        },
        default_tier="cheap",
    )
    router = Router(cfg)
    d = router.route("what is 2 + 2?", [])
    # Router would have wanted local; should escalate to cheap.
    assert d.tier == "cheap"
    assert any("escalated" in r for r in d.reasons)
    print("  router: escalates when tier is missing OK")


def test_router_falls_back_when_flagship_missing():
    """If 'flagship' isn't configured, hard prompts fall back to cheap."""
    cfg = RoutingConfig(
        tiers={
            "cheap": Tier("cheap", "fake:fake-small", 20_000),
            "flagship": Tier("flagship", None, 200_000),
        },
        default_tier="cheap",
    )
    router = Router(cfg)
    d = router.route("Please analyze this carefully.", [])
    assert d.tier == "cheap"
    assert any("fell back" in r for r in d.reasons)
    print("  router: falls back when flagship is missing OK")


def test_router_code_with_complexity_to_flagship():
    cfg = _fake_routing_config()
    router = Router(cfg)
    prompt = (
        "Why does this hang?\n"
        "```python\n"
        "async def worker():\n"
        "    async with mutex:\n"
        "        await channel.send(x)\n"
        "```")
    d = router.route(prompt, [])
    assert d.tier == "flagship"
    print("  router: complex code -> flagship OK")


# ----- CLI ------------------------------------------------------------------

def test_cli_route_test_command():
    from click.testing import CliRunner
    from llmchat.cli import cli

    runner = CliRunner()
    result = runner.invoke(cli, ["route", "test", "what is 2 + 2?"])
    assert result.exit_code == 0, result.output
    assert "tier:" in result.output

    result = runner.invoke(cli, ["route", "test", "Analyze step by step."])
    assert result.exit_code == 0, result.output
    assert "flagship" in result.output

    result = runner.invoke(cli, ["route", "show"])
    assert result.exit_code == 0, result.output
    print("  CLI route test/show OK")


def test_cli_list_and_show_unchanged():
    from click.testing import CliRunner
    from llmchat.cli import cli

    store = Store()
    sess = store.create_session(model="fake:fake-small", title="cli-test")
    store.add_message(sess.id, "user", "ping", model="fake:fake-small")
    store.add_message(sess.id, "assistant", "pong", model="fake:fake-small")

    runner = CliRunner()
    result = runner.invoke(cli, ["list"])
    assert result.exit_code == 0, result.output
    assert "cli-test" in result.output

    result = runner.invoke(cli, ["show", sess.id[:6]])
    assert result.exit_code == 0, result.output
    assert "ping" in result.output and "pong" in result.output

    result = runner.invoke(cli, ["export", sess.id[:6]])
    assert result.exit_code == 0, result.output
    assert "# cli-test" in result.output
    print("  CLI list/show/export still OK")


# ----- run ------------------------------------------------------------------

if __name__ == "__main__":
    print("running smoke tests against", os.environ["LLMCHAT_DB"])
    print()
    print("storage:")
    test_storage_roundtrip()
    print()
    print("router unit tests:")
    test_router_short_factual_to_local()
    test_router_reasoning_words_to_flagship()
    test_router_long_prompt_to_flagship()
    test_router_long_history_to_flagship()
    test_router_default_for_ambiguous()
    test_router_escalates_when_tier_missing()
    test_router_falls_back_when_flagship_missing()
    test_router_code_with_complexity_to_flagship()
    print()
    print("chat loop integration:")
    test_chat_loop_uses_router_by_default()
    test_chat_loop_routes_hard_to_flagship()
    test_pinned_model_overrides_router()
    test_resume_picks_up_history()
    print()
    print("CLI:")
    test_cli_route_test_command()
    test_cli_list_and_show_unchanged()
    print()
    print("all smoke tests passed.")
