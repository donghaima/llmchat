"""Interactive chat REPL.

Wires together the storage layer, the provider registry, the router, and the
user's terminal. Every user prompt is persisted before the model is even
called, and every assistant response is persisted as soon as the stream
finishes — so a crash mid-stream costs at most one in-flight reply.

Routing
-------
By default, each turn passes through the heuristic Router, which picks the
cheapest tier (local / cheap / flagship) that can plausibly handle the prompt.
The decision is shown to the user when ``explain`` is on. The user can:

  /model X      -> pin to model X for all subsequent turns (router off)
  /auto         -> re-engage the router
  /route off    -> turn routing off globally (sticks to current model)
  /route on     -> turn it back on
  /route show   -> dump the current routing config

The router's pick is *not* persisted as `current_model` on the session;
`current_model` reflects what the user has explicitly pinned (or the initial
model). This way, resuming a session doesn't lock in an automatic choice from
a prior turn.
"""

from __future__ import annotations

from dataclasses import dataclass

from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from rich.console import Console
from rich.rule import Rule

from .providers import (
    ChatMessage,
    ProviderError,
    get_provider,
    parse_model,
)
from .routing import Router, RoutingConfig
from .storage import Session, Store


HELP_TEXT = """\
Commands:
  /model <provider>:<name>   Pin a specific model (disables auto-routing)
  /auto                      Re-engage automatic routing
  /route on|off|show         Turn routing on/off, or show current config
  /system <text>             Set or replace the system prompt (empty to clear)
  /title <text>              Rename this session
  /clear                     Start a new session (current one stays saved)
  /history                   Show all messages in this session
  /help                      Show this help
  /exit                      Quit (Ctrl-D also works)
"""


@dataclass
class ChatState:
    session: Session
    pinned_model: str | None  # set by /model, cleared by /auto


class ChatLoop:
    def __init__(self, store: Store, session: Session,
                 routing_config: RoutingConfig,
                 console: Console | None = None) -> None:
        self.store = store
        self.console = console or Console()
        self.routing_config = routing_config
        self.router = Router(routing_config)
        self.state = ChatState(session=session, pinned_model=None)

        history_dir = store.db_path.parent
        history_dir.mkdir(parents=True, exist_ok=True)
        self.prompt_session = PromptSession(
            history=FileHistory(str(history_dir / "input_history")),
            key_bindings=self._bindings(),
        )

    def set_pinned_model(self, model: str | None) -> None:
        self.state.pinned_model = model

    def _bindings(self) -> KeyBindings:
        return KeyBindings()

    # ----- main loop --------------------------------------------------------

    def run(self) -> None:
        self._print_header()
        while True:
            try:
                line = self.prompt_session.prompt(self._prompt_str())
            except (EOFError, KeyboardInterrupt):
                self.console.print("[dim]bye[/dim]")
                return

            line = line.strip()
            if not line:
                continue

            if line.startswith("/"):
                if self._handle_command(line):
                    return
                continue

            self._handle_user_turn(line)

    # ----- model resolution -------------------------------------------------

    def _resolve_model_for_turn(self, prompt: str) -> tuple[str, str | None]:
        """Decide which model to use for this turn.

        Returns (model, explanation). The explanation is None when the model
        was pinned (manual choice) or when explanations are turned off.
        """
        # 1. Explicit user pin always wins.
        if self.state.pinned_model:
            return self.state.pinned_model, None

        # 2. Routing disabled -> fall back to whatever the session had.
        if not self.routing_config.enabled:
            return self.state.session.current_model, None

        # 3. Run the router.
        history = self.store.get_messages(self.state.session.id)
        chat_history = [ChatMessage(role=m.role, content=m.content)
                        for m in history if m.role in ("user", "assistant")]
        decision = self.router.route(prompt, chat_history)
        explanation = (decision.explanation()
                       if self.routing_config.explain else None)
        return decision.model, explanation

    # ----- commands ---------------------------------------------------------

    def _handle_command(self, line: str) -> bool:
        parts = line.split(maxsplit=1)
        cmd = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ""

        if cmd in ("/exit", "/quit", "/q"):
            return True

        if cmd == "/help":
            self.console.print(HELP_TEXT)
            return False

        if cmd == "/model":
            if not arg:
                pinned = self.state.pinned_model
                if pinned:
                    self.console.print(f"Pinned model: [cyan]{pinned}[/cyan]")
                else:
                    self.console.print("[dim]No model pinned (auto-routing).[/dim]")
                return False
            try:
                provider_name, model_name = parse_model(arg)
                get_provider(provider_name)
            except ProviderError as e:
                self.console.print(f"[red]{e}[/red]")
                return False
            full = f"{provider_name}:{model_name}"
            self.state.pinned_model = full
            self.store.update_session_model(self.state.session.id, full)
            self.console.print(
                f"[green]Pinned to[/green] [cyan]{full}[/cyan] "
                f"[dim](use /auto to re-enable routing)[/dim]")
            return False

        if cmd == "/auto":
            self.state.pinned_model = None
            if not self.routing_config.enabled:
                self.console.print(
                    "[yellow]Routing is currently disabled in config — "
                    "use /route on to enable it.[/yellow]")
                return False
            self.console.print("[green]Auto-routing re-enabled.[/green]")
            return False

        if cmd == "/route":
            sub = arg.strip().lower()
            if sub == "on":
                self.routing_config.enabled = True
                self.console.print("[green]Routing enabled.[/green]")
            elif sub == "off":
                self.routing_config.enabled = False
                self.console.print(
                    "[yellow]Routing disabled. "
                    "Pinned model will be used for every turn.[/yellow]")
            elif sub in ("", "show", "status"):
                self._print_routing_config()
            else:
                self.console.print(
                    "[red]/route expects: on, off, or show[/red]")
            return False

        if cmd == "/system":
            new_system = arg or None
            self.store.update_system_prompt(self.state.session.id, new_system)
            self.state.session.system_prompt = new_system
            if new_system:
                self.console.print("[green]System prompt set.[/green]")
            else:
                self.console.print("[green]System prompt cleared.[/green]")
            return False

        if cmd == "/title":
            if not arg:
                self.console.print(
                    f"Current title: [cyan]{self.state.session.title}[/cyan]")
                return False
            self.store.update_session_title(self.state.session.id, arg)
            self.state.session.title = arg
            self.console.print(f"[green]Title set to[/green] [cyan]{arg}[/cyan]")
            return False

        if cmd == "/clear":
            new_sess = self.store.create_session(
                model=self.state.session.current_model,
                title="New chat",
                system_prompt=self.state.session.system_prompt,
            )
            self.state.session = new_sess
            self.console.print(
                f"[green]New session[/green] [dim]{new_sess.id}[/dim]")
            self._print_header()
            return False

        if cmd == "/history":
            msgs = self.store.get_messages(self.state.session.id)
            if not msgs:
                self.console.print("[dim](no messages yet)[/dim]")
                return False
            for m in msgs:
                style = "cyan" if m.role == "user" else "green"
                tag = m.role
                if m.model:
                    tag += f" · {m.model}"
                self.console.print(Rule(f"[{style}]{tag}[/{style}]"))
                self.console.print(m.content)
            return False

        self.console.print(f"[red]Unknown command:[/red] {cmd}. Type /help.")
        return False

    # ----- chat turn --------------------------------------------------------

    def _handle_user_turn(self, text: str) -> None:
        # Pick a model BEFORE persisting, so the router has a chance to look
        # at the prompt. We persist the user message immediately afterward so
        # the prompt is durable even if the model call crashes.
        model, explanation = self._resolve_model_for_turn(text)

        self.store.add_message(self.state.session.id, "user", text, model=model)

        provider_name, model_name = parse_model(model)
        try:
            provider = get_provider(provider_name)
        except ProviderError as e:
            self.console.print(f"[red]{e}[/red]")
            return

        history = self.store.get_messages(self.state.session.id)
        api_messages = [
            ChatMessage(role=m.role, content=m.content)
            for m in history
            if m.role in ("user", "assistant")
        ]

        self.console.print()
        if explanation:
            self.console.print(f"[dim]→ {model} · {explanation}[/dim]")
        else:
            self.console.print(f"[dim]assistant · {model}[/dim]")

        chunks: list[str] = []
        try:
            for chunk in provider.stream(
                model=model_name,
                messages=api_messages,
                system=self.state.session.system_prompt,
            ):
                chunks.append(chunk)
                self.console.print(chunk, end="", soft_wrap=True, highlight=False)
            self.console.print()
        except ProviderError as e:
            self.console.print(f"\n[red]Error: {e}[/red]")
        except KeyboardInterrupt:
            self.console.print("\n[yellow](interrupted)[/yellow]")

        full = "".join(chunks)
        if full:
            self.store.add_message(
                self.state.session.id, "assistant", full, model=model)
        self.console.print()

    # ----- presentation -----------------------------------------------------

    def _prompt_str(self) -> str:
        short_id = self.state.session.id[:6]
        if self.state.pinned_model:
            tag = self.state.pinned_model.split(":", 1)[-1][:14]
            return f"[{short_id} · {tag}] you> "
        if self.routing_config.enabled:
            return f"[{short_id} · auto] you> "
        return f"[{short_id}] you> "

    def _print_header(self) -> None:
        s = self.state.session
        self.console.print(Rule(f"[bold]{s.title}[/bold] [dim]{s.id}[/dim]"))
        if self.state.pinned_model:
            self.console.print(
                f"model: [cyan]{self.state.pinned_model}[/cyan] [dim](pinned)[/dim]")
        elif self.routing_config.enabled:
            avail = self.routing_config.available_tiers()
            tiers_str = " → ".join(
                f"[cyan]{t}[/cyan]={self.routing_config.tiers[t].model}"
                for t in avail)
            self.console.print(f"routing: {tiers_str}")
        else:
            self.console.print(
                f"model: [cyan]{s.current_model}[/cyan] [dim](routing off)[/dim]")
        self.console.print("[dim]/help for commands, /exit to quit[/dim]")
        if s.system_prompt:
            self.console.print(f"[dim]system: {s.system_prompt[:80]}...[/dim]")
        msgs = self.store.get_messages(s.id)
        if msgs:
            self.console.print(
                f"[dim]({len(msgs)} message(s) in history — "
                f"continuing where you left off)[/dim]")
        self.console.print()

    def _print_routing_config(self) -> None:
        cfg = self.routing_config
        self.console.print(
            f"routing: [cyan]{'on' if cfg.enabled else 'off'}[/cyan]   "
            f"default tier: [cyan]{cfg.default_tier}[/cyan]   "
            f"explain: [cyan]{'on' if cfg.explain else 'off'}[/cyan]")
        for tier_name in ("local", "cheap", "flagship"):
            tier = cfg.tiers.get(tier_name)
            if not tier or not tier.model:
                self.console.print(f"  [dim]{tier_name}: (not configured)[/dim]")
                continue
            self.console.print(
                f"  [cyan]{tier_name}[/cyan]: {tier.model} "
                f"[dim](≤ {tier.max_input_chars} chars)[/dim]")


def run_chat(store: Store, session: Session,
             routing_config: RoutingConfig,
             pinned_model: str | None = None) -> None:
    loop = ChatLoop(store=store, session=session,
                    routing_config=routing_config)
    if pinned_model:
        loop.set_pinned_model(pinned_model)
    loop.run()
