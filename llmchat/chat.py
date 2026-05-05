"""Interactive chat REPL with extensions and tool calling.

Architecture in one paragraph: each user turn (1) optionally goes through
``pre_user_message`` hooks, (2) gets matched against slash commands and
expanded if applicable, (3) selects relevant skills and assembles a system
prompt, (4) routes to a model tier, (5) runs the model — possibly through
multiple agentic iterations if the model calls MCP tools — and (6) finally
runs ``post_assistant_message`` hooks. Storage writes happen at every
boundary so a crash anywhere costs at most one in-flight reply.

The agentic loop is bounded: at most ``MAX_TOOL_ITERATIONS`` round trips per
turn, after which we surface what we have and let the user continue.
"""

from __future__ import annotations

from dataclasses import dataclass

from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.syntax import Syntax

from .extensions import (
    HookEvent, HookRegistry, SkillRegistry, SlashCommand,
)
from .mcp import ApprovalState, MCPManager, ToolCall, ToolResult
from .mcp.approval import ApprovalDecision
from .mcp.types import StreamEvent, TextDelta, ToolUseRequest, TurnEnd
from .providers import (
    ChatMessage, ProviderError, RateLimitError, get_provider, parse_model,
)
from .routing import Router, RoutingConfig
from .storage import Session, Store


MAX_TOOL_ITERATIONS = 10
"""Hard cap on agentic round trips per user turn. If the model keeps asking
for more tool calls past this, we surface the partial result and stop.
This protects against runaway loops."""


HELP_TEXT = """\
Commands:
  /model <provider>:<name>           Pin a specific model (disables auto-routing)
  /auto                              Re-engage automatic routing
  /route on|off|show                 Turn routing on/off, or show current config
  /compare <model1> <model2> [...]   Send next message to multiple models in parallel
  /tools                             List MCP tools currently available
  /plugins                           List loaded plugins/skills/commands/hooks
  /system <text>                     Set or replace the system prompt
  /title <text>                      Rename this session
  /clear                             Start a new session (current one stays saved)
  /history                           Show all messages in this session
  /<custom>                          Run a custom slash command (see /plugins)
  /help                              Show this help
  /exit                              Quit (Ctrl-D also works)
"""


@dataclass
class ExtensionsBundle:
    """Everything loaded from disk for this session."""
    skills: SkillRegistry
    commands: dict[str, SlashCommand]
    hooks: HookRegistry
    mcp_manager: MCPManager  # may be configured-but-empty if no servers


@dataclass
class ChatState:
    session: Session
    pinned_model: str | None  # set by /model, cleared by /auto
    approval: ApprovalState
    compare_models: list[str] | None = None  # set by /compare, cleared after use


class ChatLoop:
    def __init__(self, store: Store, session: Session,
                 routing_config: RoutingConfig,
                 extensions: ExtensionsBundle,
                 console: Console | None = None) -> None:
        self.store = store
        self.console = console or Console()
        self.routing_config = routing_config
        self.router = Router(routing_config)
        self.ext = extensions
        self.state = ChatState(session=session, pinned_model=None,
                               approval=ApprovalState())

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
                consumed = self._handle_command(line)
                if consumed == "exit":
                    return
                if consumed == "handled":
                    continue
                # consumed == "expanded": fall through with a transformed text
                line = self._last_expanded_text or line

            self._handle_user_turn(line)

    _last_expanded_text: str | None = None

    # ----- model resolution -------------------------------------------------

    def _resolve_model_for_turn(self, prompt: str, need_tools: bool
                                 ) -> tuple[str, str | None]:
        if self.state.pinned_model:
            return self.state.pinned_model, None
        if not self.routing_config.enabled:
            return self.state.session.current_model, None
        history = self.store.get_messages(self.state.session.id)
        chat_history = [ChatMessage(role=m.role, content=m.content)
                        for m in history if m.role in ("user", "assistant")]
        decision = self.router.route(prompt, chat_history,
                                      need_tools=need_tools)
        explanation = (decision.explanation()
                       if self.routing_config.explain else None)
        return decision.model, explanation

    # ----- commands ---------------------------------------------------------

    def _handle_command(self, line: str) -> str:
        """Returns 'exit' to terminate, 'handled' if the command was a no-op
        or status command, 'expanded' if a slash command was expanded into
        a regular user prompt that should now be sent to the model."""
        parts = line.split(maxsplit=1)
        cmd = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ""

        if cmd in ("/exit", "/quit", "/q"):
            return "exit"

        if cmd == "/help":
            self.console.print(HELP_TEXT)
            return "handled"

        if cmd == "/model":
            self._cmd_model(arg)
            return "handled"

        if cmd == "/auto":
            self.state.pinned_model = None
            if not self.routing_config.enabled:
                self.console.print(
                    "[yellow]Routing is disabled — /route on to enable.[/yellow]")
                return "handled"
            self.console.print("[green]Auto-routing re-enabled.[/green]")
            return "handled"

        if cmd == "/route":
            self._cmd_route(arg)
            return "handled"

        if cmd == "/compare":
            self._cmd_compare(arg)
            return "handled"

        if cmd == "/system":
            self._cmd_system(arg)
            return "handled"

        if cmd == "/title":
            self._cmd_title(arg)
            return "handled"

        if cmd == "/clear":
            self._cmd_clear()
            return "handled"

        if cmd == "/history":
            self._cmd_history()
            return "handled"

        if cmd == "/tools":
            self._cmd_tools()
            return "handled"

        if cmd == "/plugins":
            self._cmd_plugins()
            return "handled"

        # User-defined slash command?
        custom_name = cmd.lstrip("/")
        custom = self.ext.commands.get(custom_name)
        if custom is not None:
            expanded = custom.expand(arg)
            self.console.print(
                f"[dim]→ /{custom.name}: expanded to "
                f"{len(expanded)} chars[/dim]")
            self._last_expanded_text = expanded
            return "expanded"

        self.console.print(f"[red]Unknown command:[/red] {cmd}. Type /help.")
        return "handled"

    # Detail handlers split out for readability.

    def _cmd_model(self, arg: str) -> None:
        if not arg:
            pinned = self.state.pinned_model
            if pinned:
                self.console.print(f"Pinned model: [cyan]{pinned}[/cyan]")
            else:
                self.console.print("[dim]No model pinned (auto-routing).[/dim]")
            return
        try:
            provider_name, model_name = parse_model(arg)
            get_provider(provider_name)
        except ProviderError as e:
            self.console.print(f"[red]{e}[/red]")
            return
        full = f"{provider_name}:{model_name}"
        self.state.pinned_model = full
        self.store.update_session_model(self.state.session.id, full)
        self.console.print(
            f"[green]Pinned to[/green] [cyan]{full}[/cyan] "
            f"[dim](use /auto to re-enable routing)[/dim]")

    def _cmd_route(self, arg: str) -> None:
        sub = arg.strip().lower()
        if sub == "on":
            self.routing_config.enabled = True
            self.console.print("[green]Routing enabled.[/green]")
        elif sub == "off":
            self.routing_config.enabled = False
            self.console.print(
                "[yellow]Routing disabled. Pinned model used for every turn.[/yellow]")
        elif sub in ("", "show", "status"):
            self._print_routing_config()
        else:
            self.console.print("[red]/route expects: on, off, or show[/red]")

    def _cmd_compare(self, arg: str) -> None:
        if not arg:
            if self.state.compare_models:
                self.state.compare_models = None
                self.console.print("[dim]Compare mode off.[/dim]")
            else:
                self.console.print(
                    "[dim]Usage: /compare model1 model2 ...[/dim]")
            return
        specs = arg.split()
        valid: list[str] = []
        for spec in specs:
            try:
                pname, mname = parse_model(spec)
                get_provider(pname)
                valid.append(f"{pname}:{mname}")
            except ProviderError as e:
                self.console.print(f"[red]{spec}: {e}[/red]")
                return
        if len(valid) < 2:
            self.console.print("[red]/compare needs at least 2 models.[/red]")
            return
        self.state.compare_models = valid
        models_str = ", ".join(f"[cyan]{m}[/cyan]" for m in valid)
        self.console.print(
            f"[green]Compare mode set.[/green] Next message → {models_str}")

    def _cmd_system(self, arg: str) -> None:
        new_system = arg or None
        self.store.update_system_prompt(self.state.session.id, new_system)
        self.state.session.system_prompt = new_system
        if new_system:
            self.console.print("[green]System prompt set.[/green]")
        else:
            self.console.print("[green]System prompt cleared.[/green]")

    def _cmd_title(self, arg: str) -> None:
        if not arg:
            self.console.print(
                f"Current title: [cyan]{self.state.session.title}[/cyan]")
            return
        self.store.update_session_title(self.state.session.id, arg)
        self.state.session.title = arg
        self.console.print(f"[green]Title set to[/green] [cyan]{arg}[/cyan]")

    def _cmd_clear(self) -> None:
        new_sess = self.store.create_session(
            model=self.state.session.current_model,
            title="New chat",
            system_prompt=self.state.session.system_prompt,
        )
        self.state.session = new_sess
        self.state.approval = ApprovalState()  # fresh approvals per session
        self.console.print(f"[green]New session[/green] [dim]{new_sess.id}[/dim]")
        self._print_header()

    def _cmd_history(self) -> None:
        msgs = self.store.get_messages(self.state.session.id)
        if not msgs:
            self.console.print("[dim](no messages yet)[/dim]")
            return
        for m in msgs:
            style = "cyan" if m.role == "user" else "green"
            tag = m.role
            if m.model:
                tag += f" · {m.model}"
            self.console.print(Rule(f"[{style}]{tag}[/{style}]"))
            self.console.print(m.content)

    def _cmd_tools(self) -> None:
        tools = self.ext.mcp_manager.list_tools()
        status = self.ext.mcp_manager.server_status()
        if not status:
            self.console.print(
                "[dim]No MCP servers configured. Add an mcp.json under "
                "~/.llmchat/ or .llmchat/ to expose tools.[/dim]")
            return
        for srv, st in status.items():
            self.console.print(f"  [cyan]{srv}[/cyan]: {st}")
        if tools:
            self.console.print()
            for t in tools:
                desc = t.description.splitlines()[0] if t.description else ""
                self.console.print(
                    f"  [green]{t.fq_name}[/green]  [dim]{desc}[/dim]")

    def _cmd_plugins(self) -> None:
        sk = self.ext.skills.all()
        cmds = list(self.ext.commands.values())
        hk_total = sum(len(v) for v in self.ext.hooks.by_event.values())
        self.console.print(
            f"skills: [cyan]{len(sk)}[/cyan]   "
            f"commands: [cyan]{len(cmds)}[/cyan]   "
            f"hooks: [cyan]{hk_total}[/cyan]")
        if sk:
            self.console.print("\n[bold]skills[/bold]")
            for s in sk:
                trigger = ("always" if s.is_always_on() else
                           f"triggers={s.triggers}" if s.triggers else
                           f"patterns={s.file_patterns}")
                self.console.print(
                    f"  [cyan]{s.name}[/cyan]  [dim]{trigger}[/dim]")
        if cmds:
            self.console.print("\n[bold]commands[/bold]")
            for c in cmds:
                self.console.print(
                    f"  [cyan]/{c.name}[/cyan]  [dim]{c.description}[/dim]")
        if hk_total:
            self.console.print("\n[bold]hooks[/bold]")
            for event, hooks in self.ext.hooks.by_event.items():
                for h in hooks:
                    self.console.print(
                        f"  [cyan]{event.value}[/cyan]: {' '.join(h.command)}")

    # ----- the user turn ----------------------------------------------------

    def _handle_user_turn(self, raw_text: str) -> None:
        # 1. pre_user_message hooks.
        hook_result = self.ext.hooks.dispatch(
            HookEvent.PRE_USER_MESSAGE, raw_text)
        if hook_result.blocked:
            self.console.print(
                f"[red]message blocked by hook:[/red] {hook_result.stderr.strip()}")
            return
        text = hook_result.transformed_payload or raw_text

        # 2. Decide whether tools should be in play.
        tools = self.ext.mcp_manager.list_tools()
        need_tools = bool(tools)

        # 5 (early). Build system prompt — needed by both paths below.
        active_skills = self.ext.skills.select_for(text)
        system_prompt = self._compose_system_prompt(active_skills)

        # Compare mode: send this prompt to multiple models in parallel,
        # then clear the flag so the next turn is normal.
        if self.state.compare_models:
            compare_models = self.state.compare_models
            self.state.compare_models = None
            self.store.add_message(
                self.state.session.id, "user", text, model=compare_models[0])
            self.console.print()
            self._handle_compare_turn(text, compare_models, system_prompt)
            self.console.print()
            return

        # 3. Pick a model via routing (or pinned override).
        try:
            model, explanation = self._resolve_model_for_turn(text, need_tools)
        except ValueError as e:
            self.console.print(f"[red]{e}[/red]")
            return

        # 4. Persist the user message.
        self.store.add_message(self.state.session.id, "user", text, model=model)

        self.console.print()
        if explanation:
            self.console.print(f"[dim]→ {model} · {explanation}[/dim]")
        else:
            self.console.print(f"[dim]assistant · {model}[/dim]")

        # 6. Run agentic loop, escalating to the next tier on rate limits.
        tried: list[str] = []
        current_model = model
        final_text = ""
        while True:
            provider_name, model_name = parse_model(current_model)
            try:
                provider = get_provider(provider_name)
            except ProviderError as e:
                self.console.print(f"[red]{e}[/red]")
                return
            tried.append(current_model)
            try:
                final_text = self._run_agentic_loop(
                    provider=provider, model_name=model_name,
                    model_label=current_model,
                    system_prompt=system_prompt, tools=tools)
                break
            except RateLimitError:
                next_model = self._failover_model(current_model, tried)
                if next_model is None:
                    self.console.print(
                        f"[red]Rate limited on {current_model} "
                        f"and no fallback available.[/red]")
                    return
                self.console.print(
                    f"[yellow]Rate limited on {current_model} "
                    f"→ failing over to {next_model}[/yellow]")
                current_model = next_model

        # 7. post_assistant_message hooks.
        if final_text:
            self.ext.hooks.dispatch(
                HookEvent.POST_ASSISTANT_MESSAGE, final_text)

        self.console.print()

    def _compose_system_prompt(self, active_skills) -> str | None:
        base = self.state.session.system_prompt or ""
        addition = self.ext.skills.render_system_prompt_addition(active_skills)
        combined = "\n\n".join(p for p in (base, addition) if p)
        return combined or None

    def _failover_model(self, current_model: str,
                        already_tried: list[str]) -> str | None:
        """Return the next-tier model to try after a rate limit, or None."""
        order = ["local", "cheap", "flagship"]
        current_idx = -1
        for i, t in enumerate(order):
            tier = self.routing_config.tiers.get(t)
            if tier and tier.model == current_model:
                current_idx = i
                break
        for t in order[current_idx + 1:]:
            tier = self.routing_config.tiers.get(t)
            if tier and tier.model and tier.model not in already_tried:
                return tier.model
        return None

    def _handle_compare_turn(self, text: str, models: list[str],
                              system_prompt: str | None) -> None:
        """Query multiple models in parallel and display results as panels."""
        import threading

        api_messages = self._load_history_as_messages()
        results: dict[str, str] = {}
        errors: dict[str, str] = {}
        lock = threading.Lock()

        def run_one(model_label: str) -> None:
            provider_name, model_name = parse_model(model_label)
            try:
                provider = get_provider(provider_name)
            except ProviderError as e:
                with lock:
                    errors[model_label] = str(e)
                return
            chunks: list[str] = []
            try:
                for chunk in provider.stream(
                        model=model_name, messages=api_messages,
                        system=system_prompt, tools=None):
                    if isinstance(chunk, str):
                        chunks.append(chunk)
            except ProviderError as e:
                with lock:
                    errors[model_label] = str(e)
                return
            with lock:
                results[model_label] = "".join(chunks)

        self.console.print(
            f"[dim]Querying {len(models)} models in parallel…[/dim]")
        threads = [threading.Thread(target=run_one, args=(m,), daemon=True)
                   for m in models]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        for model_label in models:
            if model_label in errors:
                self.console.print(Panel(
                    f"[red]{errors[model_label]}[/red]",
                    title=f"[red]{model_label} — error[/red]",
                    border_style="red"))
            elif model_label in results:
                self.console.print(Panel(
                    results[model_label],
                    title=f"[cyan]{model_label}[/cyan]",
                    border_style="cyan"))
                self._store_assistant_turn(results[model_label], [], model_label)

    # ----- agentic loop -----------------------------------------------------

    def _run_agentic_loop(self, provider, model_name: str, model_label: str,
                          system_prompt: str | None, tools) -> str:
        """Run one user turn, possibly across multiple model invocations
        if the model requests tool calls. Returns the final assistant text
        for use by post-message hooks."""
        # Pull persisted history into our internal format. The loop adds
        # to this list as the model responds.
        api_messages: list[ChatMessage] = self._load_history_as_messages()

        last_assistant_text = ""
        for iteration in range(MAX_TOOL_ITERATIONS):
            # Stream from the provider.
            text_chunks: list[str] = []
            tool_calls: list[ToolCall] = []
            stop_reason: str | None = None

            try:
                stream = provider.stream(
                    model=model_name,
                    messages=api_messages,
                    system=system_prompt,
                    tools=tools if tools else None,
                )
                if not tools:
                    # back-compat path: stream is just strings
                    for chunk in stream:
                        text_chunks.append(chunk)
                        self.console.print(chunk, end="", soft_wrap=True,
                                            highlight=False)
                    self.console.print()
                else:
                    for ev in stream:
                        if isinstance(ev, TextDelta):
                            text_chunks.append(ev.text)
                            self.console.print(ev.text, end="",
                                                soft_wrap=True, highlight=False)
                        elif isinstance(ev, ToolUseRequest):
                            tool_calls = ev.calls
                        elif isinstance(ev, TurnEnd):
                            stop_reason = ev.stop_reason
                    self.console.print()
            except RateLimitError:
                raise  # propagate so _handle_user_turn can attempt failover
            except ProviderError as e:
                self.console.print(f"\n[red]Error: {e}[/red]")
                break
            except KeyboardInterrupt:
                self.console.print("\n[yellow](interrupted)[/yellow]")
                break

            assistant_text = "".join(text_chunks)
            last_assistant_text = assistant_text

            # Persist the assistant turn (with any tool_calls embedded so a
            # future resume can re-run the conversation correctly).
            if assistant_text or tool_calls:
                self._store_assistant_turn(
                    assistant_text, tool_calls, model_label)

            # Update in-memory message list with the assistant's turn so it's
            # included in the next iteration's request.
            api_messages.append(ChatMessage(
                role="assistant", content=assistant_text,
                tool_calls=tool_calls or None))

            if not tool_calls:
                break  # done — model gave a text response with no tool requests

            # Execute tool calls (with approval).
            results = self._execute_tool_calls(tool_calls)
            self._store_tool_results(results)
            api_messages.append(ChatMessage(
                role="tool", content="", tool_results=results))

            # Continue the loop — model gets to see results and respond.
        else:
            self.console.print(
                f"[yellow]Stopped after {MAX_TOOL_ITERATIONS} tool iterations. "
                f"Send another message to continue.[/yellow]")

        return last_assistant_text

    def _load_history_as_messages(self) -> list[ChatMessage]:
        """Reconstitute persisted messages into the format providers want.

        Persisted messages may carry tool data in their ``content`` (we
        store tool calls and results as JSON). We deserialize and rebuild
        the ChatMessage with structured fields."""
        import json
        out: list[ChatMessage] = []
        for m in self.store.get_messages(self.state.session.id):
            if m.role == "user":
                out.append(ChatMessage(role="user", content=m.content))
            elif m.role == "assistant":
                # Check for tool_calls metadata stored as a JSON suffix.
                msg = ChatMessage(role="assistant", content=m.content)
                tool_calls_meta = _extract_meta(m.content, "tool_calls")
                if tool_calls_meta:
                    msg.content = _strip_meta(m.content)
                    msg.tool_calls = [
                        ToolCall(id=c["id"], fq_name=c["fq_name"],
                                  arguments=c.get("arguments") or {})
                        for c in tool_calls_meta
                    ]
                out.append(msg)
            elif m.role == "tool":
                tool_results_meta = _extract_meta(m.content, "tool_results")
                if tool_results_meta:
                    out.append(ChatMessage(
                        role="tool", content="",
                        tool_results=[
                            ToolResult(id=r["id"], fq_name=r["fq_name"],
                                        content=r["content"],
                                        is_error=r.get("is_error", False))
                            for r in tool_results_meta
                        ]))
        return out

    def _store_assistant_turn(self, text: str,
                               tool_calls: list[ToolCall],
                               model_label: str) -> None:
        # Persist as a single assistant message. If there are tool calls,
        # we attach them as a JSON-encoded metadata block at the end of the
        # content so we can reconstruct on resume without changing the
        # storage schema.
        content = text
        if tool_calls:
            content = _attach_meta(content, "tool_calls", [
                {"id": c.id, "fq_name": c.fq_name, "arguments": c.arguments}
                for c in tool_calls
            ])
        self.store.add_message(
            self.state.session.id, "assistant", content, model=model_label)

    def _store_tool_results(self, results: list[ToolResult]) -> None:
        if not results:
            return
        content = _attach_meta("", "tool_results", [
            {"id": r.id, "fq_name": r.fq_name,
             "content": r.content, "is_error": r.is_error}
            for r in results
        ])
        self.store.add_message(self.state.session.id, "tool", content)

    def _execute_tool_calls(self, calls: list[ToolCall]) -> list[ToolResult]:
        results: list[ToolResult] = []
        for call in calls:
            # Approval gate.
            if not self.state.approval.is_approved(call.fq_name):
                decision = self._prompt_for_approval(call)
                self.state.approval.record(call.fq_name, decision)
                if decision == ApprovalDecision.DENY:
                    results.append(ToolResult(
                        id=call.id, fq_name=call.fq_name,
                        content="user denied tool call", is_error=True))
                    continue

            # Pre-tool-call hook.
            import json
            hook_payload = json.dumps({
                "tool": call.fq_name, "arguments": call.arguments})
            hook_res = self.ext.hooks.dispatch(
                HookEvent.PRE_TOOL_CALL, hook_payload)
            if hook_res.blocked:
                results.append(ToolResult(
                    id=call.id, fq_name=call.fq_name,
                    content=f"hook blocked tool: {hook_res.stderr.strip()}",
                    is_error=True))
                continue

            self.console.print(
                f"[dim]→ tool: [cyan]{call.fq_name}[/cyan] "
                f"args={_short_args(call.arguments)}[/dim]")

            result = self.ext.mcp_manager.call_tool(
                call.fq_name, call.arguments)
            # Stamp the call id onto the result so providers can correlate.
            result = ToolResult(
                id=call.id, fq_name=call.fq_name,
                content=result.content, is_error=result.is_error)
            results.append(result)

            self.ext.hooks.dispatch(
                HookEvent.POST_TOOL_CALL,
                json.dumps({"tool": call.fq_name,
                             "result": result.content,
                             "is_error": result.is_error}))

            # Show a short preview.
            preview = result.content[:200].replace("\n", " ")
            color = "red" if result.is_error else "dim"
            self.console.print(f"[{color}]   {preview}[/{color}]")

        return results

    def _prompt_for_approval(self, call: ToolCall) -> ApprovalDecision:
        if self.state.approval.auto_approve_all:
            return ApprovalDecision.APPROVE_SESSION
        self.console.print(Panel(
            f"[bold]{call.fq_name}[/bold]\n"
            f"args: {_short_args(call.arguments, max_len=400)}",
            title="Tool call requested",
            border_style="yellow"))
        try:
            answer = self.prompt_session.prompt(
                "Allow? [y/Y/n] (y=once, Y=session, n=deny) > ").strip()
        except (EOFError, KeyboardInterrupt):
            return ApprovalDecision.DENY
        if answer == "Y":
            return ApprovalDecision.APPROVE_SESSION
        if answer.lower() == "y":
            return ApprovalDecision.APPROVE_ONCE
        return ApprovalDecision.DENY

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
        # MCP/extensions summary.
        ext_summary = []
        sk_count = len(self.ext.skills.all())
        cmd_count = len(self.ext.commands)
        tool_count = len(self.ext.mcp_manager.list_tools())
        if sk_count:
            ext_summary.append(f"{sk_count} skill(s)")
        if cmd_count:
            ext_summary.append(f"{cmd_count} command(s)")
        if tool_count:
            ext_summary.append(f"{tool_count} tool(s)")
        if ext_summary:
            self.console.print(
                f"[dim]extensions: {', '.join(ext_summary)} — /plugins for details[/dim]")
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
            tools_tag = "" if tier.supports_tools else " [dim](no tools)[/dim]"
            self.console.print(
                f"  [cyan]{tier_name}[/cyan]: {tier.model} "
                f"[dim](≤ {tier.max_input_chars} chars)[/dim]{tools_tag}")


def run_chat(store: Store, session: Session,
             routing_config: RoutingConfig,
             extensions: ExtensionsBundle,
             pinned_model: str | None = None) -> None:
    extensions.mcp_manager.start()
    try:
        loop = ChatLoop(store=store, session=session,
                        routing_config=routing_config,
                        extensions=extensions)
        if pinned_model:
            loop.set_pinned_model(pinned_model)
        loop.run()
    finally:
        extensions.mcp_manager.stop()


# ----- helpers --------------------------------------------------------------

# Tool-call/result metadata is appended to message content as a JSON-encoded
# block delimited by a sentinel. This keeps the SQLite schema unchanged while
# still letting us reconstruct structured tool data on resume.
_META_PREFIX = "\n\n<!--llmchat:"
_META_SUFFIX = "-->"


def _attach_meta(content: str, key: str, value) -> str:
    import json
    return content + _META_PREFIX + key + ":" + json.dumps(value) + _META_SUFFIX


def _extract_meta(content: str, key: str):
    """Return parsed metadata for ``key``, or None if not present."""
    import json
    needle = _META_PREFIX + key + ":"
    idx = content.find(needle)
    if idx < 0:
        return None
    end = content.find(_META_SUFFIX, idx)
    if end < 0:
        return None
    payload = content[idx + len(needle):end]
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        return None


def _strip_meta(content: str) -> str:
    """Remove all metadata blocks for display purposes."""
    out = content
    while True:
        idx = out.find(_META_PREFIX)
        if idx < 0:
            return out.rstrip()
        end = out.find(_META_SUFFIX, idx)
        if end < 0:
            return out.rstrip()
        out = out[:idx] + out[end + len(_META_SUFFIX):]


def _short_args(args: dict, max_len: int = 80) -> str:
    import json
    s = json.dumps(args, separators=(",", ":"), ensure_ascii=False)
    return s if len(s) <= max_len else s[:max_len] + "..."
