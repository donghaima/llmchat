"""Top-level CLI entry point.

Subcommands:
  llmchat                  Start a new chat with auto-routing
  llmchat new              Same, explicit
  llmchat resume <id>      Resume an existing session
  llmchat list             List recent sessions
  llmchat show <id>        Print a full transcript
  llmchat export <id>      Print transcript as Markdown
  llmchat models           List available models per provider
  llmchat delete <id>      Delete a session
  llmchat route show       Show current routing config
  llmchat route init       Write a starter routing.toml
  llmchat route test "..." Dry-run the router on a sample prompt
"""

from __future__ import annotations

import shutil
import sys
from datetime import datetime
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from .chat import ExtensionsBundle, run_chat
from .config import (
    config_path_for,
    load_routing_config,
    write_default_config,
)
from .extensions import (
    HookEvent,
    discover_roots,
    load_commands,
    load_hooks,
    load_plugins,
    load_skills,
)
from .extensions.plugins import merge_plugins_into
from .mcp.manager import MCPManager, load_mcp_configs
from .providers import (
    ChatMessage,
    all_provider_names,
    get_provider,
    parse_model,
    ProviderError,
)
from .routing import Router
from .storage import Store


console = Console()


@click.group(invoke_without_command=True)
@click.option("--model", "-m", default=None,
              help="Pin a specific model (disables auto-routing for this run).")
@click.option("--system", "-s", default=None,
              help="System prompt for a new session.")
@click.option("--title", "-t", default="New chat",
              help="Title for a new session.")
@click.option("--no-route", is_flag=True, default=False,
              help="Disable auto-routing for this run.")
@click.pass_context
def cli(ctx: click.Context, model: str | None, system: str | None,
        title: str, no_route: bool) -> None:
    """Multi-provider LLM chat CLI with persistent sessions and smart routing."""
    ctx.ensure_object(dict)
    store = Store()
    ctx.obj["store"] = store
    ctx.obj["routing_config"] = load_routing_config(store.db_path)
    ctx.obj["pinned_model"] = model
    ctx.obj["system"] = system
    ctx.obj["title"] = title

    if no_route:
        ctx.obj["routing_config"].enabled = False

    if ctx.invoked_subcommand is None:
        ctx.invoke(new)


@cli.command()
@click.pass_context
def new(ctx: click.Context) -> None:
    """Start a new chat session."""
    store: Store = ctx.obj["store"]
    pinned: str | None = ctx.obj["pinned_model"]
    routing_config = ctx.obj["routing_config"]

    # Validate the pinned model up front, if there is one.
    if pinned:
        provider_name, model_name = parse_model(pinned)
        full_model = f"{provider_name}:{model_name}"
        try:
            get_provider(provider_name)
        except ProviderError as e:
            console.print(f"[red]{e}[/red]")
            sys.exit(1)
    else:
        # Use the routing default tier as the session's "current_model" — it's
        # only there for resume metadata; the router still runs per-turn.
        full_model = _default_session_model(routing_config)

    session = store.create_session(
        model=full_model,
        title=ctx.obj["title"],
        system_prompt=ctx.obj["system"],
    )
    extensions = _build_extensions()
    run_chat(store, session, routing_config, extensions, pinned_model=pinned)


@cli.command()
@click.argument("session_id")
@click.pass_context
def resume(ctx: click.Context, session_id: str) -> None:
    """Resume an existing session (id prefix is fine)."""
    store: Store = ctx.obj["store"]
    routing_config = ctx.obj["routing_config"]
    session = store.get_session(session_id)
    if not session:
        console.print(f"[red]No session found matching {session_id!r}.[/red]")
        sys.exit(1)
    pinned: str | None = ctx.obj.get("pinned_model")
    if pinned:
        provider_name, model_name = parse_model(pinned)
        full = f"{provider_name}:{model_name}"
        store.update_session_model(session.id, full)
        session.current_model = full
    extensions = _build_extensions()
    run_chat(store, session, routing_config, extensions, pinned_model=pinned)


@cli.command(name="list")
@click.option("--limit", "-n", default=20, help="How many sessions to show.")
@click.pass_context
def list_sessions(ctx: click.Context, limit: int) -> None:
    """List recent sessions."""
    store: Store = ctx.obj["store"]
    sessions = store.list_sessions(limit=limit)
    if not sessions:
        console.print("[dim]No sessions yet. Run `llmchat` to start one.[/dim]")
        return

    table = Table(show_header=True, header_style="bold")
    table.add_column("id", style="dim", no_wrap=True)
    table.add_column("title")
    table.add_column("model", style="cyan")
    table.add_column("updated")
    table.add_column("msgs", justify="right")

    for s in sessions:
        updated = datetime.fromtimestamp(s.updated_at).strftime("%Y-%m-%d %H:%M")
        msg_count = len(store.get_messages(s.id))
        table.add_row(s.id[:8], s.title, s.current_model, updated, str(msg_count))

    console.print(table)


@cli.command()
@click.argument("session_id")
@click.pass_context
def show(ctx: click.Context, session_id: str) -> None:
    """Print the full transcript of a session."""
    store: Store = ctx.obj["store"]
    session = store.get_session(session_id)
    if not session:
        console.print(f"[red]No session found matching {session_id!r}.[/red]")
        sys.exit(1)

    console.print(f"[bold]{session.title}[/bold] [dim]{session.id}[/dim]")
    console.print(f"model: [cyan]{session.current_model}[/cyan]")
    if session.system_prompt:
        console.print(f"[dim]system:[/dim] {session.system_prompt}")
    console.print()

    for m in store.get_messages(session.id):
        ts = datetime.fromtimestamp(m.created_at).strftime("%H:%M:%S")
        if m.role == "user":
            console.print(f"[cyan]you[/cyan] [dim]{ts}[/dim]")
        else:
            tag = m.role
            if m.model:
                tag += f" · {m.model}"
            console.print(f"[green]{tag}[/green] [dim]{ts}[/dim]")
        console.print(m.content)
        console.print()


@cli.command()
@click.argument("session_id")
@click.pass_context
def export(ctx: click.Context, session_id: str) -> None:
    """Print a session as Markdown (pipe to a file)."""
    store: Store = ctx.obj["store"]
    session = store.get_session(session_id)
    if not session:
        console.print(f"[red]No session found matching {session_id!r}.[/red]")
        sys.exit(1)

    out = sys.stdout
    out.write(f"# {session.title}\n\n")
    out.write(f"- **session id:** `{session.id}`\n")
    out.write(f"- **model:** `{session.current_model}`\n")
    if session.system_prompt:
        out.write(f"- **system prompt:** {session.system_prompt}\n")
    out.write("\n---\n\n")
    for m in store.get_messages(session.id):
        ts = datetime.fromtimestamp(m.created_at).isoformat(timespec="seconds")
        if m.role == "user":
            out.write(f"### user · {ts}\n\n")
        else:
            tag = m.role
            if m.model:
                tag += f" · {m.model}"
            out.write(f"### {tag} · {ts}\n\n")
        out.write(m.content)
        out.write("\n\n")


@cli.command()
@click.pass_context
def models(ctx: click.Context) -> None:
    """List available models from each configured provider."""
    table = Table(show_header=True, header_style="bold")
    table.add_column("provider", style="cyan")
    table.add_column("model")
    table.add_column("status")

    any_configured = False
    for name in all_provider_names():
        try:
            provider = get_provider(name)
        except ProviderError as e:
            table.add_row(name, "—", f"[red]{e}[/red]")
            continue

        if not provider.is_configured():
            table.add_row(name, "—", "[yellow]not configured[/yellow]")
            continue
        any_configured = True

        try:
            ids = provider.list_models()
        except Exception as e:
            table.add_row(name, "—", f"[red]listing failed: {e}[/red]")
            continue

        if not ids:
            table.add_row(name, "—", "[yellow]no models found[/yellow]")
            continue

        for i, mid in enumerate(ids):
            table.add_row(name if i == 0 else "", mid,
                          "[green]ok[/green]" if i == 0 else "")

    console.print(table)
    if not any_configured:
        console.print(
            "\n[yellow]No providers configured.[/yellow] "
            "Set ANTHROPIC_API_KEY, OPENAI_API_KEY, GOOGLE_API_KEY, or run "
            "Ollama locally.")


@cli.command()
@click.argument("session_id")
@click.confirmation_option(prompt="Delete this session permanently?")
@click.pass_context
def delete(ctx: click.Context, session_id: str) -> None:
    """Delete a session and all its messages."""
    store: Store = ctx.obj["store"]
    session = store.get_session(session_id)
    if not session:
        console.print(f"[red]No session found matching {session_id!r}.[/red]")
        sys.exit(1)
    store.delete_session(session.id)
    console.print(f"[green]Deleted session[/green] [dim]{session.id}[/dim]")


# ----- routing subcommands --------------------------------------------------

@cli.group()
def route() -> None:
    """Inspect and manage routing config."""


@route.command(name="show")
@click.pass_context
def route_show(ctx: click.Context) -> None:
    """Show the current routing config."""
    store: Store = ctx.obj["store"]
    cfg = ctx.obj["routing_config"]
    path = config_path_for(store.db_path)
    if path.exists():
        console.print(f"[dim]config:[/dim] {path}")
    else:
        console.print(f"[dim]config:[/dim] (using defaults — "
                      f"run `llmchat route init` to create {path})")
    console.print(
        f"enabled: [cyan]{cfg.enabled}[/cyan]   "
        f"default tier: [cyan]{cfg.default_tier}[/cyan]   "
        f"explain: [cyan]{cfg.explain}[/cyan]")
    table = Table(show_header=True, header_style="bold")
    table.add_column("tier", style="cyan")
    table.add_column("model")
    table.add_column("max input chars", justify="right")
    table.add_column("description", style="dim")
    for tier_name in ("local", "cheap", "flagship"):
        tier = cfg.tiers.get(tier_name)
        if not tier or not tier.model:
            table.add_row(tier_name, "[yellow](not configured)[/yellow]", "—", "")
            continue
        table.add_row(tier_name, tier.model,
                      f"{tier.max_input_chars:,}", tier.description)
    console.print(table)


@route.command(name="init")
@click.pass_context
def route_init(ctx: click.Context) -> None:
    """Write a starter routing.toml the user can edit."""
    store: Store = ctx.obj["store"]
    path = config_path_for(store.db_path)
    existed = path.exists()
    write_default_config(store.db_path)
    if existed:
        console.print(
            f"Config already exists at [cyan]{path}[/cyan] [dim](unchanged)[/dim]")
    else:
        console.print(f"[green]Created[/green] [cyan]{path}[/cyan]")


@route.command(name="test")
@click.argument("prompt")
@click.option("--history", "-h", multiple=True,
              help="Add a turn of fake history (repeatable). "
                   "Format: 'user:hello' or 'assistant:hi'.")
@click.pass_context
def route_test(ctx: click.Context, prompt: str, history: tuple[str, ...]) -> None:
    """Dry-run the router on a sample prompt — useful for tuning."""
    cfg = ctx.obj["routing_config"]
    router = Router(cfg)

    msgs: list[ChatMessage] = []
    for h in history:
        if ":" not in h:
            console.print(f"[red]Bad --history format: {h!r} "
                          f"(expected role:content)[/red]")
            sys.exit(1)
        role, content = h.split(":", 1)
        msgs.append(ChatMessage(role=role.strip(), content=content.strip()))

    decision = router.route(prompt, msgs)
    console.print(f"prompt: [cyan]{prompt[:80]}{'...' if len(prompt) > 80 else ''}[/cyan]")
    console.print(f"  history: {len(msgs)} message(s), "
                  f"{sum(len(m.content) for m in msgs)} chars total")
    console.print(f"  → tier: [cyan]{decision.tier}[/cyan]")
    console.print(f"  → model: [cyan]{decision.model}[/cyan]")
    if decision.reasons:
        console.print(f"  → reasons: {', '.join(decision.reasons)}")
    else:
        console.print(f"  → reasons: (default tier)")


# ----- plugins subcommands --------------------------------------------------

@cli.group()
def plugins() -> None:
    """Manage extensions: plugins, skills, commands, and hooks."""


@plugins.command(name="list")
def plugins_list_cmd() -> None:
    """List all loaded plugins, skills, commands, and hooks."""
    roots = discover_roots()
    skills = load_skills(roots)
    commands = load_commands(roots)
    hooks = load_hooks(roots)
    loaded_plugins = load_plugins(roots)
    merge_plugins_into(loaded_plugins, skills, commands, hooks)

    table = Table(show_header=True, header_style="bold")
    table.add_column("type", style="cyan")
    table.add_column("name")
    table.add_column("detail", style="dim")

    for sk in skills.all():
        trigger = ("always" if sk.is_always_on() else
                   f"triggers: {', '.join(sk.triggers)}" if sk.triggers else
                   f"patterns: {', '.join(sk.file_patterns)}")
        table.add_row("skill", sk.name, trigger)

    for cmd in commands.values():
        table.add_row("command", f"/{cmd.name}", cmd.description)

    for event, hook_list in hooks.by_event.items():
        for h in hook_list:
            table.add_row("hook", event.value, " ".join(h.command))

    for plugin in loaded_plugins:
        table.add_row("plugin", plugin.name,
                      f"v{plugin.version}  {plugin.description}")

    if table.row_count == 0:
        console.print(
            "[dim]No extensions found. Add skills/commands/hooks under "
            "~/.llmchat/ or .llmchat/ in your project directory.[/dim]")
    else:
        console.print(table)


@plugins.command(name="show")
@click.argument("name")
def plugins_show_cmd(name: str) -> None:
    """Show details of a specific plugin."""
    roots = discover_roots()
    loaded_plugins = load_plugins(roots)
    plugin = next((p for p in loaded_plugins if p.name == name), None)
    if plugin is None:
        console.print(f"[red]Plugin {name!r} not found.[/red]")
        console.print("Run `llmchat plugins list` to see what's loaded.")
        sys.exit(1)

    console.print(f"[bold]{plugin.name}[/bold]  v{plugin.version}")
    if plugin.description:
        console.print(plugin.description)
    console.print(f"[dim]root:[/dim] {plugin.root}")
    if plugin.manifest_path:
        console.print(f"[dim]manifest:[/dim] {plugin.manifest_path}")

    plugin_roots = plugin.as_extension_root()
    plugin_skills = load_skills(plugin_roots)
    plugin_cmds = load_commands(plugin_roots)
    plugin_hooks = load_hooks(plugin_roots)

    if plugin_skills.skills:
        console.print(f"\n[bold]skills ({len(plugin_skills.skills)})[/bold]")
        for sk in plugin_skills.all():
            console.print(f"  [cyan]{sk.name}[/cyan]  [dim]{sk.description}[/dim]")

    if plugin_cmds:
        console.print(f"\n[bold]commands ({len(plugin_cmds)})[/bold]")
        for cmd in plugin_cmds.values():
            console.print(f"  [cyan]/{cmd.name}[/cyan]  [dim]{cmd.description}[/dim]")

    hook_total = sum(len(v) for v in plugin_hooks.by_event.values())
    if hook_total:
        console.print(f"\n[bold]hooks ({hook_total})[/bold]")
        for event, hook_list in plugin_hooks.by_event.items():
            for h in hook_list:
                console.print(f"  [cyan]{event.value}[/cyan]: {' '.join(h.command)}")

    mcp_path = plugin.mcp_config_path()
    if mcp_path:
        console.print(f"\n[bold]MCP config[/bold]  [dim]{mcp_path}[/dim]")
        import json
        try:
            data = json.loads(mcp_path.read_text(encoding="utf-8"))
            servers = data.get("mcpServers") or data.get("mcp_servers") or {}
            for sname in servers:
                console.print(f"  [cyan]{sname}[/cyan]")
        except Exception:
            pass


@plugins.command(name="install")
@click.argument("source")
@click.option("--project", "scope", flag_value="project",
              help="Install into .llmchat/plugins/ in the current directory.")
@click.option("--user", "scope", flag_value="user", default=True,
              help="Install into ~/.llmchat/plugins/ (default).")
def plugins_install_cmd(source: str, scope: str) -> None:
    """Install a plugin from a local directory.

    SOURCE is the path to the plugin directory. The directory is copied into
    the user (~/.llmchat/plugins/) or project (.llmchat/plugins/) extension
    root. The plugin name is taken from the directory name.
    """
    src = Path(source).resolve()
    if not src.is_dir():
        console.print(f"[red]Not a directory: {src}[/red]")
        sys.exit(1)
    if scope == "project":
        dest_root = Path.cwd() / ".llmchat" / "plugins"
    else:
        dest_root = Path.home() / ".llmchat" / "plugins"

    dest = dest_root / src.name
    if dest.exists():
        console.print(
            f"[red]Plugin {src.name!r} already exists at {dest}.[/red] "
            f"Remove it first or rename the source directory.")
        sys.exit(1)

    dest_root.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dest)
    console.print(f"[green]Installed[/green] [cyan]{src.name!r}[/cyan] → {dest}")
    console.print("Run `llmchat plugins list` to verify it loaded correctly.")


# ----- helpers --------------------------------------------------------------

def _build_extensions() -> ExtensionsBundle:
    """Discover and load all extensions (skills, commands, hooks, plugins, MCP)."""
    roots = discover_roots()
    skills = load_skills(roots)
    commands = load_commands(roots)
    hooks = load_hooks(roots)
    loaded_plugins = load_plugins(roots)
    merge_plugins_into(loaded_plugins, skills, commands, hooks)

    # Collect MCP configs from roots and plugins (last-wins for same server name).
    mcp_paths: list[Path] = []
    for root in roots.existing():
        for name in (".mcp.json", "mcp.json"):
            p = root / name
            if p.is_file():
                mcp_paths.append(p)
    for plugin in loaded_plugins:
        p = plugin.mcp_config_path()
        if p:
            mcp_paths.append(p)

    mcp_configs = load_mcp_configs(mcp_paths)
    manager = MCPManager(mcp_configs)

    return ExtensionsBundle(
        skills=skills,
        commands=commands,
        hooks=hooks,
        mcp_manager=manager,
    )


def _default_session_model(routing_config) -> str:
    """Pick a reasonable 'current_model' value for new sessions when no
    --model flag was given. We use the default tier's model if available,
    otherwise the first configured tier."""
    default_tier = routing_config.tiers.get(routing_config.default_tier)
    if default_tier and default_tier.model:
        return default_tier.model
    avail = routing_config.available_tiers()
    if avail:
        return routing_config.tiers[avail[0]].model
    # Last resort.
    return "anthropic:claude-haiku-4-5"


def main() -> None:
    cli(obj={})


if __name__ == "__main__":
    main()
