# llmchat: Multi-Provider LLM Chat with Smart Routing

A minimal, no-nonsense CLI for chatting with multiple LLM providers (Anthropic Claude, OpenAI GPT, Google Gemini, Ollama local) in one terminal. Sessions are persisted to SQLite, and **every turn is automatically routed** to the cheapest model that can plausibly handle it — local for quick lookups, cheap paid for everyday tasks, flagship for hard problems.

## Overview

**Core Philosophy**: Smart routing without extra LLM calls. A heuristic-based router (no classifier) reads the prompt length, conversation depth, and semantic signals (reasoning words, code complexity) to pick a tier (local/cheap/flagship) in microseconds. Deliberately simple to understand and tune.

**Architecture**: Modular provider abstraction, persistent SQLite storage, extensible via skills/commands/hooks, MCP tool support.

**Current State**: Version 0.1.0. Initial commit with core chat, routing, multi-provider support, and extension framework.

---

## Project Structure

```
llmchat/
├── chat.py              # Interactive REPL + agentic loop with tool support
├── cli.py               # Click commands: new, resume, list, show, export, route, models
├── config.py            # Load/parse routing.toml config files
├── routing.py           # Heuristic router: pick tier based on prompt signals
├── storage.py           # SQLite schema + Store class (sessions, messages)
├── providers/
│   ├── base.py          # Provider ABC, ChatMessage, ProviderError
│   ├── anthropic_provider.py   # Anthropic with tool-use support
│   ├── openai_provider.py      # OpenAI (text-only initially)
│   ├── google_provider.py      # Google Gemini
│   ├── ollama_provider.py      # Local Ollama
│   └── __init__.py      # Registry: get_provider(), parse_model()
├── extensions/
│   ├── skills.py        # Skills: markdown files injected into system prompt
│   ├── commands.py      # Slash commands: /name templates with $ARGS
│   ├── hooks.py         # Subprocess hooks at lifecycle events
│   ├── plugins.py       # Plugin bundles (skills + commands + hooks)
│   ├── discovery.py     # Find extension roots (user + project)
│   └── __init__.py
├── mcp/
│   ├── manager.py       # MCPManager: sync facade over async MCP SDK
│   ├── types.py         # ToolDescriptor, ToolCall, ToolResult, StreamEvent
│   ├── approval.py      # User approval gates for tool calls
│   └── __init__.py
└── tests/
    └── test_smoke.py    # Storage, router, chat loop, CLI smoke tests
```

---

## Build and Run

### Install

```bash
cd llmchat
pip install -e ".[all]"
```

- `.[all]` installs all optional provider SDKs (anthropic, openai, google-genai, httpx)
- Individual providers are optional; missing SDKs just disable that provider
- Python 3.10+ required

### Configure

Set API keys as environment variables. Missing keys disable that provider.

```bash
export ANTHROPIC_API_KEY=sk-ant-...
export OPENAI_API_KEY=sk-...
export GOOGLE_API_KEY=...
export OLLAMA_HOST=http://localhost:11434  # default
```

Generate starter routing config:
```bash
llmchat route init        # writes ~/.llmchat/routing.toml
llmchat route show        # inspect current routing
```

### Run

```bash
# New chat with auto-routing
llmchat

# Pin a model (disables routing)
llmchat --model openai:gpt-5.5

# Disable routing for this run
llmchat --no-route

# Resume an old session
llmchat resume 7f3a

# List sessions
llmchat list

# Export to markdown
llmchat export 7f3a > chat.md

# Dry-run routing on a sample prompt
llmchat route test "Analyze this step by step"
```

### Tests

```bash
python tests/test_smoke.py
```

Runs 15+ tests covering storage, router heuristics, chat loop, CLI, and pinned models. Uses a fake provider so no real API calls.

---

## High-Level Architecture

### 1. **Entry Point: CLI** (`cli.py`)

Click-based command group with subcommands:
- `llmchat [--model M] [--system S] [--title T] [--no-route]` → `new`
- `llmchat resume <id>`
- `llmchat list`, `llmchat show <id>`, `llmchat export <id>`, `llmchat delete <id>`
- `llmchat models` → list available models per provider
- `llmchat route show|init|test`

**Key Interaction**: CLI creates a `Store`, loads `RoutingConfig`, creates a `Session`, and calls `run_chat()` which spins up `ChatLoop`.

---

### 2. **Storage Layer** (`storage.py`)

**SQLite Schema** (intentionally minimal):
```sql
sessions(id, title, created_at, updated_at, current_model, system_prompt)
messages(id, session_id, role, content, model, created_at)
```

**Key Details**:
- Sessions and messages are persisted **immediately on arrival** so crashes lose at most one in-flight reply
- WAL mode + foreign keys for safety
- `role` is restricted to 'user', 'assistant', 'system' (tools use JSON metadata in content)
- Tool calls and results are stored as JSON-encoded metadata blocks in message content (preserves schema)

**Store API**:
```python
store.create_session(model, title, system_prompt) → Session
store.get_session(id_prefix) → Session | None  # allows prefix matches
store.list_sessions(limit)
store.add_message(session_id, role, content, model)
store.get_messages(session_id) → list[Message]
store.update_session_model/title/system_prompt(id, value)
```

---

### 3. **Routing: Heuristic Tier Selection** (`routing.py`, `config.py`)

**Three Tiers** (user-configurable):
- **local**: Ollama or similar. Free, small, good for factual Q&A.
- **cheap**: Haiku, GPT-4 mini, Gemini Flash. Fast, low cost, everyday work.
- **flagship**: Opus, GPT-5.5, Gemini Pro. Most capable, for hard reasoning + long context.

**Routing Logic** (in order of precedence):
1. **Long context** → flagship. Prompt ≥ 4,000 chars OR history ≥ 8,000 chars.
2. **Reasoning words** → flagship. Regex match for: analyze, debug, refactor, prove, step-by-step, trade-offs, etc.
3. **Complex code** → flagship. Code block + complexity hints (async, class, mutex) OR code > 800 chars.
4. **Deep conversation** → cheap. 10+ user turns → don't downgrade to local.
5. **Easy short Q&A** → local. "What is X", "summarize", "translate" + short prompt + early conversation.
6. **Ambiguous** → default tier (cheap).

**Why No Classifier?** Classifiers are themselves LLM calls, which is the exact token tax we're avoiding. Heuristics get the easy 80% right for free. The ambiguous 20% intentionally goes to the configured default tier, a deliberate choice.

**Tier Configuration** (`routing.toml` under `~/.llmchat/`):
```toml
default_tier = "cheap"
enabled = true
explain = true  # print routing decision on each turn

[tiers.local]
model = "ollama:llama3.2"
max_input_chars = 4000
supports_tools = false  # Ollama can't call MCP tools

[tiers.cheap]
model = "anthropic:claude-haiku-4-5"
max_input_chars = 20000

[tiers.flagship]
model = "anthropic:claude-opus-4-7"
max_input_chars = 200000
```

**Router API**:
```python
router = Router(config)
decision = router.route(prompt, chat_history, need_tools=False)
# decision.tier, decision.model, decision.reasons
```

---

### 4. **Chat Loop & Agentic Iteration** (`chat.py`)

**Main Loop Structure** (see `ChatLoop.run()` at line 107):
1. **Command Check**: If line starts with `/`, handle slash commands (`/model`, `/auto`, `/route`, `/system`, etc.)
2. **Pre-Hooks**: Run `PRE_USER_MESSAGE` hooks (can block or transform).
3. **Model Resolution**: Call `_resolve_model_for_turn()` → pin overrides router if set.
4. **Persist User Message**: Write to SQLite immediately.
5. **System Prompt Assembly**: Base session prompt + active skills selected by `SkillRegistry.select_for()`.
6. **Agentic Loop** (`_run_agentic_loop()`, max `MAX_TOOL_ITERATIONS=10`):
   - Stream from provider (text deltas + tool calls).
   - Persist assistant text + any tool calls (as JSON metadata).
   - If model requested tool calls:
     - Show tool call (with approval gate).
     - Run tool via `MCPManager.call_tool()`.
     - Feed results back into next iteration.
   - Repeat until `stop_reason != "tool_use"` or iteration limit.
7. **Post-Hooks**: Run `POST_ASSISTANT_MESSAGE` hooks.

**Key Design Decisions**:
- **Metadata in Content**: Tool calls and results are JSON-encoded and appended to message `content` so the schema stays simple. On resume, `_load_history_as_messages()` reconstructs the structured fields.
- **Streaming Events**: Providers yield `TextDelta | ToolUseRequest | TurnEnd` when tools are involved; plain strings otherwise (backward compat).
- **Approval Gates**: Before running any tool, the user is prompted to approve (once, per-session, or deny). Approval state is tracked per session.
- **Crash Safety**: Every boundary (user message, assistant turn, tool result) is persisted to disk before the next step.

**In-Chat Commands** (slash-command handlers):
```
/model <provider>:<name>   — pin to a specific model
/auto                      — re-engage routing
/route on|off|show         — toggle routing or inspect config
/system <text>             — set/replace system prompt
/title <text>              — rename session
/clear                     — start a new session
/history                   — show all messages
/tools                     — list MCP tools
/plugins                   — show loaded skills/commands/hooks
/help, /exit
```

---

### 5. **Provider Abstraction** (`providers/base.py`, `providers/*_provider.py`)

**Base Interface**:
```python
class Provider(ABC):
    name: str
    supports_tools: bool
    
    def stream(model, messages, system=None, tools=None) 
        → Iterator[str | StreamEvent]
    
    def list_models() → list[str]
    def is_configured() → bool
```

**Key Points**:
- **Back-Compat Dual-Mode**: When `tools=None`, providers yield plain `str` chunks. When tools are passed, they yield `StreamEvent` objects.
- **Message Format**: `ChatMessage(role, content, tool_calls=None, tool_results=None)` carries structured tool data.
- **Provider Implementations**:
  - **Anthropic** (`anthropic_provider.py`): Full tool-use support. Translates tool calls to `tool_use` blocks, tool results to `tool_result` blocks.
  - **OpenAI** (`openai_provider.py`): Text-only initially; tool support pending.
  - **Google** (`google_provider.py`): Gemini models.
  - **Ollama** (`ollama_provider.py`): Local models. No tools (local models can't reach MCP servers).
- **Registry** (`providers/__init__.py`): Lazy-loads providers by name. Missing SDKs just disable that provider.

---

### 6. **Extensions: Skills, Commands, Hooks** (`extensions/*`)

**Discovery** (`discovery.py`):
Searches four roots in order (later wins):
1. `~/.claude/` (Claude Code user-level)
2. `~/.llmchat/` (llmchat user-level)
3. `./.claude/` (project-level)
4. `./.llmchat/` (project-level)

Within each root:
- `skills/` — folders with `SKILL.md`
- `commands/` — `.md` files
- `hooks/` — `.json` or `.toml` files
- `plugins/` — folders with manifests

**Skills** (`skills.py`):
Markdown files with optional YAML frontmatter. Body is injected into the system prompt when the skill activates.

```yaml
---
name: code-reviewer
description: Reviews Python code.
triggers:
  - "review this code"
  - "/review"
auto_invoke: true          # default true
file_patterns:
  - "*.py"
---
You are a senior Python code reviewer. Look for:
- ...
```

Activation logic:
- **Always-on**: `auto_invoke: true` + no triggers/patterns → inject every turn.
- **Trigger-based**: Any trigger phrase in user message.
- **Pattern-based**: User references matching file (heuristic path extraction).

**Commands** (`commands.py`):
Slash-command templates. `/name args` expands to the template with `$ARGS` replaced.

```markdown
---
description: Write a pull request description
---
Summarize the changes from `git diff main`. Propose:
- PR title
- Description with: $ARGS
```

User types: `/pr fix the auth race condition` → expands to the template with `$ARGS = "fix the auth race condition"` → sent as regular user message.

**Hooks** (`hooks.py`):
Subprocess scripts that run at lifecycle events. Declared in `hooks.toml`:

```toml
[[hooks]]
event = "pre_user_message"
command = ["./scripts/redact-secrets.sh"]
timeout_seconds = 5
```

Hooks can:
- **Transform** (pre_* events): stdout becomes the new payload.
- **Block**: Non-zero exit blocks the event.
- **Informational** (post_* events): just see the data, no effect.

Events:
- `pre_user_message` — before model sees the text
- `post_assistant_message` — after model responds
- `pre_tool_call` — before tool execution
- `post_tool_call` — after tool result

---

### 7. **MCP (Model Context Protocol) Support** (`mcp/*`)

**Philosophy**: MCP servers are long-lived processes (stdio or HTTP) that expose tools. llmchat runs them in a background thread and provides a synchronous facade.

**Manager** (`mcp/manager.py`):
- **Lifecycle**: `manager.start()` spins up a daemon thread with an asyncio loop, connects to all servers from `mcp.json`.
- **Failure Model**: If a server fails to connect, log and continue (don't crash chat).
- **Tool Listing**: `list_tools()` returns `ToolDescriptor` list from all servers.
- **Tool Execution**: `call_tool(fq_name, arguments)` submits work to the loop thread and blocks for result.

**Configuration** (`.mcp.json` or `mcp.json` under extension roots):
```json
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/path"]
    },
    "weather": {
      "type": "http",
      "url": "https://example.com/mcp"
    }
  }
}
```

**Types** (`mcp/types.py`):
- `ToolDescriptor`: Advertised to model. FQ name is `server::tool_name`.
- `ToolCall`: Model's request to invoke a tool.
- `ToolResult`: Our response to feed back.
- `StreamEvent = TextDelta | ToolUseRequest | TurnEnd`: Events providers yield.

**Approval** (`mcp/approval.py`):
Simple state machine to ask users once per tool per session, or block outright.

---

## Key Design Patterns

### 1. **Metadata in Content**
Tool calls and results are stored as JSON-encoded metadata blocks appended to message content:
```
message content:
  [actual response text]
  
  <!--llmchat:tool_calls:[{"id":"...", "fq_name":"...", "arguments":{...}}]-->
```

This keeps the SQLite schema static while supporting structured tool data. On resume, `_extract_meta()` reconstructs.

### 2. **Immediate Persistence**
Every message (user, assistant, tool result) is written to SQLite **before** the next step. Crashes lose at most one in-flight reply. No in-memory queue.

### 3. **Lazy Provider Loading**
Providers are imported on-demand via `@lru_cache` in `get_provider()`. Missing SDKs don't break the CLI for users who don't need them.

### 4. **Sync-Over-Async for MCP**
The async MCP SDK runs in a background thread. Public methods use `_submit()` to send coroutines to that loop and block for results. Allows the rest of the app to stay sync and readable.

### 5. **Heuristic-First Routing**
No LLM classifier. A handful of regex patterns and length checks handle 80% of cases instantly. Ambiguous cases go to the configured default tier (a deliberate choice, not a fallback).

### 6. **Extension Roots Override**
Project-level extensions (`./.llmchat/`) override user-level (`~/.llmchat/`). Allows projects to pin their own version of a skill or command.

---

## Common Workflows

### Adding a New Skill

1. Create `~/.llmchat/skills/my_skill/SKILL.md`:
   ```yaml
   ---
   name: my-skill
   description: Does X
   triggers:
     - "help with X"
   ---
   You are an expert in X...
   ```

2. Restart chat. Skill will auto-activate when its trigger phrase appears.

### Adding a Slash Command

1. Create `~/.llmchat/commands/mycommand.md`:
   ```yaml
   ---
   description: Does Y
   ---
   Summarize the following and propose improvements:
   
   $ARGS
   ```

2. User types: `/mycommand <text>` → expands and sent as regular message.

### Configuring MCP Tools

1. Create `~/.llmchat/mcp.json`:
   ```json
   {
     "mcpServers": {
       "browser": {
         "command": "npx",
         "args": ["-y", "@modelcontextprotocol/server-browser"]
       }
     }
   }
   ```

2. Start chat. MCP manager connects and lists tools.

3. Model can now call these tools. User approves each call (or all for session).

### Tuning the Router

1. Edit `~/.llmchat/routing.toml`:
   ```toml
   [tiers.cheap]
   model = "openai:gpt-4-turbo"  # switch provider
   max_input_chars = 30000        # increase limit
   ```

2. Dry-run: `llmchat route test "Analyze trade-offs carefully"`

3. Adjust thresholds in `routing.py` if needed (constants like `LONG_PROMPT_CHARS`).

---

## Testing Strategy

**Test Suite** (`tests/test_smoke.py`):
- **Storage**: Create session, add messages, retrieve by ID prefix.
- **Router**: Test each heuristic (short factual → local, reasoning → flagship, etc.).
- **Chat Loop**: Full integration: routing, pinned models, /commands, resume.
- **CLI**: `route test`, `route show`, `list`, `show`, `export`.

All tests use a `FakeProvider` so no real API calls. Run with:
```bash
python tests/test_smoke.py
```

**Running Real Chats**:
Set API keys and run `llmchat`. Every turn persists, so you can interrupt and resume.

---

## Non-Obvious Design Choices

1. **No Provider Classes in Tool Flow**: Tool calls and results flow through the provider's native format, not a second abstraction. Keeps translation code localized.

2. **No Skill/Command Execution**: Skills and commands are templates only; they don't run scripts or tools themselves. Tool execution is MCP's job (one mechanism, not two).

3. **Role is Stored as String, Not Enum**: Simplifies schema. Constraints are in `CHECK (role IN (...))` on the database side.

4. **Metadata Blocks, Not Schema Changes**: Tool data in JSON-encoded content blocks instead of new tables. Keeps schema minimal, backward-compatible, and easy to read in exports.

5. **Tier Names Are Conventions, Not Code**: The router just picks a tier name (a string). Your config maps it to a model. Lets users call them whatever makes sense for their setup.

6. **Session Model is Metadata Only**: `session.current_model` is just for resume context. The actual model used per turn is determined by routing (or /model pin), not fetched from the session.

---

## Future Work / TODOs

- Tool support for OpenAI, Google providers (currently Anthropic only)
- Streaming token counts (for cost estimation)
- Conversation pruning (drop old messages when context gets large)
- Batch export to jsonl
- Plugin dependency resolution
- Custom routing thresholds in config (not just constants in code)
- Conversation search

---

## File Locations

- **Database & Config**: `~/.llmchat/` or `LLMCHAT_DB` env var
  - `sessions.db` — SQLite
  - `routing.toml` — routing config
  - `input_history` — readline history
  - `skills/`, `commands/`, `hooks/`, `plugins/` — extensions
  - `mcp.json` — MCP server config

- **Project Extensions**: `./.llmchat/` (overrides user-level)

- **Claude Code Compat**: `./.claude/` and `~/.claude/` are also searched

---

## Dependencies

**Core**:
- `click>=8.1` — CLI
- `rich>=13.7` — formatting
- `prompt_toolkit>=3.0` — interactive prompt
- `tomli>=2.0` (Python <3.11) — config parsing

**Optional**:
- `anthropic>=0.40`
- `openai>=1.50`
- `google-genai>=0.3`
- `httpx>=0.27`
- `mcp` (for MCP servers)

---

## Contact / Questions

See README.md for usage examples. This is a living document; update as the codebase evolves.
