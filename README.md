# llmchat

A minimal, no-nonsense CLI for chatting with multiple LLM providers in one
terminal. Sessions are persisted to SQLite, and every turn is **automatically
routed** to the cheapest model that can plausibly handle it — local for quick
lookups, cheap paid for everyday tasks, flagship for hard problems.

## Features

- **Multi-provider:** Anthropic (Claude), OpenAI (GPT), Google (Gemini), Ollama (local).
- **Smart routing by default:** the router picks a model tier per turn based on
  prompt content, length, and conversation depth. No extra LLM call required.
- **Persistent by default:** every session and message is stored in SQLite.
- **Switch models mid-session:** `/model openai:gpt-5.5` pins a specific model;
  `/auto` re-engages routing.
- **Streaming output**, **resume by id**, **markdown export**.

## Install

Requires Python 3.10+.

```bash
cd llmchat
pip install -e ".[all]"
```

This installs the `llmchat` command plus all provider SDKs.

## Configure

### API keys

Set whichever you have. Missing keys just disable that provider.

```bash
export ANTHROPIC_API_KEY=sk-ant-...
export OPENAI_API_KEY=sk-...
export GOOGLE_API_KEY=...                # for Gemini
export OLLAMA_HOST=http://localhost:11434  # optional, this is the default
```

### Routing config

Generate a starter file you can edit:

```bash
llmchat route init        # writes ~/.llmchat/routing.toml
llmchat route show        # see what's currently in effect
```

Default config maps three tiers to three models:

```toml
default_tier = "cheap"
enabled = true
explain = true

[tiers.local]
model = "ollama:llama3.2"
max_input_chars = 4000

[tiers.cheap]
model = "anthropic:claude-haiku-4-5"
max_input_chars = 20000

[tiers.flagship]
model = "anthropic:claude-opus-4-7"
max_input_chars = 200000
```

You can:
- **Replace any model** with one you prefer (`gpt-5.4-mini` for cheap, etc.).
- **Drop a tier** entirely. If you don't run Ollama, remove `[tiers.local]`
  and the router will escalate to `cheap` instead.
- **Change `default_tier`** to control where ambiguous prompts go.
- **Set `explain = false`** to suppress per-turn routing messages.
- **Set `enabled = false`** to turn routing off globally; `/model` rules.

## How routing works

The router scores each turn and picks a tier in this order of precedence:

1. **Long context →  flagship.** Prompt ≥ 4,000 chars or history ≥ 8,000
   chars. Small models can't handle the load.
2. **Hard reasoning words → flagship.** Words like *analyze, debug, refactor,
   trade-offs, step-by-step, derive, prove*. These signal the user wants
   careful work.
3. **Substantial code → flagship.** Code blocks plus complexity hints
   (`async`, `class`, `mutex`, `goroutine`, …) or any code block over 800
   chars.
4. **Deep conversation → cheap.** After 10+ user turns, even short messages
   probably depend on accumulated context.
5. **Short factual question → local.** "What is X", "translate", "summarize",
   "rephrase" + short prompt + early in conversation.
6. **Otherwise → default tier (cheap).**

A picked tier is **clamped to what's available**. If you haven't configured
`local`, the router escalates to `cheap`. If `flagship` is missing, it falls
back to `cheap`. The reason is shown in the explanation.

### Why no classifier?

Some routers use a small LLM to classify each prompt before routing. We
deliberately don't:

- It adds an extra API call to every turn (the thing we're trying to avoid).
- A classifier needs to be capable enough to be reliable, which costs real
  money.
- A misclassification on a hard problem wastes far more time than the
  classifier saves on easy ones.

The heuristic gets the easy 80% right for free. The middle 20% goes to the
default tier, which is a deliberate, predictable choice.

## Usage

```bash
# Start a new chat with auto-routing
llmchat

# Pin a model for this session (skips routing)
llmchat --model openai:gpt-5.5

# Disable routing for this run
llmchat --no-route

# List saved sessions
llmchat list

# Resume a session
llmchat resume 7f3a

# Export to markdown
llmchat export 7f3a > chat.md

# See what models are available
llmchat models

# Inspect / test routing without making API calls
llmchat route show
llmchat route test "what is 2 + 2?"
llmchat route test "Analyze the trade-offs step by step."
llmchat route test "fix this bug" --history "user:huge codebase context here"
```

### In-chat commands

While chatting:

- `/model <provider>:<name>` — pin to a specific model (disables routing)
- `/auto` — re-engage routing
- `/route on|off|show` — toggle routing globally, or inspect config
- `/system <text>` — set or replace the system prompt
- `/title <text>` — rename the session
- `/clear` — start a new session (the old one stays in the DB)
- `/history` — show all messages in this session
- `/help` — show commands
- `/exit` or Ctrl-D — quit

### Example chat flow

```
$ llmchat
─── New chat 7f3a... ───
routing: local=ollama:llama3.2 → cheap=anthropic:claude-haiku-4-5 → flagship=anthropic:claude-opus-4-7

[7f3a · auto] you> what's the capital of Sweden?
→ ollama:llama3.2 · local (short factual question)
Stockholm.

[7f3a · auto] you> Now help me debug this race condition in my Go code: ```go
                   func (c *Cache) Get(k string) Value { ... } ```
→ anthropic:claude-opus-4-7 · flagship (non-trivial code)
Looking at your code, the race appears to be...

[7f3a · auto] you> /model openai:gpt-5.5
Pinned to openai:gpt-5.5 (use /auto to re-enable routing)

[7f3a · gpt-5.5] you> what about goroutine leaks?
assistant · openai:gpt-5.5
Goroutine leaks happen when...

[7f3a · gpt-5.5] you> /auto
Auto-routing re-enabled.
```

## Storage layout

By default, everything lives under `~/.llmchat/`:

- `sessions.db` — SQLite database with `sessions` and `messages` tables
- `routing.toml` — your routing config (created on demand)
- `input_history` — readline history for the prompt

Override the directory with `LLMCHAT_DB=/path/to/sessions.db`.

## Model strings

Format is `provider:model`. Examples:

| Provider  | Example                                                   |
|-----------|-----------------------------------------------------------|
| anthropic | `anthropic:claude-opus-4-7`, `anthropic:claude-haiku-4-5` |
| openai    | `openai:gpt-5.5`, `openai:gpt-5.4-mini`                   |
| google    | `google:gemini-2.5-pro`, `google:gemini-2.5-flash`        |
| ollama    | `ollama:llama3.2`, `ollama:qwen2.5`                       |

You can omit the provider for Anthropic models: `--model claude-opus-4-7`.

## Tuning the router

Run `llmchat route test "<prompt>"` (optionally with `--history`) to dry-run
the router without burning tokens. Iterate on your prompt or your config
until the decisions look right. The thresholds are constants on the `Router`
class in `llmchat/routing.py` — tune them there if the defaults don't fit
your workload.
