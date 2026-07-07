# Backend configuration

Roundtable can either drive an agent router such as `pi`/`omp`, shell out directly
to AI CLIs, call LiteLLM, or run against the deterministic scripted backend used by
tests.

## pi / omp backend

Roundtable works best with a **pi-family coding agent**: upstream
[pi](https://github.com/earendil-works/pi) or
[oh-my-pi (`omp`)](https://github.com/can1357/oh-my-pi). Both speak many model
providers and share the same basic CLI contract.

With `provider: pi`, the pi-family tool handles LLM connectivity, auth, and model
routing. Roundtable handles planning, orchestration, state, validation, resuming,
and the dashboard.

Task agents run with the tool's file tools so they can edit the repo. Planner,
main, and phase roles run as pure completions without edit tools. Because pi/omp
report usage, Roundtable can show exact token counts and real dollar cost per run
instead of CLI text-length estimates.

Pick the flavor with `pi.flavor`:

- `pi`: upstream binary, usually `pi`
- `omp`: oh-my-pi binary, usually `omp`

Roundtable handles the flavor differences for you: `omp` has no
`--no-context-files`, and task agents get `--auto-approve` so they can edit
autonomously.

### Install and first run

```bash
npm install -g @earendil-works/pi-coding-agent
pi-ai login anthropic

# or install oh-my-pi instead
npm install -g @oh-my-pi/pi-coding-agent
omp login anthropic

cd my-existing-project
roundtable init
roundtable plan --goal "Add retry with backoff to the HTTP client"
roundtable approve
roundtable run
```

### Example pi config

The generated config usually routes stronger models to planning/orchestration and
cheaper/faster models to task work. Edit freely; `pi --list-models`, `omp models`,
or `roundtable models --list` show what you can assign.

```yaml
provider: pi
models:
  planner: { model: anthropic/claude-opus-4-1 }
  main:    { model: anthropic/claude-opus-4-1 }
  phase:   { model: anthropic/claude-sonnet-4-5 }
  task:    { model: anthropic/claude-haiku-4-5 }
pi:
  flavor: pi
  command: ["pi"]
  extra_args: []
  orchestrator_context_files: false
```

To use oh-my-pi:

```yaml
provider: pi
models:
  planner: { model: anthropic/claude-opus-4-1 }
  main:    { model: anthropic/claude-opus-4-1 }
  phase:   { model: opencode-go/glm-5.2 }
  task:    { model: opencode-go/mimo-v2.5-pro }
pi:
  flavor: omp
  command: ["omp"]
  extra_args: []
```

> Note: pi/omp has no per-action permission gate and task agents share the project
> directory, so keep `max_concurrency: 1` unless you isolate task work yourself.

## CLI backend

Use `provider: cli` when you want Roundtable to drive the tools already installed
on your machine, such as Claude Code, Codex, Gemini CLI, aider, `llm`, or Ollama.

### First run

Install at least one AI CLI (Claude Code, Codex, Gemini CLI, aider, …) and
authenticate it per its own docs. Then:

```bash
cd my-existing-project
roundtable init        # detects installed CLIs and writes a `provider: cli` config
roundtable agents      # shows which CLIs are on PATH + the models they expose
roundtable models      # pick a model per role interactively (or edit the config)
roundtable plan --goal "Add retry with backoff" && roundtable approve && roundtable run
```

Each role is a `{agent, model}` pair:

```yaml
provider: cli

models:
  planner: { agent: claude, model: opus-4.8 }
  main:    { agent: claude, model: opus-4.8 }
  phase:   { agent: gemini, model: gemini-3.5-flash }
  task:    { agent: codex,  model: gpt-5.3-codex }

agents:
  claude:
    command: ["claude", "-p", "{prompt}", "--model", "{model}"]
  codex:
    command: ["codex", "exec", "--model", "{model}", "{prompt}"]
  gemini:
    command: ["gemini", "-p", "{prompt}", "--model", "{model}"]
  ollama:
    command: ["ollama", "run", "{model}"]
    stdin: true
```

The command is an argv list, not a shell string. Tokens may include `{prompt}`,
`{system}`, and `{model}`. If `{system}` is absent, Roundtable prepends the system
prompt to the user prompt. With `stdin: true`, the prompt is piped on stdin.

Use `pty: true` only for CLIs that require a real terminal.

## LiteLLM backend

Use `provider: litellm` for direct API calls. In this mode, `model` is the LiteLLM
model string and `agent` is ignored.

```yaml
provider: litellm
models:
  planner: { model: anthropic/claude-3-5-sonnet-latest }
  main:    { model: openai/gpt-4o }
  phase:   { model: openai/gpt-4o-mini }
  task:    { model: ollama/llama3 }
```

Install the extra and set your provider API keys:

```bash
uv tool install "roundtable-cli[litellm]"
export OPENAI_API_KEY=…   # and/or ANTHROPIC_API_KEY, etc. — whatever your models need
```

## Scripted backend

`provider: scripted` is deterministic and offline. It is meant for tests, demos,
and development of Roundtable itself.

