"""Roundtable configuration: which backend, which model/agent per role, runtime knobs.

Loaded from ``roundtable.config.yaml`` at the project root.

Four backends (``provider``):

* ``pi``      — drive the `pi` coding agent for every role. Task agents run with
                pi's file tools (they edit the repo); every other role runs
                ``pi --no-tools`` as a pure completion. Connectivity, auth and
                model routing are handled by pi/pi-ai. The recommended backend
                when `pi` is installed — it reports exact token usage and cost.
* ``cli``     — run other LLMs through their terminal CLIs (claude, codex, gemini,
                aider, llm, ollama, ...). Each role's "model" names an entry in
                ``agents``. Agents act on the real project files via their own
                tools. This is what roundtable falls back to without pi.
* ``litellm`` — direct API calls; "model" is a litellm model string.
* ``scripted``— deterministic offline backend (demo/tests).
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from .models import AgentRef

CONFIG_FILENAME = "roundtable.config.yaml"


def _ref(agent: str) -> AgentRef:
    return AgentRef(agent=agent)


class ModelRoles(BaseModel):
    """Default (agent, model) assignment per role.

    Each value is an :class:`AgentRef` — accepts an object ``{agent, model}`` or
    the shorthand string ``agent:model`` in YAML. Per-phase / per-task ``runner``
    entries in the generated plan override these.
    """

    planner: AgentRef = Field(default_factory=lambda: _ref("claude"))
    main: AgentRef = Field(default_factory=lambda: _ref("claude"))
    phase: AgentRef = Field(default_factory=lambda: _ref("claude"))
    task: AgentRef = Field(default_factory=lambda: _ref("claude"))


class AgentSpec(BaseModel):
    """A terminal command that talks to an LLM.

    ``command`` is an argv list (no shell). Tokens may contain ``{prompt}``,
    ``{system}`` and ``{model}`` placeholders. ``{model}`` is replaced with the
    role/task's chosen model so one agent command serves many models. If
    ``{system}`` is absent, the system text is prepended to the prompt. If
    ``stdin`` is true, the prompt is piped on stdin instead of into ``{prompt}``.

    ``models_command`` is an optional argv that lists the models this CLI offers
    (e.g. ``["opencode", "models"]``, ``["agy", "models"]``, ``["ollama",
    "list"]``); ``roundtable init`` / ``roundtable agents`` runs it to show you what you
    can assign. Omit it for tools with no enumeration command (claude, codex).
    """

    command: list[str]
    stdin: bool = False
    pty: bool = False  # run in a pseudo-terminal so the CLI sees a real TTY
    models_command: list[str] | None = None


class Defaults(BaseModel):
    temperature: float = 0.2
    max_concurrency: int = Field(default=1, ge=1)
    max_retries: int = Field(default=1, ge=0)
    timeout: int = Field(default=900, ge=1)  # per agent call, seconds
    hitl_timeout: int = Field(default=0, ge=0)  # seconds to wait for HITL approval; 0 = infinite
    validate_timeout: int = Field(default=120, ge=1)  # seconds per task/phase validate_command


# No-config fallback agents. Kept model-less so the bare-agent role defaults
# (ModelRoles -> agent only) work without an empty {model} argument. The richer
# {model}-templated commands live in DEFAULT_CONFIG_YAML that `roundtable init` writes.
DEFAULT_AGENTS: dict[str, AgentSpec] = {
    # Claude Code, non-interactive print mode.
    "claude": AgentSpec(command=["claude", "-p", "{prompt}"]),
    # OpenAI Codex CLI, non-interactive exec mode.
    "codex": AgentSpec(command=["codex", "exec", "{prompt}"]),
    # Gemini CLI.
    "gemini": AgentSpec(command=["gemini", "-p", "{prompt}"]),
}


class PiOptions(BaseModel):
    """Options for ``provider: pi`` — drive a pi-family coding agent for every role.

    Two flavors are supported (both share the same CLI contract — ``--mode json``,
    ``--no-tools``, ``--model``, stdin prompt, and identical usage/cost JSON):

    * ``pi``  — upstream `pi` (`earendil-works/pi`), binary ``pi``.
    * ``omp`` — `oh-my-pi` (`can1357/oh-my-pi`), binary ``omp``; a batteries-included
      fork (LSP/DAP/subagents). It has no ``--no-context-files`` and gates edits
      behind approval, so task agents get ``--auto-approve`` for autonomous runs.

    Only the task role runs with file tools (so task agents edit the repo); every
    other role (planner, phase, main, map) runs ``--no-tools`` as a pure completion.
    Each role's ``model`` is passed to ``--model`` (a ``provider/id`` string, glob,
    or fuzzy pattern — ``<tool> --list-models`` shows what's available). Auth is the
    tool's: set a provider key in the environment (``ANTHROPIC_API_KEY`` /
    ``OPENAI_API_KEY`` / ``GEMINI_API_KEY`` / ...) or use the tool's own login.
    """

    flavor: str = "pi"  # "pi" (upstream) | "omp" (oh-my-pi)
    command: list[str] = Field(default_factory=list)  # override binary; [] -> flavor default (["pi"]/["omp"])
    extra_args: list[str] = Field(default_factory=list)  # appended to every invocation
    worker_extra_args: list[str] = Field(default_factory=list)  # extra flags for the task (worker) role
    orchestrator_extra_args: list[str] = Field(default_factory=list)  # extra flags for non-worker roles
    # (pi flavor only) orchestrator roles ignore the repo's AGENTS.md/CLAUDE.md by
    # default (roundtable injects its own context); set true to let them load those.
    orchestrator_context_files: bool = False


class Config(BaseModel):
    provider: str = "cli"  # "pi" | "cli" | "litellm" | "scripted"
    models: ModelRoles = Field(default_factory=ModelRoles)
    agents: dict[str, AgentSpec] = Field(default_factory=lambda: dict(DEFAULT_AGENTS))
    pi: PiOptions = Field(default_factory=PiOptions)
    defaults: Defaults = Field(default_factory=Defaults)
    project_context: str = ""  # optional project context injected into agent prompts


DEFAULT_CONFIG_YAML = """\
# roundtable configuration.

# Backend:
#   cli      -> reach other LLMs through their terminal CLIs (default)
#   litellm  -> direct API calls (model = litellm string, needs API keys)
#   scripted -> deterministic offline backend (demo/tests, no network)
provider: cli

# Choosable (agent, model) per role. `agent` names an entry in `agents` below;
# `model` is passed to that agent's {model} placeholder. Mix CLIs and models
# freely per role -- e.g. main on claude/opus, phase on antigravity/gemini,
# tasks on opencode. Per-phase and per-task `runner` entries in the generated
# plan override these. (With provider: litellm, `model` is the litellm string
# -- openai/gpt-4o, anthropic/claude-3-5-sonnet-latest, ... -- and `agent` is
# ignored.) Shorthand `agent:model` strings work too.
models:
  planner: { agent: claude,      model: opus }
  main:    { agent: claude,      model: opus }
  phase:   { agent: antigravity, model: gemini-3.5-flash }
  task:    { agent: opencode,    model: opencode/mimo-v2.5-free }

# Terminal commands for provider: cli. argv lists (no shell). Tokens may use
# {prompt}, {system} and {model}; {model} is replaced with the role/task's
# chosen model so one command serves many models. If {system} is absent it is
# prepended to the prompt. Set stdin: true to pipe the prompt on stdin instead
# of via {prompt}. Add whatever flags your tool needs to run non-interactively
# and edit files (these flags are illustrative -- check each tool's own docs).
# Optional models_command lists that CLI's models; `roundtable init` / `roundtable
# agents` runs it so you can see what to assign above.
agents:
  claude:
    command: ["claude", "-p", "{prompt}", "--model", "{model}"]
  codex:
    command: ["codex", "exec", "--model", "{model}", "{prompt}"]
  # Antigravity CLI (formerly the Gemini CLI; binary is `agy`), runs Gemini models.
  antigravity:
    command: ["agy", "-p", "{prompt}", "--model", "{model}"]
    models_command: ["agy", "models"]
  opencode:
    command: ["opencode", "run", "--model", "{model}", "{prompt}"]
    models_command: ["opencode", "models"]
  ollama:
    command: ["ollama", "run", "{model}"]
    models_command: ["ollama", "list"]
    stdin: true

defaults:
  temperature: 0.2
  max_concurrency: 1   # raise to run independent tasks in a phase concurrently
                       # (keep at 1 when file-editing agents might touch the same files)
  max_retries: 1       # retries on a failing/timed-out agent call
  timeout: 900         # seconds per agent call
  validate_timeout: 120  # seconds per task/phase validate_command
"""


# Recommended backend when a pi-family CLI is installed: it drives every role and
# handles LLM connectivity/auth itself. `roundtable init` writes this when it finds
# `pi` or `omp` on PATH. Model defaults are a mixed strong+cheap start -- edit freely.
def pi_config_yaml(flavor: str = "pi") -> str:
    tool = "omp" if flavor == "omp" else "pi"
    other = "pi" if flavor == "omp" else "omp"
    models_hint = "`pi --list-models`" if tool == "pi" else "`omp --help` / omp's model config"
    return f"""\
# roundtable configuration ({tool} backend -- recommended).

# Backend:
#   pi       -> drive a pi-family coding agent ({tool}) for every role (this file).
#               It handles LLM connectivity + auth; task agents edit files,
#               orchestrator roles run `{tool} --no-tools`. Exact usage & cost reported.
#               flavor `pi` = upstream `pi`; flavor `omp` = oh-my-pi (binary `omp`).
#   cli      -> reach other LLMs through their terminal CLIs (claude, codex, ...)
#   litellm  -> direct API calls (model = litellm string, needs API keys)
#   scripted -> deterministic offline backend (demo/tests, no network)
provider: pi

# Per-role model (passed to `{tool} --model`: a `provider/id` string, glob, or fuzzy
# pattern -- see {models_hint} for what you can use). `agent` is ignored
# on this backend. Mixed strong+cheap defaults: strong models plan/orchestrate, a
# cheap/fast model does the task work. Per-phase/per-task `runner` entries in the
# generated plan override these.
models:
  planner: {{ model: anthropic/claude-opus-4-1 }}     # strong: breaks the goal into phases/tasks
  main:    {{ model: anthropic/claude-opus-4-1 }}      # strong: keeps project docs coherent
  phase:   {{ model: anthropic/claude-sonnet-4-5 }}    # mid: defines & summarizes each phase
  task:    {{ model: anthropic/claude-haiku-4-5 }}     # cheap/fast: does the actual task work
  # Cheaper/free task work? Point `task` (and `phase`) at another provider the tool
  # supports, e.g. `openrouter/...` or `opencode/...` -- see {models_hint}.

# Connect {tool} to your LLMs (pick one per provider you use):
#   * env key: export ANTHROPIC_API_KEY=... (or OPENAI_API_KEY / GEMINI_API_KEY / ...)
#   * or use {tool}'s own login (e.g. `pi-ai login anthropic` for pi)

# Options for the pi backend (all optional).
pi:
  flavor: {flavor}          # "pi" (upstream) or "omp" (oh-my-pi); switch to use `{other}`
  command: ["{tool}"]       # base binary; use ["npx", "{tool}"] if not installed globally
  extra_args: []            # appended to every call, e.g. ["--thinking", "medium"]
  worker_extra_args: []     # extra flags for the task role{_omp_worker_note(flavor)}
  orchestrator_extra_args: []  # extra flags for planner/main/phase/map roles
  orchestrator_context_files: false   # (pi flavor) true -> orchestrator roles read AGENTS.md/CLAUDE.md

defaults:
  temperature: 0.2
  max_concurrency: 1   # raise to run independent tasks in a phase concurrently
                       # ({tool} has no per-action permission gate and tasks share the
                       #  repo dir, so keep at 1 unless you isolate tasks yourself)
  max_retries: 1       # retries on a failing/timed-out {tool} call
  timeout: 900         # seconds per {tool} call
  validate_timeout: 120  # seconds per task/phase validate_command
"""


def _omp_worker_note(flavor: str) -> str:
    if flavor == "omp":
        return " (omp task agents already get --auto-approve)"
    return ""


def load_config(project_dir: Path) -> Config:
    path = Path(project_dir) / CONFIG_FILENAME
    if not path.exists():
        return Config()
    raw = yaml.safe_load(path.read_text()) or {}
    return Config.model_validate(raw)


def write_default_config(project_dir: Path, *, backend: str = "cli", flavor: str = "pi") -> Path:
    """Write ``roundtable.config.yaml``. ``backend='pi'`` writes the recommended
    pi-family template for ``flavor`` ('pi' or 'omp'); anything else writes the CLI
    template."""
    path = Path(project_dir) / CONFIG_FILENAME
    path.write_text(pi_config_yaml(flavor) if backend == "pi" else DEFAULT_CONFIG_YAML)
    return path
