"""Harness configuration: which backend, which model/agent per role, runtime knobs.

Loaded from ``harness.config.yaml`` at the project root.

Three backends (``provider``):

* ``cli``     — run other LLMs through their terminal CLIs (claude, codex, gemini,
                aider, llm, ollama, ...). Each role's "model" names an entry in
                ``agents``. This is the primary mode: agents act on the real
                project files via their own tools.
* ``litellm`` — direct API calls; "model" is a litellm model string.
* ``scripted``— deterministic offline backend (demo/tests).
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from .models import AgentRef

CONFIG_FILENAME = "harness.config.yaml"


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
    "list"]``); ``harness init`` / ``harness agents`` runs it to show you what you
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


# No-config fallback agents. Kept model-less so the bare-agent role defaults
# (ModelRoles -> agent only) work without an empty {model} argument. The richer
# {model}-templated commands live in DEFAULT_CONFIG_YAML that `harness init` writes.
DEFAULT_AGENTS: dict[str, AgentSpec] = {
    # Claude Code, non-interactive print mode.
    "claude": AgentSpec(command=["claude", "-p", "{prompt}"]),
    # OpenAI Codex CLI, non-interactive exec mode.
    "codex": AgentSpec(command=["codex", "exec", "{prompt}"]),
    # Gemini CLI.
    "gemini": AgentSpec(command=["gemini", "-p", "{prompt}"]),
}


class Config(BaseModel):
    provider: str = "cli"  # "cli" | "litellm" | "scripted"
    models: ModelRoles = Field(default_factory=ModelRoles)
    agents: dict[str, AgentSpec] = Field(default_factory=lambda: dict(DEFAULT_AGENTS))
    defaults: Defaults = Field(default_factory=Defaults)
    project_context: str = ""  # optional project context injected into agent prompts


DEFAULT_CONFIG_YAML = """\
# llm-harness configuration.

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
# Optional models_command lists that CLI's models; `harness init` / `harness
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
"""


def load_config(project_dir: Path) -> Config:
    path = Path(project_dir) / CONFIG_FILENAME
    if not path.exists():
        return Config()
    raw = yaml.safe_load(path.read_text()) or {}
    return Config.model_validate(raw)


def write_default_config(project_dir: Path) -> Path:
    path = Path(project_dir) / CONFIG_FILENAME
    path.write_text(DEFAULT_CONFIG_YAML)
    return path
