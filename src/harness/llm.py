"""LLM transport layer.

A small ``LLMProvider`` protocol decouples the orchestration engine from how
calls are made. Three backends ship:

* ``CLIProvider`` — reach other LLMs through their terminal CLIs (claude, codex,
  gemini, aider, llm, ollama, ...). Each "model" names a configured agent
  command; the command runs in the project directory so the agent acts on real
  files. This is the primary backend.
* ``LiteLLMProvider`` — direct API calls via litellm, so any provider/model
  string works (``openai/...``, ``anthropic/...``, ``ollama/...``, ...).
* ``ScriptedProvider`` — a deterministic, offline backend that produces
  structured, inspectable output for every role. Used for the offline demo
  (``provider: scripted``) and for tests; the orchestration engine itself runs
  for real against it.

``role`` and ``meta`` are optional call metadata: real backends use them for
logging, while the scripted backend uses ``role`` to branch.
"""

from __future__ import annotations

import asyncio
import json
import shutil
from pathlib import Path
from typing import Any, Awaitable, Callable, Protocol, runtime_checkable

from .config import AgentSpec
from .errors import HarnessError


@runtime_checkable
class LLMProvider(Protocol):
    async def complete(
        self,
        *,
        model: str,
        system: str,
        user: str,
        agent: str | None = None,
        json_mode: bool = False,
        temperature: float = 0.2,
        role: str | None = None,
        meta: dict[str, Any] | None = None,
    ) -> str: ...


# --------------------------------------------------------------------------- #
# JSON extraction
# --------------------------------------------------------------------------- #
def extract_json(text: str) -> dict[str, Any]:
    """Best-effort extraction of a single JSON object from model output.

    Handles raw JSON, ```json fenced blocks, and prose-wrapped objects by
    scanning for the first balanced ``{...}`` (string-aware).
    """
    s = text.strip()
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass

    # Strip a leading/trailing code fence if present.
    if s.startswith("```"):
        s = s.split("\n", 1)[1] if "\n" in s else s
        if s.rstrip().endswith("```"):
            s = s.rstrip()[: -3]
    s = s.strip()
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass

    obj = _first_balanced_object(s)
    if obj is None:
        raise ValueError(f"no JSON object found in model output:\n{text[:500]}")
    return json.loads(obj)


def _first_balanced_object(s: str) -> str | None:
    start = s.find("{")
    if start == -1:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(s)):
        c = s[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return s[start : i + 1]
    return None


async def _retry(coro_factory: Callable[[], Awaitable[Any]], *, attempts: int) -> Any:
    last: Exception | None = None
    for n in range(attempts):
        try:
            return await coro_factory()
        except Exception as e:  # noqa: BLE001 - transport-agnostic retry
            last = e
            if n < attempts - 1:
                await asyncio.sleep(min(2.0 * (n + 1), 8.0))
    assert last is not None
    raise last


# --------------------------------------------------------------------------- #
# Real backend
# --------------------------------------------------------------------------- #
class LiteLLMProvider:
    """Real LLM calls through litellm (any provider/model string)."""

    def __init__(self, max_retries: int = 2):
        self.max_retries = max_retries

    async def complete(
        self,
        *,
        model: str,
        system: str,
        user: str,
        agent: str | None = None,
        json_mode: bool = False,
        temperature: float = 0.2,
        role: str | None = None,
        meta: dict[str, Any] | None = None,
    ) -> str:
        try:
            import litellm  # lazy: only the litellm backend needs it
        except ModuleNotFoundError as e:
            raise HarnessError(
                "provider 'litellm' needs the litellm package: pip install 'llm-harness[litellm]' "
                "(the default 'cli' provider needs no extra deps)"
            ) from e

        # litellm has no separate command; the model string is the litellm model.
        # Fall back to `agent` so a bare `agent`-only ref still works as the model.
        litellm_model = model or (agent or "")
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        kwargs: dict[str, Any] = {"model": litellm_model, "messages": messages, "temperature": temperature}
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        async def call_with_json() -> str:
            resp = await litellm.acompletion(**kwargs)
            return resp.choices[0].message.content or ""

        async def call_plain() -> str:
            plain = {k: v for k, v in kwargs.items() if k != "response_format"}
            resp = await litellm.acompletion(**plain)
            return resp.choices[0].message.content or ""

        try:
            return await _retry(call_with_json, attempts=self.max_retries + 1)
        except Exception:
            if not json_mode:
                raise
            # Some providers reject response_format; fall back to plain prompting.
            return await _retry(call_plain, attempts=self.max_retries + 1)


# --------------------------------------------------------------------------- #
# Terminal / CLI backend (the main mode): reach other LLMs via their CLIs
# --------------------------------------------------------------------------- #
class CLIProvider:
    """Run other LLMs through their terminal CLIs.

    Each ``model`` names an entry in ``agents``; its ``command`` (an argv list)
    is executed in ``cwd`` (the target project), with ``{prompt}``/``{system}``
    substituted or the prompt piped on stdin. stdout is the response.
    """

    def __init__(
        self,
        agents: dict[str, AgentSpec],
        *,
        cwd: str | Path | None = None,
        timeout: int = 900,
        max_retries: int = 1,
    ):
        self.agents = agents
        self.cwd = str(cwd) if cwd else None
        self.timeout = timeout
        self.max_retries = max_retries

    async def complete(
        self,
        *,
        model: str,
        system: str,
        user: str,
        agent: str | None = None,
        json_mode: bool = False,
        temperature: float = 0.2,
        role: str | None = None,
        meta: dict[str, Any] | None = None,
    ) -> str:
        # When `agent` is given, it names the command and `model` is the {model}
        # token. With only `model` (back-compat), `model` itself names the agent.
        if agent:
            agent_key, model_token = agent, model
        else:
            agent_key, model_token = model, ""

        spec = self.agents.get(agent_key)
        if spec is None:
            raise HarnessError(
                f"agent {agent_key!r} is not defined under `agents:` in harness.config.yaml; "
                f"available: {', '.join(sorted(self.agents)) or '(none)'}"
            )
        if json_mode:
            user = user + "\n\nRespond with ONLY the JSON object, no prose or code fences."

        argv, stdin_data = _render_command(spec, system, user, model_token)
        exe = shutil.which(argv[0])
        if exe is None:
            raise HarnessError(
                f"agent {agent_key!r}: command {argv[0]!r} not found on PATH "
                f"(install it or fix `agents.{agent_key}.command`)"
            )
        argv[0] = exe
        return await _retry(lambda: self._run(argv, stdin_data, agent_key), attempts=self.max_retries + 1)

    async def _run(self, argv: list[str], stdin_data: str | None, model: str) -> str:
        proc = await asyncio.create_subprocess_exec(
            *argv,
            cwd=self.cwd,
            stdin=asyncio.subprocess.PIPE if stdin_data is not None else None,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            out, err = await asyncio.wait_for(
                proc.communicate(stdin_data.encode() if stdin_data is not None else None),
                timeout=self.timeout,
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            raise TimeoutError(f"agent {model!r} timed out after {self.timeout}s")
        if proc.returncode != 0:
            # Many CLIs print the actual error to stdout, not stderr (e.g. claude's
            # "model ... may not exist"), so surface whichever stream has content.
            err_text = err.decode("utf-8", "replace").strip()
            out_text = out.decode("utf-8", "replace").strip()
            detail = (err_text or out_text or "(no output on stdout/stderr)")[-800:]
            raise RuntimeError(f"agent {model!r} exited {proc.returncode}: {detail}")
        return out.decode("utf-8", "replace")


def _render_command(
    spec: AgentSpec, system: str, user: str, model: str = ""
) -> tuple[list[str], str | None]:
    """Build argv + optional stdin payload from a command template.

    ``{model}`` is replaced with the chosen model token (empty string if none).
    """
    has_system = any("{system}" in tok for tok in spec.command)
    payload = user if has_system else (f"{system}\n\n{user}" if system else user)
    argv: list[str] = []
    for tok in spec.command:
        t = tok.replace("{system}", system or "").replace("{model}", model or "")
        if "{prompt}" in t:
            t = t.replace("{prompt}", "" if spec.stdin else payload)
        argv.append(t)
    return argv, (payload if spec.stdin else None)

# --------------------------------------------------------------------------- #
# Offline / deterministic backend
# --------------------------------------------------------------------------- #
Responder = Callable[[str, str, str, str, dict[str, Any]], str]


class ScriptedProvider:
    """Deterministic, network-free backend.

    Pass a custom ``responder(role, model, system, user, meta) -> str`` to script
    exact outputs (tests), or rely on the built-in responder which derives a
    coherent plan and per-task results from the inputs (offline demo).
    """

    def __init__(self, responder: Responder | None = None):
        self.responder = responder or default_responder
        self.calls: list[dict[str, Any]] = []  # observability for tests

    async def complete(
        self,
        *,
        model: str,
        system: str,
        user: str,
        agent: str | None = None,
        json_mode: bool = False,
        temperature: float = 0.2,
        role: str | None = None,
        meta: dict[str, Any] | None = None,
    ) -> str:
        role = role or "unknown"
        meta = meta or {}
        label = model or (agent or "")  # human-readable target for responders/labels
        self.calls.append(
            {"role": role, "model": label, "agent": agent or "", "user": user, "meta": meta}
        )
        out = self.responder(role, label, system, user, meta)
        if json_mode:
            extract_json(out)  # validate determinism early
        return out


def default_responder(role: str, model: str, system: str, user: str, meta: dict[str, Any]) -> str:
    """Built-in deterministic responses, one branch per role."""
    if role == "planner":
        return _scripted_plan_json(user, meta)
    if role == "phase_define":
        task = meta.get("task_title", "Task")
        return (
            f"# Work definition\n\n## Objective\nDeliver: {task}\n\n"
            "## Steps\n1. Analyze the requirement.\n2. Implement the change.\n"
            "3. Self-check the result.\n\n## Acceptance\n- Output addresses the objective.\n"
        )
    if role == "task_exec":
        task = meta.get("task_title", "Task")
        phase = meta.get("phase_title", "Phase")
        return (
            f"# Result: {task}\n\n_Phase: {phase} · model: {model}_\n\n"
            f"Completed work for **{task}**. (Deterministic offline output.)\n\n"
            "## Notes\n- Implemented per the work definition.\n- No blockers.\n"
        )
    if role == "phase_summary":
        phase = meta.get("phase_title", "Phase")
        n = meta.get("task_count", 0)
        return (
            f"# Phase summary: {phase}\n\nCompleted {n} task(s).\n\n"
            "## Outcome\nAll task objectives met.\n\n## For the main orchestrator\n"
            f"Phase '{phase}' is done; docs can be updated accordingly.\n"
        )
    if role == "main_kickoff":
        goal = meta.get("goal", "")
        return f"# Project overview\n\n**Goal:** {goal}\n\nExecution started.\n\n## Progress log\n"
    if role == "main_integrate":
        phase = meta.get("phase_title", "Phase")
        return f"- Integrated phase **{phase}**: objectives met, docs updated.\n"
    if role == "main_finalize":
        goal = meta.get("goal", "")
        return f"# Final report\n\n**Goal:** {goal}\n\nAll phases complete. See per-phase summaries.\n"
    if role == "map_arch":
        return (
            "# Architecture overview\n\n## Purpose\nDeterministic offline mapping output.\n\n"
            "## Module / component map\n| Path | Responsibility |\n| --- | --- |\n"
            "| (scripted) | (offline backend) |\n\n## Risks / unknowns\n- Scripted output.\n"
        )
    if role == "map_prd":
        return (
            "# PRD (reverse-engineered)\n\n> Review and edit before planning.\n\n"
            "## Summary\nDeterministic offline PRD output.\n\n"
            "## Current Features & Capabilities\n- Offline scripted capability.\n\n"
            "## Open questions & assumptions\n- All inferences are scripted placeholders.\n"
        )
    return f"[scripted:{role}] {user[:120]}"


def _scripted_plan_json(goal: str, meta: dict[str, Any]) -> str:
    models = meta.get("models", {})
    phase_model = models.get("phase", "scripted/phase")
    task_model = models.get("task", "scripted/task")
    goal_line = goal.strip().splitlines()[0] if goal.strip() else "the project"
    plan = {
        "goal": goal_line,
        "phases": [
            {
                "id": "p1",
                "title": "Foundation",
                "objective": f"Establish the groundwork for {goal_line}.",
                "runner": phase_model,
                "tasks": [
                    {
                        "id": "p1-t1",
                        "title": "Define scope and structure",
                        "description": "Lay out the structure needed for the goal.",
                        "runner": task_model,
                        "depends_on": [],
                        "subtasks": [{"id": "p1-t1-s1", "description": "Outline components"}],
                    },
                    {
                        "id": "p1-t2",
                        "title": "Implement core",
                        "description": "Build the core pieces.",
                        "runner": task_model,
                        "depends_on": ["p1-t1"],
                        "subtasks": [],
                    },
                ],
            },
            {
                "id": "p2",
                "title": "Finish",
                "objective": f"Complete and verify {goal_line}.",
                "runner": phase_model,
                "tasks": [
                    {
                        "id": "p2-t1",
                        "title": "Verify and document",
                        "description": "Validate the result and write docs.",
                        "runner": task_model,
                        "depends_on": [],
                        "subtasks": [],
                    }
                ],
            },
        ],
    }
    return json.dumps(plan, indent=2)
