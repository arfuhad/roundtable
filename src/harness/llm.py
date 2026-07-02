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
import time
from dataclasses import dataclass
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
        on_output: Callable[[str], None] | None = None,
    ) -> str: ...


@dataclass
class RunStats:
    """Accumulated usage statistics for a provider across an entire run."""
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    total_duration_s: float = 0.0

    def snapshot(self) -> dict[str, Any]:
        return {
            "calls": self.calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "total_duration_s": round(self.total_duration_s, 2),
        }


# --------------------------------------------------------------------------- #
# JSON extraction
# --------------------------------------------------------------------------- #
def extract_json(text: str) -> dict[str, Any] | list:
    """Best-effort extraction of a JSON object or array from model output.

    Handles raw JSON, ```json fenced blocks (including double-fenced), and
    prose-wrapped objects/arrays by scanning for the first balanced ``{...}``
    or ``[...]`` (string-aware).
    """
    s = text.strip()
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass

    # Strip code fences (handles double-fenced blocks too).
    s = _strip_fences(s)
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass

    # Find first balanced object or array.
    obj_start = s.find("{")
    arr_start = s.find("[")

    # Pick whichever comes first; try both if present.
    candidates: list[str | None] = []
    if obj_start != -1 and (arr_start == -1 or obj_start <= arr_start):
        candidates.append(_first_balanced(s, "{", "}"))
        if arr_start != -1:
            candidates.append(_first_balanced(s, "[", "]"))
    elif arr_start != -1:
        candidates.append(_first_balanced(s, "[", "]"))
        if obj_start != -1:
            candidates.append(_first_balanced(s, "{", "}"))

    for c in candidates:
        if c is not None:
            try:
                return json.loads(c)
            except json.JSONDecodeError:
                continue

    raise ValueError(f"no JSON object found in model output:\n{text[:500]}")


def _strip_fences(s: str) -> str:
    """Strip one or two layers of code fences from a string."""
    for _ in range(2):  # handle double-fenced blocks
        stripped = s.strip()
        if stripped.startswith("```"):
            stripped = stripped.split("\n", 1)[1] if "\n" in stripped else stripped
            if stripped.rstrip().endswith("```"):
                stripped = stripped.rstrip()[:-3]
            s = stripped.strip()
        else:
            break
    return s


def _first_balanced(s: str, open_ch: str, close_ch: str) -> str | None:
    """Find the first balanced substring delimited by open_ch/close_ch.

    String-aware: braces inside JSON strings are not counted.
    """
    start = s.find(open_ch)
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
        elif c == open_ch:
            depth += 1
        elif c == close_ch:
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
        self.stats = RunStats()

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
        on_output: Callable[[str], None] | None = None,  # accepted for interface parity; unused
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

        t0 = time.monotonic()

        async def call_with_json() -> str:
            resp = await litellm.acompletion(**kwargs)
            self._record_usage(resp)
            return resp.choices[0].message.content or ""

        async def call_plain() -> str:
            plain = {k: v for k, v in kwargs.items() if k != "response_format"}
            resp = await litellm.acompletion(**plain)
            self._record_usage(resp)
            return resp.choices[0].message.content or ""

        try:
            result = await _retry(call_with_json, attempts=self.max_retries + 1)
        except Exception:
            if not json_mode:
                raise
            # Some providers reject response_format; fall back to plain prompting.
            result = await _retry(call_plain, attempts=self.max_retries + 1)

        self.stats.total_duration_s += time.monotonic() - t0
        return result

    def _record_usage(self, resp: Any) -> None:
        """Accumulate token counts from a litellm response."""
        self.stats.calls += 1
        usage = getattr(resp, "usage", None)
        if usage:
            self.stats.prompt_tokens += getattr(usage, "prompt_tokens", 0) or 0
            self.stats.completion_tokens += getattr(usage, "completion_tokens", 0) or 0
            self.stats.total_tokens += getattr(usage, "total_tokens", 0) or 0


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
        self.stats = RunStats()

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
        on_output: Callable[[str], None] | None = None,
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
        use_pty = spec.pty
        t0 = time.monotonic()
        result = await _retry(
            lambda: self._run(argv, stdin_data, agent_key, use_pty, on_output),
            attempts=self.max_retries + 1,
        )
        self.stats.calls += 1
        self.stats.total_duration_s += time.monotonic() - t0
        return result

    async def _run(
        self, argv: list[str], stdin_data: str | None, model: str,
        use_pty: bool = False, on_output: Callable[[str], None] | None = None,
    ) -> str:
        if use_pty:
            return await self._run_pty(argv, stdin_data, model, on_output)
        proc = await asyncio.create_subprocess_exec(
            *argv,
            cwd=self.cwd,
            stdin=asyncio.subprocess.PIPE if stdin_data is not None else None,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        if on_output and proc.stdout:
            # Stream stdout line-by-line while also collecting the full output.
            buf: list[str] = []
            batch: list[str] = []
            batch_count = 0
            last_flush = time.monotonic()

            async def drain_stderr() -> bytes:
                assert proc.stderr is not None
                return await proc.stderr.read()

            stderr_task = asyncio.create_task(drain_stderr())

            # Feed stdin if needed.
            if stdin_data is not None and proc.stdin is not None:
                proc.stdin.write(stdin_data.encode())
                await proc.stdin.drain()
                proc.stdin.close()

            try:
                async with asyncio.timeout(self.timeout):
                    async for raw_line in proc.stdout:
                        line = raw_line.decode("utf-8", "replace")
                        buf.append(line)
                        batch.append(line)
                        batch_count += 1
                        now = time.monotonic()
                        if batch_count >= 10 or (now - last_flush) >= 5.0:
                            on_output("".join(batch))
                            batch.clear()
                            batch_count = 0
                            last_flush = now
                    # Flush remaining lines.
                    if batch:
                        on_output("".join(batch))
                    await proc.wait()
            except TimeoutError:
                proc.kill()
                await proc.wait()
                raise TimeoutError(f"agent {model!r} timed out after {self.timeout}s")

            err = await stderr_task
            out_text = "".join(buf)
            if proc.returncode != 0:
                err_text = err.decode("utf-8", "replace").strip()
                detail = (err_text or out_text.strip() or "(no output on stdout/stderr)")[-800:]
                raise RuntimeError(f"agent {model!r} exited {proc.returncode}: {detail}")
            return out_text

        # Non-streaming path (original behavior).
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

    async def _run_pty(
        self, argv: list[str], stdin_data: str | None, model: str,
        on_output: Callable[[str], None] | None = None,
    ) -> str:
        """Run argv in a pseudo-terminal so the subprocess sees a real TTY."""
        try:
            import pty as _pty
            import os as _os
        except ImportError:
            raise HarnessError("pty mode requires the 'pty' stdlib module (Unix/macOS only)")

        master_fd, slave_fd = _pty.openpty()
        proc = await asyncio.create_subprocess_exec(
            *argv,
            cwd=self.cwd,
            stdin=slave_fd,
            stdout=slave_fd,
            stderr=slave_fd,
        )
        _os.close(slave_fd)  # parent does not need the slave end

        if stdin_data is not None:
            try:
                _os.write(master_fd, stdin_data.encode())
            except OSError:
                pass

        loop = asyncio.get_running_loop()
        chunks: list[bytes] = []

        def _drain() -> None:
            batch_bytes: list[bytes] = []
            batch_size = 0
            last_flush = time.monotonic()
            while True:
                try:
                    data = _os.read(master_fd, 4096)
                    if not data:
                        break
                    chunks.append(data)
                    if on_output:
                        batch_bytes.append(data)
                        batch_size += len(data)
                        now = time.monotonic()
                        if batch_size >= 4096 or (now - last_flush) >= 5.0:
                            text = b"".join(batch_bytes).decode("utf-8", "replace")
                            on_output(text.replace("\r\n", "\n").replace("\r", ""))
                            batch_bytes.clear()
                            batch_size = 0
                            last_flush = now
                except OSError:
                    break  # EIO/ENXIO when slave closes — normal PTY EOF
            # Flush remaining.
            if on_output and batch_bytes:
                text = b"".join(batch_bytes).decode("utf-8", "replace")
                on_output(text.replace("\r\n", "\n").replace("\r", ""))
            try:
                _os.close(master_fd)
            except OSError:
                pass

        drain_fut = loop.run_in_executor(None, _drain)
        try:
            await asyncio.wait_for(proc.wait(), timeout=self.timeout)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            await drain_fut
            raise TimeoutError(f"agent {model!r} timed out after {self.timeout}s")

        await drain_fut

        # PTY driver does CR+LF translation; normalise to LF only.
        output = b"".join(chunks).decode("utf-8", "replace").replace("\r\n", "\n").replace("\r", "")
        if proc.returncode != 0:
            raise RuntimeError(f"agent {model!r} exited {proc.returncode}: {output.strip()[-800:]}")
        return output


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
        self.stats = RunStats()

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
        on_output: Callable[[str], None] | None = None,  # accepted for interface parity; unused
    ) -> str:
        role = role or "unknown"
        meta = meta or {}
        label = model or (agent or "")  # human-readable target for responders/labels
        self.calls.append(
            {"role": role, "model": label, "agent": agent or "", "user": user, "meta": meta}
        )
        out = self.responder(role, label, system, user, meta)
        self.stats.calls += 1
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
                    },
                    {
                        "id": "p1-t2",
                        "title": "Implement core",
                        "description": "Build the core pieces.",
                        "runner": task_model,
                        "depends_on": ["p1-t1"],
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
                        "depends_on": ["p1-t2"],  # cross-phase: builds on phase 1's core
                    }
                ],
            },
        ],
    }
    return json.dumps(plan, indent=2)
