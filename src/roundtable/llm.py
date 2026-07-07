"""LLM transport layer.

A small ``LLMProvider`` protocol decouples the orchestration engine from how
calls are made. Four backends ship:

* ``PiProvider`` — drive the ``pi`` coding agent for every role (recommended).
  pi handles LLM connectivity/auth; task agents run with pi's file tools,
  orchestrator roles run ``pi --no-tools``. Reports exact token usage and cost.
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
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Protocol, runtime_checkable

from .config import AgentSpec, PiOptions
from .errors import RoundtableError


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
    """Accumulated usage statistics for a provider across an entire run.

    ``estimated`` is True when the token counts are approximations rather than
    exact figures reported by the provider — the case for the CLI backend,
    which only sees stdout and has no usage metadata to read.
    """
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    total_duration_s: float = 0.0
    estimated: bool = False
    cost_usd: float = 0.0  # real dollar cost when the backend reports it (pi); 0 otherwise

    def snapshot(self) -> dict[str, Any]:
        return {
            "calls": self.calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "total_duration_s": round(self.total_duration_s, 2),
            "estimated": self.estimated,
            "cost_usd": round(self.cost_usd, 6),
        }


def estimate_tokens(text: str) -> int:
    """Rough token count (~4 chars/token) for providers that report no usage.

    Used by the CLI backend so a run surfaces an approximate token tally
    instead of zeros. Flagged via ``RunStats.estimated`` so callers can label
    it as an estimate rather than an exact count.
    """
    if not text:
        return 0
    return max(1, len(text) // 4)


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


async def _terminate_process(proc: asyncio.subprocess.Process) -> None:
    """Terminate a subprocess, escalating to kill if it does not exit promptly."""
    if proc.returncode is not None:
        return
    try:
        proc.terminate()
    except ProcessLookupError:
        return
    try:
        await asyncio.wait_for(proc.wait(), timeout=5)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()


async def _cancel_reader(task: "asyncio.Task[Any]") -> None:
    """Cancel a background stream-reader task and wait for it to settle, so it is
    not left orphaned (which can log 'Task was destroyed but it is pending')."""
    if task.done():
        return
    task.cancel()
    try:
        await task
    except (asyncio.CancelledError, Exception):
        pass


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
            raise RoundtableError(
                "provider 'litellm' needs the litellm package: pip install 'roundtable-cli[litellm]' "
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
            raise RoundtableError(
                f"agent {agent_key!r} is not defined under `agents:` in roundtable.config.yaml; "
                f"available: {', '.join(sorted(self.agents)) or '(none)'}"
            )
        if json_mode:
            user = user + "\n\nRespond with ONLY the JSON object, no prose or code fences."

        argv, stdin_data = _render_command(spec, system, user, model_token)
        exe = shutil.which(argv[0])
        if exe is None:
            raise RoundtableError(
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
        # A CLI returns only stdout — no usage metadata — so approximate tokens
        # from text length and mark the run's stats as estimated.
        prompt_est = estimate_tokens(system) + estimate_tokens(user)
        completion_est = estimate_tokens(result)
        self.stats.prompt_tokens += prompt_est
        self.stats.completion_tokens += completion_est
        self.stats.total_tokens += prompt_est + completion_est
        self.stats.estimated = True
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
                await _terminate_process(proc)
                await _cancel_reader(stderr_task)
                raise TimeoutError(f"agent {model!r} timed out after {self.timeout}s")
            except asyncio.CancelledError:
                await _terminate_process(proc)
                await _cancel_reader(stderr_task)
                raise

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
            await _terminate_process(proc)
            raise TimeoutError(f"agent {model!r} timed out after {self.timeout}s")
        except asyncio.CancelledError:
            await _terminate_process(proc)
            raise
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
            raise RoundtableError("pty mode requires the 'pty' stdlib module (Unix/macOS only)")

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
            await _terminate_process(proc)
            await drain_fut
            raise TimeoutError(f"agent {model!r} timed out after {self.timeout}s")
        except asyncio.CancelledError:
            await _terminate_process(proc)
            await drain_fut
            raise

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
# Pi backend (recommended): drive the `pi` coding agent for every role
# --------------------------------------------------------------------------- #
# Only this role gets pi's file tools (it edits the repo). Every other role runs
# `pi --no-tools` as a pure completion.
PI_WORKER_ROLE = "task_exec"


def _pi_flavor(options: PiOptions) -> tuple[list[str], list[str], list[str]]:
    """Resolve (command, worker_flags, orchestrator_flags) for a pi flavor.

    Both flavors share the core contract; only two things differ:
    * ``pi``  — orchestrator roles skip the repo's context files (``--no-context-files``)
      unless ``orchestrator_context_files`` is set.
    * ``omp`` — has no ``--no-context-files``, and gates edits behind approval, so
      task (worker) agents get ``--auto-approve`` to run autonomously.
    User ``worker_extra_args`` / ``orchestrator_extra_args`` are appended on top.
    """
    flavor = (options.flavor or "pi").lower()
    if flavor == "omp":
        command = list(options.command) or ["omp"]
        worker = ["--auto-approve"]
        orch: list[str] = []
    else:  # "pi"
        command = list(options.command) or ["pi"]
        worker = []
        orch = [] if options.orchestrator_context_files else ["--no-context-files"]
    worker += list(options.worker_extra_args)
    orch += list(options.orchestrator_extra_args)
    return command, worker, orch


def _build_pi_argv(
    options: PiOptions, *, model: str, system: str, is_worker: bool
) -> list[str]:
    """Build the pi/omp flag argv for a role (without the user prompt — that is
    delivered per flavor by :func:`_pi_prompt_delivery`). The system prompt goes
    via a flag.

    Worker (``task_exec``) keeps file tools and *appends* roundtable's task
    instructions to the tool's coding system prompt. Orchestrator roles disable
    tools and *replace* the system prompt (they only reason/write).
    """
    command, worker_flags, orch_flags = _pi_flavor(options)
    argv = list(command) + ["--mode", "json"]
    if not is_worker:
        argv += ["--no-tools"]
    if model:
        argv += ["--model", model]
    if system:
        argv += (["--append-system-prompt", system] if is_worker else ["--system-prompt", system])
    argv += (worker_flags if is_worker else orch_flags)
    argv += list(options.extra_args)
    return argv


def _pi_prompt_delivery(flavor: str, argv: list[str], payload: str) -> tuple[list[str], str | None]:
    """Attach the user prompt to the invocation the way the flavor expects.

    Upstream ``pi`` reads a piped prompt from stdin; ``omp`` ignores stdin and
    takes the prompt as a positional argument (``--`` guards prompts that start
    with ``-``/``@``). Returns ``(argv, stdin_data)`` where ``stdin_data`` is None
    when the prompt is passed as an argument.
    """
    if (flavor or "pi").lower() == "omp":
        return argv + ["-p", "--", payload], None
    return argv, payload


def _assistant_text_usage(msg: Any) -> tuple[str, dict[str, float], str | None] | None:
    """Extract (text, usage, error) from a pi assistant message, or None if the
    message is not an assistant turn. ``usage`` keys: input/output/total/cost."""
    if not isinstance(msg, dict) or msg.get("role") != "assistant":
        return None
    text = "".join(
        c.get("text", "")
        for c in msg.get("content", [])
        if isinstance(c, dict) and c.get("type") == "text"
    )
    usage = msg.get("usage") or {}
    cost = usage.get("cost") or {}
    u = {
        "input": float(usage.get("input", 0) or 0),
        "output": float(usage.get("output", 0) or 0),
        "total": float(usage.get("totalTokens", 0) or 0),
        "cost": float(cost.get("total", 0.0) or 0.0),
    }
    if not u["total"]:
        u["total"] = u["input"] + u["output"]
    stop = msg.get("stopReason")
    err = msg.get("errorMessage") or f"pi stopped: {stop}" if stop in ("error", "aborted") else None
    return text, u, err


def parse_pi_events(stdout: str) -> tuple[str, dict[str, float]]:
    """Parse a ``pi --mode json`` stdout stream into (final_text, usage).

    Usage is summed across assistant ``message_end`` events (each message counted
    once); the final text is the last assistant message that carried text. Raises
    :class:`RoundtableError` if any assistant turn reported an error/abort. Falls
    back to the terminal ``agent_end`` message list if no ``message_end`` events
    were seen (older/edge pi builds).
    """
    totals = {"input": 0.0, "output": 0.0, "total": 0.0, "cost": 0.0}
    final_text = ""
    saw_message_end = False
    error: str | None = None
    agent_end_msgs: list[Any] | None = None

    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(ev, dict):
            continue
        etype = ev.get("type")
        if etype == "message_end":
            parsed = _assistant_text_usage(ev.get("message"))
            if parsed is None:
                continue
            saw_message_end = True
            text, u, err = parsed
            for k in totals:
                totals[k] += u[k]
            if text.strip():
                final_text = text
            if err and not error:
                error = err
        elif etype == "agent_end":
            msgs = ev.get("messages")
            if isinstance(msgs, list):
                agent_end_msgs = msgs

    if not saw_message_end and agent_end_msgs is not None:
        for msg in agent_end_msgs:
            parsed = _assistant_text_usage(msg)
            if parsed is None:
                continue
            text, u, err = parsed
            for k in totals:
                totals[k] += u[k]
            if text.strip():
                final_text = text
            if err and not error:
                error = err

    if error:
        raise RoundtableError(f"pi agent error: {error}")
    return final_text, totals


class PiProvider:
    """Drive the ``pi`` coding agent for every role (recommended backend).

    pi handles LLM connectivity, auth and model routing. Task agents run with pi's
    file tools so they edit the real project; all other roles run ``pi --no-tools``
    as pure completions. pi is invoked with ``--mode json`` and the event stream is
    parsed for the final answer plus **exact** token usage and dollar cost.
    """

    def __init__(
        self,
        options: PiOptions,
        *,
        cwd: str | Path | None = None,
        timeout: int = 900,
        max_retries: int = 1,
    ):
        self.options = options
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
        agent: str | None = None,  # ignored: pi is the agent
        json_mode: bool = False,
        temperature: float = 0.2,  # accepted for interface parity; pi controls its own sampling
        role: str | None = None,
        meta: dict[str, Any] | None = None,
        on_output: Callable[[str], None] | None = None,
    ) -> str:
        is_worker = role == PI_WORKER_ROLE
        argv = _build_pi_argv(self.options, model=model, system=system, is_worker=is_worker)
        exe = shutil.which(argv[0])
        if exe is None:
            flavor = (self.options.flavor or "pi").lower()
            install = (
                "npm install -g @oh-my-pi/pi-coding-agent"
                if flavor == "omp"
                else "npm install -g @earendil-works/pi-coding-agent"
            )
            raise RoundtableError(
                f"provider 'pi' (flavor {flavor!r}) needs the {argv[0]!r} CLI on PATH. "
                f"Install it ({install}) and connect an LLM (set ANTHROPIC_API_KEY / "
                "OPENAI_API_KEY / ... or use the tool's login), or switch `provider:` to "
                "'cli'/'litellm' in roundtable.config.yaml."
            )
        argv[0] = exe

        payload = user
        if json_mode:
            payload = user + "\n\nRespond with ONLY the JSON object, no prose or code fences."
        run_argv, stdin_data = _pi_prompt_delivery(self.options.flavor, argv, payload)

        t0 = time.monotonic()
        raw = await _retry(
            lambda: self._run(run_argv, stdin_data, role or "pi", on_output),
            attempts=self.max_retries + 1,
        )
        text, usage = parse_pi_events(raw)
        self.stats.calls += 1
        self.stats.total_duration_s += time.monotonic() - t0
        self.stats.prompt_tokens += int(usage["input"])
        self.stats.completion_tokens += int(usage["output"])
        self.stats.total_tokens += int(usage["total"])
        self.stats.cost_usd += usage["cost"]
        return text

    async def _run(
        self, argv: list[str], stdin_data: str | None, role: str,
        on_output: Callable[[str], None] | None = None,
    ) -> str:
        proc = await asyncio.create_subprocess_exec(
            *argv,
            cwd=self.cwd,
            stdin=asyncio.subprocess.PIPE if stdin_data is not None else asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        assert proc.stdout is not None and proc.stderr is not None
        stderr_task = asyncio.create_task(proc.stderr.read())

        if stdin_data is not None and proc.stdin is not None:
            proc.stdin.write(stdin_data.encode())
            await proc.stdin.drain()
            proc.stdin.close()

        # Read in chunks and split lines ourselves. pi/omp emit one JSON event per
        # line, and a single event (an assistant message or tool result carrying a
        # whole generated file) can be many MB — far past asyncio's default 64KB
        # readline limit, which would raise "Separator is found, but chunk is longer
        # than limit". Chunked reads have no such cap.
        lines: list[str] = []
        buf = b""

        def _emit(line: str) -> None:
            lines.append(line)
            if on_output:
                summary = _pi_event_summary(line)
                if summary:
                    on_output(summary)

        try:
            async with asyncio.timeout(self.timeout):
                while True:
                    chunk = await proc.stdout.read(65536)
                    if not chunk:
                        break
                    buf += chunk
                    nl = buf.rfind(b"\n")
                    if nl != -1:
                        complete, buf = buf[: nl + 1], buf[nl + 1 :]
                        for raw in complete.splitlines(keepends=True):
                            _emit(raw.decode("utf-8", "replace"))
                if buf:  # trailing line without a newline
                    _emit(buf.decode("utf-8", "replace"))
                await proc.wait()
        except TimeoutError:
            await _terminate_process(proc)
            await _cancel_reader(stderr_task)
            raise TimeoutError(f"pi ({role}) timed out after {self.timeout}s")
        except asyncio.CancelledError:
            await _terminate_process(proc)
            await _cancel_reader(stderr_task)
            raise

        err = await stderr_task
        out = "".join(lines)
        if proc.returncode != 0:
            err_text = err.decode("utf-8", "replace").strip()
            detail = (err_text or out.strip() or "(no output on stdout/stderr)")[-800:]
            auth = _auth_error(detail)
            if auth:
                raise RoundtableError(
                    f"pi ({role}) could not authenticate provider {auth!r}; "
                    f"run the pi/omp login flow for {auth} or set its API key, "
                    "then retry."
                )
            raise RuntimeError(f"pi ({role}) exited {proc.returncode}: {detail}")
        return out


def _auth_error(text: str) -> str | None:
    m = re.search(r"No API key found for ([A-Za-z0-9_.-]+)", text)
    return m.group(1) if m else None


def _pi_event_summary(line: str) -> str | None:
    """A short human line for the live dashboard from one pi json event (or None)."""
    line = line.strip()
    if not line:
        return None
    try:
        ev = json.loads(line)
    except json.JSONDecodeError:
        return None
    if not isinstance(ev, dict):
        return None
    etype = ev.get("type")
    if etype == "message_end":
        parsed = _assistant_text_usage(ev.get("message"))
        if parsed and parsed[0].strip():
            return parsed[0]
    elif etype in ("tool_execution_start", "tool_call"):
        name = ev.get("toolName") or ev.get("name")
        if isinstance(name, str) and name:
            return f"→ {name}"
    return None


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
