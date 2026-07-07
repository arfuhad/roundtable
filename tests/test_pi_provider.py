"""PiProvider: drive the `pi` coding agent per role, parsing its --mode json stream.

Pure helpers (argv building, event parsing) are unit-tested; the subprocess path
is exercised end-to-end with a fake `pi` shell script that records its argv and
emits a canned pi event stream.
"""

import json
import os
from pathlib import Path

import pytest

from roundtable.config import Config, PiOptions, load_config, write_default_config
from roundtable.errors import RoundtableError
from roundtable.llm import (
    PiProvider,
    _build_pi_argv,
    _pi_event_summary,
    _pi_flavor,
    _pi_prompt_delivery,
    parse_pi_events,
)


# --------------------------------------------------------------------------- #
# argv builder
# --------------------------------------------------------------------------- #
def test_build_pi_argv_worker_keeps_tools_and_appends_system():
    argv = _build_pi_argv(PiOptions(), model="anthropic/claude-haiku", system="SYS", is_worker=True)
    assert argv[:3] == ["pi", "--mode", "json"]
    assert "--no-tools" not in argv
    assert "--no-context-files" not in argv
    assert "--model" in argv and "anthropic/claude-haiku" in argv
    assert "--append-system-prompt" in argv and "--system-prompt" not in argv


def test_build_pi_argv_orchestrator_disables_tools_and_replaces_system():
    argv = _build_pi_argv(PiOptions(), model="anthropic/claude-opus", system="SYS", is_worker=False)
    assert "--no-tools" in argv
    assert "--no-context-files" in argv  # default: orchestrator_context_files=False
    assert "--system-prompt" in argv and "--append-system-prompt" not in argv


def test_build_pi_argv_orchestrator_context_files_opt_in():
    opts = PiOptions(orchestrator_context_files=True)
    argv = _build_pi_argv(opts, model="", system="", is_worker=False)
    assert "--no-tools" in argv
    assert "--no-context-files" not in argv  # opted back in
    assert "--model" not in argv  # empty model -> flag omitted


def test_build_pi_argv_uses_command_and_extra_args():
    opts = PiOptions(command=["npx", "pi"], extra_args=["--thinking", "medium"])
    argv = _build_pi_argv(opts, model="m", system="", is_worker=True)
    assert argv[:2] == ["npx", "pi"]
    assert argv[-2:] == ["--thinking", "medium"]


# --------------------------------------------------------------------------- #
# oh-my-pi (omp) flavor
# --------------------------------------------------------------------------- #
def test_pi_flavor_defaults():
    cmd, worker, orch = _pi_flavor(PiOptions())  # pi
    assert cmd == ["pi"] and worker == [] and orch == ["--no-context-files"]
    cmd, worker, orch = _pi_flavor(PiOptions(flavor="omp"))
    assert cmd == ["omp"] and worker == ["--auto-approve"] and orch == []


def test_pi_flavor_appends_user_extra_flags():
    opts = PiOptions(flavor="omp", worker_extra_args=["--no-lsp"], orchestrator_extra_args=["--no-rules"])
    _cmd, worker, orch = _pi_flavor(opts)
    assert worker == ["--auto-approve", "--no-lsp"]
    assert orch == ["--no-rules"]


def test_build_pi_argv_omp_worker_gets_auto_approve_no_context_flag():
    argv = _build_pi_argv(PiOptions(flavor="omp"), model="anthropic/claude-haiku", system="SYS", is_worker=True)
    assert argv[0] == "omp"
    assert "--auto-approve" in argv
    assert "--no-context-files" not in argv  # omp has no such flag
    assert "--no-tools" not in argv
    assert "--append-system-prompt" in argv


def test_build_pi_argv_omp_orchestrator_no_tools_no_autoapprove():
    argv = _build_pi_argv(PiOptions(flavor="omp"), model="m", system="SYS", is_worker=False)
    assert argv[0] == "omp"
    assert "--no-tools" in argv
    assert "--no-context-files" not in argv
    assert "--auto-approve" not in argv  # orchestrator can't edit anyway
    assert "--system-prompt" in argv


def test_pi_flavor_command_override_wins():
    cmd, _w, _o = _pi_flavor(PiOptions(flavor="omp", command=["bunx", "omp"]))
    assert cmd == ["bunx", "omp"]


def test_pi_prompt_delivery_pi_uses_stdin():
    argv, stdin = _pi_prompt_delivery("pi", ["pi", "--mode", "json"], "PROMPT")
    assert argv == ["pi", "--mode", "json"]
    assert stdin == "PROMPT"  # pi reads the prompt from stdin


def test_pi_prompt_delivery_omp_uses_arg_after_dashdash():
    argv, stdin = _pi_prompt_delivery("omp", ["omp", "--mode", "json"], "PROMPT")
    assert argv == ["omp", "--mode", "json", "-p", "--", "PROMPT"]  # omp reads the prompt as an arg
    assert stdin is None


# --------------------------------------------------------------------------- #
# event stream parsing
# --------------------------------------------------------------------------- #
def _stream(*events: dict) -> str:
    return "\n".join(json.dumps(e) for e in events)


def test_parse_pi_events_aggregates_usage_and_takes_last_text():
    stdout = _stream(
        {"type": "session_start", "reason": "startup"},
        {"type": "message_end", "message": {
            "role": "assistant",
            "content": [{"type": "toolCall", "name": "edit"}],
            "stopReason": "toolUse",
            "usage": {"input": 100, "output": 20, "totalTokens": 120, "cost": {"total": 0.002}},
        }},
        {"type": "message_end", "message": {"role": "toolResult"}},  # non-assistant, skipped
        {"type": "message_end", "message": {
            "role": "assistant",
            "content": [{"type": "text", "text": "Done: edited file."}],
            "stopReason": "stop",
            "usage": {"input": 200, "output": 40, "totalTokens": 240, "cost": {"total": 0.004}},
        }},
        {"type": "agent_end", "messages": []},
    )
    text, usage = parse_pi_events(stdout)
    assert text == "Done: edited file."
    assert usage["input"] == 300 and usage["output"] == 60 and usage["total"] == 360
    assert usage["cost"] == pytest.approx(0.006)


def test_parse_pi_events_totals_fallback_when_no_totalTokens():
    stdout = _stream({"type": "message_end", "message": {
        "role": "assistant", "content": [{"type": "text", "text": "hi"}],
        "stopReason": "stop", "usage": {"input": 5, "output": 7},  # no totalTokens/cost
    }})
    _text, usage = parse_pi_events(stdout)
    assert usage["total"] == 12 and usage["cost"] == 0.0


def test_parse_pi_events_falls_back_to_agent_end_messages():
    # No message_end events -> aggregate from the terminal agent_end message list.
    stdout = _stream({"type": "agent_end", "messages": [
        {"role": "assistant", "content": [{"type": "text", "text": "from agent_end"}],
         "stopReason": "stop", "usage": {"input": 9, "output": 1, "totalTokens": 10, "cost": {"total": 0.5}}},
    ]})
    text, usage = parse_pi_events(stdout)
    assert text == "from agent_end"
    assert usage["total"] == 10 and usage["cost"] == pytest.approx(0.5)


def test_parse_pi_events_raises_on_error_stop_reason():
    stdout = _stream({"type": "message_end", "message": {
        "role": "assistant", "content": [], "stopReason": "error", "errorMessage": "rate limit",
    }})
    with pytest.raises(RoundtableError, match="rate limit"):
        parse_pi_events(stdout)


def test_parse_pi_events_ignores_non_json_and_non_dict_lines():
    stdout = "not json\n[1,2,3]\n" + _stream({"type": "message_end", "message": {
        "role": "assistant", "content": [{"type": "text", "text": "ok"}], "stopReason": "stop",
        "usage": {"input": 1, "output": 1, "totalTokens": 2, "cost": {"total": 0}},
    }})
    text, usage = parse_pi_events(stdout)
    assert text == "ok" and usage["total"] == 2


def test_pi_event_summary_text_tool_and_none():
    assert _pi_event_summary(json.dumps({"type": "message_end", "message": {
        "role": "assistant", "content": [{"type": "text", "text": "hello"}], "stopReason": "stop",
    }})) == "hello"
    assert _pi_event_summary(json.dumps({"type": "tool_execution_start", "toolName": "bash"})) == "→ bash"
    assert _pi_event_summary("not json") is None
    assert _pi_event_summary(json.dumps({"type": "session_start"})) is None


# --------------------------------------------------------------------------- #
# config + wiring
# --------------------------------------------------------------------------- #
def test_write_pi_config_selects_pi_backend(tmp_path):
    write_default_config(tmp_path, backend="pi")
    cfg = load_config(tmp_path)
    assert cfg.provider == "pi"
    assert cfg.models.task.model  # mixed template assigns a task model
    assert cfg.pi.command == ["pi"]


def test_write_default_config_still_cli(tmp_path):
    write_default_config(tmp_path)  # default backend
    assert load_config(tmp_path).provider == "cli"


def test_write_pi_config_omp_flavor(tmp_path):
    write_default_config(tmp_path, backend="pi", flavor="omp")
    cfg = load_config(tmp_path)
    assert cfg.provider == "pi"
    assert cfg.pi.flavor == "omp"
    assert cfg.pi.command == ["omp"]
    # sanity: an omp-flavored provider builds omp argv with auto-approve
    argv = _build_pi_argv(cfg.pi, model="m", system="S", is_worker=True)
    assert argv[0] == "omp" and "--auto-approve" in argv


def test_build_provider_returns_pi_provider():
    from roundtable.cli import build_provider

    prov = build_provider(Config(provider="pi"), cwd=".")
    assert type(prov).__name__ == "PiProvider"


async def test_pi_provider_missing_binary_raises_helpful_error():
    prov = PiProvider(PiOptions(command=["definitely-not-a-real-pi-xyz"]), max_retries=0)
    with pytest.raises(RoundtableError, match="needs the .* CLI on PATH"):
        await prov.complete(model="m", system="", user="hi", role="task_exec")


async def test_pi_provider_auth_error_is_concise(tmp_path):
    script = tmp_path / "fake-omp.sh"
    script.write_text(
        "#!/bin/sh\n"
        "echo 'error: No API key found for fireworks.' >&2\n"
        "exit 1\n"
    )
    script.chmod(0o755)
    prov = PiProvider(PiOptions(flavor="omp", command=[str(script)]), max_retries=0)

    with pytest.raises(RoundtableError) as ei:
        await prov.complete(model="opencode-go/glm-5.2", system="", user="hi", role="phase_define")

    msg = str(ei.value)
    assert "fireworks" in msg
    assert "authenticate provider" in msg
    assert "dist/cli.js" not in msg


# --------------------------------------------------------------------------- #
# end-to-end subprocess (fake `pi`)
# --------------------------------------------------------------------------- #
_FAKE_PI = r"""#!/bin/sh
# Record argv (one per line) so the test can assert role -> flags.
if [ -n "$PI_ARGV_OUT" ]; then
  : > "$PI_ARGV_OUT"
  for a in "$@"; do printf '%s\n' "$a" >> "$PI_ARGV_OUT"; done
fi
# Drain the piped prompt (record it if asked) so the writer end isn't broken.
if [ -n "$PI_STDIN_OUT" ]; then cat > "$PI_STDIN_OUT"; else cat > /dev/null; fi
# Emit a pi --mode json event stream: a tool turn then a final text answer.
cat <<'JSON'
{"type":"session_start","reason":"startup"}
{"type":"message_end","message":{"role":"assistant","content":[{"type":"toolCall","name":"edit"}],"stopReason":"toolUse","usage":{"input":100,"output":10,"totalTokens":110,"cost":{"total":0.001}}}}
{"type":"message_end","message":{"role":"assistant","content":[{"type":"text","text":"pi did the work"}],"stopReason":"stop","usage":{"input":50,"output":20,"totalTokens":70,"cost":{"total":0.002}}}}
{"type":"agent_end","messages":[]}
JSON
"""


def _fake_pi(tmp_path: Path) -> Path:
    script = tmp_path / "fake-pi.sh"
    script.write_text(_FAKE_PI)
    script.chmod(0o755)
    return script


async def test_pi_provider_end_to_end_worker(tmp_path, monkeypatch):
    argv_out = tmp_path / "argv.txt"
    stdin_out = tmp_path / "stdin.txt"
    monkeypatch.setenv("PI_ARGV_OUT", str(argv_out))
    monkeypatch.setenv("PI_STDIN_OUT", str(stdin_out))

    prov = PiProvider(PiOptions(command=[str(_fake_pi(tmp_path))]), max_retries=0)
    out = await prov.complete(
        model="anthropic/claude-haiku", system="SYS", user="do the thing", role="task_exec"
    )

    assert out == "pi did the work"
    # exact usage/cost recorded, not estimated
    assert prov.stats.total_tokens == 180  # 110 + 70
    assert prov.stats.prompt_tokens == 150 and prov.stats.completion_tokens == 30
    assert prov.stats.cost_usd == pytest.approx(0.003)
    assert prov.stats.estimated is False
    assert prov.stats.calls == 1
    # prompt was piped on stdin
    assert stdin_out.read_text() == "do the thing"
    # worker invocation keeps tools
    flags = argv_out.read_text().splitlines()
    assert "--mode" in flags and "json" in flags
    assert "--no-tools" not in flags
    assert "--append-system-prompt" in flags


async def test_pi_provider_end_to_end_orchestrator_role_disables_tools(tmp_path, monkeypatch):
    argv_out = tmp_path / "argv.txt"
    monkeypatch.setenv("PI_ARGV_OUT", str(argv_out))
    monkeypatch.delenv("PI_STDIN_OUT", raising=False)

    prov = PiProvider(PiOptions(command=[str(_fake_pi(tmp_path))]), max_retries=0)
    await prov.complete(model="anthropic/claude-opus", system="SYS", user="plan it", role="planner")

    flags = argv_out.read_text().splitlines()
    assert "--no-tools" in flags
    assert "--no-context-files" in flags
    assert "--system-prompt" in flags and "--append-system-prompt" not in flags


async def test_pi_provider_end_to_end_omp_prompt_as_arg(tmp_path, monkeypatch):
    argv_out = tmp_path / "argv.txt"
    stdin_out = tmp_path / "stdin.txt"
    monkeypatch.setenv("PI_ARGV_OUT", str(argv_out))
    monkeypatch.setenv("PI_STDIN_OUT", str(stdin_out))

    prov = PiProvider(PiOptions(flavor="omp", command=[str(_fake_pi(tmp_path))]), max_retries=0)
    out = await prov.complete(model="m", system="SYS", user="do the thing", role="task_exec")

    assert out == "pi did the work"
    flags = argv_out.read_text().splitlines()
    assert "--auto-approve" in flags
    assert "--" in flags
    assert flags[-1] == "do the thing"  # prompt passed as the final argument
    assert stdin_out.read_text() == ""  # omp gets no stdin (DEVNULL)


async def test_pi_provider_handles_huge_single_json_line(tmp_path, monkeypatch):
    # A single event line carrying a whole generated file easily exceeds asyncio's
    # default 64KB readline limit; the chunked reader must handle it without crashing.
    monkeypatch.delenv("PI_ARGV_OUT", raising=False)
    monkeypatch.delenv("PI_STDIN_OUT", raising=False)
    big = "X" * 500_000  # ~500KB of text on one line (>> 64KB readline limit)
    msg = {"role": "assistant", "content": [{"type": "text", "text": big}],
           "stopReason": "stop", "usage": {"input": 1, "output": 1, "totalTokens": 2, "cost": {"total": 0.0}}}
    payload = tmp_path / "payload.jsonl"
    payload.write_text(json.dumps({"type": "message_end", "message": msg}) + "\n")
    script = tmp_path / "big-pi.sh"
    script.write_text(f"#!/bin/sh\ncat > /dev/null 2>&1\ncat {payload}\n")  # drain stdin, emit the big line
    script.chmod(0o755)

    prov = PiProvider(PiOptions(command=[str(script)]), max_retries=0)
    out = await prov.complete(model="m", system="", user="go", role="task_exec")
    assert out == big  # full large text returned intact
    assert prov.stats.total_tokens == 2


async def test_pi_provider_json_mode_appends_instruction(tmp_path, monkeypatch):
    stdin_out = tmp_path / "stdin.txt"
    monkeypatch.setenv("PI_STDIN_OUT", str(stdin_out))
    monkeypatch.delenv("PI_ARGV_OUT", raising=False)

    prov = PiProvider(PiOptions(command=[str(_fake_pi(tmp_path))]), max_retries=0)
    await prov.complete(model="m", system="", user="give me json", role="planner", json_mode=True)
    piped = stdin_out.read_text()
    assert piped.startswith("give me json")
    assert "ONLY the JSON object" in piped
