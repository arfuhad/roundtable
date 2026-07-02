import pytest

from harness.llm import ScriptedProvider, extract_json
from harness.models import Plan


def test_extract_json_raw_fenced_and_prose():
    assert extract_json('{"a": 1}') == {"a": 1}
    assert extract_json('```json\n{"a": 2}\n```') == {"a": 2}
    assert extract_json('blah {"a": {"b": 3}} trailing') == {"a": {"b": 3}}
    # string-aware: braces inside strings don't confuse the scanner
    assert extract_json('note: {"k": "a}b{c"}') == {"k": "a}b{c"}


def test_extract_json_arrays_and_double_fence():
    # top-level array, raw and prose-wrapped
    assert extract_json("[1, 2, 3]") == [1, 2, 3]
    assert extract_json('prefix [{"a": 1}] suffix') == [{"a": 1}]
    # object that appears before an array in prose -> object wins
    assert extract_json('see {"a": 1} and [9]') == {"a": 1}
    # double-fenced code block
    assert extract_json('```\n```json\n{"a": 9}\n```\n```') == {"a": 9}


def test_extract_json_failure():
    with pytest.raises(ValueError):
        extract_json("no json here at all")


async def test_cli_provider_streams_output(tmp_path):
    from harness.config import AgentSpec
    from harness.llm import CLIProvider

    spec = AgentSpec(command=["sh", "-c", "printf 'alpha\\nbeta\\ngamma\\n'"])
    provider = CLIProvider({"echo": spec}, cwd=tmp_path)
    chunks: list[str] = []

    out = await provider.complete(
        model="echo", system="s", user="u", role="task_exec", on_output=chunks.append
    )

    assert "alpha" in out and "gamma" in out          # full output still returned
    assert "beta" in "".join(chunks)                  # and it was streamed via on_output
    assert provider.stats.calls == 1                  # usage stats tracked (Task 3.1)


async def test_cli_provider_nonzero_exit_raises(tmp_path):
    from harness.config import AgentSpec
    from harness.llm import CLIProvider

    spec = AgentSpec(command=["sh", "-c", "echo boom >&2; exit 2"])
    provider = CLIProvider({"bad": spec}, cwd=tmp_path, max_retries=0)
    with pytest.raises(RuntimeError, match="exited 2"):
        await provider.complete(model="bad", system="s", user="u", role="task_exec")


async def test_scripted_planner_produces_valid_plan():
    p = ScriptedProvider()
    out = await p.complete(
        model="m", system="s", user="Build X", json_mode=True, role="planner",
        meta={"models": {"phase": "x/phase", "task": "y/task"}},
    )
    plan = Plan.model_validate(extract_json(out))
    assert [ph.id for ph in plan.phases] == ["p1", "p2"]
    assert plan.phases[0].tasks[1].depends_on == ["p1-t1"]
    assert str(plan.phases[0].runner) == "x/phase"
    assert str(plan.phases[0].tasks[0].runner) == "y/task"


async def test_scripted_provider_records_calls():
    p = ScriptedProvider()
    await p.complete(model="m", system="s", user="hi", role="task_exec",
                     meta={"task_title": "T", "phase_title": "P"})
    assert p.calls[-1]["role"] == "task_exec"
    assert p.calls[-1]["meta"]["task_title"] == "T"
