import pytest
from pydantic import ValidationError

from harness.models import AgentRef, Phase, Plan, Status, Subtask, Task, slugify
from harness.errors import HarnessError


def test_agentref_coercion_object_string_and_bare():
    # object form
    assert AgentRef.model_validate({"agent": "opencode", "model": "mimo-v2.5-pro"}) == AgentRef(
        agent="opencode", model="mimo-v2.5-pro"
    )
    # shorthand "agent:model" string (split on first colon only)
    r = AgentRef.model_validate("antigravity:gemini-3.5-flash")
    assert (r.agent, r.model) == ("antigravity", "gemini-3.5-flash")
    assert AgentRef.model_validate("ollama:llama3:8b").model == "llama3:8b"
    # bare agent, no model
    assert AgentRef.model_validate("claude") == AgentRef(agent="claude")
    # str() round-trips, empty ref is falsy (so `or default` backfill works)
    assert str(AgentRef(agent="claude", model="opus-4.8")) == "claude:opus-4.8"
    assert str(AgentRef(agent="claude")) == "claude"
    assert not AgentRef()


def test_task_runner_coerces_from_string():
    # plan.json / hand-edits may use either an object or the shorthand string
    t = Task(id="p1-t1", title="T", runner="opencode:mimo-v2.5-pro")
    assert t.runner.agent == "opencode" and t.runner.model == "mimo-v2.5-pro"


def test_slugify():
    assert slugify("Hello, World!") == "hello-world"
    assert slugify("  Trailing -- dashes  ") == "trailing-dashes"
    assert slugify("") == "item"
    assert len(slugify("x" * 200)) <= 40


def test_task_default_slug():
    t = Task(id="p1-t1", title="Build the Thing")
    assert t.slug == "build-the-thing"
    assert t.status == Status.pending


def test_topological_order_respects_deps():
    phase = Phase(
        id="p1", index=1, title="P",
        tasks=[
            Task(id="c", title="C", depends_on=["a", "b"]),
            Task(id="b", title="B", depends_on=["a"]),
            Task(id="a", title="A"),
        ],
    )
    order = [t.id for t in phase.topological_order()]
    assert order.index("a") < order.index("b") < order.index("c")


def test_cycle_detected():
    with pytest.raises(HarnessError):
        Plan(goal="g", phases=[Phase(id="p1", index=1, title="P", tasks=[
            Task(id="a", title="A", depends_on=["b"]),
            Task(id="b", title="B", depends_on=["a"]),
        ])])


def test_unknown_dependency_rejected():
    with pytest.raises(HarnessError):
        Plan(goal="g", phases=[Phase(id="p1", index=1, title="P", tasks=[
            Task(id="a", title="A", depends_on=["does-not-exist"]),
        ])])


def test_duplicate_ids_rejected():
    with pytest.raises(HarnessError):
        Plan(goal="g", phases=[Phase(id="p1", index=1, title="P", tasks=[
            Task(id="a", title="A"),
            Task(id="a", title="A2"),
        ])])


def test_plan_roundtrip_and_lookup():
    plan = Plan(goal="g", main_runner="m", phases=[
        Phase(id="p1", index=1, title="P", runner="pm", tasks=[
            Task(id="p1-t1", title="T", runner="tm",
                 subtasks=[Subtask(id="p1-t1-s1", description="s")]),
        ]),
    ])
    restored = Plan.model_validate_json(plan.model_dump_json())
    assert restored == plan
    found = restored.task_by_id("p1-t1")
    assert found is not None and found[1].title == "T"
    assert restored.task_by_id("nope") is None
