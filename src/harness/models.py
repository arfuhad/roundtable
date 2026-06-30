"""Plan data model: Plan -> Phase -> Task -> Subtask.

The plan manifest (``plan.json``) is the source of truth for status, model
assignments, and task dependencies. These pydantic models (de)serialize it and
enforce structural invariants (unique ids, intra-phase deps, no cycles).
"""

from __future__ import annotations

import re
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .errors import HarnessError


class Status(str, Enum):
    pending = "pending"
    in_progress = "in_progress"
    done = "done"
    failed = "failed"
    skipped = "skipped"


class AgentRef(BaseModel):
    """How to run a role: which configured CLI agent, and which model it uses.

    ``agent`` names an entry in the config ``agents`` map (for ``provider: cli``);
    ``model`` is the model token passed to that agent's ``{model}`` placeholder
    (e.g. ``opus-4.8``, ``gemini-3.5-flash``). For ``provider: litellm`` there is
    no separate command, so ``model`` (falling back to ``agent``) is the litellm
    model string.

    Accepts either an object ``{agent, model}`` or a shorthand string on input:
    ``"opencode:mimo-v2.5-pro"`` -> ``agent=opencode, model=mimo-v2.5-pro``, and a
    bare ``"claude"`` -> ``agent=claude`` (no model). Always serializes as an
    object so the manifest is explicit.
    """

    model_config = ConfigDict(protected_namespaces=())

    agent: str = ""
    model: str = ""

    @model_validator(mode="before")
    @classmethod
    def _coerce(cls, v: object) -> object:
        if isinstance(v, str):
            s = v.strip()
            if ":" in s:
                a, m = s.split(":", 1)
                return {"agent": a.strip(), "model": m.strip()}
            return {"agent": s}
        return v

    def __bool__(self) -> bool:
        return bool(self.agent or self.model)

    def __str__(self) -> str:
        return f"{self.agent}:{self.model}" if self.model else self.agent


_SLUG_RE = re.compile(r"[^a-z0-9]+")


def slugify(text: str, max_len: int = 40) -> str:
    """Lowercase, hyphenated, filesystem-safe slug."""
    s = _SLUG_RE.sub("-", text.strip().lower()).strip("-")
    if len(s) > max_len:
        s = s[:max_len].rstrip("-")
    return s or "item"


class Subtask(BaseModel):
    id: str
    description: str
    status: Status = Status.pending


class Task(BaseModel):
    id: str
    title: str
    slug: str = ""
    description: str = ""
    runner: AgentRef = Field(default_factory=AgentRef)  # choosable (agent, model) for this task; backfilled if empty
    depends_on: list[str] = Field(default_factory=list)
    subtasks: list[Subtask] = Field(default_factory=list)
    status: Status = Status.pending
    result_path: str | None = None

    @model_validator(mode="after")
    def _default_slug(self) -> Task:
        if not self.slug:
            self.slug = slugify(self.title)
        return self


class Phase(BaseModel):
    id: str
    index: int = 0
    title: str
    slug: str = ""
    objective: str = ""
    runner: AgentRef = Field(default_factory=AgentRef)  # choosable (agent, model) for this phase orchestrator
    tasks: list[Task] = Field(default_factory=list)
    status: Status = Status.pending
    summary_path: str | None = None

    @model_validator(mode="after")
    def _default_slug(self) -> Phase:
        if not self.slug:
            self.slug = slugify(self.title)
        return self

    @property
    def dir_name(self) -> str:
        return f"phase-{self.index:02d}-{self.slug}"

    def task_dir_name(self, task: Task, task_index: int) -> str:
        return f"task-{task_index:02d}-{task.slug}"

    def topological_order(self) -> list[Task]:
        """Tasks ordered so dependencies precede dependents.

        Raises ValueError on unknown dep ids or cycles.
        """
        by_id = {t.id: t for t in self.tasks}
        for t in self.tasks:
            for dep in t.depends_on:
                if dep not in by_id:
                    raise HarnessError(
                        f"task {t.id!r} depends on unknown task {dep!r} in phase {self.id!r}"
                    )
        ordered: list[Task] = []
        seen: set[str] = set()
        visiting: set[str] = set()

        def visit(t: Task) -> None:
            if t.id in seen:
                return
            if t.id in visiting:
                raise HarnessError(f"dependency cycle detected at task {t.id!r}")
            visiting.add(t.id)
            for dep in t.depends_on:
                visit(by_id[dep])
            visiting.discard(t.id)
            seen.add(t.id)
            ordered.append(t)

        for t in self.tasks:
            visit(t)
        return ordered


class Plan(BaseModel):
    goal: str
    created_at: str = ""
    main_runner: AgentRef = Field(default_factory=AgentRef)  # choosable (agent, model) for the main orchestrator
    runners: dict[str, AgentRef] = Field(default_factory=dict)  # role defaults snapshot
    phases: list[Phase] = Field(default_factory=list)
    status: Status = Status.pending
    approved: bool = False

    @model_validator(mode="after")
    def _check_unique_ids(self) -> Plan:
        pids = [p.id for p in self.phases]
        if len(pids) != len(set(pids)):
            raise HarnessError("duplicate phase ids in plan")
        for p in self.phases:
            tids = [t.id for t in p.tasks]
            if len(tids) != len(set(tids)):
                raise HarnessError(f"duplicate task ids in phase {p.id!r}")
            # Validate dependency graph eagerly (raises on cycles / unknown deps).
            p.topological_order()
        return self

    def task_by_id(self, task_id: str) -> tuple[Phase, Task] | None:
        for p in self.phases:
            for t in p.tasks:
                if t.id == task_id:
                    return p, t
        return None
