# llm-harness Improvement Plan

> Written: 2026-07-02  
> Scope: Fix all critical bugs, flawed mechanisms, and architectural gaps identified in the expert review.  
> Approach: Phase 1 fixes correctness bugs (ship-blockers). Phase 2 hardens the runtime. Phase 3 adds missing capabilities.

---

## Phase 1 — Critical Correctness Fixes

These are bugs that cause silent failures, data corruption, or incorrect terminal states. Fix these before anything else.

### Task 1.1 — Implement real failure state propagation

**Problem:** `Status.failed` is defined but never set. When a task fails after all retries, it is marked `Status.done`, the error string is written to `result.md`, and the run continues. A failed run and a successful run produce identical final state.

**Fix:**
- In `engine.py / _run_task`: after 3 retries produce a failure result, set `task.status = Status.failed`, save, and raise a typed exception (`TaskFailed`) rather than returning the error string.
- In `engine.py / _schedule`: catch `TaskFailed` per-task. Mark the task failed. Collect all failed task IDs for the wave. Do not add them to `results` — remove them from `remaining` into a `failed` set instead.
- Block dependents: any task whose `depends_on` includes a failed task ID should be set to `Status.skipped` (not run).
- After each wave, if `failed` is non-empty: call `po.replan()` only for tasks that are neither failed nor skipped. If the phase cannot complete (all remaining tasks are skipped), set `phase.status = Status.failed`.
- In `engine.py / run`: if any phase ends with `Status.failed`, set `plan.status = Status.failed` after the loop. Do not call `main.finalize()` on a failed run (or call a `main.fail()` variant).
- Add `Status.failed` and `Status.skipped` rendering to `cmd_status` and the dashboard.

**Files:** `engine.py`, `models.py` (no model changes needed), `insights.py`, `dashboard.py`

---

### Task 1.2 — Replace heuristic failure detection with explicit exit codes

**Problem:** Task success/failure is detected by checking if stdout starts with `"error:"` or `"exception:"`. This misfires in both directions and is completely unreliable.

**Fix:**
- `CLIProvider._run` already raises `RuntimeError` on non-zero exit code. Surface this up through `TaskAgent.execute()` → `engine._run_task`.
- Change `engine._run_task` to catch `RuntimeError` / `TimeoutError` from the provider as the definitive failure signal, not the content of the returned string.
- The retry loop should retry on provider exceptions, not on string heuristics. Remove the `startswith("error:")` check entirely.
- Keep `validate_command` as an additional post-success check (it runs only after the provider call succeeds), which can downgrade a pass to a failure.
- For the scripted provider (tests), the `ScriptedProvider.complete()` call never raises, so keep the heuristic only there (it is fine for tests since the scripted backend controls its outputs exactly).

**Files:** `engine.py`, `llm.py` (no changes needed — it already raises)

---

### Task 1.3 — Fix silent error cascade through the DAG

**Problem:** When a task in a wave fails, its result (an error string) is stored in `results` and passed to dependent tasks as upstream context. Downstream tasks receive garbage input and produce garbage output, but none are marked failed.

**Fix:** Follows directly from Task 1.2. Once `_run_task` raises on failure instead of returning an error string, the `gather` in `_schedule` will propagate the exception. Use `asyncio.gather(*coros, return_exceptions=True)` to collect results and exceptions per-task. Any task returning an exception is marked `Status.failed` and its ID is excluded from `results`. Dependents that cannot run (missing a dependency in `results`) are set to `Status.skipped`.

**Files:** `engine.py`

---

### Task 1.4 — Fix the HITL busy-poll: add timeout, orphan detection, and status visibility

**Problem:** `_wait_for_approval` polls a file every 2 seconds forever. On process restart, the waiting task re-writes the checkpoint, invalidating any previous `harness resume` call. `harness status` shows the task as `in_progress`, not waiting.

**Fix:**
- Add a `waiting` status to `Status` enum (or reuse `in_progress` with a flag — a `Status.waiting` variant is cleaner).
- In `_wait_for_approval`: set `task.status = Status.waiting`, save, then poll. On restart, `_schedule` should detect `Status.waiting` checkpoints on disk and re-enter the poll loop rather than re-running the task from scratch.
- Add a `--timeout` option to the wait (configurable in `defaults`, default `0` = infinite). If the timeout fires, set `task.status = Status.failed` and raise.
- In `harness status` output and the dashboard, show waiting tasks distinctly (e.g. `[?]` marker and "waiting for approval").
- In `Store`: add `list_waiting_checkpoints()` to enumerate all `hitl/*.json` files with `status: waiting` so `harness status` can surface them even without reading the plan.

**Files:** `engine.py`, `models.py`, `store.py`, `cli.py` (status rendering), `insights.py`

---

### Task 1.5 — Fix event log thread safety under concurrency

**Problem:** `store.record_event` appends to `run.log` without a lock. With `max_concurrency > 1`, concurrent asyncio tasks (running in the same thread via `asyncio.gather`) can interleave writes.

**Fix:**
- Add an `asyncio.Lock` to `Store.__init__`: `self._log_lock = asyncio.Lock()`.
- Make `record_event` an `async def` that acquires the lock before the file write.
- Update all callers in `engine.py` to `await store.record_event(...)`.
- Alternatively, keep `record_event` synchronous but use a threading lock (`threading.Lock`) since asyncio coroutines in a single-threaded event loop are already cooperative — they cannot truly interleave between `await` points, so the current append is safe as long as `record_event` does not `await`. Verify this is the case and add a comment.

**Recommendation:** Verify the cooperative scheduling guarantee holds (no `await` between `open()` and `write()`), add a `# safe: no await between open and write` comment, and document the assumption. Only switch to a real lock if you add async I/O (aiofiles).

**Files:** `store.py`

---

## Phase 2 — Hardening

These fix unsafe behaviors and operational blind spots that don't cause silent incorrect results but will cause real pain in production use.

### Task 2.1 — Guard MCP `harness_run` against duplicate launches

**Problem:** `harness_run` in the MCP server spawns a fully detached process with no pid tracking. Calling it twice starts two competing processes writing to the same `.harness/` directory.

**Fix:**
- On `harness_run` entry: check if a `run.pid` file exists under `.harness/runs/`. If it does, check if that pid is still alive (`os.kill(pid, 0)`). If alive, return an error: "run already in progress (pid=N); call harness_status to monitor."
- On process launch: write `proc.pid` to `.harness/runs/run.pid`.
- On run completion (inside the engine's `run()` method, in a `finally` block): delete `run.pid`.
- Add `harness_stop` MCP tool: reads `run.pid`, sends `SIGTERM`, removes the file.
- This also fixes the "process crashes immediately" case — the pid file won't be cleaned up, but `os.kill(pid, 0)` will raise `ProcessLookupError`, signaling a crash.

**Files:** `mcp.py`, `engine.py`, `store.py` (add `run_pid_path` property)

---

### Task 2.2 — Prevent concurrent agents from racing on shared files

**Problem:** With `max_concurrency > 1`, multiple file-editing CLI agents run in the same working directory simultaneously with no coordination.

**Fix (pragmatic):**
- Default `max_concurrency` stays at 1 (it already is). Do not change that default.
- Add a hard warning in `_schedule`: if `max_concurrency > 1` and the agent type is known to edit files (i.e. provider is `cli`), emit a warning to stderr at run start: "warning: max_concurrency > 1 with provider: cli — concurrent agents edit the same directory; ensure tasks touch disjoint files."
- Document the constraint clearly in `harness.config.yaml` comments and README.
- Future-proof: add an optional `working_dir` field to `Task` so tasks can be routed to different directories when concurrency is > 1.

**Files:** `engine.py`, `config.py` (comment), `README.md`

---

### Task 2.3 — Add proper logging (replace bare `print()` calls)

**Problem:** The codebase uses `print()` throughout for internal messages. There is no way to control verbosity, redirect output, or integrate with monitoring.

**Fix:**
- Add `import logging; logger = logging.getLogger("harness")` to each module that currently uses `print()`.
- Replace internal/debug prints with `logger.info/debug/warning`. Keep user-facing CLI output (`cmd_*` functions) as `print()` — those are intentional.
- In `cli.py / main()`, configure basicConfig if `--verbose` flag is passed (default level WARNING).
- This does not change external behavior but makes the tool scriptable and debuggable.

**Files:** `engine.py`, `store.py`, `llm.py`, `cli.py`

---

### Task 2.4 — Harden `extract_json` for arrays and nested fences

**Problem:** `extract_json` scans for the first `{`. If a model returns a JSON array `[{...}]`, it fails. Some models wrap in double-fenced blocks.

**Fix:**
- Before scanning for `{`, also scan for `[` and return the first-balanced array if `{` is absent or appears later.
- Add a second fence-strip pass for double-fenced blocks.
- Add tests for these cases.

**Files:** `llm.py`, `tests/test_llm.py`

---

### Task 2.5 — Prompts: inject project context and tool guidance

**Problem:** `TASK_EXEC_SYSTEM` and `PHASE_DEFINE_SYSTEM` are 3 sentences each. They provide no project context, no tool guidance, and no output format requirements. Agent behavior varies wildly by provider.

**Fix:**
- Add a `project_context` string to `Config` (optional, default empty). Users populate it with stack, conventions, and working directory notes.
- Thread `config.project_context` through `Engine` → `agents.py` method signatures → prompt construction.
- Expand `TASK_EXEC_SYSTEM` to include: what tools the agent can use, how to signal completion vs. failure, and that the working directory is the project root.
- Expand `PHASE_DEFINE_SYSTEM` to instruct the orchestrator to be explicit about expected output artifacts, not just steps.
- Keep prompts in `prompts.py` as templates with `{project_context}` interpolation.

**Files:** `prompts.py`, `config.py`, `agents.py`, `engine.py`

---

## Phase 3 — Missing Capabilities

Significant features that are absent and limit the tool's real-world utility.

### Task 3.1 — Cost and token tracking

**Problem:** No tracking of API calls, tokens, or estimated cost per run. Users have no budget visibility.

**Fix:**
- For `LiteLLMProvider`: litellm responses include `usage.prompt_tokens`, `usage.completion_tokens`, `usage.total_tokens`. Accumulate these in a `RunStats` dataclass on the provider instance.
- For `CLIProvider`: tokens are unknowable (opaque subprocess). Track call count and duration instead.
- Store stats in `run.log` as a `usage` event type after each `complete()` call.
- Expose totals in `harness status` output and dashboard.
- Add `harness_usage` MCP tool that returns the running token/cost tally.

**Files:** `llm.py`, `store.py`, `insights.py`, `mcp.py`

---

### Task 3.2 — Cross-phase task dependencies

**Problem:** Phases are strictly sequential. A task in phase 2 cannot start until every task in phase 1 is done, even if it only depends on one phase-1 task.

**Fix:**
- Add cross-phase `depends_on` syntax: `"p1-t2"` in a phase-2 task's `depends_on` list references a task from a different phase.
- Validate in `Plan._check_unique_ids`: cross-phase deps must reference earlier phases only (no forward or same-phase cross-references, which remain intra-phase).
- In `engine.py / run`: replace the simple phase-by-phase loop with a global scheduler that knows the full DAG. A phase-level wrapper task is "done" when all its tasks are done.
- This is a significant refactor. Ship phases 1 and 2 first; treat this as a v2 feature.

**Files:** `models.py`, `engine.py`, `prompts.py` (update planner schema)

---

### Task 3.3 — Streaming output from CLI agents

**Problem:** CLI agent calls buffer all output until exit. With 15-minute timeouts, users see nothing while waiting. Debugging hangs is impossible.

**Fix:**
- In `CLIProvider._run`: switch from `proc.communicate()` to streaming stdout line-by-line via `proc.stdout.readline()` in a loop. Buffer lines; emit each line to the event log as a `task_output` event (truncated to avoid log bloat — e.g. emit every 10 lines or every 5 seconds as a batch).
- The final accumulated output is returned as before.
- The dashboard and `harness watch` can render the latest `task_output` event lines as a live tail.
- PTY mode already streams but the drain runs in a thread executor — add periodic log flushes there too.

**Files:** `llm.py`, `store.py`, `insights.py`, `dashboard.py`

---

### Task 3.4 — Remove dead code and back-compat shims

**Problem:** `store.log_event()` is a back-compat shim with no callers. `Subtask` model fields are never used by the engine (subtasks appear in prompts only). `Status.failed` and `Status.skipped` are never set.

**Fix (after Phase 1 is complete):**
- Delete `store.log_event()`.
- Decide on `Subtask`: either wire it into the engine (track completion per subtask) or remove it from the model and prompts. The latter is simpler and more honest.
- `Status.failed` and `Status.skipped` will be live after Task 1.1 is done.

**Files:** `store.py`, `models.py`, `prompts.py`

---

## Execution Order

```
Phase 1 (correctness — do first, in order):
  1.2  Replace heuristic failure detection with exit codes   ← enables everything below
  1.1  Implement real failure state propagation              ← depends on 1.2
  1.3  Fix silent error cascade through DAG                 ← depends on 1.1 + 1.2
  1.4  Fix HITL busy-poll                                   ← independent
  1.5  Event log thread safety                              ← independent (verify-only likely)

Phase 2 (hardening — after Phase 1):
  2.1  Guard MCP duplicate launches
  2.2  Concurrent agent file-race warning
  2.3  Logging
  2.4  extract_json hardening
  2.5  Prompt improvements + project context

Phase 3 (new capabilities — after Phase 2):
  3.1  Cost/token tracking
  3.3  Streaming output
  3.4  Dead code removal
  3.2  Cross-phase dependencies  ← largest; ship last
```

---

## Test Coverage Targets

Each fix must ship with tests:

| Task | Test type | What to cover |
|---|---|---|
| 1.1 | Unit (ScriptedProvider) | Plan/phase/task end in `failed`; dependent tasks `skipped` |
| 1.2 | Unit | Provider exception → task failure; no heuristic string check |
| 1.3 | Unit | DAG with failing task; downstream tasks are skipped not run |
| 1.4 | Unit | Waiting status visible in state; timeout fires correctly |
| 1.5 | Verify existing | Confirm no await between open/write; add comment |
| 2.1 | Unit | Second `harness_run` call returns error when pid alive |
| 2.4 | Unit | `extract_json` on array, double-fence, prose-wrapped object |
| 2.5 | Unit | Project context string appears in rendered prompt |
| 3.1 | Unit | Token counts accumulate in run.log |

---

## Out of Scope

- GUI / web UI beyond the existing polling dashboard
- Distributed execution across machines
- Agent sandboxing / security isolation
- Built-in git integration (agents handle this themselves via their own tools)

---

## Implementation status — 2026-07-02

All tasks complete. Full suite: **67 passing**.

| Task | Status | Notes |
|---|---|---|
| 1.1 Failure state propagation | ✅ | task/phase/plan → `failed`; dependents → `skipped` |
| 1.2 Exit-code failure detection | ✅ | provider exceptions are the failure signal; heuristic removed |
| 1.3 DAG error cascade | ✅ | failed tasks excluded from results; dependents skipped |
| 1.4 HITL busy-poll | ✅ | `Status.waiting`, `hitl_timeout`, `list_waiting_checkpoints` |
| 1.5 Event-log thread safety | ✅ | now lock-guarded (needed once streaming added a worker-thread writer) |
| 2.1 MCP duplicate-launch guard | ✅ | `run.pid` + `harness_stop`; guards MCP-launched runs |
| 2.2 Concurrency file-race warning | ✅ | logged when `max_concurrency > 1` |
| 2.3 Logging | ✅ | `-v/--verbose`; internal `print`s → `logger` |
| 2.4 `extract_json` hardening | ✅ | arrays + double-fenced blocks |
| 2.5 Prompt/project context | ✅ | `Config.project_context` threaded into prompts |
| 3.1 Cost/token tracking | ✅ | `RunStats` on all providers; `usage` event; `harness_usage` |
| 3.3 Streaming output | ✅ | `on_output` wired engine→providers; `task_output` events |
| 3.4 Dead-code removal | ✅ | `Subtask`, `log_event` removed |
| 3.2 Cross-phase dependencies | ✅ | see design note below |

### Design note on 3.2

Implemented as **cross-phase data dependencies with phases kept sequential**, not
the "global DAG scheduler" originally sketched. A task may `depends_on` a task in
the same or any earlier phase; validation enforces earlier-phase-only refs,
plan-wide unique task ids, and no cycles. At run time the dependent receives the
upstream task's result as context, and a failed/skipped cross-phase dependency
skips the dependent (which marks its phase failed).

Deliberately **not** done: starting a later phase's task before its phase begins
(cross-phase *parallelism*). A global scheduler would break the context-cleaning
invariant (per-phase orchestrator created fresh and discarded after its summary;
Main only sees summaries) for little gain, since `max_concurrency` defaults to 1
for file-editing agents. If cross-phase parallelism is ever needed, it should be
designed alongside a rework of the phase-orchestrator lifecycle, not bolted on.
