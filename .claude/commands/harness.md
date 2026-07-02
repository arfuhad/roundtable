# /harness

Run the harness multi-agent orchestration tool against the current project.

**Usage:** `/harness <goal>`

Drive the full harness workflow end-to-end using the `harness` CLI:

## Workflow

1. **Check for existing config** — look for `harness.config.yaml` in the project root.
   - If missing, run `harness init .` to scaffold `.harness/` and the default config.
   - Remind the user to set their agent/model in `harness.config.yaml` if it's freshly created.

2. **Generate a plan** — run `harness plan --goal "$ARGUMENTS"` where `$ARGUMENTS` is the
   goal text the user passed to this command.
   - Show the user the plan summary printed by the command.
   - If a plan already exists and the user didn't say to redo it, ask whether to overwrite.

3. **Review** — show the user `.harness/plan/PLAN.md` (use `cat` or Read) so they can
   review phases and tasks before approving.

4. **Approve** — run `harness approve` to mark the plan ready for execution.

5. **Run** — run `harness run --approve` to execute all phases with live progress.
   - The command streams inline progress and starts a web dashboard; share the dashboard URL.
   - Wait for it to complete (it prints "run complete" when done).

6. **Report** — after completion, run `harness status` and summarise results for the user.
   Point them to `.harness/docs/FINAL.md` for the main orchestrator's wrap-up.

## Notes

- All harness artifacts live under `.harness/` — the project's actual files are not touched
  except by the agents during task execution.
- If a run is interrupted (Ctrl-C), it can be resumed with `harness run` — completed phases
  are skipped automatically.
- To only generate and review the plan without running: stop after step 3.
- To map an existing codebase first: run `harness map` before `harness plan`, then pass
  `--prd .harness/docs/PRD.md` to `harness plan` instead of `--goal`.
