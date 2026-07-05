# /roundtable

Run the roundtable multi-agent orchestration tool against the current project.

**Usage:** `/roundtable <goal>`

Drive the full roundtable workflow end-to-end using the `roundtable` CLI:

## Workflow

1. **Check for existing config** — look for `roundtable.config.yaml` in the project root.
   - If missing, run `roundtable init .` to scaffold `.roundtable/` and the default config.
   - Remind the user to set their agent/model in `roundtable.config.yaml` if it's freshly created.

2. **Generate a plan** — run `roundtable plan --goal "$ARGUMENTS"` where `$ARGUMENTS` is the
   goal text the user passed to this command.
   - Show the user the plan summary printed by the command.
   - If a plan already exists and the user didn't say to redo it, ask whether to overwrite.

3. **Review** — show the user `.roundtable/plan/PLAN.md` (use `cat` or Read) so they can
   review phases and tasks before approving.

4. **Approve** — run `roundtable approve` to mark the plan ready for execution.

5. **Run** — run `roundtable run --approve` to execute all phases with live progress.
   - The command streams inline progress and starts a web dashboard; share the dashboard URL.
   - Wait for it to complete (it prints "run complete" when done).

6. **Report** — after completion, run `roundtable status` and summarise results for the user.
   Point them to `.roundtable/docs/FINAL.md` for the main orchestrator's wrap-up.

## Notes

- All roundtable artifacts live under `.roundtable/` — the project's actual files are not touched
  except by the agents during task execution.
- If a run is interrupted (Ctrl-C), it can be resumed with `roundtable run` — completed phases
  are skipped automatically.
- To only generate and review the plan without running: stop after step 3.
- To map an existing codebase first: run `roundtable map` before `roundtable plan`, then pass
  `--prd .roundtable/docs/PRD.md` to `roundtable plan` instead of `--goal`.
