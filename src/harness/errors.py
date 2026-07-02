"""Harness-specific exceptions."""

from __future__ import annotations


class HarnessError(Exception):
    """Base class for expected, user-facing harness errors."""


class TaskFailed(HarnessError):
    """A task exhausted its retries or failed validation.

    Raised inside ``_run_task`` and caught per-task in ``_schedule`` so that
    dependent tasks can be skipped and the phase can be marked failed.
    """

    def __init__(self, task_id: str, detail: str = ""):
        self.task_id = task_id
        self.detail = detail
        super().__init__(f"task {task_id!r} failed: {detail}" if detail else f"task {task_id!r} failed")
