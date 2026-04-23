"""Shared reporting helpers for simulation benchmark outputs."""

from __future__ import annotations

import json
from pathlib import Path

from .tasks import TaskConfig


def task_label(task: TaskConfig) -> str:
    return f"{task.task_id}/{task.description}"


def reference_force_center(task: TaskConfig) -> float:
    return (task.reference_force_range[0] + task.reference_force_range[1]) / 2.0


def reference_approach_height(task: TaskConfig) -> float | None:
    if task.profile is None:
        return None
    return task.profile.preferred_approach_height


def approach_height_error(task: TaskConfig, approach_height: float | None) -> float | None:
    reference = reference_approach_height(task)
    if reference is None or approach_height is None:
        return None
    return abs(approach_height - reference)


def reference_force_deviation(task: TaskConfig, gripper_force: float) -> float:
    return abs(gripper_force - reference_force_center(task))


def write_json(path: str | Path, payload) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_table(path: str | Path, lines: list[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
