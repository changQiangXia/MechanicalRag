"""Closed-loop feedback based on environment observations rather than shared ground-truth ranges."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


FORCE_MIN, FORCE_MAX = 5.0, 50.0


@dataclass
class FeedbackSignal:
    success: bool
    gripper_force: float
    distance: float
    steps: int
    transport_velocity: float = 0.3
    lift_clearance: float = 0.06
    slip_risk: float = 0.0
    compression_risk: float = 0.0
    stability_score: float = 0.0
    velocity_risk: float = 0.0
    clearance_risk: float = 0.0


def suggest_force_adjustment(signal: FeedbackSignal) -> str:
    """Return "increase", "decrease" or "none" based on observed failure modes."""
    if signal.success:
        return "none"
    if signal.clearance_risk > 0.2:
        return "increase"
    if signal.velocity_risk > 0.25 and signal.stability_score < 0.5:
        return "decrease" if signal.gripper_force > 18.0 else "increase"
    if signal.slip_risk > signal.compression_risk + 0.08:
        return "increase"
    if signal.compression_risk > signal.slip_risk + 0.08:
        return "decrease"
    if signal.distance > 0.05 and signal.stability_score < 0.45:
        return "increase" if signal.gripper_force < 25.0 else "decrease"
    return "increase" if signal.gripper_force < 20.0 else "decrease"


def adjust_params_by_feedback(
    params: dict[str, Any],
    suggestion: str,
    step: float = 4.0,
    force_min: float = FORCE_MIN,
    force_max: float = FORCE_MAX,
) -> dict[str, Any]:
    out = dict(params)
    force = float(out.get("gripper_force", 25.0))
    velocity = float(out.get("transport_velocity", 0.3))
    clearance = float(out.get("lift_clearance", 0.06))
    if suggestion == "increase":
        force = min(force_max, force + step)
        velocity = max(0.12, velocity - 0.03)
        clearance = min(0.14, clearance + 0.005)
    elif suggestion == "decrease":
        force = max(force_min, force - step)
        velocity = max(0.12, velocity - 0.02)
    out["transport_velocity"] = round(velocity, 3)
    out["lift_clearance"] = round(clearance, 3)
    out["gripper_force"] = round(force, 2)
    out["feedback_adjusted"] = True
    return out


def build_feedback_signal(
    success: bool,
    gripper_force: float,
    info: dict,
) -> FeedbackSignal:
    return FeedbackSignal(
        success=success,
        gripper_force=gripper_force,
        distance=float(info.get("distance", 0.0)),
        steps=int(info.get("steps", 0)),
        transport_velocity=float(info.get("transport_velocity", 0.3)),
        lift_clearance=float(info.get("lift_clearance", 0.06)),
        slip_risk=float(info.get("slip_risk", 0.0)),
        compression_risk=float(info.get("compression_risk", 0.0)),
        stability_score=float(info.get("stability_score", 0.0)),
        velocity_risk=float(info.get("velocity_risk", 0.0)),
        clearance_risk=float(info.get("clearance_risk", 0.0)),
    )
