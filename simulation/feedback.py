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
    lift_hold_risk: float = 0.0
    transfer_sway_risk: float = 0.0
    placement_settle_risk: float = 0.0
    failure_bucket: str = "unknown_failure"
    dynamic_transport_mode: str = "static"


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
    signal: FeedbackSignal | None = None,
    step: float = 4.0,
    force_min: float = FORCE_MIN,
    force_max: float = FORCE_MAX,
) -> dict[str, Any]:
    out = dict(params)
    force = float(out.get("gripper_force", 25.0))
    lift_force = float(out.get("lift_force", force))
    transfer_force = float(out.get("transfer_force", force))
    transfer_alignment = float(out.get("transfer_alignment", 0.0))
    velocity = float(out.get("transport_velocity", 0.3))
    placement_velocity = float(out.get("placement_velocity", velocity))
    clearance = float(out.get("lift_clearance", 0.06))
    stage_adjustments: list[str] = []
    if suggestion == "increase":
        force = min(force_max, force + step)
        velocity = max(0.12, velocity - 0.03)
        clearance = min(0.14, clearance + 0.005)
    elif suggestion == "decrease":
        force = max(force_min, force - step)
        velocity = max(0.12, velocity - 0.02)
    placement_velocity = min(placement_velocity, velocity)
    lift_force = max(lift_force, force)
    transfer_force = max(transfer_force, force)

    if signal is not None:
        if signal.lift_hold_risk > 0.18 or signal.failure_bucket == "lift_hold_fail":
            lift_force = min(force_max, max(lift_force, force) + max(0.8, step * 0.18))
            clearance = min(0.14, clearance + 0.005)
            stage_adjustments.append("lift_stage_guard")
        if signal.transfer_sway_risk > 0.18 or signal.failure_bucket == "transfer_sway_fail":
            transfer_force = min(force_max, max(transfer_force, force) + max(0.6, step * 0.15))
            transfer_alignment = min(1.0, transfer_alignment + 0.08)
            velocity = max(0.12, velocity - 0.04)
            placement_velocity = min(placement_velocity, velocity)
            stage_adjustments.append("transfer_stage_guard")
        if signal.placement_settle_risk > 0.18 or signal.failure_bucket == "placement_settle_fail":
            placement_velocity = max(0.12, min(placement_velocity, velocity) - 0.04)
            clearance = min(0.14, clearance + 0.003)
            stage_adjustments.append("placement_stage_guard")

    out["transport_velocity"] = round(velocity, 3)
    out["placement_velocity"] = round(min(placement_velocity, velocity), 3)
    out["lift_clearance"] = round(clearance, 3)
    out["gripper_force"] = round(force, 2)
    out["lift_force"] = round(max(force, lift_force), 2)
    out["transfer_force"] = round(max(force, transfer_force), 2)
    out["transfer_alignment"] = round(transfer_alignment, 3)
    out["feedback_adjusted"] = True
    out["feedback_adjustment_type"] = suggestion
    if stage_adjustments:
        out["feedback_stage_adjustments"] = stage_adjustments
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
        lift_hold_risk=float(info.get("lift_hold_risk", 0.0)),
        transfer_sway_risk=float(info.get("transfer_sway_risk", 0.0)),
        placement_settle_risk=float(info.get("placement_settle_risk", 0.0)),
        failure_bucket=str(info.get("failure_bucket", "unknown_failure")),
        dynamic_transport_mode=str(info.get("dynamic_transport_mode", "static")),
    )
