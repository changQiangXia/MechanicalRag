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


def build_feedback_signal_from_observation(
    previous_params: dict[str, Any],
    observation: dict[str, Any],
) -> FeedbackSignal:
    stage = str(observation.get("stage", "unknown"))
    stage_failure_bucket = {
        "grasp": "lift_hold_fail",
        "lift": "lift_hold_fail",
        "transfer": "transfer_sway_fail",
        "place": "placement_settle_fail",
    }.get(stage, "unknown_failure")
    slip_indicator = float(observation.get("slip_indicator", 0.0))
    compression_indicator = float(observation.get("compression_indicator", 0.0))
    velocity_margin = float(observation.get("velocity_margin", 0.0))
    clearance_margin = float(observation.get("clearance_margin", 0.0))
    stage_risk = float(observation.get("risk_score", 0.0))
    failure_bucket = str(observation.get("estimated_failure_stage", "none"))
    if failure_bucket in {"none", "success"}:
        failure_bucket = stage_failure_bucket
    elif not failure_bucket.endswith("_fail"):
        failure_bucket = f"{failure_bucket}_fail"
    transport_velocity = float(previous_params.get("transport_velocity", 0.3))
    placement_velocity = float(previous_params.get("placement_velocity", transport_velocity))
    if stage == "place":
        velocity_risk = max(0.0, -velocity_margin)
        placement_settle_risk = stage_risk
    elif stage == "transfer":
        velocity_risk = max(0.0, -velocity_margin)
        placement_settle_risk = 0.0
    else:
        velocity_risk = max(0.0, -velocity_margin) * 0.5
        placement_settle_risk = 0.0
    return FeedbackSignal(
        success=False,
        gripper_force=float(previous_params.get("gripper_force", 25.0)),
        distance=float(observation.get("distance_to_target", 0.0)),
        steps=int(observation.get("observation_index", 0)),
        transport_velocity=transport_velocity,
        lift_clearance=float(previous_params.get("lift_clearance", 0.06)),
        slip_risk=slip_indicator,
        compression_risk=compression_indicator,
        stability_score=float(observation.get("stability_score", 0.0)),
        velocity_risk=velocity_risk,
        clearance_risk=max(0.0, -clearance_margin),
        lift_hold_risk=stage_risk if stage in {"grasp", "lift"} else 0.0,
        transfer_sway_risk=stage_risk if stage == "transfer" else 0.0,
        placement_settle_risk=placement_settle_risk,
        failure_bucket=failure_bucket,
        dynamic_transport_mode=str(previous_params.get("dynamic_transport_mode", "static")),
    )


def build_feedback_replan_request(
    previous_params: dict[str, Any],
    signal: FeedbackSignal,
    suggestion: str,
    step: float = 4.0,
) -> dict[str, Any]:
    stage_scores = {
        "lift": max(signal.lift_hold_risk, 0.35 if signal.failure_bucket == "lift_hold_fail" else 0.0),
        "transfer": max(signal.transfer_sway_risk, 0.35 if signal.failure_bucket == "transfer_sway_fail" else 0.0),
        "place": max(signal.placement_settle_risk, 0.35 if signal.failure_bucket == "placement_settle_fail" else 0.0),
    }
    stage_bias = max(stage_scores, key=stage_scores.get) if any(value > 0.0 for value in stage_scores.values()) else "lift"
    uncertainty_reasons: list[str] = [f"feedback_{stage_bias}_stage"]
    if signal.slip_risk > signal.compression_risk + 0.08:
        uncertainty_reasons.append("feedback_slip_risk")
    if signal.compression_risk > signal.slip_risk + 0.08:
        uncertainty_reasons.append("feedback_compression_risk")
    if signal.velocity_risk > 0.18:
        uncertainty_reasons.append("feedback_velocity_risk")
    if signal.clearance_risk > 0.18:
        uncertainty_reasons.append("feedback_clearance_risk")

    param_deltas: dict[str, float] = {}
    if suggestion == "increase":
        param_deltas["gripper_force"] = min(2.0, max(0.6, step * 0.20))
        param_deltas["transport_velocity"] = -0.02
    elif suggestion == "decrease":
        param_deltas["gripper_force"] = -min(1.5, max(0.4, step * 0.15))
        param_deltas["transport_velocity"] = -0.01

    if stage_bias == "lift":
        param_deltas["lift_force"] = max(0.8, step * 0.18)
        param_deltas["lift_clearance"] = 0.005
        param_deltas["transport_velocity"] = min(param_deltas.get("transport_velocity", 0.0), -0.02)
    elif stage_bias == "transfer":
        param_deltas["transfer_force"] = max(0.7, step * 0.16)
        param_deltas["transfer_alignment"] = 0.08
        param_deltas["transport_velocity"] = min(param_deltas.get("transport_velocity", 0.0), -0.04)
        param_deltas["placement_velocity"] = -0.02
    elif stage_bias == "place":
        param_deltas["placement_velocity"] = -0.04
        param_deltas["lift_clearance"] = 0.003

    failure_attribution = {
        "lift": round(stage_scores["lift"], 4),
        "transfer": round(stage_scores["transfer"], 4),
        "place": round(stage_scores["place"], 4),
        "slip_risk": round(signal.slip_risk, 4),
        "compression_risk": round(signal.compression_risk, 4),
        "velocity_risk": round(signal.velocity_risk, 4),
        "clearance_risk": round(signal.clearance_risk, 4),
    }
    phase_observation = {
        "phase": stage_bias,
        "contact_stability_obs": round(max(0.0, 1.0 - signal.slip_risk), 4),
        "micro_slip_obs": round(signal.slip_risk, 4),
        "payload_ratio_obs": round(1.0 + signal.lift_hold_risk, 4),
        "lift_progress_obs": round(max(0.0, 1.0 - signal.distance), 4),
        "lift_reserve_obs": round(-signal.lift_hold_risk if stage_bias == "lift" else 0.0, 4),
        "tilt_obs": round(signal.clearance_risk if stage_bias == "lift" else 0.0, 4),
        "sway_obs": round(signal.transfer_sway_risk if stage_bias == "transfer" else 0.0, 4),
        "velocity_stress_obs": round(signal.velocity_risk, 4),
        "settle_obs": round(signal.placement_settle_risk if stage_bias == "place" else 0.0, 4),
        "placement_error_obs": round(signal.distance, 4),
        "observation_confidence": 0.76,
        "trigger_reason": signal.failure_bucket,
    }
    return {
        "failure_bucket": signal.failure_bucket,
        "stage_bias": stage_bias,
        "suggestion": suggestion,
        "uncertainty_reasons": uncertainty_reasons,
        "failure_attribution": failure_attribution,
        "param_deltas": {key: round(value, 4) for key, value in param_deltas.items()},
        "phase_observation": phase_observation,
        "requested_suffix_start": stage_bias,
        "previous_solver_mode": previous_params.get("solver_mode"),
        "previous_solver_selected_candidate": previous_params.get("solver_selected_candidate"),
    }
