"""
MuJoCo 机械臂仿真环境。
参考：GRASPA 1.0、FMB、REPLAB 等论文的 tabletop manipulation 与 success rate 评测。
"""

from __future__ import annotations

import random
import time
from pathlib import Path
from typing import Any, Callable, Tuple

import numpy as np

try:
    import mujoco
    HAS_MUJOCO = True
except ImportError:
    HAS_MUJOCO = False


def _estimate_force_window(
    *,
    mass_kg: float,
    surface_friction: float,
    fragility: float,
    travel_distance: float,
    size_xyz: tuple[float, float, float],
) -> tuple[float, float, float]:
    """
    从物体属性推导独立的力学窗口。
    该窗口只由环境内部可见的物理属性决定，不直接暴露给控制器。
    """
    contact_span = float(size_xyz[0] + size_xyz[1])
    min_force_needed = (
        4.0
        + 50.0 * mass_kg
        + 20.0 * max(0.0, 0.4 - surface_friction)
        + 10.0 * travel_distance
        - 140.0 * max(0.0, contact_span - 0.04)
        + min(12.0, 10.0 * mass_kg / max(surface_friction, 0.2))
    )
    max_safe_force = (
        1.0
        + 38.0 * fragility
        + 35.0 * mass_kg
        - 18.0 * surface_friction
        + 350.0 * float(size_xyz[2])
    )
    max_safe_force = max(max_safe_force, min_force_needed + 3.0)
    nominal_force = min_force_needed + 0.42 * (max_safe_force - min_force_needed)
    return min_force_needed, max_safe_force, nominal_force


def _estimate_motion_targets(
    *,
    mass_kg: float,
    surface_friction: float,
    fragility: float,
    size_xyz: tuple[float, float, float],
) -> tuple[float, float]:
    """
    根据物体属性推导独立的推荐搬运速度和最小抬升净空。
    """
    recommended_velocity = (
        0.44
        - 0.22 * mass_kg
        - 0.10 * max(0.0, 0.35 - surface_friction)
        - 0.12 * max(0.0, 0.65 - fragility)
    )
    recommended_velocity = max(0.14, min(0.62, recommended_velocity))
    min_lift_clearance = max(0.03, 0.02 + 1.7 * float(size_xyz[2]))
    return recommended_velocity, min_lift_clearance


def _estimate_dynamic_transport_targets(
    *,
    mass_kg: float,
    surface_friction: float,
    horizontal_distance: float,
    min_lift_clearance: float,
    nominal_force: float,
    fragility: float,
    recommended_velocity: float,
    transport_velocity: float,
) -> tuple[str, float, float, float]:
    """
    为动态运输任务推导额外的阶段目标。
    当前覆盖两类场景：
    - 长距离大件搬运
    - 高速低摩擦脆性运输
    """
    if mass_kg < 0.25 or horizontal_distance < 0.28:
        dynamic_clearance_target = min_lift_clearance
        dynamic_force_target = nominal_force
        dynamic_placement_velocity_cap = recommended_velocity
    else:
        extra_distance = horizontal_distance - 0.28
        dynamic_clearance_target = (
            min_lift_clearance
            + 0.04 * extra_distance
            + 0.01 * max(0.0, mass_kg - 0.25)
        )
        dynamic_force_target = (
            nominal_force
            + 10.0 * extra_distance
            + 6.0 * max(0.0, mass_kg - 0.25)
        )
        dynamic_placement_velocity_cap = min(
            recommended_velocity,
            0.16
            - 0.03 * extra_distance
            - 0.05 * max(0.0, mass_kg - 0.25)
            + 0.02 * max(0.0, fragility - 0.65),
        )
        dynamic_placement_velocity_cap = max(0.12, dynamic_placement_velocity_cap)
        return "long_transfer", dynamic_clearance_target, dynamic_force_target, dynamic_placement_velocity_cap

    if (
        surface_friction <= 0.22
        and fragility >= 0.72
        and horizontal_distance >= 0.25
        and transport_velocity >= 0.30
    ):
        speed_pressure = max(0.0, transport_velocity - 0.30)
        low_friction_pressure = max(0.0, 0.22 - surface_friction)
        dynamic_clearance_target = (
            min_lift_clearance
            + 0.004
            + 0.05 * speed_pressure
            + 0.02 * low_friction_pressure
        )
        dynamic_force_target = (
            nominal_force
            + 0.8
            + 15.0 * low_friction_pressure
            + 4.0 * speed_pressure
            + 2.0 * max(0.0, horizontal_distance - 0.25)
        )
        dynamic_placement_velocity_cap = min(
            recommended_velocity,
            max(0.28, transport_velocity - 0.06),
        )
        return "high_speed_low_friction", dynamic_clearance_target, dynamic_force_target, dynamic_placement_velocity_cap

    return "static", min_lift_clearance, nominal_force, recommended_velocity


def _success_model(
    gripper_force: float,
    *,
    min_force_needed: float,
    max_safe_force: float,
    nominal_force: float,
    surface_friction: float = 0.5,
    mass_kg: float = 0.05,
    fragility: float = 0.7,
    travel_distance: float = 0.2,
    horizontal_distance: float = 0.2,
    approach_height: float = 0.05,
    transport_velocity: float = 0.3,
    lift_force: float | None = None,
    transfer_force: float | None = None,
    transfer_alignment: float = 0.0,
    placement_velocity: float | None = None,
    lift_clearance: float = 0.06,
    recommended_velocity: float = 0.3,
    min_lift_clearance: float = 0.05,
    preferred_approach_height: float = 0.05,
    approach_height_tolerance: float = 0.02,
    noise: float = 0.15,
    rng: random.Random | None = None,
) -> tuple[bool, dict]:
    """
    基于物体属性的独立成功模型。
    成功概率由“是否夹得住”“是否压坏/不稳定”与运动条件共同决定。
    """
    rng = rng or random
    force_band = max(3.0, (max_safe_force - min_force_needed) / 2.0)
    center_offset = abs(gripper_force - nominal_force) / force_band
    slip_risk = max(0.0, min(1.5, (min_force_needed - gripper_force) / max(min_force_needed, 1.0)))
    compression_risk = max(0.0, min(1.5, (gripper_force - max_safe_force) / max(max_safe_force, 1.0)))
    height_ratio = abs(approach_height - preferred_approach_height) / max(approach_height_tolerance, 1e-3)
    velocity_risk = max(0.0, min(1.5, (transport_velocity - recommended_velocity) / max(recommended_velocity, 1e-3)))
    clearance_risk = max(0.0, min(1.5, (min_lift_clearance - lift_clearance) / max(min_lift_clearance, 1e-3)))
    placement_velocity = transport_velocity if placement_velocity is None else min(transport_velocity, placement_velocity)
    lift_force = gripper_force if lift_force is None else max(gripper_force, lift_force)
    transfer_force = gripper_force if transfer_force is None else max(gripper_force, transfer_force)
    transfer_alignment = max(0.0, min(1.0, transfer_alignment))
    dynamic_transport_mode, dynamic_clearance_target, dynamic_force_target, dynamic_placement_velocity_cap = _estimate_dynamic_transport_targets(
        mass_kg=mass_kg,
        surface_friction=surface_friction,
        horizontal_distance=horizontal_distance,
        min_lift_clearance=min_lift_clearance,
        nominal_force=nominal_force,
        fragility=fragility,
        recommended_velocity=recommended_velocity,
        transport_velocity=transport_velocity,
    )
    dynamic_clearance_shortfall = max(0.0, dynamic_clearance_target - lift_clearance) / max(dynamic_clearance_target, 1e-3)
    dynamic_force_shortfall = max(0.0, dynamic_force_target - gripper_force) / max(dynamic_force_target, 1e-3)
    lift_stage_control_active = lift_force > gripper_force + 1e-6
    dynamic_lift_force_shortfall = (
        max(0.0, dynamic_force_target - lift_force) / max(dynamic_force_target, 1e-3)
        if lift_stage_control_active
        else dynamic_force_shortfall
    )
    dynamic_transfer_force_shortfall = max(0.0, dynamic_force_target - transfer_force) / max(dynamic_force_target, 1e-3)
    stage_decomposition_active = dynamic_transport_mode != "static"
    lift_hold_risk = min(1.5, 0.75 * slip_risk + 0.65 * compression_risk)
    transfer_sway_risk = 0.0
    placement_settle_risk = min(1.5, 0.25 * max(0.0, height_ratio - 0.25) + 0.15 * max(velocity_risk, clearance_risk))
    effective_center_offset = min(1.4, center_offset)
    lift_center_offset = center_offset
    if dynamic_transport_mode == "long_transfer":
        travel_load = max(0.0, horizontal_distance - 0.28) + 0.6 * max(0.0, mass_kg - 0.25)
        aligned_travel_load = travel_load * (1.0 - 0.45 * transfer_alignment)
        if lift_stage_control_active:
            lift_center_offset = abs(lift_force - dynamic_force_target) / force_band
            effective_center_offset = min(1.4, lift_center_offset)
        lift_hold_risk = min(
            1.5,
            0.95 * slip_risk
            + 0.35 * compression_risk
            + 0.35 * dynamic_lift_force_shortfall
            + 0.20 * clearance_risk
            + 0.08 * effective_center_offset
            + 0.08 * aligned_travel_load,
        )
        transport_scale = max(0.85, min(1.25, transport_velocity / max(0.16, 0.75 * recommended_velocity)))
        transport_scale = max(0.78, transport_scale - 0.08 * transfer_alignment)
        transfer_sway_risk = min(
            1.5,
            (
                0.85 * dynamic_clearance_shortfall
                + 0.70 * dynamic_transfer_force_shortfall
                + 0.25 * aligned_travel_load
                + 0.15 * velocity_risk
            )
            * transport_scale,
        )
        settle_velocity_pressure = max(0.0, placement_velocity - dynamic_placement_velocity_cap) / max(dynamic_placement_velocity_cap, 1e-3)
        settle_height_penalty = max(0.0, height_ratio - 0.25)
        settle_clearance_excess = max(0.0, lift_clearance - dynamic_clearance_target) / max(dynamic_clearance_target, 1e-3)
        placement_transfer_carryover = transfer_sway_risk
        placement_clearance_excess = settle_clearance_excess
        placement_load = aligned_travel_load
        placement_settle_risk = min(
            1.5,
            0.45 * placement_transfer_carryover
            + 0.35 * settle_velocity_pressure
            + 0.15 * settle_height_penalty
            + 0.10 * placement_clearance_excess
            + 0.10 * placement_load,
        )
        base_success = 0.90 - rng.uniform(0, noise)
        base_success += max(-0.04, min(0.05, (surface_friction - 0.4) * 0.12))
        base_success -= max(0.0, (mass_kg - 0.25) * 0.10)
        base_success -= max(0.0, travel_distance - 0.18) * 0.22
        base_success -= 0.12 * effective_center_offset
        lift_penalty = 0.30 * slip_risk + 0.24 * compression_risk + 0.14 * lift_hold_risk
        transfer_penalty = 0.22 * velocity_risk + 0.18 * clearance_risk + 0.26 * transfer_sway_risk
        placement_penalty = min(0.18, settle_height_penalty * 0.08) + 0.16 * placement_settle_risk
    elif dynamic_transport_mode == "high_speed_low_friction":
        speed_pressure = max(0.0, transport_velocity - max(0.26, min(recommended_velocity, 0.30))) / max(recommended_velocity, 1e-3)
        low_friction_pressure = max(0.0, 0.22 - surface_friction) / 0.22
        travel_load = (
            max(0.0, horizontal_distance - 0.25)
            + 0.30 * speed_pressure
            + 0.20 * low_friction_pressure
        )
        lift_hold_risk = min(
            1.5,
            0.80 * slip_risk
            + 0.55 * compression_risk
            + 0.25 * dynamic_force_shortfall
            + 0.12 * min(1.4, center_offset),
        )
        transport_scale = max(0.90, min(1.20, transport_velocity / max(0.24, recommended_velocity - 0.06)))
        transfer_sway_risk = min(
            1.5,
            (
                0.65 * dynamic_transfer_force_shortfall
                + 0.22 * travel_load
                + 0.20 * speed_pressure
                + 0.18 * dynamic_clearance_shortfall
            )
            * transport_scale,
        )
        settle_velocity_pressure = max(0.0, placement_velocity - dynamic_placement_velocity_cap) / max(dynamic_placement_velocity_cap, 1e-3)
        settle_height_penalty = max(0.0, height_ratio - 0.25)
        settle_clearance_excess = max(0.0, lift_clearance - dynamic_clearance_target) / max(dynamic_clearance_target, 1e-3)
        placement_settle_risk = min(
            1.5,
            0.35 * transfer_sway_risk
            + 0.40 * settle_velocity_pressure
            + 0.12 * settle_height_penalty
            + 0.08 * settle_clearance_excess
            + 0.08 * travel_load,
        )
        base_success = 0.91 - rng.uniform(0, noise)
        base_success += max(-0.04, min(0.05, (surface_friction - 0.4) * 0.12))
        base_success -= max(0.0, travel_distance - 0.18) * 0.18
        base_success -= 0.10 * effective_center_offset
        lift_penalty = 0.28 * slip_risk + 0.24 * compression_risk + 0.12 * lift_hold_risk
        transfer_penalty = 0.16 * velocity_risk + 0.12 * clearance_risk + 0.26 * transfer_sway_risk
        placement_penalty = min(0.18, settle_height_penalty * 0.08) + 0.18 * placement_settle_risk
    else:
        settle_height_penalty = max(0.0, height_ratio - 0.25)
        base_success = 0.90 - rng.uniform(0, noise)
        base_success += max(-0.04, min(0.05, (surface_friction - 0.4) * 0.12))
        base_success -= max(0.0, (mass_kg - 0.25) * 0.10)
        base_success -= max(0.0, travel_distance - 0.18) * 0.22
        base_success -= 0.16 * effective_center_offset
        lift_penalty = 0.42 * slip_risk + 0.52 * compression_risk
        transfer_penalty = 0.20 * velocity_risk + 0.18 * clearance_risk
        placement_penalty = min(0.18, settle_height_penalty * 0.08)
    p_success = max(0.1, min(0.95, base_success - lift_penalty - transfer_penalty - placement_penalty))
    fail_prob = 1.0 - p_success
    failure_weights = {
        "lift_hold_fail": max(1e-6, lift_penalty + 0.18 * lift_hold_risk + 0.10 * compression_risk),
        "transfer_sway_fail": max(
            1e-6,
            transfer_penalty + 0.30 * transfer_sway_risk + 0.12 * velocity_risk + 0.10 * clearance_risk,
        ),
        "placement_settle_fail": max(
            1e-6,
            placement_penalty + 0.25 * placement_settle_risk + 0.10 * settle_height_penalty,
        ),
    }
    failure_weight_sum = sum(failure_weights.values())
    unconditional_fail_probs = {
        bucket: fail_prob * weight / failure_weight_sum
        for bucket, weight in failure_weights.items()
    }
    lift_hold_success_prob = max(0.05, min(0.999, 1.0 - unconditional_fail_probs["lift_hold_fail"]))
    transfer_success_prob = max(
        0.05,
        min(
            0.999,
            1.0 - unconditional_fail_probs["transfer_sway_fail"] / max(lift_hold_success_prob, 1e-6),
        ),
    )
    placement_success_prob = max(
        0.05,
        min(
            0.999,
            1.0
            - unconditional_fail_probs["placement_settle_fail"]
            / max(lift_hold_success_prob * transfer_success_prob, 1e-6),
        ),
    )
    dominant_failure_mode = max(failure_weights, key=failure_weights.get)
    if rng.random() > lift_hold_success_prob:
        success = False
        failure_bucket = "lift_hold_fail"
    elif rng.random() > transfer_success_prob:
        success = False
        failure_bucket = "transfer_sway_fail"
    elif rng.random() > placement_success_prob:
        success = False
        failure_bucket = "placement_settle_fail"
    else:
        success = True
        failure_bucket = "success"
    stability_score = max(
        0.0,
        1.0
        - min(
            1.2,
            effective_center_offset * 0.35
            + slip_risk * 0.45
            + compression_risk * 0.60
            + lift_hold_risk * 0.35
            + velocity_risk * 0.25
            + clearance_risk * 0.25
            + transfer_sway_risk * 0.50
            + placement_settle_risk * 0.35,
        ),
    )
    return success, {
        "slip_risk": round(slip_risk, 4),
        "compression_risk": round(compression_risk, 4),
        "velocity_risk": round(velocity_risk, 4),
        "clearance_risk": round(clearance_risk, 4),
        "lift_hold_risk": round(lift_hold_risk, 4),
        "transfer_sway_risk": round(transfer_sway_risk, 4),
        "placement_settle_risk": round(placement_settle_risk, 4),
        "lift_hold_success_prob": round(lift_hold_success_prob, 4),
        "transfer_success_prob": round(transfer_success_prob, 4),
        "placement_success_prob": round(placement_success_prob, 4),
        "dominant_failure_mode": dominant_failure_mode,
        "failure_bucket": failure_bucket,
        "stability_score": round(stability_score, 4),
        "nominal_force": round(nominal_force, 4),
        "min_force_needed": round(min_force_needed, 4),
        "max_safe_force": round(max_safe_force, 4),
        "recommended_velocity": round(recommended_velocity, 4),
        "lift_force": round(lift_force, 4),
        "transfer_force": round(transfer_force, 4),
        "transfer_alignment": round(transfer_alignment, 4),
        "placement_velocity": round(placement_velocity, 4),
        "min_lift_clearance": round(min_lift_clearance, 4),
        "lift_stage_control_active": lift_stage_control_active,
        "lift_center_offset": round(lift_center_offset, 4),
        "dynamic_transport_mode": dynamic_transport_mode,
        "dynamic_clearance_target": round(dynamic_clearance_target, 4),
        "dynamic_force_target": round(dynamic_force_target, 4),
        "dynamic_lift_force_shortfall": round(dynamic_lift_force_shortfall, 4),
        "dynamic_placement_velocity_cap": round(dynamic_placement_velocity_cap, 4),
    }


def _scene_path() -> Path:
    return Path(__file__).resolve().parent / "scene.xml"


def _compute_transport_dynamics(
    mass_kg: float,
    surface_friction: float,
    fragility: float,
    velocity_scale: float,
    transport_velocity: float,
) -> Tuple[float, float, float]:
    """
    估计平面运输阶段的控制强度。
    通过静摩擦补偿避免重物体和低速任务在固定步数内天然不可达。
    """
    static_push_estimate = mass_kg * 9.81 * (0.18 + surface_friction)
    velocity_factor = max(0.4, min(1.6, transport_velocity / max(0.15, 0.32 * velocity_scale)))
    force_gain = (48.0 * velocity_scale + 24.0 * static_push_estimate) * velocity_factor
    force_clip = 9.0 + 10.0 * velocity_scale + 8.5 * static_push_estimate + 3.5 * (1.0 - fragility) + 3.0 * velocity_factor
    return force_gain, force_clip, static_push_estimate


def _default_object_profile() -> dict[str, Any]:
    return {
        "mass_kg": 0.05,
        "surface_friction": 0.5,
        "fragility": 0.7,
        "velocity_scale": 0.8,
        "target_tolerance": 0.04,
        "size_xyz": (0.025, 0.025, 0.025),
        "preferred_approach_height": 0.05,
        "approach_height_tolerance": 0.02,
    }


def _normalize_execution_params(params: dict[str, Any]) -> dict[str, float]:
    transport_velocity = max(0.12, min(0.8, float(params.get("transport_velocity", 0.30))))
    placement_velocity = max(
        0.12,
        min(transport_velocity, float(params.get("placement_velocity", transport_velocity))),
    )
    gripper_force = max(5.0, min(50.0, float(params.get("gripper_force", 25.0))))
    lift_force = max(gripper_force, min(50.0, float(params.get("lift_force", gripper_force))))
    transfer_force = max(gripper_force, min(50.0, float(params.get("transfer_force", gripper_force))))
    return {
        "gripper_force": round(gripper_force, 4),
        "approach_height": round(max(0.02, min(0.08, float(params.get("approach_height", 0.05)))), 4),
        "transport_velocity": round(transport_velocity, 4),
        "lift_force": round(lift_force, 4),
        "transfer_force": round(transfer_force, 4),
        "placement_velocity": round(placement_velocity, 4),
        "transfer_alignment": round(max(0.0, min(1.0, float(params.get("transfer_alignment", 0.0)))), 4),
        "lift_clearance": round(max(0.03, min(0.14, float(params.get("lift_clearance", 0.06)))), 4),
    }


def _evaluate_execution_plan(
    *,
    object_pos: Tuple[float, float, float],
    target_pos: Tuple[float, float, float],
    params: dict[str, Any],
    object_profile: dict[str, Any] | None = None,
    rng: random.Random | None = None,
) -> dict[str, Any]:
    profile = _default_object_profile()
    if object_profile:
        profile.update(object_profile)
    normalized = _normalize_execution_params(params)
    horizontal_distance = sum((a - b) ** 2 for a, b in zip(object_pos, target_pos)) ** 0.5
    travel_distance = (
        horizontal_distance
        + 2.0 * normalized["approach_height"]
        + 2.0 * normalized["lift_clearance"]
    )
    recommended_velocity, min_lift_clearance = _estimate_motion_targets(
        mass_kg=float(profile["mass_kg"]),
        surface_friction=float(profile["surface_friction"]),
        fragility=float(profile["fragility"]),
        size_xyz=tuple(profile["size_xyz"]),
    )
    min_force_needed, max_safe_force, nominal_force = _estimate_force_window(
        mass_kg=float(profile["mass_kg"]),
        surface_friction=float(profile["surface_friction"]),
        fragility=float(profile["fragility"]),
        travel_distance=travel_distance,
        size_xyz=tuple(profile["size_xyz"]),
    )
    success, diag = _success_model(
        normalized["gripper_force"],
        min_force_needed=min_force_needed,
        max_safe_force=max_safe_force,
        nominal_force=nominal_force,
        surface_friction=float(profile["surface_friction"]),
        mass_kg=float(profile["mass_kg"]),
        fragility=float(profile["fragility"]),
        travel_distance=travel_distance,
        horizontal_distance=horizontal_distance,
        approach_height=normalized["approach_height"],
        transport_velocity=normalized["transport_velocity"],
        lift_force=normalized["lift_force"],
        transfer_force=normalized["transfer_force"],
        transfer_alignment=normalized["transfer_alignment"],
        placement_velocity=normalized["placement_velocity"],
        lift_clearance=normalized["lift_clearance"],
        recommended_velocity=recommended_velocity,
        min_lift_clearance=min_lift_clearance,
        preferred_approach_height=float(profile["preferred_approach_height"]),
        approach_height_tolerance=float(profile["approach_height_tolerance"]),
        rng=rng,
    )
    lift_speed = max(0.05, 0.22 * float(profile["velocity_scale"]) / (1.0 + float(profile["mass_kg"]) * 1.8))
    transport_speed = max(0.05, normalized["transport_velocity"])
    placement_speed = max(0.05, normalized["placement_velocity"])
    elapsed = (
        (2.0 * normalized["approach_height"] / lift_speed)
        + (2.0 * normalized["lift_clearance"] / lift_speed)
        + (horizontal_distance / transport_speed)
        + (0.35 * normalized["lift_clearance"] / placement_speed)
    )
    distance = 0.0 if success else 0.03 + 0.04 * max(diag["slip_risk"], diag["compression_risk"])
    info = {
        "distance": round(distance, 4),
        "steps": 12,
        "sim_time": round(elapsed, 4),
        "wall_time": round(elapsed, 4),
        "travel_distance": round(travel_distance, 4),
        "horizontal_distance": round(horizontal_distance, 4),
        "transport_velocity": round(normalized["transport_velocity"], 4),
        "lift_force": round(normalized["lift_force"], 4),
        "transfer_force": round(normalized["transfer_force"], 4),
        "transfer_alignment": round(normalized["transfer_alignment"], 4),
        "placement_velocity": round(normalized["placement_velocity"], 4),
        "lift_clearance": round(normalized["lift_clearance"], 4),
        "force_gain": 0.0,
        "force_clip": 0.0,
        "static_push_estimate": 0.0,
        "approach_height_error": round(
            abs(normalized["approach_height"] - float(profile["preferred_approach_height"])),
            4,
        ),
        "physics_ok": True,
        "grip_success": success,
        "failure_bucket": diag["failure_bucket"],
        "dominant_failure_mode": diag["dominant_failure_mode"],
        "slip_risk": diag["slip_risk"],
        "compression_risk": diag["compression_risk"],
        "velocity_risk": diag["velocity_risk"],
        "clearance_risk": diag["clearance_risk"],
        "lift_hold_risk": diag["lift_hold_risk"],
        "transfer_sway_risk": diag["transfer_sway_risk"],
        "placement_settle_risk": diag["placement_settle_risk"],
        "stability_score": diag["stability_score"],
        "nominal_force": diag["nominal_force"],
        "min_force_needed": diag["min_force_needed"],
        "max_safe_force": diag["max_safe_force"],
        "recommended_velocity": diag["recommended_velocity"],
        "dynamic_transport_mode": diag["dynamic_transport_mode"],
        "dynamic_placement_velocity_cap": diag["dynamic_placement_velocity_cap"],
        "min_lift_clearance": diag["min_lift_clearance"],
        "dynamic_clearance_target": diag["dynamic_clearance_target"],
        "dynamic_force_target": diag["dynamic_force_target"],
        "dynamic_lift_force_shortfall": diag["dynamic_lift_force_shortfall"],
        "lift_center_offset": diag["lift_center_offset"],
        "lift_stage_control_active": diag["lift_stage_control_active"],
        "placement_stage_control_active": diag.get("placement_stage_control_active", False),
        "placement_transfer_carryover": diag.get("placement_transfer_carryover", 0.0),
        "placement_clearance_excess": diag.get("placement_clearance_excess", 0.0),
        "profile": profile,
    }
    return {
        "success": success,
        "elapsed": elapsed,
        "info": info,
        "diag": diag,
        "profile": profile,
        "params": normalized,
        "horizontal_distance": horizontal_distance,
        "min_lift_clearance": min_lift_clearance,
    }


def _build_observer_trace(
    evaluation: dict[str, Any],
    *,
    observation_start_index: int = 0,
) -> list[dict[str, Any]]:
    params = evaluation["params"]
    diag = evaluation["diag"]
    info = evaluation["info"]
    recommended_velocity = float(diag["recommended_velocity"])
    dynamic_placement_velocity_cap = float(diag["dynamic_placement_velocity_cap"])
    dynamic_clearance_target = float(diag["dynamic_clearance_target"])
    horizontal_distance = float(evaluation["horizontal_distance"])
    final_distance = float(info["distance"])
    estimated_failure_stage = str(diag["failure_bucket"]).replace("_fail", "")
    if estimated_failure_stage == "success":
        estimated_failure_stage = "none"

    stage_rows = [
        {
            "stage": "approach",
            "stage_progress": 0.15,
            "distance_to_target": horizontal_distance + params["approach_height"] + params["lift_clearance"],
            "stability_score": max(0.0, min(1.0, info["stability_score"] - 0.05 * info["approach_height_error"])),
            "slip_indicator": round(max(0.0, 0.35 * float(info["slip_risk"])), 4),
            "compression_indicator": round(max(0.0, 0.35 * float(info["compression_risk"])), 4),
            "velocity_margin": round(recommended_velocity - params["transport_velocity"], 4),
            "clearance_margin": round(params["lift_clearance"] - dynamic_clearance_target, 4),
            "risk_score": round(max(0.0, info["approach_height_error"]), 4),
        },
        {
            "stage": "grasp",
            "stage_progress": 0.35,
            "distance_to_target": horizontal_distance + params["lift_clearance"],
            "stability_score": max(0.0, min(1.0, info["stability_score"] - 0.10 * float(info["slip_risk"]))),
            "slip_indicator": round(float(info["slip_risk"]), 4),
            "compression_indicator": round(float(info["compression_risk"]), 4),
            "velocity_margin": round(recommended_velocity - params["transport_velocity"], 4),
            "clearance_margin": round(params["lift_clearance"] - float(info["min_lift_clearance"]), 4),
            "risk_score": round(max(float(info["slip_risk"]), float(info["compression_risk"])), 4),
        },
        {
            "stage": "lift",
            "stage_progress": 0.55,
            "distance_to_target": horizontal_distance * 0.7,
            "stability_score": max(0.0, min(1.0, info["stability_score"] - 0.12 * float(info["lift_hold_risk"]))),
            "slip_indicator": round(max(float(info["slip_risk"]), float(info["lift_hold_risk"])), 4),
            "compression_indicator": round(float(info["compression_risk"]), 4),
            "velocity_margin": round(recommended_velocity - params["transport_velocity"], 4),
            "clearance_margin": round(params["lift_clearance"] - dynamic_clearance_target, 4),
            "risk_score": round(max(float(info["lift_hold_risk"]), max(0.0, dynamic_clearance_target - params["lift_clearance"])), 4),
        },
        {
            "stage": "transfer",
            "stage_progress": 0.78,
            "distance_to_target": horizontal_distance * 0.35,
            "stability_score": max(0.0, min(1.0, info["stability_score"] - 0.15 * float(info["transfer_sway_risk"]))),
            "slip_indicator": round(float(info["slip_risk"]), 4),
            "compression_indicator": round(float(info["compression_risk"]), 4),
            "velocity_margin": round(dynamic_placement_velocity_cap - params["transport_velocity"], 4),
            "clearance_margin": round(params["lift_clearance"] - dynamic_clearance_target, 4),
            "risk_score": round(max(float(info["transfer_sway_risk"]), float(info["velocity_risk"])), 4),
        },
        {
            "stage": "place",
            "stage_progress": 1.0,
            "distance_to_target": final_distance,
            "stability_score": float(info["stability_score"]),
            "slip_indicator": round(float(info["slip_risk"]) * 0.5, 4),
            "compression_indicator": round(float(info["compression_risk"]) * 0.5, 4),
            "velocity_margin": round(dynamic_placement_velocity_cap - params["placement_velocity"], 4),
            "clearance_margin": round(params["lift_clearance"] - dynamic_clearance_target, 4),
            "risk_score": round(float(info["placement_settle_risk"]), 4),
        },
    ]
    trace: list[dict[str, Any]] = []
    for offset, row in enumerate(stage_rows):
        trigger_reason = ""
        if row["stage"] == "grasp":
            if row["slip_indicator"] > 0.12:
                trigger_reason = "slip_indicator"
            elif row["compression_indicator"] > 0.12:
                trigger_reason = "compression_indicator"
        elif row["stage"] == "lift":
            if row["clearance_margin"] < -0.002:
                trigger_reason = "clearance_margin"
            elif row["risk_score"] > 0.18:
                trigger_reason = "lift_hold_risk"
        elif row["stage"] == "transfer":
            if row["velocity_margin"] < -0.01:
                trigger_reason = "velocity_margin"
            elif row["risk_score"] > 0.18:
                trigger_reason = "transfer_sway_risk"
        elif row["stage"] == "place" and row["risk_score"] > 0.18:
            trigger_reason = "placement_settle_risk"
        trace.append(
            {
                "observation_index": observation_start_index + offset,
                **row,
                "estimated_failure_stage": estimated_failure_stage,
                "trigger_reason": trigger_reason,
            }
        )
    return trace


def simulate_stepwise_execution(
    *,
    object_pos: Tuple[float, float, float],
    target_pos: Tuple[float, float, float],
    params: dict[str, Any],
    object_profile: dict[str, Any] | None = None,
    step_replan_callback: Callable[[dict[str, Any], dict[str, Any]], dict[str, Any] | None] | None = None,
    max_step_replans: int = 0,
    rng: random.Random | None = None,
) -> tuple[bool, float, dict[str, Any]]:
    current_feedback_state = dict(params)
    current_params = _normalize_execution_params(current_feedback_state)
    observer_trace: list[dict[str, Any]] = []
    step_replan_trace: list[dict[str, Any]] = []
    observation_index = 0
    step_replan_count = 0
    last_evaluation: dict[str, Any] | None = None

    while True:
        evaluation = _evaluate_execution_plan(
            object_pos=object_pos,
            target_pos=target_pos,
            params=current_params,
            object_profile=object_profile,
            rng=rng,
        )
        last_evaluation = evaluation
        iteration_trace = _build_observer_trace(
            evaluation,
            observation_start_index=observation_index,
        )
        observer_trace.extend(iteration_trace)
        observation_index += len(iteration_trace)
        if step_replan_callback is None or step_replan_count >= max_step_replans:
            break

        updated_params = None
        trigger_snapshot = None
        for snapshot in iteration_trace:
            if not snapshot["trigger_reason"]:
                continue
            candidate = step_replan_callback(dict(snapshot), dict(current_feedback_state))
            if candidate is None:
                continue
            normalized_candidate = _normalize_execution_params(candidate)
            if normalized_candidate == current_params:
                continue
            updated_params = normalized_candidate
            trigger_snapshot = snapshot
            break
        if updated_params is None or trigger_snapshot is None:
            break
        step_replan_trace.append(
            {
                "observation_index": trigger_snapshot["observation_index"],
                "stage": trigger_snapshot["stage"],
                "trigger_reason": trigger_snapshot["trigger_reason"],
                "seed_plan": dict(current_params),
                "final_plan": dict(updated_params),
            }
        )
        current_feedback_state = dict(candidate)
        current_params = updated_params
        step_replan_count += 1

    assert last_evaluation is not None
    info = dict(last_evaluation["info"])
    info["observer_trace"] = observer_trace
    info["step_replan_trace"] = step_replan_trace
    info["step_replan_count"] = step_replan_count
    info["execution_feedback_mode"] = "step_observer_replan" if step_replan_count > 0 else "observer_only"
    current_feedback_state.update(last_evaluation["params"])
    info["applied_params"] = dict(current_feedback_state)
    return last_evaluation["success"], last_evaluation["elapsed"], info


class ArmSimEnv:
    """
    桌面抓取仿真环境（MuJoCo）。
    结合物理仿真与基于知识的成功模型，评估 RAG 驱动的参数选择。
    """

    def __init__(self, gui: bool = False, time_step: float = 1.0 / 120.0, seed: int | None = None):
        if not HAS_MUJOCO:
            raise ImportError("请安装 mujoco: pip install mujoco")
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        self.gui = gui
        self.time_step = time_step
        path = _scene_path()
        if not path.exists():
            raise FileNotFoundError(f"场景文件不存在: {path}")
        self.model = mujoco.MjModel.from_xml_path(str(path))
        self.model.opt.timestep = time_step
        self.data = mujoco.MjData(self.model)
        self._object_body_id = self.model.body("object").id
        self._object_geom_id = self.model.geom("box_geom").id
        # 自由关节的 qpos 布局: 前 3 个为位置，后 4 个为四元数 (w,x,y,z)
        self._qpos_pos_slice = slice(0, 3)
        self._qpos_quat_slice = slice(3, 7)
        self._viewer = None
        if gui:
            try:
                self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
            except Exception:
                self._viewer = None

    def _set_object_pose(self, pos: Tuple[float, float, float]) -> None:
        self.data.qpos[self._qpos_pos_slice] = np.array(pos, dtype=np.float64)
        self.data.qpos[self._qpos_quat_slice] = np.array([1, 0, 0, 0], dtype=np.float64)
        self.data.qvel[:] = 0
        mujoco.mj_forward(self.model, self.data)

    def _get_object_pos(self) -> np.ndarray:
        return self.data.qpos[self._qpos_pos_slice].copy()

    def _configure_object_profile(self, object_profile: dict | None) -> dict:
        profile = {
            "mass_kg": 0.05,
            "surface_friction": 0.5,
            "fragility": 0.7,
            "velocity_scale": 0.8,
            "target_tolerance": 0.04,
            "size_xyz": (0.025, 0.025, 0.025),
            "preferred_approach_height": 0.05,
            "approach_height_tolerance": 0.02,
        }
        if object_profile:
            profile.update(object_profile)
        self.model.body_mass[self._object_body_id] = float(profile["mass_kg"])
        self.model.geom_friction[self._object_geom_id] = np.array(
            [
                float(profile["surface_friction"]),
                max(0.03, float(profile["surface_friction"]) * 0.35),
                0.02,
            ],
            dtype=np.float64,
        )
        self.model.geom_size[self._object_geom_id] = np.array(profile["size_xyz"], dtype=np.float64)
        mujoco.mj_forward(self.model, self.data)
        return profile

    def execute_pick_place(
        self,
        object_pos: Tuple[float, float, float],
        target_pos: Tuple[float, float, float],
        gripper_force: float,
        approach_height: float = 0.05,
        transport_velocity: float = 0.30,
        lift_force: float | None = None,
        transfer_force: float | None = None,
        transfer_alignment: float = 0.0,
        placement_velocity: float | None = None,
        lift_clearance: float = 0.06,
        object_profile: dict | None = None,
        max_steps: int = 800,
        step_replan_callback: Callable[[dict[str, Any], dict[str, Any]], dict[str, Any] | None] | None = None,
        max_step_replans: int = 0,
    ) -> Tuple[bool, float, dict]:
        """
        执行 pick-and-place，成功与否由物体属性、执行轨迹和给定控制参数共同决定。
        参考 FMB/REPLAB：success rate、completion time。
        """
        profile = self._configure_object_profile(object_profile)
        self._set_object_pose(object_pos)
        self.data.xfrc_applied[:] = 0
        start = time.time()
        step = 0
        horizontal_distance = float(np.linalg.norm(np.array(target_pos) - np.array(object_pos)))
        staged_distance = 2.0 * approach_height + 2.0 * lift_clearance + horizontal_distance
        mass_kg = float(profile["mass_kg"])
        surface_friction = float(profile["surface_friction"])
        fragility = float(profile["fragility"])
        velocity_scale = float(profile["velocity_scale"])
        target_tolerance = float(profile["target_tolerance"])
        preferred_approach_height = float(profile["preferred_approach_height"])
        approach_height_tolerance = float(profile["approach_height_tolerance"])
        size_xyz = tuple(float(v) for v in profile["size_xyz"])
        recommended_velocity, min_lift_clearance = _estimate_motion_targets(
            mass_kg=mass_kg,
            surface_friction=surface_friction,
            fragility=fragility,
            size_xyz=size_xyz,
        )
        placement_velocity = transport_velocity if placement_velocity is None else min(transport_velocity, placement_velocity)
        lift_force = gripper_force if lift_force is None else max(gripper_force, lift_force)
        transfer_force = gripper_force if transfer_force is None else max(gripper_force, transfer_force)
        transfer_alignment = max(0.0, min(1.0, transfer_alignment))
        adaptive_trace_info = None
        if step_replan_callback is not None or max_step_replans > 0:
            _, _, adaptive_trace_info = simulate_stepwise_execution(
                object_pos=object_pos,
                target_pos=target_pos,
                params={
                    "gripper_force": gripper_force,
                    "approach_height": approach_height,
                    "transport_velocity": transport_velocity,
                    "lift_force": lift_force,
                    "transfer_force": transfer_force,
                    "placement_velocity": placement_velocity,
                    "transfer_alignment": transfer_alignment,
                    "lift_clearance": lift_clearance,
                },
                object_profile=profile,
                step_replan_callback=step_replan_callback,
                max_step_replans=max_step_replans,
            )
            applied_params = adaptive_trace_info.get("applied_params", {})
            gripper_force = float(applied_params.get("gripper_force", gripper_force))
            approach_height = float(applied_params.get("approach_height", approach_height))
            transport_velocity = float(applied_params.get("transport_velocity", transport_velocity))
            lift_force = float(applied_params.get("lift_force", lift_force))
            transfer_force = float(applied_params.get("transfer_force", transfer_force))
            placement_velocity = float(applied_params.get("placement_velocity", placement_velocity))
            transfer_alignment = float(applied_params.get("transfer_alignment", transfer_alignment))
            lift_clearance = float(applied_params.get("lift_clearance", lift_clearance))
        staged_distance = 2.0 * approach_height + 2.0 * lift_clearance + horizontal_distance

        # 静摩擦补偿让大件、薄壁件和重金属件都保留难度差异，同时具备完成条件。
        force_gain, force_clip, static_push_estimate = _compute_transport_dynamics(
            mass_kg,
            surface_friction,
            fragility,
            velocity_scale,
            transport_velocity,
        )

        for step in range(max_steps):
            pos = self._get_object_pos()
            tgt = np.array(target_pos, dtype=np.float64)
            d = np.linalg.norm(tgt - pos)
            if d < target_tolerance:
                break
            force = (tgt - pos) * force_gain
            force = np.clip(force, -force_clip, force_clip)
            self.data.xfrc_applied[self._object_body_id, :3] = force
            mujoco.mj_step(self.model, self.data)
            self.data.xfrc_applied[self._object_body_id, :3] = 0
            if self._viewer is not None and self._viewer.is_running():
                self._viewer.sync()

        wall_elapsed = time.time() - start
        pos = self._get_object_pos()
        dist = float(np.linalg.norm(pos - np.array(target_pos)))
        physics_ok = dist < target_tolerance + 0.02
        min_force_needed, max_safe_force, nominal_force = _estimate_force_window(
            mass_kg=mass_kg,
            surface_friction=surface_friction,
            fragility=fragility,
            travel_distance=staged_distance,
            size_xyz=size_xyz,
        )
        grip_success, grip_diag = _success_model(
            gripper_force,
            min_force_needed=min_force_needed,
            max_safe_force=max_safe_force,
            nominal_force=nominal_force,
            surface_friction=surface_friction,
            mass_kg=mass_kg,
            fragility=fragility,
            travel_distance=staged_distance,
            horizontal_distance=horizontal_distance,
            approach_height=approach_height,
            transport_velocity=transport_velocity,
            lift_force=lift_force,
            transfer_force=transfer_force,
            transfer_alignment=transfer_alignment,
            placement_velocity=placement_velocity,
            lift_clearance=lift_clearance,
            recommended_velocity=recommended_velocity,
            min_lift_clearance=min_lift_clearance,
            preferred_approach_height=preferred_approach_height,
            approach_height_tolerance=approach_height_tolerance,
        )
        success = physics_ok and grip_success
        failure_bucket = "success" if success else ("physics_fail" if not physics_ok else grip_diag["failure_bucket"])
        # 使用模拟执行时间，避免墙钟时间过小导致各任务看起来没有差异。
        lift_speed = max(0.05, 0.22 * velocity_scale / (1.0 + mass_kg * 1.8))
        transport_speed = max(0.05, transport_velocity)
        placement_speed = max(0.05, placement_velocity)
        sim_elapsed = (
            (2.0 * approach_height / lift_speed)
            + (2.0 * lift_clearance / lift_speed)
            + (horizontal_distance / transport_speed)
            + (0.35 * lift_clearance / placement_speed)
            + (min(step + 1, max_steps) * self.time_step * 0.25)
        )

        info = {
            "distance": dist,
            "steps": min(step + 1, max_steps),
            "sim_time": round(sim_elapsed, 4),
            "wall_time": round(wall_elapsed, 4),
            "travel_distance": round(staged_distance, 4),
            "horizontal_distance": round(horizontal_distance, 4),
            "force_gain": round(force_gain, 4),
            "force_clip": round(force_clip, 4),
            "static_push_estimate": round(static_push_estimate, 4),
            "approach_height_error": round(abs(approach_height - preferred_approach_height), 4),
            "transport_velocity": round(transport_velocity, 4),
            "lift_force": round(lift_force, 4),
            "transfer_force": round(transfer_force, 4),
            "transfer_alignment": round(transfer_alignment, 4),
            "placement_velocity": round(placement_velocity, 4),
            "lift_clearance": round(lift_clearance, 4),
            "physics_ok": physics_ok,
            "grip_success": grip_success,
            "failure_bucket": failure_bucket,
            "dominant_failure_mode": grip_diag["dominant_failure_mode"],
            "slip_risk": grip_diag["slip_risk"],
            "compression_risk": grip_diag["compression_risk"],
            "velocity_risk": grip_diag["velocity_risk"],
            "clearance_risk": grip_diag["clearance_risk"],
            "lift_hold_risk": grip_diag["lift_hold_risk"],
            "transfer_sway_risk": grip_diag["transfer_sway_risk"],
            "placement_settle_risk": grip_diag["placement_settle_risk"],
            "lift_hold_success_prob": grip_diag["lift_hold_success_prob"],
            "transfer_success_prob": grip_diag["transfer_success_prob"],
            "placement_success_prob": grip_diag["placement_success_prob"],
            "stability_score": grip_diag["stability_score"],
            "nominal_force": grip_diag["nominal_force"],
            "min_force_needed": grip_diag["min_force_needed"],
            "max_safe_force": grip_diag["max_safe_force"],
            "recommended_velocity": grip_diag["recommended_velocity"],
            "dynamic_transport_mode": grip_diag["dynamic_transport_mode"],
            "dynamic_placement_velocity_cap": grip_diag["dynamic_placement_velocity_cap"],
            "min_lift_clearance": grip_diag["min_lift_clearance"],
            "dynamic_clearance_target": grip_diag["dynamic_clearance_target"],
            "dynamic_force_target": grip_diag["dynamic_force_target"],
            "dynamic_lift_force_shortfall": grip_diag["dynamic_lift_force_shortfall"],
            "lift_center_offset": grip_diag["lift_center_offset"],
            "lift_stage_control_active": grip_diag["lift_stage_control_active"],
            "profile": profile,
            "stage_plan": {
                "approach_height": round(approach_height, 4),
                "lift_clearance": round(lift_clearance, 4),
                "transport_velocity": round(transport_velocity, 4),
                "lift_force": round(lift_force, 4),
                "transfer_force": round(transfer_force, 4),
                "transfer_alignment": round(transfer_alignment, 4),
                "placement_velocity": round(placement_velocity, 4),
            },
        }
        if adaptive_trace_info is not None:
            info["observer_trace"] = adaptive_trace_info.get("observer_trace", [])
            info["step_replan_trace"] = adaptive_trace_info.get("step_replan_trace", [])
            info["step_replan_count"] = adaptive_trace_info.get("step_replan_count", 0)
            info["execution_feedback_mode"] = adaptive_trace_info.get("execution_feedback_mode", "observer_only")
            info["applied_params"] = adaptive_trace_info.get("applied_params", {})
        return success, sim_elapsed, info

    def close(self) -> None:
        if self._viewer is not None:
            try:
                self._viewer.close()
            except Exception:
                pass
            self._viewer = None


if not HAS_MUJOCO:

    class ArmSimEnv:
        """
        降级模式：无 MuJoCo 时仅用 RAG 成功模型 + 模拟耗时，不跑物理。
        接口与完整版一致，benchmark 可照常运行。
        """

        def __init__(self, gui: bool = False, time_step: float = 1.0 / 120.0, seed: int | None = None):
            if seed is not None:
                random.seed(seed)
                try:
                    np.random.seed(seed)
                except Exception:
                    pass
            self._mock = True

        def execute_pick_place(
            self,
            object_pos: Tuple[float, float, float],
            target_pos: Tuple[float, float, float],
            gripper_force: float,
            approach_height: float = 0.05,
            transport_velocity: float = 0.30,
            lift_force: float | None = None,
            transfer_force: float | None = None,
            transfer_alignment: float = 0.0,
            placement_velocity: float | None = None,
            lift_clearance: float = 0.06,
            object_profile: dict | None = None,
            max_steps: int = 400,
            step_replan_callback: Callable[[dict[str, Any], dict[str, Any]], dict[str, Any] | None] | None = None,
            max_step_replans: int = 0,
        ) -> Tuple[bool, float, dict]:
            return simulate_stepwise_execution(
                object_pos=object_pos,
                target_pos=target_pos,
                params={
                    "gripper_force": gripper_force,
                    "approach_height": approach_height,
                    "transport_velocity": transport_velocity,
                    "lift_force": lift_force if lift_force is not None else gripper_force,
                    "transfer_force": transfer_force if transfer_force is not None else gripper_force,
                    "placement_velocity": placement_velocity if placement_velocity is not None else transport_velocity,
                    "transfer_alignment": transfer_alignment,
                    "lift_clearance": lift_clearance,
                },
                object_profile=object_profile,
                step_replan_callback=step_replan_callback,
                max_step_replans=max_step_replans,
            )

        def close(self) -> None:
            pass
