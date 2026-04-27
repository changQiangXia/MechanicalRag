"""Belief-state and lightweight solver utilities for the high-level controller."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


STAGE_ORDER = ("approach", "grasp", "lift", "transfer", "place")


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


@dataclass
class ObjectBeliefState:
    mass_band: str
    friction_band: str
    fragility_band: str
    size_band: str
    dynamic_load_band: str
    center_of_mass_risk: float
    support_contact_expected: bool
    specific_force_supported: bool
    numeric_motion_supported: bool


@dataclass
class TaskConstraintSet:
    speed_priority: float
    stability_priority: float
    precision_priority: float
    safety_priority: float
    required_alignment: bool
    required_lift_margin: bool
    preferred_transport_mode: str
    stage_order: tuple[str, ...] = STAGE_ORDER


@dataclass
class UncertaintyProfile:
    support_score: float
    state_coverage: float
    conflict_count: int
    missing_specific_force: bool
    missing_numeric_motion: bool
    missing_alignment: bool
    missing_lift_stage: bool
    conservative_mode: bool
    reasons: list[str] = field(default_factory=list)


@dataclass
class StageIntent:
    name: str
    primary_goal: str
    risk_focus: list[str] = field(default_factory=list)
    control_bias: str = "balanced"
    exit_condition: str = ""


@dataclass
class ControlBeliefBundle:
    object_state: ObjectBeliefState
    task_constraints: TaskConstraintSet
    uncertainty: UncertaintyProfile
    stage_plan: list[StageIntent]

    def to_trace_dict(self) -> dict[str, Any]:
        return {
            "belief_state": asdict(self.object_state),
            "task_constraints": asdict(self.task_constraints),
            "uncertainty_profile": asdict(self.uncertainty),
            "stage_plan": [asdict(stage) for stage in self.stage_plan],
            "belief_state_coverage": round(self.uncertainty.state_coverage, 4),
            "uncertainty_conservative_mode": self.uncertainty.conservative_mode,
            "uncertainty_reasons": list(self.uncertainty.reasons),
        }


@dataclass
class CandidateControlPlan:
    label: str
    gripper_force: float
    approach_height: float
    transport_velocity: float
    lift_force: float
    transfer_force: float
    placement_velocity: float
    transfer_alignment: float
    lift_clearance: float

    def to_trace_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "gripper_force": round(self.gripper_force, 4),
            "approach_height": round(self.approach_height, 4),
            "transport_velocity": round(self.transport_velocity, 4),
            "lift_force": round(self.lift_force, 4),
            "transfer_force": round(self.transfer_force, 4),
            "placement_velocity": round(self.placement_velocity, 4),
            "transfer_alignment": round(self.transfer_alignment, 4),
            "lift_clearance": round(self.lift_clearance, 4),
        }


def build_control_belief(
    *,
    features: dict[str, bool],
    dynamic_transport_mode: str,
    support_score: float,
    conflict_count: int,
    force_rule_mode: str,
    motion_rule_mode: str,
    available_specific_force_rules: bool,
    available_motion_rules: bool,
    available_numeric_motion_rules: bool,
    available_alignment_rules: bool,
    available_lift_stage_rules: bool,
    available_support_contact_rules: bool,
) -> ControlBeliefBundle:
    mass_band = "heavy" if features["heavy"] else "large" if features["large"] else "light"
    if features["smooth_metal"]:
        friction_band = "low"
    elif features["metal"]:
        friction_band = "medium_low"
    elif features["rubber"]:
        friction_band = "high"
    else:
        friction_band = "medium"
    if features["thin_wall"]:
        fragility_band = "high"
    elif features["smooth_metal"]:
        fragility_band = "medium"
    else:
        fragility_band = "low"
    size_band = "small" if features["small"] else "large" if features["large"] else "medium"
    dynamic_load_band = "high" if features["high_speed"] or features["long_transfer"] else "medium" if features["heavy"] else "low"
    center_of_mass_risk = 0.85 if features["long_transfer"] and features["large"] else 0.65 if features["heavy"] else 0.3

    speed_priority = 0.9 if features["high_speed"] and not (features["long_transfer"] and features["large"]) else 0.55
    stability_priority = 0.9 if features["long_transfer"] or features["heavy"] or features["smooth_metal"] else 0.6
    precision_priority = 0.88 if features["long_transfer"] and features["large"] else 0.78 if features["thin_wall"] else 0.45
    safety_priority = 0.92 if features["thin_wall"] or features["heavy"] else 0.68

    required_alignment = features["long_transfer"] and features["large"]
    required_lift_margin = required_alignment or (
        features["high_speed"] and features["smooth_metal"] and features["heavy"]
    )

    coverage_checks = [
        1.0 if available_motion_rules or motion_rule_mode == "disabled" else 0.0,
        1.0 if available_specific_force_rules or force_rule_mode == "generic_only" else 0.0,
        1.0 if not required_alignment or available_alignment_rules else 0.0,
        1.0 if not required_lift_margin or available_lift_stage_rules else 0.0,
        1.0 if not features["thin_wall"] or available_support_contact_rules else 0.0,
        1.0 if not (features["long_transfer"] and features["large"]) or available_numeric_motion_rules else 0.0,
    ]
    state_coverage = sum(coverage_checks) / len(coverage_checks)

    reasons: list[str] = []
    missing_specific_force = force_rule_mode == "all" and not available_specific_force_rules
    missing_numeric_motion = features["long_transfer"] and features["large"] and not available_numeric_motion_rules
    missing_alignment = required_alignment and not available_alignment_rules
    missing_lift_stage = required_lift_margin and not available_lift_stage_rules
    if support_score < 2.2:
        reasons.append("low_support_score")
    if conflict_count > 0:
        reasons.append("evidence_conflict")
    if state_coverage < 0.75:
        reasons.append("state_coverage_gap")
    if missing_specific_force:
        reasons.append("missing_specific_force")
    if missing_numeric_motion:
        reasons.append("missing_numeric_motion")
    if missing_alignment:
        reasons.append("missing_alignment")
    if missing_lift_stage:
        reasons.append("missing_lift_stage")

    critical_reasons = [reason for reason in reasons if reason != "evidence_conflict"]
    uncertainty = UncertaintyProfile(
        support_score=round(support_score, 4),
        state_coverage=round(state_coverage, 4),
        conflict_count=conflict_count,
        missing_specific_force=missing_specific_force,
        missing_numeric_motion=missing_numeric_motion,
        missing_alignment=missing_alignment,
        missing_lift_stage=missing_lift_stage,
        conservative_mode=bool(critical_reasons),
        reasons=reasons,
    )

    object_state = ObjectBeliefState(
        mass_band=mass_band,
        friction_band=friction_band,
        fragility_band=fragility_band,
        size_band=size_band,
        dynamic_load_band=dynamic_load_band,
        center_of_mass_risk=round(center_of_mass_risk, 4),
        support_contact_expected=available_support_contact_rules,
        specific_force_supported=available_specific_force_rules,
        numeric_motion_supported=available_numeric_motion_rules,
    )
    constraints = TaskConstraintSet(
        speed_priority=round(speed_priority, 4),
        stability_priority=round(stability_priority, 4),
        precision_priority=round(precision_priority, 4),
        safety_priority=round(safety_priority, 4),
        required_alignment=required_alignment,
        required_lift_margin=required_lift_margin,
        preferred_transport_mode=dynamic_transport_mode,
    )
    stage_plan = [
        StageIntent(
            name="approach",
            primary_goal="establish_safe_contact",
            risk_focus=["fragility", "misalignment"] if features["thin_wall"] else ["contact_setup"],
            control_bias="safety",
            exit_condition="stable_pregrasp_pose",
        ),
        StageIntent(
            name="grasp",
            primary_goal="secure_initial_hold",
            risk_focus=["slip"] if friction_band in {"low", "medium_low"} else ["compression"],
            control_bias="force_margin" if friction_band in {"low", "medium_low"} else "balanced",
            exit_condition="contact_force_stable",
        ),
        StageIntent(
            name="lift",
            primary_goal="survive_vertical_detach",
            risk_focus=["lift_hold", "center_of_mass_shift"] if required_lift_margin else ["lift_hold"],
            control_bias="stability",
            exit_condition="object_cleared",
        ),
        StageIntent(
            name="transfer",
            primary_goal="suppress_sway_and_slip",
            risk_focus=["transfer_sway", "slip"] if required_alignment else ["velocity"],
            control_bias="speed" if speed_priority > stability_priority and not uncertainty.conservative_mode else "stability",
            exit_condition="target_zone_reached",
        ),
        StageIntent(
            name="place",
            primary_goal="settle_without_rebound",
            risk_focus=["placement_settle", "precision"],
            control_bias="precision",
            exit_condition="object_stable_at_target",
        ),
    ]
    return ControlBeliefBundle(
        object_state=object_state,
        task_constraints=constraints,
        uncertainty=uncertainty,
        stage_plan=stage_plan,
    )


def solve_control_plan(
    base_plan: dict[str, float],
    belief: ControlBeliefBundle,
) -> tuple[dict[str, float], dict[str, Any]]:
    base = CandidateControlPlan(label="rule_aggregate", **base_plan)
    candidates = [base]
    notes: list[str] = []

    if belief.task_constraints.stability_priority >= 0.75:
        candidates.append(
            CandidateControlPlan(
                label="stability_bias",
                gripper_force=base.gripper_force + 0.6,
                approach_height=base.approach_height,
                transport_velocity=base.transport_velocity - 0.02,
                lift_force=base.lift_force + 0.4,
                transfer_force=base.transfer_force + 0.5,
                placement_velocity=base.placement_velocity - 0.02,
                transfer_alignment=base.transfer_alignment + (0.04 if belief.task_constraints.required_alignment else 0.0),
                lift_clearance=base.lift_clearance + 0.004,
            )
        )
    if belief.task_constraints.precision_priority >= 0.75:
        candidates.append(
            CandidateControlPlan(
                label="precision_bias",
                gripper_force=base.gripper_force,
                approach_height=base.approach_height,
                transport_velocity=base.transport_velocity,
                lift_force=base.lift_force,
                transfer_force=base.transfer_force,
                placement_velocity=base.placement_velocity - 0.03,
                transfer_alignment=base.transfer_alignment + (0.05 if belief.task_constraints.required_alignment else 0.0),
                lift_clearance=base.lift_clearance,
            )
        )
    if belief.task_constraints.speed_priority >= 0.85 and not belief.uncertainty.conservative_mode:
        candidates.append(
            CandidateControlPlan(
                label="speed_bias",
                gripper_force=base.gripper_force,
                approach_height=base.approach_height,
                transport_velocity=base.transport_velocity + 0.015,
                lift_force=base.lift_force,
                transfer_force=base.transfer_force,
                placement_velocity=base.placement_velocity + 0.015,
                transfer_alignment=base.transfer_alignment,
                lift_clearance=base.lift_clearance,
            )
        )
    if belief.uncertainty.conservative_mode:
        candidates.append(
            CandidateControlPlan(
                label="uncertainty_guard",
                gripper_force=base.gripper_force + 0.8,
                approach_height=base.approach_height,
                transport_velocity=base.transport_velocity - 0.03,
                lift_force=base.lift_force + 0.8,
                transfer_force=base.transfer_force + 0.8,
                placement_velocity=base.placement_velocity - 0.03,
                transfer_alignment=base.transfer_alignment + (0.08 if belief.task_constraints.required_alignment else 0.0),
                lift_clearance=base.lift_clearance + 0.006,
            )
        )

    preferred_transport_cap = base.transport_velocity
    preferred_placement_cap = base.placement_velocity
    preferred_clearance_floor = base.lift_clearance
    preferred_force_floor = base.gripper_force
    preferred_lift_force_floor = max(base.gripper_force, base.lift_force)
    preferred_transfer_force_floor = max(base.gripper_force, base.transfer_force)
    preferred_alignment_floor = base.transfer_alignment if belief.task_constraints.required_alignment else 0.0
    preferred_force_cap = 50.0

    if belief.uncertainty.conservative_mode:
        preferred_transport_cap = max(0.12, preferred_transport_cap - 0.02)
        preferred_placement_cap = max(0.12, preferred_placement_cap - 0.02)
        preferred_clearance_floor += 0.004
        preferred_force_floor += 0.4
        preferred_lift_force_floor += 0.6
        preferred_transfer_force_floor += 0.5
        if belief.task_constraints.required_alignment:
            preferred_alignment_floor = max(preferred_alignment_floor, 0.82)
    if belief.object_state.friction_band == "low":
        preferred_transfer_force_floor = max(preferred_transfer_force_floor, base.gripper_force + 0.6)
    if belief.task_constraints.precision_priority >= 0.8:
        preferred_placement_cap = min(preferred_placement_cap, max(0.12, base.placement_velocity))
    if belief.object_state.fragility_band == "high":
        preferred_force_cap = max(base.gripper_force + 1.5, 14.0)

    def _score(plan: CandidateControlPlan) -> tuple[float, dict[str, float]]:
        transport_velocity = _clamp(plan.transport_velocity, 0.12, 0.8)
        placement_velocity = _clamp(min(plan.placement_velocity, transport_velocity), 0.12, 0.8)
        gripper_force = _clamp(plan.gripper_force, 5.0, 50.0)
        lift_force = _clamp(max(plan.lift_force, gripper_force), gripper_force, 50.0)
        transfer_force = _clamp(max(plan.transfer_force, gripper_force), gripper_force, 50.0)
        transfer_alignment = _clamp(plan.transfer_alignment, 0.0, 1.0)
        lift_clearance = _clamp(plan.lift_clearance, 0.03, 0.14)

        breakdown = {
            "force_shortfall": max(0.0, preferred_force_floor - gripper_force) * 3.0,
            "force_overdrive": max(0.0, gripper_force - preferred_force_cap) * 1.5,
            "lift_shortfall": max(0.0, preferred_lift_force_floor - lift_force) * 2.8,
            "transfer_shortfall": max(0.0, preferred_transfer_force_floor - transfer_force) * 2.6,
            "transport_overspeed": max(0.0, transport_velocity - preferred_transport_cap) * 18.0,
            "placement_overspeed": max(0.0, placement_velocity - preferred_placement_cap) * 22.0,
            "clearance_shortfall": max(0.0, preferred_clearance_floor - lift_clearance) * 40.0,
            "alignment_shortfall": max(0.0, preferred_alignment_floor - transfer_alignment) * 6.0,
            "stage_inconsistency": max(0.0, placement_velocity - transport_velocity) * 6.0,
        }
        if belief.uncertainty.conservative_mode and plan.label == "speed_bias":
            breakdown["uncertainty_speed_penalty"] = 2.0
        return sum(breakdown.values()), breakdown

    candidate_scores: list[dict[str, Any]] = []
    best_plan = base
    best_score = float("inf")
    best_breakdown: dict[str, float] = {}
    for candidate in candidates:
        score, breakdown = _score(candidate)
        candidate_scores.append(
            {
                "label": candidate.label,
                "score": round(score, 4),
                "breakdown": {key: round(value, 4) for key, value in breakdown.items()},
            }
        )
        if score < best_score:
            best_plan = candidate
            best_score = score
            best_breakdown = breakdown

    solved_plan = {
        "gripper_force": round(_clamp(best_plan.gripper_force, 5.0, 50.0), 4),
        "approach_height": round(_clamp(best_plan.approach_height, 0.02, 0.08), 4),
        "transport_velocity": round(_clamp(best_plan.transport_velocity, 0.12, 0.8), 4),
        "lift_clearance": round(_clamp(best_plan.lift_clearance, 0.03, 0.14), 4),
        "lift_force": round(_clamp(max(best_plan.lift_force, best_plan.gripper_force), best_plan.gripper_force, 50.0), 4),
        "transfer_force": round(_clamp(max(best_plan.transfer_force, best_plan.gripper_force), best_plan.gripper_force, 50.0), 4),
        "placement_velocity": round(
            _clamp(min(best_plan.placement_velocity, best_plan.transport_velocity), 0.12, 0.8),
            4,
        ),
        "transfer_alignment": round(_clamp(best_plan.transfer_alignment, 0.0, 1.0), 4),
    }
    if best_plan.label != base.label:
        notes.append(f"solver_selected:{best_plan.label}")
    if belief.uncertainty.conservative_mode and best_plan.label == "uncertainty_guard":
        notes.append("solver_uncertainty_guard")

    return solved_plan, {
        "solver_mode": "belief_scored_candidates",
        "solver_selected_candidate": best_plan.label,
        "solver_selected_score": round(best_score, 4),
        "solver_score_breakdown": {key: round(value, 4) for key, value in best_breakdown.items()},
        "solver_candidate_scores": candidate_scores,
        "solver_adjustment_notes": notes,
    }
