"""Evidence-derived symbolic control-state and local re-optimization utilities."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


STAGE_ORDER = ("approach", "grasp", "lift", "transfer", "place")


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _weighted_mean(values: list[float], weights: list[float], default: float) -> float:
    if not values or not weights or sum(weights) <= 0.0:
        return default
    return sum(value * weight for value, weight in zip(values, weights)) / sum(weights)


def _weighted_std(values: list[float], weights: list[float], default: float) -> float:
    if len(values) <= 1 or not weights or sum(weights) <= 0.0:
        return default
    center = _weighted_mean(values, weights, values[0])
    variance = sum(weight * ((value - center) ** 2) for value, weight in zip(values, weights)) / sum(weights)
    return variance**0.5


def _support_weight(rule: dict[str, Any]) -> float:
    return max(0.35, float(rule.get("score", 0.0)) + 0.35 * float(rule.get("specificity", 0.0)))


def _estimate_signal(
    *,
    prior: float,
    weighted_values: list[tuple[float, float]],
    fallback_confidence: float = 0.25,
) -> tuple[float, float]:
    usable = [(value, weight) for value, weight in weighted_values if weight > 1e-6]
    if usable:
        total_weight = sum(weight for _, weight in usable)
        signal = sum(value * weight for value, weight in usable) / total_weight
        confidence = _clamp(total_weight / (total_weight + 0.75), 0.15, 1.0)
        return _clamp(signal, 0.0, 1.0), confidence
    return _clamp(prior, 0.0, 1.0), fallback_confidence


def _mass_band(signal: float) -> str:
    if signal >= 0.78:
        return "heavy"
    if signal >= 0.55:
        return "large"
    return "light"


def _friction_band(signal: float) -> str:
    if signal >= 0.78:
        return "low"
    if signal >= 0.58:
        return "medium_low"
    if signal <= 0.22:
        return "high"
    return "medium"


def _fragility_band(signal: float) -> str:
    if signal >= 0.75:
        return "high"
    if signal >= 0.45:
        return "medium"
    return "low"


def _size_band(signal: float) -> str:
    if signal >= 0.72:
        return "large"
    if signal <= 0.35:
        return "small"
    return "medium"


def _dynamic_load_band(signal: float) -> str:
    if signal >= 0.72:
        return "high"
    if signal >= 0.42:
        return "medium"
    return "low"


def _clone_stage_plan(stages: list["StageIntent"]) -> list["StageIntent"]:
    return [StageIntent(**asdict(stage)) for stage in stages]


@dataclass
class EvidenceStateSummary:
    mass_signal: float
    mass_confidence: float
    friction_signal: float
    friction_confidence: float
    fragility_signal: float
    fragility_confidence: float
    size_signal: float
    size_confidence: float
    dynamic_load_signal: float
    dynamic_load_confidence: float
    center_of_mass_risk: float
    force_center: float
    force_std: float
    motion_confidence: float
    numeric_motion_confidence: float
    alignment_confidence: float
    lift_stage_confidence: float
    support_contact_confidence: float
    specific_force_confidence: float
    preferred_transport_mode: str
    term_supports: dict[str, float] = field(default_factory=dict)

    def to_trace_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "term_supports": {key: round(value, 4) for key, value in self.term_supports.items()},
        }


@dataclass
class ObjectBeliefState:
    mass_band: str
    mass_confidence: float
    friction_band: str
    friction_confidence: float
    fragility_band: str
    fragility_confidence: float
    size_band: str
    size_confidence: float
    dynamic_load_band: str
    dynamic_load_confidence: float
    center_of_mass_risk: float
    force_center: float
    force_std: float
    motion_confidence: float
    alignment_confidence: float
    lift_stage_confidence: float
    support_contact_confidence: float
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
    replan_stage_bias: str = "none"
    stage_order: tuple[str, ...] = STAGE_ORDER


@dataclass
class UncertaintyProfile:
    support_score: float
    state_coverage: float
    conflict_count: int
    force_std: float
    attribute_confidence: float
    motion_confidence: float
    alignment_confidence: float
    lift_stage_confidence: float
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
    evidence_state: EvidenceStateSummary
    object_state: ObjectBeliefState
    task_constraints: TaskConstraintSet
    uncertainty: UncertaintyProfile
    stage_plan: list[StageIntent]

    def to_trace_dict(self) -> dict[str, Any]:
        return {
            "evidence_state_summary": self.evidence_state.to_trace_dict(),
            "belief_state": asdict(self.object_state),
            "task_constraints": asdict(self.task_constraints),
            "uncertainty_profile": asdict(self.uncertainty),
            "stage_plan": [asdict(stage) for stage in self.stage_plan],
            "belief_state_coverage": round(self.uncertainty.state_coverage, 4),
            "uncertainty_conservative_mode": self.uncertainty.conservative_mode,
            "uncertainty_reasons": list(self.uncertainty.reasons),
        }


@dataclass
class PhaseObservation:
    phase: str
    contact_stability_obs: float = 0.0
    micro_slip_obs: float = 0.0
    payload_ratio_obs: float = 0.0
    lift_progress_obs: float = 0.0
    lift_reserve_obs: float = 0.0
    tilt_obs: float = 0.0
    sway_obs: float = 0.0
    velocity_stress_obs: float = 0.0
    settle_obs: float = 0.0
    placement_error_obs: float = 0.0
    observation_confidence: float = 0.6
    trigger_reason: str = ""


@dataclass
class ExecutionBelief:
    phase: str
    load_support_margin: float
    load_support_uncertainty: float
    grip_hold_margin: float
    grip_hold_uncertainty: float
    pose_alignment_error: float
    pose_alignment_uncertainty: float
    lift_reserve: float
    lift_reserve_uncertainty: float
    transfer_disturbance: float
    transfer_disturbance_uncertainty: float
    preferred_transport_mode: str
    conservative_mode: bool

    def to_trace_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CounterfactualIntervention:
    label: str
    param_deltas: dict[str, float]
    targeted_latents: list[str]
    predicted_flip_gain: float


@dataclass
class CounterfactualDiagnosis:
    diagnosed_cause: str
    candidate_interventions: list[CounterfactualIntervention]
    selected_intervention: CounterfactualIntervention
    predicted_flip_gain: float


def build_execution_prior(params: dict[str, Any], *, phase: str) -> ExecutionBelief:
    belief_state = dict(params.get("belief_state", {}))
    uncertainty = dict(params.get("uncertainty_profile", {}))
    task_constraints = dict(params.get("task_constraints", {}))
    mass_band = str(belief_state.get("mass_band", "light"))
    dynamic_load_band = str(belief_state.get("dynamic_load_band", "low"))
    load_support_margin = -0.18 if mass_band == "heavy" else -0.05 if dynamic_load_band == "high" else 0.12
    lift_reserve = -0.12 if mass_band == "heavy" else 0.08
    return ExecutionBelief(
        phase=phase,
        load_support_margin=round(load_support_margin, 4),
        load_support_uncertainty=0.22,
        grip_hold_margin=round(0.14 - 0.04 * float(belief_state.get("center_of_mass_risk", 0.4)), 4),
        grip_hold_uncertainty=0.18,
        pose_alignment_error=round(0.16 + 0.12 * (1.0 - float(belief_state.get("alignment_confidence", 0.3))), 4),
        pose_alignment_uncertainty=0.20,
        lift_reserve=round(lift_reserve, 4),
        lift_reserve_uncertainty=0.24,
        transfer_disturbance=0.22
        if str(params.get("dynamic_transport_mode", task_constraints.get("preferred_transport_mode", "static"))) == "static"
        else 0.48,
        transfer_disturbance_uncertainty=0.18,
        preferred_transport_mode=str(
            params.get("dynamic_transport_mode", task_constraints.get("preferred_transport_mode", "static"))
        ),
        conservative_mode=bool(uncertainty.get("conservative_mode", False)),
    )


def apply_phase_observation(
    belief: ExecutionBelief,
    observation: PhaseObservation,
) -> tuple[ExecutionBelief, dict[str, Any]]:
    updated = ExecutionBelief(**asdict(belief))
    updated.phase = observation.phase
    updates: list[dict[str, Any]] = []
    weight = _clamp(observation.observation_confidence, 0.15, 1.0)

    if observation.phase == "lift":
        updated.load_support_margin = round(
            updated.load_support_margin - weight * max(0.0, observation.payload_ratio_obs - 0.75) * 0.30,
            4,
        )
        updated.lift_reserve = round(updated.lift_reserve + weight * observation.lift_reserve_obs, 4)
        updated.pose_alignment_error = round(
            updated.pose_alignment_error + weight * abs(observation.tilt_obs) * 0.35,
            4,
        )
        updates.extend(
            [
                {"name": "load_support_margin", "value": updated.load_support_margin},
                {"name": "lift_reserve", "value": updated.lift_reserve},
                {"name": "pose_alignment_error", "value": updated.pose_alignment_error},
            ]
        )
    elif observation.phase == "grasp":
        updated.grip_hold_margin = round(
            updated.grip_hold_margin
            + weight * observation.contact_stability_obs
            - weight * observation.micro_slip_obs,
            4,
        )
        updates.append({"name": "grip_hold_margin", "value": updated.grip_hold_margin})
    elif observation.phase == "transfer":
        updated.transfer_disturbance = round(
            updated.transfer_disturbance
            + weight * (observation.sway_obs + observation.velocity_stress_obs) * 0.25,
            4,
        )
        updates.append({"name": "transfer_disturbance", "value": updated.transfer_disturbance})

    return updated, {
        "phase": observation.phase,
        "trigger_reason": observation.trigger_reason,
        "updated_latents": updates,
        "posterior": updated.to_trace_dict(),
    }


def _heavy_static_candidates() -> list[CounterfactualIntervention]:
    return [
        CounterfactualIntervention(
            "gripper_force_up",
            {"gripper_force": 1.2},
            ["grip_hold_margin", "load_support_margin"],
            0.0,
        ),
        CounterfactualIntervention(
            "lift_force_up",
            {"lift_force": 1.6},
            ["load_support_margin", "lift_reserve"],
            0.0,
        ),
        CounterfactualIntervention(
            "transfer_force_up",
            {"transfer_force": 1.2},
            ["load_support_margin"],
            0.0,
        ),
        CounterfactualIntervention(
            "transport_velocity_down",
            {"transport_velocity": -0.02, "placement_velocity": -0.02},
            ["lift_reserve"],
            0.0,
        ),
        CounterfactualIntervention(
            "lift_clearance_up",
            {"lift_clearance": 0.006},
            ["lift_reserve"],
            0.0,
        ),
        CounterfactualIntervention(
            "lift_force_plus_velocity",
            {"lift_force": 1.4, "transport_velocity": -0.02, "placement_velocity": -0.02},
            ["load_support_margin", "lift_reserve"],
            0.0,
        ),
        CounterfactualIntervention(
            "lift_force_plus_clearance",
            {"lift_force": 2.0, "lift_clearance": 0.02},
            ["load_support_margin", "lift_reserve"],
            0.0,
        ),
        CounterfactualIntervention(
            "gripper_plus_lift",
            {"gripper_force": 1.0, "lift_force": 1.2},
            ["grip_hold_margin", "load_support_margin"],
            0.0,
        ),
    ]


def _predicted_latent_gains(
    belief: ExecutionBelief,
    deltas: dict[str, float],
) -> dict[str, float]:
    velocity_drop = max(0.0, -float(deltas.get("transport_velocity", 0.0)))
    return {
        "load_support_margin": round(
            0.14 * float(deltas.get("lift_force", 0.0))
            + 0.07 * float(deltas.get("gripper_force", 0.0))
            + 0.06 * float(deltas.get("transfer_force", 0.0)),
            4,
        ),
        "lift_reserve": round(
            0.10 * float(deltas.get("lift_force", 0.0))
            + 4.0 * float(deltas.get("lift_clearance", 0.0))
            + 1.0 * velocity_drop
            + 0.04 * float(deltas.get("gripper_force", 0.0)),
            4,
        ),
        "grip_hold_margin": round(0.07 * float(deltas.get("gripper_force", 0.0)), 4),
    }


def diagnose_failure_cause(
    belief: ExecutionBelief,
    observation: PhaseObservation,
    current_plan: dict[str, float],
) -> CounterfactualDiagnosis:
    del current_plan
    if belief.load_support_margin <= min(belief.grip_hold_margin, belief.lift_reserve):
        diagnosed_cause = "under_supported_load"
    elif belief.grip_hold_margin < 0.0:
        diagnosed_cause = "load_induced_slip"
    else:
        diagnosed_cause = "alignment_coupled_overload"

    required_support = max(0.0, -belief.load_support_margin)
    required_lift = max(0.0, -belief.lift_reserve)
    required_grip = max(0.0, -belief.grip_hold_margin)

    ranked: list[tuple[bool, CounterfactualIntervention]] = []
    for candidate in _heavy_static_candidates():
        gains = _predicted_latent_gains(belief, candidate.param_deltas)
        sufficient = (
            gains["load_support_margin"] >= required_support
            and gains["lift_reserve"] >= required_lift
            and gains["grip_hold_margin"] >= required_grip
        )
        flip_gain = round(
            gains["load_support_margin"]
            + gains["lift_reserve"]
            + 0.5 * gains["grip_hold_margin"]
            - required_support
            - required_lift
            - 0.5 * required_grip,
            4,
        )
        ranked.append(
            (
                sufficient,
                CounterfactualIntervention(
                    label=candidate.label,
                    param_deltas=dict(candidate.param_deltas),
                    targeted_latents=list(candidate.targeted_latents),
                    predicted_flip_gain=flip_gain,
                ),
            )
        )

    ranked.sort(
        key=lambda item: (
            0 if item[0] else 1,
            len(item[1].param_deltas),
            -item[1].predicted_flip_gain,
            item[1].label,
        )
    )
    candidates = [item[1] for item in ranked]
    selected = candidates[0]
    return CounterfactualDiagnosis(
        diagnosed_cause=diagnosed_cause,
        candidate_interventions=candidates,
        selected_intervention=selected,
        predicted_flip_gain=selected.predicted_flip_gain,
    )


def repair_suffix_plan(
    current_plan: dict[str, float],
    *,
    current_phase: str,
    diagnosis: CounterfactualDiagnosis,
) -> dict[str, Any]:
    next_plan = dict(current_plan)
    for key, delta in diagnosis.selected_intervention.param_deltas.items():
        next_plan[key] = round(float(next_plan.get(key, 0.0)) + float(delta), 4)
    next_plan["suffix_replan_start_stage"] = current_phase
    next_plan["execution_feedback_mode"] = "suffix_counterfactual_replan"
    next_plan["counterfactual_replan_trace"] = [
        {
            "start_phase": current_phase,
            "diagnosed_cause": diagnosis.diagnosed_cause,
            "selected_intervention": diagnosis.selected_intervention.label,
            "predicted_flip_gain": diagnosis.predicted_flip_gain,
            "candidate_interventions": [
                {
                    "label": item.label,
                    "param_deltas": item.param_deltas,
                    "predicted_flip_gain": item.predicted_flip_gain,
                }
                for item in diagnosis.candidate_interventions
            ],
        }
    ]
    return next_plan


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


@dataclass
class EvidenceConstraintHints:
    force_floor: float | None = None
    force_cap: float | None = None
    transport_velocity_floor: float | None = None
    transport_velocity_cap: float | None = None
    placement_velocity_cap: float | None = None
    clearance_floor: float | None = None
    clearance_target: float | None = None
    approach_height_target: float | None = None
    alignment_target: float | None = None
    lift_force_margin: float = 0.0
    transfer_force_margin: float = 0.0
    gripper_force_bias: float = 0.0
    source_notes: list[str] = field(default_factory=list)

    def to_trace_dict(self) -> dict[str, Any]:
        return {
            "force_floor": None if self.force_floor is None else round(self.force_floor, 4),
            "force_cap": None if self.force_cap is None else round(self.force_cap, 4),
            "transport_velocity_floor": None if self.transport_velocity_floor is None else round(self.transport_velocity_floor, 4),
            "transport_velocity_cap": None if self.transport_velocity_cap is None else round(self.transport_velocity_cap, 4),
            "placement_velocity_cap": None if self.placement_velocity_cap is None else round(self.placement_velocity_cap, 4),
            "clearance_floor": None if self.clearance_floor is None else round(self.clearance_floor, 4),
            "clearance_target": None if self.clearance_target is None else round(self.clearance_target, 4),
            "approach_height_target": None if self.approach_height_target is None else round(self.approach_height_target, 4),
            "alignment_target": None if self.alignment_target is None else round(self.alignment_target, 4),
            "lift_force_margin": round(self.lift_force_margin, 4),
            "transfer_force_margin": round(self.transfer_force_margin, 4),
            "gripper_force_bias": round(self.gripper_force_bias, 4),
            "source_notes": list(self.source_notes),
        }


def summarize_control_evidence(
    *,
    features: dict[str, bool],
    rules: list[dict[str, Any]],
    selected_force_rules: list[dict[str, Any]],
    specific_force_rules: list[dict[str, Any]],
    motion_rules: list[dict[str, Any]],
    numeric_motion_rules: list[dict[str, Any]],
    alignment_rules: list[dict[str, Any]],
    lift_stage_rules: list[dict[str, Any]],
    support_contact_rules: list[dict[str, Any]],
    default_force: float,
) -> EvidenceStateSummary:
    weighted_rules = [(_support_weight(rule), rule) for rule in rules]
    support_den = max(1.0, sum(weight for weight, _ in weighted_rules))
    term_supports: dict[str, float] = {}
    for weight, rule in weighted_rules:
        for term in rule.get("matched_terms", []):
            term_supports[term] = term_supports.get(term, 0.0) + weight

    def _term_score(term: str) -> float:
        return _clamp(term_supports.get(term, 0.0) / support_den, 0.0, 1.0)

    force_centers = [
        sum(rule["force_candidates"]) / len(rule["force_candidates"])
        for rule in selected_force_rules
        if rule.get("force_candidates")
    ]
    force_weights = [_support_weight(rule) for rule in selected_force_rules if rule.get("force_candidates")]
    specific_force_confidence = _clamp(sum(force_weights) / support_den, 0.0, 1.0)
    force_center = _weighted_mean(force_centers, force_weights, default_force)
    force_std = _weighted_std(force_centers, force_weights, 4.5 if not force_centers else 2.0)

    motion_confidence = _clamp(sum(_support_weight(rule) for rule in motion_rules) / support_den, 0.0, 1.0)
    numeric_motion_confidence = _clamp(sum(_support_weight(rule) for rule in numeric_motion_rules) / support_den, 0.0, 1.0)
    alignment_confidence = _clamp(
        (sum(_support_weight(rule) for rule in alignment_rules) / support_den) + 0.15 * _term_score("long_transfer"),
        0.0,
        1.0,
    )
    lift_stage_confidence = _clamp(
        (sum(_support_weight(rule) for rule in lift_stage_rules) / support_den)
        + 0.08 * _term_score("heavy")
        + 0.08 * _term_score("large"),
        0.0,
        1.0,
    )
    support_contact_confidence = _clamp(
        sum(_support_weight(rule) for rule in support_contact_rules) / support_den,
        0.0,
        1.0,
    )

    mass_prior = 0.82 if features["heavy"] else 0.66 if features["large"] else 0.22 if features["small"] else 0.48
    mass_signal, mass_confidence = _estimate_signal(
        prior=mass_prior,
        weighted_values=[
            (0.92, _term_score("heavy")),
            (0.72, _term_score("large") * 0.9),
            (_clamp((force_center - 12.0) / 32.0, 0.0, 1.0), specific_force_confidence + 0.2),
        ],
        fallback_confidence=0.28,
    )

    friction_prior = 0.92 if features["smooth_metal"] else 0.68 if features["metal"] else 0.12 if features["rubber"] else 0.45
    friction_signal, friction_confidence = _estimate_signal(
        prior=friction_prior,
        weighted_values=[
            (0.92, _term_score("metal") + (0.2 if features["smooth_metal"] else 0.0)),
            (0.10, _term_score("rubber")),
            (0.82, 0.5 * motion_confidence if features["high_speed"] and features["smooth_metal"] else 0.0),
        ],
        fallback_confidence=0.26,
    )

    fragility_prior = 0.92 if features["thin_wall"] else 0.55 if features["smooth_metal"] else 0.28
    fragility_signal, fragility_confidence = _estimate_signal(
        prior=fragility_prior,
        weighted_values=[
            (0.96, _term_score("thin_wall")),
            (0.62, support_contact_confidence * (1.0 if features["thin_wall"] else 0.4)),
            (0.55, _term_score("metal") * (0.4 if features["smooth_metal"] else 0.15)),
        ],
        fallback_confidence=0.24,
    )

    size_prior = 0.86 if features["large"] else 0.18 if features["small"] else 0.5
    size_signal, size_confidence = _estimate_signal(
        prior=size_prior,
        weighted_values=[
            (0.88, _term_score("large")),
            (0.16, _term_score("small")),
            (0.72, 0.35 * numeric_motion_confidence if features["long_transfer"] else 0.0),
        ],
        fallback_confidence=0.24,
    )

    dynamic_load_prior = 0.9 if (features["high_speed"] or features["long_transfer"]) else 0.58 if features["heavy"] else 0.24
    dynamic_load_signal, dynamic_load_confidence = _estimate_signal(
        prior=dynamic_load_prior,
        weighted_values=[
            (0.92, _term_score("high_speed") + 0.35 * motion_confidence),
            (0.94, _term_score("long_transfer") + 0.4 * numeric_motion_confidence),
            (0.72, 0.35 * alignment_confidence),
        ],
        fallback_confidence=0.25,
    )

    center_of_mass_risk = _clamp(
        0.20
        + 0.28 * size_signal
        + 0.22 * mass_signal
        + 0.25 * alignment_confidence
        + 0.18 * _term_score("long_transfer"),
        0.0,
        1.0,
    )

    preferred_transport_mode = "static"
    if (
        _term_score("long_transfer") >= 0.12
        and size_signal >= 0.60
        and numeric_motion_confidence >= 0.25
    ):
        preferred_transport_mode = "long_transfer"
    elif (
        features["high_speed"]
        and friction_signal >= 0.62
        and motion_confidence >= 0.20
        and dynamic_load_signal >= 0.65
    ):
        preferred_transport_mode = "high_speed_low_friction"

    return EvidenceStateSummary(
        mass_signal=round(mass_signal, 4),
        mass_confidence=round(mass_confidence, 4),
        friction_signal=round(friction_signal, 4),
        friction_confidence=round(friction_confidence, 4),
        fragility_signal=round(fragility_signal, 4),
        fragility_confidence=round(fragility_confidence, 4),
        size_signal=round(size_signal, 4),
        size_confidence=round(size_confidence, 4),
        dynamic_load_signal=round(dynamic_load_signal, 4),
        dynamic_load_confidence=round(dynamic_load_confidence, 4),
        center_of_mass_risk=round(center_of_mass_risk, 4),
        force_center=round(force_center, 4),
        force_std=round(force_std, 4),
        motion_confidence=round(motion_confidence, 4),
        numeric_motion_confidence=round(numeric_motion_confidence, 4),
        alignment_confidence=round(alignment_confidence, 4),
        lift_stage_confidence=round(lift_stage_confidence, 4),
        support_contact_confidence=round(support_contact_confidence, 4),
        specific_force_confidence=round(specific_force_confidence, 4),
        preferred_transport_mode=preferred_transport_mode,
        term_supports={key: round(value / support_den, 4) for key, value in term_supports.items()},
    )


def build_control_belief(
    *,
    features: dict[str, bool],
    evidence_summary: EvidenceStateSummary,
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
    mass_band = _mass_band(evidence_summary.mass_signal)
    friction_band = _friction_band(evidence_summary.friction_signal)
    fragility_band = _fragility_band(evidence_summary.fragility_signal)
    size_band = _size_band(evidence_summary.size_signal)
    dynamic_load_band = _dynamic_load_band(evidence_summary.dynamic_load_signal)
    attribute_confidence = (
        evidence_summary.mass_confidence
        + evidence_summary.friction_confidence
        + evidence_summary.fragility_confidence
        + evidence_summary.size_confidence
        + evidence_summary.dynamic_load_confidence
    ) / 5.0
    force_std_pressure = _clamp(evidence_summary.force_std / 6.0, 0.0, 1.0)

    speed_priority = _clamp(
        0.25
        + 0.55 * evidence_summary.dynamic_load_signal
        + 0.10 * evidence_summary.motion_confidence
        - 0.15 * evidence_summary.alignment_confidence,
        0.20,
        0.95,
    )
    if dynamic_transport_mode == "long_transfer":
        speed_priority = min(speed_priority, 0.65)
    stability_priority = _clamp(
        0.30
        + 0.28 * evidence_summary.mass_signal
        + 0.24 * evidence_summary.friction_signal
        + 0.22 * evidence_summary.center_of_mass_risk
        + 0.10 * force_std_pressure,
        0.35,
        0.98,
    )
    precision_priority = _clamp(
        0.25
        + 0.28 * evidence_summary.fragility_signal
        + 0.25 * evidence_summary.alignment_confidence
        + 0.16 * evidence_summary.size_signal,
        0.25,
        0.95,
    )
    safety_priority = _clamp(
        0.32
        + 0.28 * evidence_summary.fragility_signal
        + 0.18 * (1.0 - attribute_confidence)
        + 0.14 * force_std_pressure
        + 0.08 * evidence_summary.support_contact_confidence,
        0.35,
        0.98,
    )

    required_alignment = (
        evidence_summary.alignment_confidence >= 0.40
        or (features["long_transfer"] and evidence_summary.size_signal >= 0.60)
    )
    required_lift_margin = (
        required_alignment
        or evidence_summary.lift_stage_confidence >= 0.38
        or (evidence_summary.dynamic_load_signal >= 0.75 and evidence_summary.friction_signal >= 0.70)
    )

    missing_specific_force = force_rule_mode == "all" and evidence_summary.specific_force_confidence < 0.35
    missing_numeric_motion = (
        dynamic_transport_mode == "long_transfer"
        and evidence_summary.numeric_motion_confidence < 0.30
    )
    missing_alignment = required_alignment and evidence_summary.alignment_confidence < 0.35
    missing_lift_stage = required_lift_margin and evidence_summary.lift_stage_confidence < 0.30

    coverage_values = [
        evidence_summary.motion_confidence if motion_rule_mode == "all" else 1.0,
        evidence_summary.specific_force_confidence if force_rule_mode == "all" else 1.0,
        evidence_summary.alignment_confidence if required_alignment else 1.0,
        evidence_summary.lift_stage_confidence if required_lift_margin else 1.0,
        evidence_summary.support_contact_confidence if fragility_band == "high" else 1.0,
        evidence_summary.numeric_motion_confidence if dynamic_transport_mode == "long_transfer" else 1.0,
        attribute_confidence,
    ]
    state_coverage = sum(coverage_values) / len(coverage_values)

    reasons: list[str] = []
    if support_score < 2.2:
        reasons.append("low_support_score")
    if conflict_count > 0:
        reasons.append("evidence_conflict")
    if state_coverage < 0.72:
        reasons.append("state_coverage_gap")
    if attribute_confidence < 0.55:
        reasons.append("low_attribute_confidence")
    if evidence_summary.force_std > 5.0:
        reasons.append("high_force_dispersion")
    if dynamic_transport_mode != "static" and evidence_summary.motion_confidence < 0.45:
        reasons.append("low_motion_confidence")
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
        force_std=round(evidence_summary.force_std, 4),
        attribute_confidence=round(attribute_confidence, 4),
        motion_confidence=round(evidence_summary.motion_confidence, 4),
        alignment_confidence=round(evidence_summary.alignment_confidence, 4),
        lift_stage_confidence=round(evidence_summary.lift_stage_confidence, 4),
        missing_specific_force=missing_specific_force,
        missing_numeric_motion=missing_numeric_motion,
        missing_alignment=missing_alignment,
        missing_lift_stage=missing_lift_stage,
        conservative_mode=bool(critical_reasons),
        reasons=reasons,
    )

    object_state = ObjectBeliefState(
        mass_band=mass_band,
        mass_confidence=round(evidence_summary.mass_confidence, 4),
        friction_band=friction_band,
        friction_confidence=round(evidence_summary.friction_confidence, 4),
        fragility_band=fragility_band,
        fragility_confidence=round(evidence_summary.fragility_confidence, 4),
        size_band=size_band,
        size_confidence=round(evidence_summary.size_confidence, 4),
        dynamic_load_band=dynamic_load_band,
        dynamic_load_confidence=round(evidence_summary.dynamic_load_confidence, 4),
        center_of_mass_risk=round(evidence_summary.center_of_mass_risk, 4),
        force_center=round(evidence_summary.force_center, 4),
        force_std=round(evidence_summary.force_std, 4),
        motion_confidence=round(evidence_summary.motion_confidence, 4),
        alignment_confidence=round(evidence_summary.alignment_confidence, 4),
        lift_stage_confidence=round(evidence_summary.lift_stage_confidence, 4),
        support_contact_confidence=round(evidence_summary.support_contact_confidence, 4),
        support_contact_expected=available_support_contact_rules or evidence_summary.support_contact_confidence >= 0.40,
        specific_force_supported=available_specific_force_rules or evidence_summary.specific_force_confidence >= 0.40,
        numeric_motion_supported=available_numeric_motion_rules or evidence_summary.numeric_motion_confidence >= 0.40,
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
            risk_focus=["fragility", "misalignment"] if fragility_band == "high" else ["contact_setup"],
            control_bias="safety" if safety_priority >= 0.7 else "balanced",
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
        evidence_state=evidence_summary,
        object_state=object_state,
        task_constraints=constraints,
        uncertainty=uncertainty,
        stage_plan=stage_plan,
    )


def _normalize_plan(plan: CandidateControlPlan) -> CandidateControlPlan:
    transport_velocity = _clamp(plan.transport_velocity, 0.12, 0.8)
    placement_velocity = _clamp(min(plan.placement_velocity, transport_velocity), 0.12, 0.8)
    gripper_force = _clamp(plan.gripper_force, 5.0, 50.0)
    lift_force = _clamp(max(plan.lift_force, gripper_force), gripper_force, 50.0)
    transfer_force = _clamp(max(plan.transfer_force, gripper_force), gripper_force, 50.0)
    transfer_alignment = _clamp(plan.transfer_alignment, 0.0, 1.0)
    lift_clearance = _clamp(plan.lift_clearance, 0.03, 0.14)
    approach_height = _clamp(plan.approach_height, 0.02, 0.08)
    return CandidateControlPlan(
        label=plan.label,
        gripper_force=round(gripper_force, 4),
        approach_height=round(approach_height, 4),
        transport_velocity=round(transport_velocity, 4),
        lift_force=round(lift_force, 4),
        transfer_force=round(transfer_force, 4),
        placement_velocity=round(placement_velocity, 4),
        transfer_alignment=round(transfer_alignment, 4),
        lift_clearance=round(lift_clearance, 4),
    )


def synthesize_control_seed(
    default_plan: dict[str, float],
    belief: ControlBeliefBundle,
    hints: EvidenceConstraintHints,
) -> tuple[dict[str, float], dict[str, Any]]:
    base = _normalize_plan(CandidateControlPlan(label="belief_seed", **default_plan))
    notes = list(hints.source_notes)

    gripper_force = max(
        base.gripper_force,
        belief.evidence_state.force_center - 0.20 * belief.evidence_state.force_std,
    )
    gripper_force += float(hints.gripper_force_bias)
    if belief.uncertainty.conservative_mode:
        gripper_force += 0.6
        notes.append("seed_uncertainty_force_guard")
    if belief.object_state.friction_band in {"low", "medium_low"}:
        gripper_force += 0.4
        notes.append("seed_low_friction_margin")
    if hints.force_floor is not None:
        gripper_force = max(gripper_force, float(hints.force_floor))
    if hints.force_cap is not None:
        gripper_force = min(gripper_force, float(hints.force_cap))

    approach_height = base.approach_height
    if hints.approach_height_target is not None:
        approach_height = float(hints.approach_height_target)
    elif belief.object_state.fragility_band == "high":
        approach_height = max(approach_height, 0.05)
        notes.append("seed_fragility_height_guard")

    transport_velocity = base.transport_velocity
    if belief.uncertainty.conservative_mode:
        transport_velocity -= 0.02
    if belief.task_constraints.preferred_transport_mode == "long_transfer":
        transport_velocity = min(transport_velocity, 0.22)
        notes.append("seed_long_transfer_velocity_guard")
    elif belief.task_constraints.preferred_transport_mode == "high_speed_low_friction":
        transport_velocity = min(transport_velocity, 0.30)
        notes.append("seed_high_speed_low_friction_cap")
    if hints.transport_velocity_floor is not None:
        transport_velocity = max(transport_velocity, float(hints.transport_velocity_floor))
    if hints.transport_velocity_cap is not None:
        transport_velocity = min(transport_velocity, float(hints.transport_velocity_cap))

    placement_velocity = min(base.placement_velocity, transport_velocity)
    if belief.task_constraints.precision_priority >= 0.75:
        placement_velocity = min(placement_velocity, transport_velocity - 0.01)
    if hints.placement_velocity_cap is not None:
        placement_velocity = min(placement_velocity, float(hints.placement_velocity_cap))

    lift_clearance = base.lift_clearance
    if belief.task_constraints.stability_priority >= 0.75:
        lift_clearance = max(lift_clearance, 0.06)
    if hints.clearance_floor is not None:
        lift_clearance = max(lift_clearance, float(hints.clearance_floor))
    if hints.clearance_target is not None:
        lift_clearance = max(lift_clearance, float(hints.clearance_target))

    transfer_alignment = base.transfer_alignment if belief.task_constraints.required_alignment else 0.0
    if belief.task_constraints.required_alignment:
        transfer_alignment = max(
            transfer_alignment,
            0.55 + 0.25 * belief.evidence_state.alignment_confidence,
        )
        notes.append("seed_alignment_requirement")
    if hints.alignment_target is not None:
        transfer_alignment = max(transfer_alignment, float(hints.alignment_target))

    lift_force = max(gripper_force, base.lift_force)
    transfer_force = max(gripper_force, base.transfer_force)
    if belief.task_constraints.required_lift_margin:
        lift_force = max(lift_force, gripper_force + 0.5)
        notes.append("seed_lift_margin_requirement")
    if belief.object_state.friction_band in {"low", "medium_low"}:
        transfer_force = max(transfer_force, gripper_force + 0.4)
    if hints.lift_force_margin > 0.0:
        lift_force = max(lift_force, gripper_force + float(hints.lift_force_margin))
    if hints.transfer_force_margin > 0.0:
        transfer_force = max(transfer_force, gripper_force + float(hints.transfer_force_margin))

    seed_plan = _normalize_plan(
        CandidateControlPlan(
            label="belief_seed",
            gripper_force=gripper_force,
            approach_height=approach_height,
            transport_velocity=transport_velocity,
            lift_force=lift_force,
            transfer_force=transfer_force,
            placement_velocity=placement_velocity,
            transfer_alignment=transfer_alignment,
            lift_clearance=lift_clearance,
        )
    )
    return {
        "gripper_force": seed_plan.gripper_force,
        "approach_height": seed_plan.approach_height,
        "transport_velocity": seed_plan.transport_velocity,
        "lift_force": seed_plan.lift_force,
        "transfer_force": seed_plan.transfer_force,
        "placement_velocity": seed_plan.placement_velocity,
        "transfer_alignment": seed_plan.transfer_alignment,
        "lift_clearance": seed_plan.lift_clearance,
    }, {
        "seed_mode": "belief_constraint_synthesis",
        "seed_notes": notes,
        "belief_transport_mode": belief.task_constraints.preferred_transport_mode,
        "hints": hints.to_trace_dict(),
        "seed_plan": seed_plan.to_trace_dict(),
    }


def _build_candidate_seeds(
    base: CandidateControlPlan,
    belief: ControlBeliefBundle,
    replan_request: dict[str, Any] | None = None,
) -> list[CandidateControlPlan]:
    candidates = [base]
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
    if replan_request is not None:
        deltas = dict(replan_request.get("param_deltas", {}))
        if deltas:
            candidates.append(
                CandidateControlPlan(
                    label="feedback_stage_guard",
                    gripper_force=base.gripper_force + float(deltas.get("gripper_force", 0.0)),
                    approach_height=base.approach_height + float(deltas.get("approach_height", 0.0)),
                    transport_velocity=base.transport_velocity + float(deltas.get("transport_velocity", 0.0)),
                    lift_force=base.lift_force + float(deltas.get("lift_force", 0.0)),
                    transfer_force=base.transfer_force + float(deltas.get("transfer_force", 0.0)),
                    placement_velocity=base.placement_velocity + float(deltas.get("placement_velocity", 0.0)),
                    transfer_alignment=base.transfer_alignment + float(deltas.get("transfer_alignment", 0.0)),
                    lift_clearance=base.lift_clearance + float(deltas.get("lift_clearance", 0.0)),
                )
            )
    return [_normalize_plan(candidate) for candidate in candidates]


def solve_control_plan(
    base_plan: dict[str, float],
    belief: ControlBeliefBundle,
    replan_request: dict[str, Any] | None = None,
) -> tuple[dict[str, float], dict[str, Any]]:
    base_label = "feedback_seed" if replan_request is not None else "belief_seed"
    base = _normalize_plan(CandidateControlPlan(label=base_label, **base_plan))
    candidates = _build_candidate_seeds(base, belief, replan_request=replan_request)
    notes: list[str] = []

    preferred_transport_cap = base.transport_velocity
    preferred_placement_cap = base.placement_velocity
    preferred_clearance_floor = base.lift_clearance
    preferred_force_floor = max(base.gripper_force, belief.evidence_state.force_center - 0.5 * belief.evidence_state.force_std)
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
    if belief.evidence_state.force_std >= 4.5:
        preferred_force_floor += 0.25
        preferred_clearance_floor += 0.002

    stage_bias = str(replan_request.get("stage_bias", "none")) if replan_request is not None else "none"
    if stage_bias == "lift":
        preferred_lift_force_floor += 0.8
        preferred_clearance_floor += 0.004
        preferred_transport_cap = min(preferred_transport_cap, max(0.12, base.transport_velocity - 0.02))
    elif stage_bias == "transfer":
        preferred_transfer_force_floor += 0.8
        preferred_alignment_floor = max(preferred_alignment_floor, base.transfer_alignment + 0.08)
        preferred_transport_cap = min(preferred_transport_cap, max(0.12, base.transport_velocity - 0.03))
    elif stage_bias == "place":
        preferred_placement_cap = min(preferred_placement_cap, max(0.12, base.placement_velocity - 0.03))
        preferred_clearance_floor += 0.003

    def _score(plan: CandidateControlPlan) -> tuple[float, dict[str, float]]:
        normalized = _normalize_plan(plan)
        breakdown = {
            "force_shortfall": max(0.0, preferred_force_floor - normalized.gripper_force) * 3.0,
            "force_overdrive": max(0.0, normalized.gripper_force - preferred_force_cap) * 1.5,
            "lift_shortfall": max(0.0, preferred_lift_force_floor - normalized.lift_force) * 2.8,
            "transfer_shortfall": max(0.0, preferred_transfer_force_floor - normalized.transfer_force) * 2.6,
            "transport_overspeed": max(0.0, normalized.transport_velocity - preferred_transport_cap) * 18.0,
            "placement_overspeed": max(0.0, normalized.placement_velocity - preferred_placement_cap) * 22.0,
            "clearance_shortfall": max(0.0, preferred_clearance_floor - normalized.lift_clearance) * 40.0,
            "alignment_shortfall": max(0.0, preferred_alignment_floor - normalized.transfer_alignment) * 6.0,
            "stage_inconsistency": max(0.0, normalized.placement_velocity - normalized.transport_velocity) * 6.0,
        }
        if belief.uncertainty.conservative_mode and normalized.label == "speed_bias":
            breakdown["uncertainty_speed_penalty"] = 2.0
        if stage_bias == "lift":
            breakdown["feedback_lift_penalty"] = max(0.0, preferred_lift_force_floor - normalized.lift_force) * 1.2
        elif stage_bias == "transfer":
            breakdown["feedback_transfer_penalty"] = max(0.0, preferred_alignment_floor - normalized.transfer_alignment) * 1.5
        elif stage_bias == "place":
            breakdown["feedback_place_penalty"] = max(0.0, normalized.placement_velocity - preferred_placement_cap) * 4.0
        return sum(breakdown.values()), breakdown

    candidate_scores: list[dict[str, Any]] = []
    best_seed = base
    best_seed_score = float("inf")
    best_seed_breakdown: dict[str, float] = {}
    for candidate in candidates:
        score, breakdown = _score(candidate)
        candidate_scores.append(
            {
                "label": candidate.label,
                "score": round(score, 4),
                "breakdown": {key: round(value, 4) for key, value in breakdown.items()},
            }
        )
        if score < best_seed_score:
            best_seed = candidate
            best_seed_score = score
            best_seed_breakdown = breakdown

    current = best_seed
    current_score = best_seed_score
    current_breakdown = best_seed_breakdown
    step_sizes = {
        "gripper_force": 0.6,
        "lift_force": 0.5,
        "transfer_force": 0.5,
        "approach_height": 0.002,
        "transport_velocity": 0.015,
        "placement_velocity": 0.015,
        "transfer_alignment": 0.04,
        "lift_clearance": 0.003,
    }
    min_steps = {
        "gripper_force": 0.12,
        "lift_force": 0.12,
        "transfer_force": 0.12,
        "approach_height": 0.0008,
        "transport_velocity": 0.003,
        "placement_velocity": 0.003,
        "transfer_alignment": 0.01,
        "lift_clearance": 0.0008,
    }
    tunable_fields = list(step_sizes.keys())
    search_trace: list[dict[str, Any]] = []
    max_iterations = 18

    for iteration in range(1, max_iterations + 1):
        best_trial: CandidateControlPlan | None = None
        best_trial_score = current_score
        best_trial_breakdown: dict[str, float] | None = None
        best_field = ""
        best_delta = 0.0

        for field_name in tunable_fields:
            base_value = getattr(current, field_name)
            for direction in (-1.0, 1.0):
                delta = step_sizes[field_name] * direction
                trial = CandidateControlPlan(
                    **{
                        **asdict(current),
                        "label": f"local_search:{field_name}:{'up' if direction > 0 else 'down'}",
                        field_name: base_value + delta,
                    }
                )
                trial = _normalize_plan(trial)
                score, breakdown = _score(trial)
                if score + 1e-6 < best_trial_score:
                    best_trial = trial
                    best_trial_score = score
                    best_trial_breakdown = breakdown
                    best_field = field_name
                    best_delta = delta

        if best_trial is None:
            reduced = False
            for field_name in tunable_fields:
                if step_sizes[field_name] > min_steps[field_name]:
                    step_sizes[field_name] = max(min_steps[field_name], round(step_sizes[field_name] * 0.6, 4))
                    reduced = True
            search_trace.append(
                {
                    "iteration": iteration,
                    "accepted": False,
                    "step_sizes": {key: round(value, 4) for key, value in step_sizes.items()},
                }
            )
            if not reduced:
                break
            continue

        current = best_trial
        current_score = best_trial_score
        current_breakdown = best_trial_breakdown or {}
        search_trace.append(
            {
                "iteration": iteration,
                "accepted": True,
                "field": best_field,
                "delta": round(best_delta, 4),
                "score": round(current_score, 4),
                "plan": current.to_trace_dict(),
            }
        )

    solved_plan = {
        "gripper_force": current.gripper_force,
        "approach_height": current.approach_height,
        "transport_velocity": current.transport_velocity,
        "lift_clearance": current.lift_clearance,
        "lift_force": current.lift_force,
        "transfer_force": current.transfer_force,
        "placement_velocity": current.placement_velocity,
        "transfer_alignment": current.transfer_alignment,
    }
    if current.label != base.label:
        notes.append(f"solver_selected:{current.label}")
    if belief.uncertainty.conservative_mode:
        notes.append("solver_conservative_mode")
    if current_score + 1e-6 < best_seed_score:
        notes.append("solver_local_search_improved")
    if replan_request is not None:
        notes.append(f"solver_feedback_stage:{stage_bias}")

    return solved_plan, {
        "solver_mode": "belief_seeded_local_search",
        "solver_selected_candidate": current.label,
        "solver_selected_score": round(current_score, 4),
        "solver_score_breakdown": {key: round(value, 4) for key, value in current_breakdown.items()},
        "solver_candidate_scores": candidate_scores,
        "solver_seed_candidate": best_seed.label,
        "solver_seed_score": round(best_seed_score, 4),
        "solver_local_search_iterations": len(search_trace),
        "solver_local_search_improvement": round(max(0.0, best_seed_score - current_score), 4),
        "solver_local_search_trace": search_trace,
        "solver_adjustment_notes": notes,
    }


def control_belief_from_trace(params: dict[str, Any]) -> ControlBeliefBundle:
    evidence_state_raw = dict(params.get("evidence_state_summary", {}))
    if not evidence_state_raw:
        belief_state_raw = dict(params.get("belief_state", {}))
        uncertainty_raw = dict(params.get("uncertainty_profile", {}))
        evidence_state_raw = {
            "mass_signal": 0.82 if belief_state_raw.get("mass_band") == "heavy" else 0.65 if belief_state_raw.get("mass_band") == "large" else 0.25,
            "mass_confidence": float(belief_state_raw.get("mass_confidence", uncertainty_raw.get("attribute_confidence", 0.4))),
            "friction_signal": 0.9 if belief_state_raw.get("friction_band") == "low" else 0.65 if belief_state_raw.get("friction_band") == "medium_low" else 0.1 if belief_state_raw.get("friction_band") == "high" else 0.45,
            "friction_confidence": float(belief_state_raw.get("friction_confidence", uncertainty_raw.get("attribute_confidence", 0.4))),
            "fragility_signal": 0.9 if belief_state_raw.get("fragility_band") == "high" else 0.55 if belief_state_raw.get("fragility_band") == "medium" else 0.25,
            "fragility_confidence": float(belief_state_raw.get("fragility_confidence", uncertainty_raw.get("attribute_confidence", 0.4))),
            "size_signal": 0.86 if belief_state_raw.get("size_band") == "large" else 0.18 if belief_state_raw.get("size_band") == "small" else 0.5,
            "size_confidence": float(belief_state_raw.get("size_confidence", uncertainty_raw.get("attribute_confidence", 0.4))),
            "dynamic_load_signal": 0.86 if belief_state_raw.get("dynamic_load_band") == "high" else 0.55 if belief_state_raw.get("dynamic_load_band") == "medium" else 0.25,
            "dynamic_load_confidence": float(belief_state_raw.get("dynamic_load_confidence", uncertainty_raw.get("attribute_confidence", 0.4))),
            "center_of_mass_risk": float(belief_state_raw.get("center_of_mass_risk", 0.4)),
            "force_center": float(belief_state_raw.get("force_center", params.get("gripper_force", 25.0))),
            "force_std": float(belief_state_raw.get("force_std", uncertainty_raw.get("force_std", 4.0))),
            "motion_confidence": float(belief_state_raw.get("motion_confidence", uncertainty_raw.get("motion_confidence", 0.4))),
            "numeric_motion_confidence": float(1.0 if belief_state_raw.get("numeric_motion_supported") else 0.25),
            "alignment_confidence": float(belief_state_raw.get("alignment_confidence", uncertainty_raw.get("alignment_confidence", 0.3))),
            "lift_stage_confidence": float(belief_state_raw.get("lift_stage_confidence", uncertainty_raw.get("lift_stage_confidence", 0.3))),
            "support_contact_confidence": float(belief_state_raw.get("support_contact_confidence", 0.25)),
            "specific_force_confidence": float(1.0 if belief_state_raw.get("specific_force_supported") else 0.25),
            "preferred_transport_mode": str(params.get("dynamic_transport_mode", params.get("task_constraints", {}).get("preferred_transport_mode", "static"))),
            "term_supports": {},
        }
    evidence_state = EvidenceStateSummary(**evidence_state_raw)
    object_state = ObjectBeliefState(**dict(params.get("belief_state", {})))
    task_constraints = TaskConstraintSet(**dict(params.get("task_constraints", {})))
    uncertainty = UncertaintyProfile(**dict(params.get("uncertainty_profile", {})))
    stage_plan = [StageIntent(**stage) for stage in params.get("stage_plan", [])]
    return ControlBeliefBundle(
        evidence_state=evidence_state,
        object_state=object_state,
        task_constraints=task_constraints,
        uncertainty=uncertainty,
        stage_plan=stage_plan,
    )


def update_belief_with_feedback(
    belief: ControlBeliefBundle,
    replan_request: dict[str, Any],
) -> tuple[ControlBeliefBundle, dict[str, Any]]:
    updated = ControlBeliefBundle(
        evidence_state=EvidenceStateSummary(**asdict(belief.evidence_state)),
        object_state=ObjectBeliefState(**asdict(belief.object_state)),
        task_constraints=TaskConstraintSet(**asdict(belief.task_constraints)),
        uncertainty=UncertaintyProfile(**asdict(belief.uncertainty)),
        stage_plan=_clone_stage_plan(belief.stage_plan),
    )
    stage_bias = str(replan_request.get("stage_bias", "none"))
    added_reasons = list(replan_request.get("uncertainty_reasons", []))
    updated.task_constraints.replan_stage_bias = stage_bias
    updated.uncertainty.reasons = list(updated.uncertainty.reasons)
    for reason in added_reasons:
        if reason not in updated.uncertainty.reasons:
            updated.uncertainty.reasons.append(reason)
    updated.uncertainty.conservative_mode = True
    updated.uncertainty.state_coverage = round(max(0.25, updated.uncertainty.state_coverage - 0.08), 4)
    updated.uncertainty.attribute_confidence = round(max(0.20, updated.uncertainty.attribute_confidence - 0.05), 4)
    updated.uncertainty.support_score = round(max(0.0, updated.uncertainty.support_score - 0.05), 4)

    if stage_bias == "lift":
        updated.task_constraints.stability_priority = round(max(updated.task_constraints.stability_priority, 0.97), 4)
        updated.task_constraints.safety_priority = round(max(updated.task_constraints.safety_priority, 0.93), 4)
        updated.task_constraints.required_lift_margin = True
        updated.object_state.center_of_mass_risk = round(min(1.0, updated.object_state.center_of_mass_risk + 0.05), 4)
        updated.object_state.lift_stage_confidence = round(max(0.15, updated.object_state.lift_stage_confidence - 0.08), 4)
        updated.evidence_state.lift_stage_confidence = updated.object_state.lift_stage_confidence
    elif stage_bias == "transfer":
        updated.task_constraints.stability_priority = round(max(updated.task_constraints.stability_priority, 0.97), 4)
        updated.task_constraints.required_alignment = True
        updated.object_state.alignment_confidence = round(max(0.15, updated.object_state.alignment_confidence - 0.08), 4)
        updated.object_state.motion_confidence = round(max(0.15, updated.object_state.motion_confidence - 0.04), 4)
        updated.evidence_state.alignment_confidence = updated.object_state.alignment_confidence
        updated.evidence_state.motion_confidence = updated.object_state.motion_confidence
        updated.uncertainty.alignment_confidence = updated.object_state.alignment_confidence
    elif stage_bias == "place":
        updated.task_constraints.precision_priority = round(max(updated.task_constraints.precision_priority, 0.97), 4)
        updated.task_constraints.safety_priority = round(max(updated.task_constraints.safety_priority, 0.90), 4)
        updated.object_state.motion_confidence = round(max(0.15, updated.object_state.motion_confidence - 0.04), 4)
        updated.evidence_state.motion_confidence = updated.object_state.motion_confidence

    for stage in updated.stage_plan:
        if stage.name == stage_bias:
            stage.control_bias = "feedback_replan"
            if "feedback_replan" not in stage.risk_focus:
                stage.risk_focus.append("feedback_replan")

    return updated, {
        "stage_bias": stage_bias,
        "added_uncertainty_reasons": added_reasons,
        "updated_state_coverage": updated.uncertainty.state_coverage,
        "updated_attribute_confidence": updated.uncertainty.attribute_confidence,
        "updated_stage_plan": [asdict(stage) for stage in updated.stage_plan],
    }


def replan_control_plan(
    previous_params: dict[str, Any],
    replan_request: dict[str, Any],
) -> dict[str, Any]:
    belief = control_belief_from_trace(previous_params)
    updated_belief, belief_update_trace = update_belief_with_feedback(belief, replan_request)

    base_plan = {
        "gripper_force": float(previous_params.get("gripper_force", 25.0)),
        "approach_height": float(previous_params.get("approach_height", 0.05)),
        "transport_velocity": float(previous_params.get("transport_velocity", 0.30)),
        "lift_force": float(previous_params.get("lift_force", previous_params.get("gripper_force", 25.0))),
        "transfer_force": float(previous_params.get("transfer_force", previous_params.get("gripper_force", 25.0))),
        "placement_velocity": float(previous_params.get("placement_velocity", previous_params.get("transport_velocity", 0.30))),
        "transfer_alignment": float(previous_params.get("transfer_alignment", 0.0)),
        "lift_clearance": float(previous_params.get("lift_clearance", 0.06)),
    }
    for key, delta in dict(replan_request.get("param_deltas", {})).items():
        if key in base_plan:
            base_plan[key] = float(base_plan[key]) + float(delta)
    base_plan["placement_velocity"] = min(base_plan["placement_velocity"], base_plan["transport_velocity"])

    solved_plan, solver_trace = solve_control_plan(base_plan, updated_belief, replan_request=replan_request)
    next_params = dict(previous_params)
    for key, value in solved_plan.items():
        next_params[key] = round(float(value), 4 if key != "transfer_alignment" else 4)

    trace = updated_belief.to_trace_dict()
    next_params["evidence_state_summary"] = trace["evidence_state_summary"]
    next_params["belief_state"] = trace["belief_state"]
    next_params["task_constraints"] = trace["task_constraints"]
    next_params["uncertainty_profile"] = trace["uncertainty_profile"]
    next_params["stage_plan"] = trace["stage_plan"]
    next_params["belief_state_coverage"] = trace["belief_state_coverage"]
    next_params["uncertainty_conservative_mode"] = trace["uncertainty_conservative_mode"]
    next_params["uncertainty_reasons"] = trace["uncertainty_reasons"]
    next_params["solver_mode"] = solver_trace["solver_mode"]
    next_params["solver_selected_candidate"] = solver_trace["solver_selected_candidate"]
    next_params["solver_selected_score"] = solver_trace["solver_selected_score"]
    next_params["solver_score_breakdown"] = solver_trace["solver_score_breakdown"]
    next_params["solver_candidate_scores"] = solver_trace["solver_candidate_scores"]
    next_params["solver_seed_candidate"] = solver_trace["solver_seed_candidate"]
    next_params["solver_seed_score"] = solver_trace["solver_seed_score"]
    next_params["solver_local_search_iterations"] = solver_trace["solver_local_search_iterations"]
    next_params["solver_local_search_improvement"] = solver_trace["solver_local_search_improvement"]
    next_params["solver_local_search_trace"] = solver_trace["solver_local_search_trace"]
    next_params["solver_adjustment_notes"] = list(solver_trace["solver_adjustment_notes"])

    next_params["feedback_adjusted"] = True
    next_params["feedback_adjustment_type"] = str(replan_request.get("suggestion", "replan"))
    next_params["feedback_stage_adjustments"] = [str(replan_request.get("stage_bias", "none"))]
    next_params["feedback_replan_trace"] = {
        "failure_bucket": replan_request.get("failure_bucket", "unknown_failure"),
        "stage_bias": replan_request.get("stage_bias", "none"),
        "observation_index": replan_request.get("observation_index"),
        "observation_stage": replan_request.get("observation_stage"),
        "trigger_reason": replan_request.get("trigger_reason"),
        "failure_attribution": replan_request.get("failure_attribution", {}),
        "updated_uncertainty_reasons": trace["uncertainty_reasons"],
        "belief_update": belief_update_trace,
        "seed_plan": {key: round(float(value), 4) for key, value in base_plan.items()},
        "final_plan": {key: round(float(value), 4) for key, value in solved_plan.items()},
        "param_deltas": {
            key: round(float(solved_plan[key]) - float(previous_params.get(key, 0.0)), 4)
            for key in solved_plan
        },
    }

    uncertainty_std = min(
        0.35,
        0.04
        + 0.12 * max(0.0, 1.0 - float(trace["belief_state_coverage"]))
        + 0.03 * float(trace["uncertainty_profile"]["conflict_count"])
        + (0.04 if trace["uncertainty_conservative_mode"] else 0.0),
    )
    confidence = max(
        0.05,
        min(
            1.0,
            0.55
            + 0.08 * float(trace["belief_state_coverage"])
            - (0.06 if trace["uncertainty_conservative_mode"] else 0.0),
        ),
    )
    next_params["uncertainty_std"] = round(uncertainty_std, 4)
    next_params["confidence"] = round(confidence, 3)
    return next_params
