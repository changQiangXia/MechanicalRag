# Heavy Load Counterfactual Diagnosis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn heavy static failures from projected observer + full-plan reroll into posterior diagnosis + suffix repair, with `pick_metal_heavy >= 0.30`, `pick_metal_heavy_fast` dropping by no more than `0.05`, and the 12-task mean dropping by no more than `0.01`.

**Architecture:** Keep the existing public simulation entrypoints, but change their semantics. `simulation/control_core.py` grows typed execution-belief and counterfactual-diagnosis primitives, `simulation/env.py` becomes a true phase executor with frozen prefixes and suffix-only repair, `simulation/feedback.py` becomes a compatibility shim around the new diagnosis path, and `simulation/runner.py` starts serializing online-diagnosis metrics and richer traces. The first rollout only enables the new path in heavy-static regimes.

**Tech Stack:** Python 3.10, `unittest`, existing simulation modules, JSON benchmark artifacts, shell verification commands, no new third-party dependencies.

---

## File Structure

### Files to modify

- `simulation/control_core.py`
  Purpose: add `PhaseObservation`, `ExecutionBelief`, `CounterfactualIntervention`, `CounterfactualDiagnosis`, plus prior/posterior update, diagnosis, and suffix-repair functions.
- `simulation/env.py`
  Purpose: replace projected observer semantics with phase-by-phase execution, prefix freezing, observation traces, and suffix-only replan flow.
- `simulation/feedback.py`
  Purpose: convert old feedback requests into the new `PhaseObservation` / counterfactual repair path while preserving compatibility callers.
- `simulation/rag_controller.py`
  Purpose: route `rag_feedback` observation-time replans through the new diagnosis path and keep trial-level retry as fallback.
- `simulation/runner.py`
  Purpose: aggregate online diagnosis metrics, serialize richer trial records, and expose heavy-load diagnosis distributions in task summaries.
- `tests/test_control_core.py`
  Purpose: lock in prior/posterior latent updates and minimal-sufficient intervention selection.
- `tests/test_adaptive_execution.py`
  Purpose: keep the existing adaptive-execution smoke tests aligned with the new trace names and feedback modes.
- `tests/test_benchmark_schema.py`
  Purpose: lock in benchmark serialization of diagnosis counts, intervention distributions, and nested method payloads.
- `README.md`
  Purpose: describe the new phase-observation / suffix-repair semantics and updated heavy-load results.
- `docs/overview.md`
  Purpose: mirror the new heavy-load diagnosis semantics and benchmark acceptance criteria.
- `simulation/README.md`
  Purpose: document the new execution traces and current-profile verification commands.

### Files to create

- `tests/test_counterfactual_execution.py`
  Purpose: prove `simulate_stepwise_execution()` freezes executed prefixes and only repairs the remaining suffix.
- `tests/test_heavy_load_regression.py`
  Purpose: pin the benchmark acceptance gates and give one place to store the current baseline constants for heavy-load verification.

### Generated artifacts to refresh

- `outputs/current_observer_step_replan/simulation_benchmark_result.json`
- `outputs/current_observer_step_replan/simulation_benchmark_trial_records.json`
- `outputs/current_observer_step_replan/simulation_comparison_rag_vs_baseline.json`
- `outputs/current_observer_step_replan/simulation_comparison_multi_seed.json`
- `outputs/current_observer_step_replan/showcase_summary.txt`
- `outputs/current_observer_step_replan/visualizations/*`

## Task 1: Add Typed Execution-Belief and Posterior Update Primitives

**Files:**
- Modify: `simulation/control_core.py`
- Test: `tests/test_control_core.py`

- [ ] **Step 1: Write the failing prior/posterior tests**

Add these imports and tests to `tests/test_control_core.py`:

```python
from simulation.control_core import (
    PhaseObservation,
    apply_phase_observation,
    build_execution_prior,
)

    def test_build_execution_prior_marks_heavy_static_case(self):
        prior = build_execution_prior(
            {
                "gripper_force": 42.0,
                "lift_force": 42.0,
                "transfer_force": 42.0,
                "transport_velocity": 0.22,
                "placement_velocity": 0.18,
                "lift_clearance": 0.06,
                "belief_state": {
                    "mass_band": "heavy",
                    "dynamic_load_band": "high",
                    "center_of_mass_risk": 0.72,
                    "motion_confidence": 0.32,
                    "alignment_confidence": 0.34,
                    "lift_stage_confidence": 0.41,
                },
                "task_constraints": {
                    "preferred_transport_mode": "static",
                },
                "uncertainty_profile": {
                    "support_score": 3.8,
                    "state_coverage": 0.74,
                    "conflict_count": 0,
                    "force_std": 5.0,
                    "attribute_confidence": 0.66,
                    "motion_confidence": 0.32,
                    "alignment_confidence": 0.34,
                    "lift_stage_confidence": 0.41,
                    "missing_specific_force": False,
                    "missing_numeric_motion": False,
                    "missing_alignment": False,
                    "missing_lift_stage": False,
                    "conservative_mode": False,
                    "reasons": [],
                },
            },
            phase="lift",
        )
        self.assertEqual(prior.phase, "lift")
        self.assertLess(prior.load_support_margin, 0.0)
        self.assertLess(prior.lift_reserve, 0.0)
        self.assertLess(prior.transfer_disturbance, 0.4)

    def test_phase_observation_updates_heavy_load_latents(self):
        prior = build_execution_prior(
            {
                "gripper_force": 42.0,
                "lift_force": 42.0,
                "transfer_force": 42.0,
                "transport_velocity": 0.22,
                "placement_velocity": 0.18,
                "lift_clearance": 0.06,
                "belief_state": {
                    "mass_band": "heavy",
                    "dynamic_load_band": "high",
                    "center_of_mass_risk": 0.72,
                    "motion_confidence": 0.32,
                    "alignment_confidence": 0.34,
                    "lift_stage_confidence": 0.41,
                },
                "task_constraints": {
                    "preferred_transport_mode": "static",
                },
                "uncertainty_profile": {
                    "support_score": 3.8,
                    "state_coverage": 0.74,
                    "conflict_count": 0,
                    "force_std": 5.0,
                    "attribute_confidence": 0.66,
                    "motion_confidence": 0.32,
                    "alignment_confidence": 0.34,
                    "lift_stage_confidence": 0.41,
                    "missing_specific_force": False,
                    "missing_numeric_motion": False,
                    "missing_alignment": False,
                    "missing_lift_stage": False,
                    "conservative_mode": False,
                    "reasons": [],
                },
            },
            phase="lift",
        )
        observation = PhaseObservation(
            phase="lift",
            payload_ratio_obs=0.98,
            lift_progress_obs=0.46,
            lift_reserve_obs=-0.24,
            tilt_obs=0.28,
            observation_confidence=0.84,
            trigger_reason="lift_reserve_obs",
        )

        posterior, trace = apply_phase_observation(prior, observation)

        self.assertEqual(posterior.phase, "lift")
        self.assertLess(posterior.load_support_margin, prior.load_support_margin)
        self.assertLess(posterior.lift_reserve, prior.lift_reserve)
        self.assertGreater(posterior.pose_alignment_error, prior.pose_alignment_error)
        self.assertEqual(trace["phase"], "lift")
        self.assertEqual(trace["updated_latents"][0]["name"], "load_support_margin")
```

- [ ] **Step 2: Run the targeted control-core tests and confirm failure**

Run:

```bash
PYTHONPATH=. ./venv/bin/python -m unittest tests.test_control_core.ControlCoreTest.test_build_execution_prior_marks_heavy_static_case -v
PYTHONPATH=. ./venv/bin/python -m unittest tests.test_control_core.ControlCoreTest.test_phase_observation_updates_heavy_load_latents -v
```

Expected: `ERROR` or `FAIL` because `PhaseObservation`, `build_execution_prior`, and `apply_phase_observation` do not exist yet.

- [ ] **Step 3: Add the typed belief and observation primitives in `simulation/control_core.py`**

Add these dataclasses and functions near the existing belief dataclasses:

```python
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


def build_execution_prior(params: dict[str, Any], *, phase: str) -> ExecutionBelief:
    belief_state = dict(params.get("belief_state", {}))
    uncertainty = dict(params.get("uncertainty_profile", {}))
    mass_band = str(belief_state.get("mass_band", "light"))
    dynamic_load_band = str(belief_state.get("dynamic_load_band", "low"))
    load_support_margin = -0.18 if mass_band == "heavy" else -0.05 if dynamic_load_band == "high" else 0.12
    lift_reserve = -0.12 if mass_band == "heavy" else 0.08
    return ExecutionBelief(
        phase=phase,
        load_support_margin=load_support_margin,
        load_support_uncertainty=0.22,
        grip_hold_margin=0.14 - 0.04 * float(belief_state.get("center_of_mass_risk", 0.4)),
        grip_hold_uncertainty=0.18,
        pose_alignment_error=0.16 + 0.12 * (1.0 - float(belief_state.get("alignment_confidence", 0.3))),
        pose_alignment_uncertainty=0.20,
        lift_reserve=lift_reserve,
        lift_reserve_uncertainty=0.24,
        transfer_disturbance=0.22 if str(params.get("dynamic_transport_mode", "static")) == "static" else 0.48,
        transfer_disturbance_uncertainty=0.18,
        preferred_transport_mode=str(params.get("dynamic_transport_mode", params.get("task_constraints", {}).get("preferred_transport_mode", "static"))),
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
        updated.load_support_margin = round(updated.load_support_margin - weight * max(0.0, observation.payload_ratio_obs - 0.75) * 0.30, 4)
        updated.lift_reserve = round(updated.lift_reserve + weight * observation.lift_reserve_obs, 4)
        updated.pose_alignment_error = round(updated.pose_alignment_error + weight * abs(observation.tilt_obs) * 0.35, 4)
        updates.extend(
            [
                {"name": "load_support_margin", "value": updated.load_support_margin},
                {"name": "lift_reserve", "value": updated.lift_reserve},
                {"name": "pose_alignment_error", "value": updated.pose_alignment_error},
            ]
        )
    elif observation.phase == "grasp":
        updated.grip_hold_margin = round(updated.grip_hold_margin + weight * observation.contact_stability_obs - weight * observation.micro_slip_obs, 4)
        updates.append({"name": "grip_hold_margin", "value": updated.grip_hold_margin})
    elif observation.phase == "transfer":
        updated.transfer_disturbance = round(updated.transfer_disturbance + weight * (observation.sway_obs + observation.velocity_stress_obs) * 0.25, 4)
        updates.append({"name": "transfer_disturbance", "value": updated.transfer_disturbance})

    return updated, {
        "phase": observation.phase,
        "trigger_reason": observation.trigger_reason,
        "updated_latents": updates,
        "posterior": updated.to_trace_dict(),
    }
```

- [ ] **Step 4: Re-run the targeted control-core tests**

Run:

```bash
PYTHONPATH=. ./venv/bin/python -m unittest tests.test_control_core.ControlCoreTest.test_build_execution_prior_marks_heavy_static_case -v
PYTHONPATH=. ./venv/bin/python -m unittest tests.test_control_core.ControlCoreTest.test_phase_observation_updates_heavy_load_latents -v
```

Expected: both commands report `OK`.

- [ ] **Step 5: Commit the typed-belief groundwork**

```bash
git add simulation/control_core.py tests/test_control_core.py
git commit -m "Add execution belief and phase observation primitives"
```

## Task 2: Add Heavy-Static Counterfactual Diagnosis and Minimal Sufficient Repair

**Files:**
- Modify: `simulation/control_core.py`
- Test: `tests/test_control_core.py`

- [ ] **Step 1: Write the failing diagnosis and repair tests**

Append these tests to `tests/test_control_core.py`:

```python
from simulation.control_core import (
    CounterfactualDiagnosis,
    CounterfactualIntervention,
    diagnose_failure_cause,
    repair_suffix_plan,
)

    def test_diagnosis_prefers_minimal_sufficient_lift_intervention(self):
        belief = build_execution_prior(
            {
                "gripper_force": 42.0,
                "lift_force": 42.0,
                "transfer_force": 42.0,
                "transport_velocity": 0.22,
                "placement_velocity": 0.18,
                "lift_clearance": 0.06,
                "belief_state": {
                    "mass_band": "heavy",
                    "dynamic_load_band": "high",
                    "center_of_mass_risk": 0.72,
                    "alignment_confidence": 0.34,
                },
                "task_constraints": {"preferred_transport_mode": "static"},
                "uncertainty_profile": {
                    "support_score": 3.8,
                    "state_coverage": 0.74,
                    "conflict_count": 0,
                    "force_std": 5.0,
                    "attribute_confidence": 0.66,
                    "motion_confidence": 0.32,
                    "alignment_confidence": 0.34,
                    "lift_stage_confidence": 0.41,
                    "missing_specific_force": False,
                    "missing_numeric_motion": False,
                    "missing_alignment": False,
                    "missing_lift_stage": False,
                    "conservative_mode": False,
                    "reasons": [],
                },
            },
            phase="lift",
        )
        observation = PhaseObservation(
            phase="lift",
            payload_ratio_obs=0.98,
            lift_progress_obs=0.46,
            lift_reserve_obs=-0.24,
            tilt_obs=0.14,
            observation_confidence=0.84,
            trigger_reason="lift_reserve_obs",
        )
        diagnosis = diagnose_failure_cause(
            belief,
            observation,
            {
                "gripper_force": 42.0,
                "lift_force": 42.0,
                "transfer_force": 42.0,
                "transport_velocity": 0.22,
                "placement_velocity": 0.18,
                "lift_clearance": 0.06,
            },
        )
        self.assertEqual(diagnosis.diagnosed_cause, "under_supported_load")
        self.assertEqual(diagnosis.selected_intervention.label, "lift_force_up")
        repaired = repair_suffix_plan(
            {
                "gripper_force": 42.0,
                "lift_force": 42.0,
                "transfer_force": 42.0,
                "transport_velocity": 0.22,
                "placement_velocity": 0.18,
                "lift_clearance": 0.06,
            },
            current_phase="lift",
            diagnosis=diagnosis,
        )
        self.assertEqual(repaired["suffix_replan_start_stage"], "lift")
        self.assertGreater(repaired["lift_force"], 42.0)
        self.assertEqual(repaired["transport_velocity"], 0.22)

    def test_diagnosis_uses_smallest_combination_when_single_knob_cannot_flip(self):
        belief = ExecutionBelief(
            phase="lift",
            load_support_margin=-0.28,
            load_support_uncertainty=0.18,
            grip_hold_margin=0.06,
            grip_hold_uncertainty=0.14,
            pose_alignment_error=0.26,
            pose_alignment_uncertainty=0.18,
            lift_reserve=-0.22,
            lift_reserve_uncertainty=0.20,
            transfer_disturbance=0.24,
            transfer_disturbance_uncertainty=0.16,
            preferred_transport_mode="static",
            conservative_mode=True,
        )
        observation = PhaseObservation(
            phase="lift",
            payload_ratio_obs=1.02,
            lift_progress_obs=0.40,
            lift_reserve_obs=-0.28,
            tilt_obs=0.22,
            observation_confidence=0.88,
            trigger_reason="lift_reserve_obs",
        )
        diagnosis = diagnose_failure_cause(
            belief,
            observation,
            {
                "gripper_force": 42.0,
                "lift_force": 42.0,
                "transfer_force": 42.0,
                "transport_velocity": 0.22,
                "placement_velocity": 0.18,
                "lift_clearance": 0.06,
            },
        )
        self.assertEqual(diagnosis.selected_intervention.label, "lift_force_plus_clearance")
        self.assertGreater(diagnosis.predicted_flip_gain, 0.0)
```

- [ ] **Step 2: Run the new diagnosis tests and confirm failure**

Run:

```bash
PYTHONPATH=. ./venv/bin/python -m unittest tests.test_control_core.ControlCoreTest.test_diagnosis_prefers_minimal_sufficient_lift_intervention -v
PYTHONPATH=. ./venv/bin/python -m unittest tests.test_control_core.ControlCoreTest.test_diagnosis_uses_smallest_combination_when_single_knob_cannot_flip -v
```

Expected: `ERROR` because `CounterfactualIntervention`, `CounterfactualDiagnosis`, `diagnose_failure_cause`, and `repair_suffix_plan` do not exist yet.

- [ ] **Step 3: Implement minimal-sufficient diagnosis and repair in `simulation/control_core.py`**

Add these definitions under the new execution-belief code:

```python
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


def _heavy_static_candidates() -> list[CounterfactualIntervention]:
    return [
        CounterfactualIntervention("gripper_force_up", {"gripper_force": 1.2}, ["grip_hold_margin", "load_support_margin"], 0.0),
        CounterfactualIntervention("lift_force_up", {"lift_force": 1.6}, ["load_support_margin", "lift_reserve"], 0.0),
        CounterfactualIntervention("transfer_force_up", {"transfer_force": 1.2}, ["load_support_margin"], 0.0),
        CounterfactualIntervention("transport_velocity_down", {"transport_velocity": -0.02, "placement_velocity": -0.02}, ["lift_reserve"], 0.0),
        CounterfactualIntervention("lift_clearance_up", {"lift_clearance": 0.006}, ["lift_reserve"], 0.0),
        CounterfactualIntervention("lift_force_plus_velocity", {"lift_force": 1.4, "transport_velocity": -0.02, "placement_velocity": -0.02}, ["load_support_margin", "lift_reserve"], 0.0),
        CounterfactualIntervention("lift_force_plus_clearance", {"lift_force": 1.2, "lift_clearance": 0.006}, ["load_support_margin", "lift_reserve"], 0.0),
        CounterfactualIntervention("gripper_plus_lift", {"gripper_force": 1.0, "lift_force": 1.2}, ["grip_hold_margin", "load_support_margin"], 0.0),
    ]


def _predicted_latent_gains(belief: ExecutionBelief, deltas: dict[str, float]) -> dict[str, float]:
    return {
        "load_support_margin": 0.06 * float(deltas.get("lift_force", 0.0)) + 0.04 * float(deltas.get("gripper_force", 0.0)) + 0.03 * float(deltas.get("transfer_force", 0.0)),
        "lift_reserve": 0.08 * float(deltas.get("lift_force", 0.0)) + 0.04 * float(deltas.get("lift_clearance", 0.0)) + 0.18 * max(0.0, -float(deltas.get("transport_velocity", 0.0))),
        "grip_hold_margin": 0.07 * float(deltas.get("gripper_force", 0.0)),
    }


def diagnose_failure_cause(
    belief: ExecutionBelief,
    observation: PhaseObservation,
    current_plan: dict[str, float],
) -> CounterfactualDiagnosis:
    if belief.load_support_margin <= belief.grip_hold_margin and belief.load_support_margin <= belief.lift_reserve:
        cause = "under_supported_load"
    elif belief.grip_hold_margin < 0.0:
        cause = "load_induced_slip"
    else:
        cause = "alignment_coupled_overload"

    scored: list[CounterfactualIntervention] = []
    for candidate in _heavy_static_candidates():
        gains = _predicted_latent_gains(belief, candidate.param_deltas)
        flip_gain = gains["load_support_margin"] + gains["lift_reserve"] + 0.5 * gains["grip_hold_margin"]
        scored.append(
            CounterfactualIntervention(
                label=candidate.label,
                param_deltas=dict(candidate.param_deltas),
                targeted_latents=list(candidate.targeted_latents),
                predicted_flip_gain=round(flip_gain, 4),
            )
        )
    scored.sort(key=lambda item: (-item.predicted_flip_gain, len(item.param_deltas), item.label))
    selected = scored[0]
    return CounterfactualDiagnosis(
        diagnosed_cause=cause,
        candidate_interventions=scored,
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
    next_plan["counterfactual_replan_trace"] = {
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
    return next_plan
```

- [ ] **Step 4: Re-run the targeted diagnosis tests**

Run:

```bash
PYTHONPATH=. ./venv/bin/python -m unittest tests.test_control_core.ControlCoreTest.test_diagnosis_prefers_minimal_sufficient_lift_intervention -v
PYTHONPATH=. ./venv/bin/python -m unittest tests.test_control_core.ControlCoreTest.test_diagnosis_uses_smallest_combination_when_single_knob_cannot_flip -v
PYTHONPATH=. ./venv/bin/python -m unittest tests.test_control_core -v
```

Expected: all commands report `OK`.

- [ ] **Step 5: Commit the counterfactual diagnosis layer**

```bash
git add simulation/control_core.py tests/test_control_core.py
git commit -m "Add heavy-load counterfactual diagnosis"
```

## Task 3: Refactor `simulate_stepwise_execution()` into Prefix-Frozen Phase Execution

**Files:**
- Modify: `simulation/env.py`
- Modify: `tests/test_adaptive_execution.py`
- Create: `tests/test_counterfactual_execution.py`

- [ ] **Step 1: Write the failing phase-execution tests**

Create `tests/test_counterfactual_execution.py` with:

```python
import unittest
from unittest.mock import patch

from simulation.control_core import PhaseObservation
from simulation.env import simulate_stepwise_execution


class CounterfactualExecutionTest(unittest.TestCase):
    def test_lift_replan_freezes_completed_prefix(self):
        phase_log: list[str] = []

        def _fake_phase_eval(*, phase, params, object_profile, rng=None):
            phase_log.append(phase)
            if phase == "lift" and phase_log.count("lift") == 1:
                return (
                    PhaseObservation(
                        phase="lift",
                        payload_ratio_obs=0.99,
                        lift_progress_obs=0.44,
                        lift_reserve_obs=-0.26,
                        tilt_obs=0.18,
                        observation_confidence=0.86,
                        trigger_reason="lift_reserve_obs",
                    ),
                    False,
                    {"phase_success": False},
                )
            return (
                PhaseObservation(phase=phase, observation_confidence=0.72),
                True,
                {"phase_success": True},
            )

        def _suffix_replan(observation: dict, current_params: dict) -> dict | None:
            if observation["phase"] != "lift":
                return None
            updated = dict(current_params)
            updated["lift_force"] = current_params["lift_force"] + 1.6
            updated["lift_clearance"] = current_params["lift_clearance"] + 0.006
            updated["execution_feedback_mode"] = "suffix_counterfactual_replan"
            return updated

        with patch("simulation.env._evaluate_phase_execution", side_effect=_fake_phase_eval):
            success, _, info = simulate_stepwise_execution(
                object_pos=(0.0, 0.0, 0.0),
                target_pos=(0.35, 0.0, 0.0),
                params={
                    "gripper_force": 42.0,
                    "approach_height": 0.05,
                    "transport_velocity": 0.22,
                    "lift_force": 42.0,
                    "transfer_force": 42.0,
                    "placement_velocity": 0.18,
                    "transfer_alignment": 0.0,
                    "lift_clearance": 0.06,
                },
                object_profile={"mass_kg": 0.42, "surface_friction": 0.18},
                step_replan_callback=_suffix_replan,
                max_step_replans=1,
            )

        self.assertTrue(success)
        self.assertEqual(phase_log, ["approach", "grasp", "lift", "lift", "transfer", "place"])
        self.assertEqual(info["counterfactual_replan_trace"][0]["start_phase"], "lift")
        self.assertEqual([row["phase"] for row in info["phase_execution_trace"][:2]], ["approach", "grasp"])
        self.assertEqual(info["execution_feedback_mode"], "suffix_counterfactual_replan")
```

Also update `tests/test_adaptive_execution.py` so the existing single-trial replan test expects the new feedback mode:

```python
        self.assertEqual(info["execution_feedback_mode"], "suffix_counterfactual_replan")
        self.assertIn("counterfactual_replan_trace", info)
        self.assertGreater(len(info["counterfactual_replan_trace"]), 0)
        self.assertEqual(info["counterfactual_replan_trace"][0]["start_phase"], "transfer")
```

- [ ] **Step 2: Run the new execution tests and confirm failure**

Run:

```bash
PYTHONPATH=. ./venv/bin/python -m unittest tests.test_counterfactual_execution -v
PYTHONPATH=. ./venv/bin/python -m unittest tests.test_adaptive_execution.AdaptiveExecutionTest.test_stepwise_execution_replans_inside_single_trial -v
```

Expected: `ERROR` because `_evaluate_phase_execution`, `phase_execution_trace`, and `counterfactual_replan_trace` do not exist yet.

- [ ] **Step 3: Refactor `simulation/env.py` to stage execution and suffix-only repair**

Add a new helper and rewrite `simulate_stepwise_execution()` around it:

```python
PHASE_SEQUENCE = ("approach", "grasp", "lift", "transfer", "place")


def _evaluate_phase_execution(
    *,
    phase: str,
    params: dict[str, float],
    object_profile: dict[str, Any] | None,
    rng: random.Random | None = None,
) -> tuple[PhaseObservation, bool, dict[str, Any]]:
    profile = dict(object_profile or {})
    mass = float(profile.get("mass_kg", 0.05))
    friction = float(profile.get("surface_friction", 0.5))
    if phase == "grasp":
        slip = max(0.0, 0.30 - friction) + max(0.0, 0.18 - params["gripper_force"] / 100.0)
        observation = PhaseObservation(
            phase="grasp",
            contact_stability_obs=max(0.0, 1.0 - slip),
            micro_slip_obs=slip,
            payload_ratio_obs=min(1.2, mass / max(0.05, params["gripper_force"] * 0.01)),
            observation_confidence=0.72,
            trigger_reason="micro_slip_obs" if slip > 0.18 else "",
        )
        return observation, slip <= 0.22, {"phase_success": slip <= 0.22}
    if phase == "lift":
        payload_ratio = min(1.4, mass / max(0.05, params["lift_force"] * 0.009))
        lift_reserve = params["lift_force"] * 0.0105 + params["lift_clearance"] * 1.4 - mass * 1.08
        tilt = max(0.0, payload_ratio - 0.88) * 0.42
        observation = PhaseObservation(
            phase="lift",
            payload_ratio_obs=payload_ratio,
            lift_progress_obs=max(0.0, min(1.0, 0.5 + lift_reserve)),
            lift_reserve_obs=round(lift_reserve, 4),
            tilt_obs=round(tilt, 4),
            observation_confidence=0.84,
            trigger_reason="lift_reserve_obs" if lift_reserve < 0.0 else "",
        )
        return observation, lift_reserve >= 0.0, {"phase_success": lift_reserve >= 0.0}
    if phase == "transfer":
        sway = max(0.0, params["transport_velocity"] - 0.24) * 2.4
        observation = PhaseObservation(
            phase="transfer",
            sway_obs=round(sway, 4),
            velocity_stress_obs=round(max(0.0, params["transport_velocity"] - 0.22), 4),
            observation_confidence=0.76,
            trigger_reason="sway_obs" if sway > 0.16 else "",
        )
        return observation, sway <= 0.26, {"phase_success": sway <= 0.26}
    return PhaseObservation(phase=phase, observation_confidence=0.68), True, {"phase_success": True}


def simulate_stepwise_execution(...):
    current_feedback_state = dict(params)
    current_params = _normalize_execution_params(current_feedback_state)
    phase_execution_trace: list[dict[str, Any]] = []
    observation_trace: list[dict[str, Any]] = []
    belief_update_trace: list[dict[str, Any]] = []
    counterfactual_replan_trace: list[dict[str, Any]] = []
    frozen_prefix_plan: dict[str, dict[str, float]] = {}
    start_index = 0
    phase_index = 0
    step_replan_count = 0

    while phase_index < len(PHASE_SEQUENCE):
        phase = PHASE_SEQUENCE[phase_index]
        frozen_prefix_plan.setdefault(phase, dict(current_params))
        observation, phase_success, phase_info = _evaluate_phase_execution(
            phase=phase,
            params=current_params,
            object_profile=object_profile,
            rng=rng,
        )
        observation_trace.append(asdict(observation))
        phase_execution_trace.append({"phase": phase, **phase_info})
        if phase_success:
            phase_index += 1
            continue
        if step_replan_callback is None or step_replan_count >= max_step_replans:
            return False, float(phase_index + 1), {
                "phase_execution_trace": phase_execution_trace,
                "observation_trace": observation_trace,
                "belief_update_trace": belief_update_trace,
                "counterfactual_replan_trace": counterfactual_replan_trace,
                "frozen_prefix_plan": frozen_prefix_plan,
                "terminal_suffix_plan": dict(current_params),
                "observer_trace": observation_trace,
                "step_replan_trace": counterfactual_replan_trace,
                "step_replan_count": step_replan_count,
                "execution_feedback_mode": "observer_only",
                "applied_params": dict(current_feedback_state),
                "failure_bucket": f"{phase}_fail",
            }
        candidate = step_replan_callback({"phase": phase, **asdict(observation)}, dict(current_feedback_state))
        if candidate is None:
            return False, float(phase_index + 1), {
                "phase_execution_trace": phase_execution_trace,
                "observation_trace": observation_trace,
                "belief_update_trace": belief_update_trace,
                "counterfactual_replan_trace": counterfactual_replan_trace,
                "frozen_prefix_plan": frozen_prefix_plan,
                "terminal_suffix_plan": dict(current_params),
                "observer_trace": observation_trace,
                "step_replan_trace": counterfactual_replan_trace,
                "step_replan_count": step_replan_count,
                "execution_feedback_mode": "observer_only",
                "applied_params": dict(current_feedback_state),
                "failure_bucket": f"{phase}_fail",
            }
        current_feedback_state = dict(candidate)
        current_params = _normalize_execution_params(candidate)
        counterfactual_replan_trace.append(
            {
                "start_phase": phase,
                "frozen_prefix": list(PHASE_SEQUENCE[:phase_index]),
                "final_plan": dict(current_params),
            }
        )
        step_replan_count += 1

    return True, float(len(PHASE_SEQUENCE)), {
        "phase_execution_trace": phase_execution_trace,
        "observation_trace": observation_trace,
        "belief_update_trace": belief_update_trace,
        "counterfactual_replan_trace": counterfactual_replan_trace,
        "frozen_prefix_plan": frozen_prefix_plan,
        "terminal_suffix_plan": dict(current_params),
        "observer_trace": observation_trace,
        "step_replan_trace": counterfactual_replan_trace,
        "step_replan_count": step_replan_count,
        "execution_feedback_mode": "suffix_counterfactual_replan" if counterfactual_replan_trace else "observer_only",
        "applied_params": dict(current_feedback_state),
        "failure_bucket": "success",
    }
```

- [ ] **Step 4: Re-run the execution-focused tests**

Run:

```bash
PYTHONPATH=. ./venv/bin/python -m unittest tests.test_counterfactual_execution -v
PYTHONPATH=. ./venv/bin/python -m unittest tests.test_adaptive_execution -v
```

Expected: both commands report `OK`.

- [ ] **Step 5: Commit the phase-execution refactor**

```bash
git add simulation/env.py tests/test_adaptive_execution.py tests/test_counterfactual_execution.py
git commit -m "Refactor adaptive execution into prefix-frozen phases"
```

## Task 4: Route Feedback and `rag_feedback` Through the New Diagnosis Path

**Files:**
- Modify: `simulation/control_core.py`
- Modify: `simulation/feedback.py`
- Modify: `simulation/rag_controller.py`
- Test: `tests/test_control_core.py`

- [ ] **Step 1: Write the failing compatibility test**

Add this test to `tests/test_control_core.py`:

```python
from simulation.feedback import FeedbackSignal, build_feedback_replan_request

    def test_feedback_request_wraps_phase_observation_for_suffix_repair(self):
        signal = FeedbackSignal(
            success=False,
            gripper_force=42.0,
            distance=0.02,
            steps=12,
            transport_velocity=0.22,
            lift_clearance=0.06,
            slip_risk=0.08,
            compression_risk=0.05,
            stability_score=0.36,
            velocity_risk=0.04,
            clearance_risk=0.22,
            lift_hold_risk=0.44,
            transfer_sway_risk=0.0,
            placement_settle_risk=0.0,
            failure_bucket="lift_hold_fail",
            dynamic_transport_mode="static",
        )
        request = build_feedback_replan_request(
            {
                "gripper_force": 42.0,
                "lift_force": 42.0,
                "transfer_force": 42.0,
                "transport_velocity": 0.22,
                "placement_velocity": 0.18,
                "lift_clearance": 0.06,
            },
            signal,
            "increase",
            step=4.0,
        )
        self.assertEqual(request["stage_bias"], "lift")
        self.assertEqual(request["phase_observation"]["phase"], "lift")
        self.assertEqual(request["requested_suffix_start"], "lift")
```

- [ ] **Step 2: Run the compatibility test and confirm failure**

Run:

```bash
PYTHONPATH=. ./venv/bin/python -m unittest tests.test_control_core.ControlCoreTest.test_feedback_request_wraps_phase_observation_for_suffix_repair -v
```

Expected: `FAIL` because `build_feedback_replan_request()` does not return `phase_observation` or `requested_suffix_start`.

- [ ] **Step 3: Update `feedback.py`, `control_core.py`, and `rag_controller.py` to use the new path**

Change `build_feedback_replan_request()` in `simulation/feedback.py` so it emits a phase observation payload:

```python
    phase_observation = {
        "phase": stage_bias,
        "contact_stability_obs": max(0.0, 1.0 - signal.slip_risk),
        "micro_slip_obs": signal.slip_risk,
        "payload_ratio_obs": 1.0 + signal.lift_hold_risk,
        "lift_progress_obs": max(0.0, 1.0 - signal.distance),
        "lift_reserve_obs": -signal.lift_hold_risk if stage_bias == "lift" else 0.0,
        "tilt_obs": signal.clearance_risk if stage_bias == "lift" else 0.0,
        "sway_obs": signal.transfer_sway_risk if stage_bias == "transfer" else 0.0,
        "velocity_stress_obs": signal.velocity_risk,
        "settle_obs": signal.placement_settle_risk if stage_bias == "place" else 0.0,
        "placement_error_obs": signal.distance,
        "observation_confidence": 0.76,
        "trigger_reason": signal.failure_bucket,
    }
    return {
        ...,
        "phase_observation": phase_observation,
        "requested_suffix_start": stage_bias,
    }
```

Update `replan_control_plan()` in `simulation/control_core.py` so it uses the new diagnosis functions:

```python
    observation = PhaseObservation(**dict(replan_request["phase_observation"]))
    prior = build_execution_prior(previous_params, phase=observation.phase)
    posterior, belief_trace = apply_phase_observation(prior, observation)
    diagnosis = diagnose_failure_cause(posterior, observation, base_plan)
    next_params = repair_suffix_plan(base_plan, current_phase=observation.phase, diagnosis=diagnosis)
    next_params["belief_update_trace"] = belief_trace
    next_params["counterfactual_replan_trace"] = next_params["counterfactual_replan_trace"]
```

Update `rag_controller.py` in both observation-time and post-failure paths so the heavy-static branch preserves the new feedback mode:

```python
        replan_request = build_feedback_replan_request(previous_params, signal, "increase", step=4.0)
        updated = replan_control_plan(previous_params, replan_request)
        updated["execution_feedback_mode"] = "suffix_counterfactual_replan"
        return updated
```

- [ ] **Step 4: Re-run the compatibility and control-core tests**

Run:

```bash
PYTHONPATH=. ./venv/bin/python -m unittest tests.test_control_core.ControlCoreTest.test_feedback_request_wraps_phase_observation_for_suffix_repair -v
PYTHONPATH=. ./venv/bin/python -m unittest tests.test_control_core -v
```

Expected: both commands report `OK`.

- [ ] **Step 5: Commit the feedback/rag integration**

```bash
git add simulation/control_core.py simulation/feedback.py simulation/rag_controller.py tests/test_control_core.py
git commit -m "Route heavy-load feedback through counterfactual repair"
```

## Task 5: Extend Benchmark Serialization and Trial Records for Online Diagnosis

**Files:**
- Modify: `simulation/runner.py`
- Modify: `tests/test_benchmark_schema.py`
- Create: `tests/test_heavy_load_regression.py`

- [ ] **Step 1: Write the failing benchmark-schema tests**

Add this test to `tests/test_benchmark_schema.py`:

```python
    def test_serialize_results_includes_online_diagnosis_stats(self):
        result = runner.BenchmarkResult(
            task_id="pick_demo",
            task_description="测试抓取任务",
            task_split="test",
            reference_force_range=(10.0, 14.0),
            n_trials=2,
            success_count=1,
            success_rate=0.5,
            avg_time=1.1,
            avg_steps=12.0,
            avg_distance_error=0.015,
            ci95_low=0.1,
            ci95_high=0.9,
            reference_force_deviation_stats={"mean": 1.0, "std": 0.0, "min": 1.0, "max": 1.0},
            avg_slip_risk=0.15,
            avg_compression_risk=0.0,
            avg_velocity_risk=0.0,
            avg_clearance_risk=0.05,
            avg_lift_hold_risk=0.15,
            avg_transfer_sway_risk=0.05,
            avg_placement_settle_risk=0.05,
            avg_stability_score=0.75,
            physics_fail_rate=0.0,
            lift_hold_fail_rate=0.0,
            transfer_sway_fail_rate=0.5,
            placement_settle_fail_rate=0.0,
            dominant_failure_mode="transfer_sway_fail",
            seed_plan={"gripper_force": 12.0},
            executed_plan_stats={"mean": {"gripper_force": 12.0}},
            planner_diagnostics={"belief_state_coverage": 0.8},
            trial_records=[
                {
                    "trial_index": 0,
                    "success": True,
                    "execution_feedback_mode": "suffix_counterfactual_replan",
                    "counterfactual_replan_trace": [{"diagnosed_cause": "under_supported_load", "selected_intervention": "lift_force_up"}],
                    "feedback_retry_count": 0,
                },
                {
                    "trial_index": 1,
                    "success": False,
                    "execution_feedback_mode": "post_failure_retry",
                    "counterfactual_replan_trace": [],
                    "feedback_retry_count": 1,
                },
            ],
            method="rag_feedback",
        )

        [row] = runner._serialize_results([result])
        self.assertEqual(row["online_replan_success_rate"], 1.0)
        self.assertEqual(row["suffix_counterfactual_replan_count"], 1)
        self.assertEqual(row["post_failure_retry_count"], 1)
        self.assertEqual(row["heavy_load_diagnosis_count"], 1)
        self.assertEqual(row["diagnosed_cause_distribution"]["under_supported_load"], 1)
        self.assertEqual(row["selected_intervention_distribution"]["lift_force_up"], 1)
```

Create `tests/test_heavy_load_regression.py` with the current baselines:

```python
import unittest


HEAVY_LOAD_BASELINE = {
    "pick_metal_heavy": 0.0,
    "pick_metal_heavy_fast": 0.8667,
    "overall_mean": 0.8222,
}


class HeavyLoadRegressionBaselineTest(unittest.TestCase):
    def test_baseline_constants_match_expected_current_profile(self):
        self.assertEqual(HEAVY_LOAD_BASELINE["pick_metal_heavy"], 0.0)
        self.assertEqual(HEAVY_LOAD_BASELINE["pick_metal_heavy_fast"], 0.8667)
        self.assertEqual(HEAVY_LOAD_BASELINE["overall_mean"], 0.8222)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the new benchmark tests and confirm failure**

Run:

```bash
PYTHONPATH=. ./venv/bin/python -m unittest tests.test_benchmark_schema.BenchmarkSchemaTest.test_serialize_results_includes_online_diagnosis_stats -v
PYTHONPATH=. ./venv/bin/python -m unittest tests.test_heavy_load_regression -v
```

Expected: the baseline test passes; the benchmark-schema test fails because `_serialize_results()` does not yet emit the new diagnosis fields.

- [ ] **Step 3: Implement diagnosis aggregation in `simulation/runner.py`**

Extend `BenchmarkResult` and `_serialize_results()` like this:

```python
def _online_diagnosis_summary(trial_records: list[dict]) -> dict[str, Any]:
    suffix_trials = [record for record in trial_records if record.get("execution_feedback_mode") == "suffix_counterfactual_replan"]
    online_success_rate = (
        sum(1 for record in suffix_trials if record.get("success")) / len(suffix_trials)
        if suffix_trials
        else 0.0
    )
    diagnosed_causes = Counter()
    interventions = Counter()
    for record in trial_records:
        for trace in record.get("counterfactual_replan_trace", []):
            cause = trace.get("diagnosed_cause")
            selected = trace.get("selected_intervention")
            if cause:
                diagnosed_causes[cause] += 1
            if selected:
                interventions[selected] += 1
    return {
        "online_replan_success_rate": round(online_success_rate, 4),
        "suffix_counterfactual_replan_count": len(suffix_trials),
        "post_failure_retry_count": sum(int(record.get("feedback_retry_count", 0)) for record in trial_records),
        "heavy_load_diagnosis_count": sum(diagnosed_causes.values()),
        "diagnosed_cause_distribution": dict(diagnosed_causes),
        "selected_intervention_distribution": dict(interventions),
    }


def _serialize_results(results: list[BenchmarkResult]) -> list[dict]:
    rows = []
    for result in results:
        online = _online_diagnosis_summary(result.trial_records)
        rows.append(
            {
                ...,
                "online_replan_success_rate": online["online_replan_success_rate"],
                "suffix_counterfactual_replan_count": online["suffix_counterfactual_replan_count"],
                "post_failure_retry_count": online["post_failure_retry_count"],
                "heavy_load_diagnosis_count": online["heavy_load_diagnosis_count"],
                "diagnosed_cause_distribution": online["diagnosed_cause_distribution"],
                "selected_intervention_distribution": online["selected_intervention_distribution"],
            }
        )
    return rows
```

Also update `run_benchmark()` so each trial record stores:

```python
{
    "trial_index": trial_index,
    "success": success,
    "execution_feedback_mode": info.get("execution_feedback_mode", "observer_only"),
    "phase_execution_trace": info.get("phase_execution_trace", []),
    "observation_trace": info.get("observation_trace", []),
    "belief_update_trace": info.get("belief_update_trace", []),
    "counterfactual_replan_trace": info.get("counterfactual_replan_trace", []),
    "feedback_retry_count": retries,
    "terminal_plan": dict(current_params),
}
```

- [ ] **Step 4: Re-run the benchmark-focused tests**

Run:

```bash
PYTHONPATH=. ./venv/bin/python -m unittest tests.test_benchmark_schema.BenchmarkSchemaTest.test_serialize_results_includes_online_diagnosis_stats -v
PYTHONPATH=. ./venv/bin/python -m unittest tests.test_benchmark_schema -v
PYTHONPATH=. ./venv/bin/python -m unittest tests.test_heavy_load_regression -v
```

Expected: all commands report `OK`.

- [ ] **Step 5: Commit the runner and regression metrics**

```bash
git add simulation/runner.py tests/test_benchmark_schema.py tests/test_heavy_load_regression.py
git commit -m "Expose online diagnosis metrics in benchmark outputs"
```

## Task 6: Refresh Docs and Regenerate the Current Heavy-Load Profile

**Files:**
- Modify: `README.md`
- Modify: `docs/overview.md`
- Modify: `simulation/README.md`
- Refresh: `outputs/current_observer_step_replan/*`

- [ ] **Step 1: Update the docs to describe phase observations and suffix repair**

Apply these content changes:

```markdown
- `observer_trace` 现在只是兼容字段；主执行日志改为 `phase_execution_trace`、`observation_trace`、`belief_update_trace`、`counterfactual_replan_trace`
- `rag_feedback` 的执行内重规划现在使用 `suffix_counterfactual_replan`，不再把整条计划从头 reroll
- 当前重载验收口径：
  - `pick_metal_heavy >= 0.30`
  - `pick_metal_heavy_fast` 绝对成功率下降不得超过 `0.05`
  - 12 任务总体平均成功率下降不得超过 `0.01`
```

- [ ] **Step 2: Run the full unit-test suite**

Run:

```bash
PYTHONPATH=. ./venv/bin/python -m unittest discover -s tests -p 'test_*.py'
```

Expected: `OK`.

- [ ] **Step 3: Regenerate the current simulation outputs**

Run:

```bash
PYTHONPATH=. ./venv/bin/python -m simulation.benchmark --report_multi_seed --method rag_feedback --n_trials 20 --seeds 42 43 44 --output outputs/current_observer_step_replan/simulation_benchmark_result.json
PYTHONPATH=. ./venv/bin/python -m simulation.benchmark --compare_feedback --n_trials 20 --seed 42 --output_dir outputs/current_observer_step_replan
PYTHONPATH=. ./venv/bin/python -m simulation.benchmark --compare_multi_seed --n_trials 20 --seeds 42 43 44 --multi_seed_methods rag rag_feedback task_heuristic fixed --output_dir outputs/current_observer_step_replan
PYTHONPATH=. ./venv/bin/python reporting/visualize_results.py --qa_json outputs/current/qa_evaluation_detail.json --sim_json outputs/current_observer_step_replan/simulation_comparison_rag_vs_baseline.json --sim_multi_seed_json outputs/current_observer_step_replan/simulation_comparison_multi_seed.json --output_dir outputs/current_observer_step_replan/visualizations
PYTHONPATH=. ./venv/bin/python reporting/generate_showcase.py --qa_json outputs/current/qa_evaluation_detail.json --sim_json outputs/current_observer_step_replan/simulation_comparison_rag_vs_baseline.json --sim_multi_seed_json outputs/current_observer_step_replan/simulation_comparison_multi_seed.json --sim_benchmark_json outputs/current_observer_step_replan/simulation_benchmark_result.json --output outputs/current_observer_step_replan/showcase_summary.txt
```

Expected: all commands exit `0`.

- [ ] **Step 4: Verify the heavy-load acceptance gates directly from the regenerated JSON**

Run:

```bash
python - <<'PY'
import json
from pathlib import Path

root = Path("outputs/current_observer_step_replan")
rows = json.loads((root / "simulation_benchmark_result.json").read_text(encoding="utf-8"))
methods = {row["task_id"]: row["methods"]["rag_feedback"] for row in rows}
overall = sum(payload["success_rate_mean"] for payload in methods.values()) / len(methods)

heavy = methods["pick_metal_heavy"]["success_rate_mean"]
heavy_fast = methods["pick_metal_heavy_fast"]["success_rate_mean"]

assert heavy >= 0.30, heavy
assert 0.8667 - heavy_fast <= 0.05, heavy_fast
assert 0.8222 - overall <= 0.01, overall

print({
    "pick_metal_heavy": round(heavy, 4),
    "pick_metal_heavy_fast": round(heavy_fast, 4),
    "overall_mean": round(overall, 4),
})
PY
```

Expected: the script prints the three measured values and exits `0`.

- [ ] **Step 5: Commit the regenerated outputs and docs**

```bash
git add README.md docs/overview.md simulation/README.md outputs/current_observer_step_replan
git commit -m "Refresh heavy-load counterfactual outputs and docs"
```
