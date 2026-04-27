# Simulation Execution / Schema / Current Profile Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the simulation current path internally consistent by removing surrogate double-sampling, replacing the old `params_used` result schema with explicit execution summaries, decoupling `simulation` package imports from RAG dependencies, and aligning all default scripts/docs/results with `rag_feedback + outputs/current_observer_step_replan`.

**Architecture:** Keep the existing module boundaries, but tighten semantics. `simulation/env.py` becomes the single source of truth for one surrogate execution result, `simulation/runner.py` moves to Schema V2 with explicit `seed_plan` / `executed_plan_stats` / `trial_records`, `reporting/*` and `simulation/benchmark.py` switch to nested method payloads and simulation-current defaults, and `simulation/__init__.py` stops importing `RAGController` eagerly.

**Tech Stack:** Python 3.10, `unittest`, shell scripts, JSON artifact generation, existing simulation/reporting modules, no new third-party libraries.

---

## File Structure

### Files to modify

- `simulation/env.py`
  Purpose: remove the final duplicate surrogate evaluation and ensure returned `success`, `failure_bucket`, and `observer_trace` come from the same final evaluation record.
- `simulation/runner.py`
  Purpose: replace `params_used` with Schema V2 summary fields, generate trial record artifacts, and update comparison / multi-seed aggregation to consume execution means instead of last-trial leftovers.
- `simulation/__init__.py`
  Purpose: remove eager import of `RAGController`.
- `simulation/benchmark.py`
  Purpose: expose parser defaults for testing and switch defaults to simulation current (`rag_feedback`, `outputs/current_observer_step_replan`).
- `reporting/visualize_results.py`
  Purpose: consume Schema V2 comparison / multi-seed payloads and expose parser defaults for testing.
- `reporting/generate_showcase.py`
  Purpose: consume Schema V2 benchmark / comparison / multi-seed payloads and expose parser defaults for testing.
- `scripts/run_all.sh`
  Purpose: align the default simulation pipeline with the current simulation profile.
- `README.md`
  Purpose: document Schema V2 fields, default commands, and regenerated simulation current outputs.
- `docs/overview.md`
  Purpose: mirror the new simulation current commands and result semantics.
- `docs/DESIGN.md`
  Purpose: update the design narrative from `params_used` / old current defaults to Schema V2 and the new current profile.
- `simulation/README.md`
  Purpose: mirror simulation-specific commands, field semantics, and regenerated artifact paths.

### Files to create

- `tests/test_benchmark_schema.py`
  Purpose: lock in Schema V2 summary, comparison, and multi-seed semantics.
- `tests/test_reporting_schema.py`
  Purpose: lock in Schema V2 reporting helper behavior and parser defaults.
- `tests/test_package_imports.py`
  Purpose: guarantee `simulation.control_core` and `simulation.env` import without langchain dependencies.

### Generated artifacts to refresh

- `outputs/current_observer_step_replan/simulation_benchmark_result.json`
- `outputs/current_observer_step_replan/simulation_benchmark_trial_records.json`
- `outputs/current_observer_step_replan/simulation_comparison_rag_vs_baseline.json`
- `outputs/current_observer_step_replan/simulation_comparison_multi_seed.json`
- `outputs/current_observer_step_replan/showcase_summary.txt`
- `outputs/current_observer_step_replan/visualizations/*`

## Task 1: Reuse the Final Surrogate Evaluation Instead of Re-sampling

**Files:**
- Modify: `simulation/env.py`
- Test: `tests/test_adaptive_execution.py`

- [ ] **Step 1: Write the failing execution-consistency test**

Add the import and test below to `tests/test_adaptive_execution.py`:

```python
from unittest.mock import patch

def test_stepwise_execution_reuses_last_evaluation_without_extra_sampling(self):
    evaluation_log: list[dict] = []
    original = simulate_stepwise_execution.__globals__["_evaluate_execution_plan"]

    def _tracked_evaluate(*, object_pos, target_pos, params, object_profile=None, rng=None):
        evaluation = original(
            object_pos=object_pos,
            target_pos=target_pos,
            params=params,
            object_profile=object_profile,
            rng=rng,
        )
        evaluation_log.append(
            {
                "success": evaluation["success"],
                "failure_bucket": evaluation["info"]["failure_bucket"],
                "params": dict(evaluation["params"]),
            }
        )
        return evaluation

    with patch("simulation.env._evaluate_execution_plan", side_effect=_tracked_evaluate):
        success, _, info = simulate_stepwise_execution(
            object_pos=(0.0, 0.0, 0.0),
            target_pos=(0.35, 0.0, 0.0),
            params={
                "gripper_force": 7.5,
                "approach_height": 0.03,
                "transport_velocity": 0.34,
                "lift_force": 7.5,
                "transfer_force": 7.5,
                "placement_velocity": 0.30,
                "transfer_alignment": 0.0,
                "lift_clearance": 0.045,
            },
            object_profile={
                "mass_kg": 0.06,
                "surface_friction": 0.18,
                "fragility": 0.78,
                "velocity_scale": 0.8,
                "target_tolerance": 0.04,
                "size_xyz": (0.04, 0.04, 0.02),
                "preferred_approach_height": 0.05,
                "approach_height_tolerance": 0.02,
            },
            rng=__import__("random").Random(0),
        )

    self.assertEqual(len(evaluation_log), info["step_replan_count"] + 1)
    self.assertEqual(success, evaluation_log[-1]["success"])
    self.assertEqual(info["failure_bucket"], evaluation_log[-1]["failure_bucket"])
    expected_stage = "none" if info["failure_bucket"] == "success" else info["failure_bucket"].replace("_fail", "")
    self.assertEqual(info["observer_trace"][-1]["estimated_failure_stage"], expected_stage)
```

- [ ] **Step 2: Run the targeted test and confirm failure**

Run:

```bash
./venv/bin/python -m unittest tests.test_adaptive_execution.AdaptiveExecutionTest.test_stepwise_execution_reuses_last_evaluation_without_extra_sampling -v
```

Expected: `FAIL` because the current surrogate path calls `_evaluate_execution_plan` once inside the loop and again after the loop.

- [ ] **Step 3: Implement final-evaluation reuse in `simulation/env.py`**

Update `simulate_stepwise_execution()` so the loop keeps the last accepted evaluation and returns it directly instead of making a second final call:

```python
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
```

- [ ] **Step 4: Re-run the targeted test and the adaptive execution test file**

Run:

```bash
./venv/bin/python -m unittest tests.test_adaptive_execution.AdaptiveExecutionTest.test_stepwise_execution_reuses_last_evaluation_without_extra_sampling -v
./venv/bin/python -m unittest tests.test_adaptive_execution -v
```

Expected: both commands report `OK`.

- [ ] **Step 5: Commit the execution-consistency fix**

```bash
git add simulation/env.py tests/test_adaptive_execution.py
git commit -m "Fix surrogate execution evaluation reuse"
```

## Task 2: Replace `params_used` with Schema V2 in the benchmark summary

**Files:**
- Modify: `simulation/runner.py`
- Create: `tests/test_benchmark_schema.py`

- [ ] **Step 1: Write failing Schema V2 summary tests**

Create `tests/test_benchmark_schema.py` with the exact scaffold below:

```python
import unittest
from unittest.mock import patch

from simulation import runner
from simulation.tasks import ObjectProfile, TaskConfig


def _task() -> TaskConfig:
    return TaskConfig(
        task_id="pick_demo",
        description="测试抓取任务",
        object_type="测试件",
        object_pos=(0.0, 0.0, 0.0),
        target_pos=(0.2, 0.0, 0.0),
        reference_force_range=(10.0, 14.0),
        split="test",
        profile=ObjectProfile(
            name="demo",
            mass_kg=0.08,
            surface_friction=0.4,
            fragility=0.5,
            velocity_scale=0.8,
            target_tolerance=0.04,
            size_xyz=(0.02, 0.02, 0.02),
            preferred_approach_height=0.04,
            approach_height_tolerance=0.01,
        ),
        challenge_tags=("demo",),
    )


class BenchmarkSchemaTest(unittest.TestCase):
    def test_run_benchmark_separates_seed_plan_from_executed_plan_stats(self):
        seed_plan = {
            "gripper_force": 12.0,
            "lift_force": 12.0,
            "transfer_force": 12.0,
            "transfer_alignment": 0.25,
            "approach_height": 0.04,
            "transport_velocity": 0.3,
            "placement_velocity": 0.24,
            "lift_clearance": 0.05,
            "belief_state_coverage": 0.8,
            "solver_selected_candidate": "belief_seed",
        }
        terminal_plans = iter(
            [
                (
                    True,
                    1.0,
                    {
                        "distance": 0.0,
                        "steps": 12,
                        "slip_risk": 0.1,
                        "compression_risk": 0.0,
                        "velocity_risk": 0.0,
                        "clearance_risk": 0.0,
                        "lift_hold_risk": 0.1,
                        "transfer_sway_risk": 0.0,
                        "placement_settle_risk": 0.0,
                        "stability_score": 0.9,
                        "failure_bucket": "success",
                        "applied_params": {
                            "gripper_force": 11.0,
                            "lift_force": 11.5,
                            "transfer_force": 11.2,
                            "transfer_alignment": 0.30,
                            "approach_height": 0.04,
                            "transport_velocity": 0.28,
                            "placement_velocity": 0.22,
                            "lift_clearance": 0.055,
                            "dynamic_transport_mode": "static",
                            "execution_feedback_mode": "observer_only",
                        },
                    },
                ),
                (
                    False,
                    1.2,
                    {
                        "distance": 0.03,
                        "steps": 12,
                        "slip_risk": 0.2,
                        "compression_risk": 0.0,
                        "velocity_risk": 0.0,
                        "clearance_risk": 0.1,
                        "lift_hold_risk": 0.2,
                        "transfer_sway_risk": 0.1,
                        "placement_settle_risk": 0.1,
                        "stability_score": 0.6,
                        "failure_bucket": "transfer_sway_fail",
                        "applied_params": {
                            "gripper_force": 13.0,
                            "lift_force": 13.5,
                            "transfer_force": 13.4,
                            "transfer_alignment": 0.45,
                            "approach_height": 0.045,
                            "transport_velocity": 0.24,
                            "placement_velocity": 0.20,
                            "lift_clearance": 0.060,
                            "dynamic_transport_mode": "static",
                            "execution_feedback_mode": "step_observer_replan",
                        },
                    },
                ),
            ]
        )

        with patch.object(runner, "BENCHMARK_TASKS", [_task()]), \
             patch.object(runner, "HAS_MUJOCO", False), \
             patch.object(runner, "_get_param_getter", return_value=lambda desc: dict(seed_plan)), \
             patch.object(runner, "_get_feedback_controller", return_value=None), \
             patch.object(runner, "_run_surrogate_trial", side_effect=lambda *args, **kwargs: next(terminal_plans)):
            [result] = runner.run_benchmark(
                n_trials_per_task=2,
                method="rag_feedback",
                output_path=None,
            )

        self.assertEqual(result.seed_plan["gripper_force"], 12.0)
        self.assertEqual(result.executed_plan_stats["mean"]["gripper_force"], 12.0)
        self.assertEqual(result.executed_plan_stats["min"]["gripper_force"], 11.0)
        self.assertEqual(result.executed_plan_stats["max"]["gripper_force"], 13.0)
        self.assertEqual(result.reference_force_deviation_stats["mean"], 1.0)
        self.assertEqual(result.executed_plan_stats["execution_feedback_mode_mode"], "observer_only")
        self.assertEqual(len(result.trial_records), 2)

    def test_serialize_results_omits_params_used_and_emits_schema_v2_fields(self):
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
            trial_records=[{"trial_index": 0, "terminal_plan": {"gripper_force": 11.0}}],
            method="rag_feedback",
        )

        [row] = runner._serialize_results([result])
        self.assertIn("seed_plan", row)
        self.assertIn("executed_plan_stats", row)
        self.assertIn("trial_record_count", row)
        self.assertNotIn("params_used", row)
        self.assertNotIn("params", row)
```

- [ ] **Step 2: Run the new Schema V2 tests and confirm failure**

Run:

```bash
./venv/bin/python -m unittest tests.test_benchmark_schema -v
```

Expected: `ERROR` / `FAIL` because `BenchmarkResult` still expects `params_used` and the serializer still writes the old fields.

- [ ] **Step 3: Implement Schema V2 in `simulation/runner.py`**

Refactor the dataclass and summary helpers to use explicit seed / execution / diagnostics fields:

```python
@dataclass
class BenchmarkResult:
    task_id: str
    task_description: str
    task_split: str
    reference_force_range: tuple[float, float]
    n_trials: int
    success_count: int
    success_rate: float
    avg_time: float
    avg_steps: float
    avg_distance_error: float
    ci95_low: float
    ci95_high: float
    reference_force_deviation_stats: dict
    avg_slip_risk: float
    avg_compression_risk: float
    avg_velocity_risk: float
    avg_clearance_risk: float
    avg_lift_hold_risk: float
    avg_transfer_sway_risk: float
    avg_placement_settle_risk: float
    avg_stability_score: float
    physics_fail_rate: float
    lift_hold_fail_rate: float
    transfer_sway_fail_rate: float
    placement_settle_fail_rate: float
    dominant_failure_mode: str
    seed_plan: dict
    executed_plan_stats: dict
    planner_diagnostics: dict
    trial_records: list[dict]
    method: str = "rag"


NUMERIC_PLAN_FIELDS = (
    "gripper_force",
    "lift_force",
    "transfer_force",
    "transfer_alignment",
    "approach_height",
    "transport_velocity",
    "placement_velocity",
    "lift_clearance",
)


def _extract_seed_plan(params: dict) -> dict:
    return {key: params[key] for key in NUMERIC_PLAN_FIELDS if key in params}


def _extract_planner_diagnostics(params: dict) -> dict:
    kept = {}
    for key in (
        "belief_state_coverage",
        "uncertainty_conservative_mode",
        "solver_selected_candidate",
        "evidence_rule_count",
        "evidence_support_score",
        "evidence_conflict_count",
        "force_rule_mode",
        "motion_rule_mode",
        "available_specific_force_rules",
        "suppressed_specific_force_rules",
        "available_motion_rules",
        "suppressed_motion_rules",
        "available_lift_stage_rules",
        "used_lift_stage_rules",
        "seed_mode",
        "seed_notes",
    ):
        if key in params:
            kept[key] = params[key]
    return kept
```

Inside `run_benchmark()`:

```python
seed_plan = _extract_seed_plan(params)
planner_diagnostics = _extract_planner_diagnostics(params)
terminal_plans: list[dict] = []
trial_records: list[dict] = []
force_deviations: list[float] = []
for trial_index in range(n_trials_per_task):
    current_params = dict(params)
    success, elapsed, info = _run_surrogate_trial(
        task,
        current_params,
        rng=surrogate_rng,
        step_replan_callback=step_replan_callback,
        max_step_replans=max_feedback_retries if feedback_controller is not None else 0,
    )
    current_params = dict(info.get("applied_params", current_params))
    terminal_plan = {
        field: float(current_params[field])
        for field in NUMERIC_PLAN_FIELDS
        if field in current_params
    }
    terminal_plan["dynamic_transport_mode"] = info.get("dynamic_transport_mode")
    terminal_plan["execution_feedback_mode"] = info.get("execution_feedback_mode")
    terminal_plans.append(terminal_plan)
    force_deviations.append(reference_force_deviation(task, float(terminal_plan["gripper_force"])))
    trial_records.append(
        {
            "trial_index": trial_index,
            "success": success,
            "failure_bucket": info.get("failure_bucket", "unknown_failure"),
            "elapsed_sec": round(elapsed, 4),
            "distance_error": round(info.get("distance", 0.0), 4),
            "feedback_retry_count": retries,
            "step_replan_count": int(info.get("step_replan_count", 0)),
            "terminal_plan": terminal_plan,
            "observer_trace": info.get("observer_trace", []),
            "step_replan_trace": info.get("step_replan_trace", []),
            "avg_risks": {
                "slip_risk": info.get("slip_risk", 0.0),
                "compression_risk": info.get("compression_risk", 0.0),
                "velocity_risk": info.get("velocity_risk", 0.0),
                "clearance_risk": info.get("clearance_risk", 0.0),
            },
        },
    )
executed_plan_stats = _aggregate_plan_stats(terminal_plans)
reference_force_deviation_stats = _aggregate_scalar_stats(force_deviations)
BenchmarkResult(
    task_id=task.task_id,
    task_description=task.description,
    task_split=task.split,
    reference_force_range=task.reference_force_range,
    n_trials=n_trials_per_task,
    success_count=successes,
    success_rate=successes / n_trials_per_task,
    avg_time=sum(times) / len(times),
    avg_steps=sum(step_counts) / len(step_counts),
    avg_distance_error=sum(distance_errors) / len(distance_errors),
    ci95_low=ci95_low,
    ci95_high=ci95_high,
    reference_force_deviation_stats=reference_force_deviation_stats,
    avg_slip_risk=sum(slip_risks) / len(slip_risks),
    avg_compression_risk=sum(compression_risks) / len(compression_risks),
    avg_velocity_risk=sum(velocity_risks) / len(velocity_risks),
    avg_clearance_risk=sum(clearance_risks) / len(clearance_risks),
    avg_lift_hold_risk=sum(lift_hold_risks) / len(lift_hold_risks),
    avg_transfer_sway_risk=sum(transfer_sway_risks) / len(transfer_sway_risks),
    avg_placement_settle_risk=sum(placement_settle_risks) / len(placement_settle_risks),
    avg_stability_score=sum(stability_scores) / len(stability_scores),
    physics_fail_rate=fail_rate_map["physics_fail"],
    lift_hold_fail_rate=fail_rate_map["lift_hold_fail"],
    transfer_sway_fail_rate=fail_rate_map["transfer_sway_fail"],
    placement_settle_fail_rate=fail_rate_map["placement_settle_fail"],
    dominant_failure_mode=dominant_failure_mode,
    seed_plan=seed_plan,
    executed_plan_stats=executed_plan_stats,
    planner_diagnostics=planner_diagnostics,
    trial_records=trial_records,
    method=method,
)
```

Update `_serialize_results()` to emit:

```python
{
    "task_id": result.task_id,
    "task_label": task_label(task),
    "task_description": result.task_description,
    "task_split": result.task_split,
    "challenge_tags": list(task.challenge_tags),
    "method": result.method,
    "n_trials": result.n_trials,
    "reference_force_deviation_stats": result.reference_force_deviation_stats,
    "seed_plan": result.seed_plan,
    "executed_plan_stats": result.executed_plan_stats,
    "planner_diagnostics": result.planner_diagnostics,
    "trial_record_count": len(result.trial_records),
}
```

- [ ] **Step 4: Re-run the Schema V2 summary tests**

Run:

```bash
./venv/bin/python -m unittest tests.test_benchmark_schema.BenchmarkSchemaTest.test_run_benchmark_separates_seed_plan_from_executed_plan_stats -v
./venv/bin/python -m unittest tests.test_benchmark_schema.BenchmarkSchemaTest.test_serialize_results_omits_params_used_and_emits_schema_v2_fields -v
```

Expected: both commands report `OK`.

- [ ] **Step 5: Commit the Schema V2 summary changes**

```bash
git add simulation/runner.py tests/test_benchmark_schema.py
git commit -m "Introduce schema v2 benchmark summaries"
```

## Task 3: Move comparison and multi-seed outputs to nested method payloads

**Files:**
- Modify: `simulation/runner.py`
- Modify: `tests/test_benchmark_schema.py`

- [ ] **Step 1: Add failing comparison / multi-seed tests**

Append the tests below to `tests/test_benchmark_schema.py`:

```python
    def test_comparison_uses_executed_plan_stats_mean_values(self):
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
            executed_plan_stats={
                "mean": {"gripper_force": 12.0, "transport_velocity": 0.26, "lift_clearance": 0.0575},
                "std": {"gripper_force": 1.0},
                "dynamic_transport_mode_mode": "static",
                "execution_feedback_mode_mode": "observer_only",
            },
            planner_diagnostics={"belief_state_coverage": 0.8},
            trial_records=[],
            method="rag_feedback",
        )

        with patch.object(runner, "run_benchmark", return_value=[result]):
            rows = runner.run_benchmark_comparison(
                n_trials_per_task=2,
                methods=["rag_feedback"],
                output_dir=None,
            )

        method_row = rows[0]["methods"]["rag_feedback"]
        self.assertEqual(method_row["executed_plan_stats"]["mean"]["gripper_force"], 12.0)
        self.assertNotIn("rag_feedback_gripper_force", rows[0])

    def test_multi_seed_report_keeps_executed_plan_stats_nested_under_method(self):
        base = runner.BenchmarkResult(
            task_id="pick_demo",
            task_description="测试抓取任务",
            task_split="test",
            reference_force_range=(10.0, 14.0),
            n_trials=2,
            success_count=2,
            success_rate=1.0,
            avg_time=1.0,
            avg_steps=12.0,
            avg_distance_error=0.0,
            ci95_low=0.5,
            ci95_high=1.0,
            reference_force_deviation_stats={"mean": 1.0, "std": 0.0, "min": 1.0, "max": 1.0},
            avg_slip_risk=0.1,
            avg_compression_risk=0.0,
            avg_velocity_risk=0.0,
            avg_clearance_risk=0.0,
            avg_lift_hold_risk=0.1,
            avg_transfer_sway_risk=0.0,
            avg_placement_settle_risk=0.0,
            avg_stability_score=0.9,
            physics_fail_rate=0.0,
            lift_hold_fail_rate=0.0,
            transfer_sway_fail_rate=0.0,
            placement_settle_fail_rate=0.0,
            dominant_failure_mode="none",
            seed_plan={"gripper_force": 12.0},
            executed_plan_stats={
                "mean": {"gripper_force": 12.0, "transport_velocity": 0.26, "lift_clearance": 0.0575},
                "std": {"gripper_force": 1.0},
                "dynamic_transport_mode_mode": "static",
                "execution_feedback_mode_mode": "observer_only",
            },
            planner_diagnostics={"belief_state_coverage": 0.8, "solver_selected_candidate": "belief_seed"},
            trial_records=[],
            method="rag_feedback",
        )

        with patch.object(runner, "run_benchmark", side_effect=[[base], [base], [base]]):
            rows = runner.run_benchmark_multi_seed_report(
                n_trials_per_task=2,
                seeds=[42, 43, 44],
                method="rag_feedback",
                output_path=None,
            )

        self.assertEqual(rows[0]["methods"]["rag_feedback"]["executed_plan_stats"]["mean"]["gripper_force"], 12.0)
        self.assertEqual(rows[0]["methods"]["rag_feedback"]["planner_diagnostics"]["belief_state_coverage_mean"], 0.8)
```

- [ ] **Step 2: Run the comparison / multi-seed tests and confirm failure**

Run:

```bash
./venv/bin/python -m unittest tests.test_benchmark_schema.BenchmarkSchemaTest.test_comparison_uses_executed_plan_stats_mean_values -v
./venv/bin/python -m unittest tests.test_benchmark_schema.BenchmarkSchemaTest.test_multi_seed_report_keeps_executed_plan_stats_nested_under_method -v
```

Expected: `FAIL` because `run_benchmark_comparison()` and `run_benchmark_multi_seed_report()` still flatten method fields and read from `params_used`.

- [ ] **Step 3: Implement nested method payloads in `simulation/runner.py`**

Refactor comparison and multi-seed outputs to write nested `methods` dictionaries keyed by method name:

```python
def _method_summary(result: BenchmarkResult) -> dict:
    return {
        "success_rate": round(result.success_rate, 4),
        "success_count": result.success_count,
        "success_rate_ci95": [round(result.ci95_low, 4), round(result.ci95_high, 4)],
        "avg_time_sec": round(result.avg_time, 4),
        "avg_steps": round(result.avg_steps, 2),
        "avg_distance_error": round(result.avg_distance_error, 4),
        "reference_force_deviation_stats": result.reference_force_deviation_stats,
        "avg_risks": {
            "slip_risk": round(result.avg_slip_risk, 4),
            "compression_risk": round(result.avg_compression_risk, 4),
            "velocity_risk": round(result.avg_velocity_risk, 4),
            "clearance_risk": round(result.avg_clearance_risk, 4),
            "lift_hold_risk": round(result.avg_lift_hold_risk, 4),
            "transfer_sway_risk": round(result.avg_transfer_sway_risk, 4),
            "placement_settle_risk": round(result.avg_placement_settle_risk, 4),
            "stability_score": round(result.avg_stability_score, 4),
        },
        "failure_rates": {
            "physics_fail": round(result.physics_fail_rate, 4),
            "lift_hold_fail": round(result.lift_hold_fail_rate, 4),
            "transfer_sway_fail": round(result.transfer_sway_fail_rate, 4),
            "placement_settle_fail": round(result.placement_settle_fail_rate, 4),
            "dominant_failure_mode": result.dominant_failure_mode,
        },
        "seed_plan": result.seed_plan,
        "executed_plan_stats": result.executed_plan_stats,
        "planner_diagnostics": result.planner_diagnostics,
    }
```

Use that helper in `run_benchmark_comparison()` and change the function to return `comparison` rows instead of `all_results`:

```python
comparison: list[dict] = []
all_results: dict[str, list[BenchmarkResult]] = {}
for method in methods:
    out_file = output_dir / f"simulation_benchmark_{method}.json"
    all_results[method] = run_benchmark(
        data_path=data_path,
        n_trials_per_task=n_trials_per_task,
        gui=False,
        seed=seed,
        output_path=str(out_file),
        method=method,
    )
row = {
    "task_id": task.task_id,
    "task_label": task_label(task),
    "task_description": task.description,
    "task_split": task.split,
    "challenge_tags": list(task.challenge_tags),
    "reference_force_range": list(task.reference_force_range),
    "reference_force_center": round(reference_force_center(task), 4),
    "reference_approach_height": reference_approach_height(task),
    "methods": {},
}
for method in methods:
    result = next(r for r in all_results[method] if r.task_id == task.task_id)
    row["methods"][method] = _method_summary(result)
comparison.append(row)
write_json(comparison_path, comparison)
write_table(table_path, lines)
return comparison
```

And in `run_benchmark_multi_seed_report()`:

```python
methods_payload = {}
for method in methods:
    task_runs = [
        next(result for result in per_seed_results[seed] if result.task_id == task.task_id and result.method == method)
        for seed in seeds
    ]
    success_rates = [run.success_rate for run in task_runs]
    avg_times = [run.avg_time for run in task_runs]
    avg_steps = [run.avg_steps for run in task_runs]
    distance_errors = [run.avg_distance_error for run in task_runs]
    methods_payload[method] = {
        "success_rate_mean": _mean(success_rates),
        "success_rate_std": _std(success_rates),
        "avg_time_sec_mean": _mean(avg_times),
        "avg_time_sec_std": _std(avg_times),
        "avg_steps_mean": _mean(avg_steps),
        "avg_steps_std": _std(avg_steps),
        "avg_distance_error_mean": _mean(distance_errors),
        "avg_distance_error_std": _std(distance_errors),
        "executed_plan_stats": _aggregate_nested_plan_stats(task_runs),
        "reference_force_deviation_stats": _aggregate_nested_scalar_stats(task_runs),
        "planner_diagnostics": _aggregate_planner_diagnostics(task_runs),
        "failure_rates": _aggregate_failure_rates(task_runs),
    }
summary_rows.append(
    {
        "task_id": task.task_id,
        "task_label": task_label(task),
        "task_description": task.description,
        "task_split": task.split,
        "challenge_tags": list(task.challenge_tags),
        "seeds": seeds,
        "n_trials_per_seed": n_trials_per_task,
        "methods": methods_payload,
    }
)
```

- [ ] **Step 4: Re-run the Schema V2 benchmark test file**

Run:

```bash
./venv/bin/python -m unittest tests.test_benchmark_schema -v
```

Expected: `OK`.

- [ ] **Step 5: Commit the nested comparison / multi-seed schema**

```bash
git add simulation/runner.py tests/test_benchmark_schema.py
git commit -m "Switch benchmark comparison outputs to schema v2"
```

## Task 4: Teach `visualize_results.py` and `generate_showcase.py` to read Schema V2

**Files:**
- Modify: `reporting/visualize_results.py`
- Modify: `reporting/generate_showcase.py`
- Create: `tests/test_reporting_schema.py`

- [ ] **Step 1: Write failing reporting helper tests**

Create `tests/test_reporting_schema.py` with:

```python
import unittest

from reporting import generate_showcase as showcase
from reporting import visualize_results as viz


class ReportingSchemaTest(unittest.TestCase):
    def test_visualize_helpers_read_nested_method_payload(self):
        row = {
            "task_id": "pick_demo",
            "methods": {
                "rag_feedback": {
                    "success_rate": 0.8,
                    "executed_plan_stats": {
                        "mean": {
                            "gripper_force": 13.5,
                            "transport_velocity": 0.22,
                            "lift_clearance": 0.06,
                        }
                    },
                }
            },
        }
        self.assertEqual(viz._method_metric(row, "rag_feedback", "success_rate"), 0.8)
        self.assertEqual(viz._plan_mean(row, "rag_feedback", "gripper_force"), 13.5)
        self.assertEqual(viz._plan_mean(row, "rag_feedback", "transport_velocity"), 0.22)

    def test_showcase_helpers_read_nested_multi_seed_payload(self):
        row = {
            "task_id": "pick_demo",
            "methods": {
                "rag_feedback": {"success_rate_mean": 0.84},
                "task_heuristic": {"success_rate_mean": 0.61},
            },
        }
        self.assertEqual(showcase._method_metric(row, "rag_feedback", "success_rate_mean"), 0.84)
        self.assertEqual(showcase._method_metric(row, "task_heuristic", "success_rate_mean"), 0.61)
```

- [ ] **Step 2: Run the reporting helper tests and confirm failure**

Run:

```bash
./venv/bin/python -m unittest tests.test_reporting_schema -v
```

Expected: `ERROR` because `_method_metric()` / `_plan_mean()` do not exist yet and the reporting code still expects flat keys like `rag_success_rate_mean`.

- [ ] **Step 3: Add Schema V2 access helpers and switch the reporting code**

In `reporting/visualize_results.py`, add:

```python
def _method_entry(row: dict, method: str) -> dict:
    return row.get("methods", {}).get(method, {})


def _method_metric(row: dict, method: str, field: str, default=np.nan):
    return _method_entry(row, method).get(field, default)


def _plan_mean(row: dict, method: str, field: str, default=np.nan):
    return (
        _method_entry(row, method)
        .get("executed_plan_stats", {})
        .get("mean", {})
        .get(field, default)
    )
```

Then update plot readers, for example:

```python
forces = [_plan_mean(row, method, "gripper_force") for row in sim_rows]
velocities = [_plan_mean(row, method, "transport_velocity") for row in sim_rows]
clearances = [_plan_mean(row, method, "lift_clearance") for row in sim_rows]
```

In `reporting/generate_showcase.py`, add:

```python
def _method_metric(row: dict, method: str, field: str, default=None):
    return row.get("methods", {}).get(method, {}).get(field, default)


def _plan_mean(row: dict, method: str, field: str, default=None):
    return (
        row.get("methods", {})
        .get(method, {})
        .get("executed_plan_stats", {})
        .get("mean", {})
        .get(field, default)
    )
```

Then replace flat-key access such as:

```python
row["rag_success_rate_mean"]
```

with:

```python
_method_metric(row, "rag_feedback", "success_rate_mean")
```

- [ ] **Step 4: Re-run the reporting tests**

Run:

```bash
./venv/bin/python -m unittest tests.test_reporting_schema -v
```

Expected: `OK`.

- [ ] **Step 5: Commit the reporting Schema V2 update**

```bash
git add reporting/visualize_results.py reporting/generate_showcase.py tests/test_reporting_schema.py
git commit -m "Update reporting tools for schema v2"
```

## Task 5: Remove the eager `RAGController` package import

**Files:**
- Modify: `simulation/__init__.py`
- Create: `tests/test_package_imports.py`

- [ ] **Step 1: Write the import-boundary tests**

Create `tests/test_package_imports.py`:

```python
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class SimulationPackageImportTest(unittest.TestCase):
    def _run_blocked_import(self, module_name: str):
        code = f"""
import builtins
real_import = builtins.__import__

def blocked(name, globals=None, locals=None, fromlist=(), level=0):
    if name.startswith("langchain") or name.startswith("langchain_community"):
        raise ImportError("blocked for import-boundary test")
    return real_import(name, globals, locals, fromlist, level)

builtins.__import__ = blocked
import {module_name}
print("ok")
"""
        return subprocess.run(
            [sys.executable, "-c", code],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )

    def test_importing_simulation_control_core_does_not_pull_rag_dependencies(self):
        proc = self._run_blocked_import("simulation.control_core")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("ok", proc.stdout)

    def test_importing_simulation_env_does_not_pull_rag_dependencies(self):
        proc = self._run_blocked_import("simulation.env")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("ok", proc.stdout)
```

- [ ] **Step 2: Run the import-boundary tests and confirm failure**

Run:

```bash
./venv/bin/python -m unittest tests.test_package_imports -v
```

Expected: `FAIL` because `simulation/__init__.py` currently imports `RAGController`, which pulls `langchain` and `Chroma`.

- [ ] **Step 3: Remove the eager package import**

Update `simulation/__init__.py` to:

```python
"""Simulation package exports with minimal import-time dependencies."""

from .env import ArmSimEnv, HAS_MUJOCO

__all__ = ["ArmSimEnv", "HAS_MUJOCO"]
```

Do not re-export `RAGController` from the package root.

- [ ] **Step 4: Re-run the import-boundary tests**

Run:

```bash
./venv/bin/python -m unittest tests.test_package_imports -v
```

Expected: `OK`.

- [ ] **Step 5: Commit the import-boundary fix**

```bash
git add simulation/__init__.py tests/test_package_imports.py
git commit -m "Decouple simulation package root from rag imports"
```

## Task 6: Align default parsers and the run script with the simulation current profile

**Files:**
- Modify: `simulation/benchmark.py`
- Modify: `reporting/visualize_results.py`
- Modify: `reporting/generate_showcase.py`
- Modify: `scripts/run_all.sh`
- Create: `tests/test_current_profile.py`

- [ ] **Step 1: Write the current-profile default tests**

Create `tests/test_current_profile.py`:

```python
import unittest
from pathlib import Path

from reporting import generate_showcase as showcase
from reporting import visualize_results as viz
from simulation import benchmark


class CurrentProfileTest(unittest.TestCase):
    def test_benchmark_parser_defaults_point_to_simulation_current(self):
        args = benchmark.build_parser().parse_args([])
        self.assertEqual(args.method, "rag_feedback")
        self.assertEqual(args.output_dir, "outputs/current_observer_step_replan")
        self.assertEqual(args.output, "outputs/current_observer_step_replan/simulation_benchmark_result.json")

    def test_reporting_parser_defaults_point_to_simulation_current(self):
        viz_args = viz.build_parser().parse_args([])
        self.assertEqual(
            viz_args.sim_json,
            "outputs/current_observer_step_replan/simulation_comparison_rag_vs_baseline.json",
        )
        self.assertEqual(
            viz_args.sim_multi_seed_json,
            "outputs/current_observer_step_replan/simulation_comparison_multi_seed.json",
        )
        self.assertEqual(
            viz_args.output_dir,
            "outputs/current_observer_step_replan/visualizations",
        )
        showcase_args = showcase.build_parser().parse_args([])
        self.assertEqual(
            showcase_args.sim_benchmark_json,
            "outputs/current_observer_step_replan/simulation_benchmark_result.json",
        )
        self.assertEqual(
            showcase_args.output,
            "outputs/current_observer_step_replan/showcase_summary.txt",
        )

    def test_run_all_script_targets_rag_feedback_simulation_current(self):
        text = Path("scripts/run_all.sh").read_text(encoding="utf-8")
        self.assertIn("outputs/current_observer_step_replan", text)
        self.assertIn("--method rag_feedback", text)
        self.assertIn("--multi_seed_methods rag rag_feedback", text)
```

- [ ] **Step 2: Run the default-profile tests and confirm failure**

Run:

```bash
./venv/bin/python -m unittest tests.test_current_profile -v
```

Expected: `FAIL` because the parsers and script still point to `outputs/current` and default `method=rag`.

- [ ] **Step 3: Switch the parsers and script to the simulation current profile**

In `simulation/benchmark.py`, add `build_parser()` and switch defaults:

```python
DEFAULT_OUTPUT_DIR = Path("outputs/current_observer_step_replan")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RAG 驱动机械臂仿真 Benchmark CLI")
    parser.add_argument("--data_path", default="mechanical_data.txt")
    parser.add_argument("--n_trials", type=int, default=10)
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_DIR / "simulation_benchmark_result.json"))
    parser.add_argument(
        "--method",
        choices=("rag", "rag_generic_only", "rag_no_motion_rules", "rag_multi", "rag_random", "rag_llm", "direct_llm", "rag_learned", "rag_feedback", "fixed", "task_heuristic", "random"),
        default="rag_feedback",
    )
    parser.add_argument("--max_feedback_retries", type=int, default=1)
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--compare_multi_seed", action="store_true")
    parser.add_argument("--report_multi_seed", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--multi_seed_methods", nargs="+", default=["rag", "rag_feedback", "task_heuristic", "fixed"])
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser
```

In `reporting/visualize_results.py`:

```python
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qa_json", default="outputs/current/qa_evaluation_detail.json")
    parser.add_argument("--sim_json", default="outputs/current_observer_step_replan/simulation_comparison_rag_vs_baseline.json")
    parser.add_argument("--sim_multi_seed_json", default="outputs/current_observer_step_replan/simulation_comparison_multi_seed.json")
    parser.add_argument("--output_dir", default="outputs/current_observer_step_replan/visualizations")
    return parser
```

In `reporting/generate_showcase.py`:

```python
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qa_json", default="outputs/current/qa_evaluation_detail.json")
    parser.add_argument("--sim_json", default="outputs/current_observer_step_replan/simulation_comparison_rag_vs_baseline.json")
    parser.add_argument("--sim_multi_seed_json", default="outputs/current_observer_step_replan/simulation_comparison_multi_seed.json")
    parser.add_argument("--sim_benchmark_json", default="outputs/current_observer_step_replan/simulation_benchmark_result.json")
    parser.add_argument("--output", default="outputs/current_observer_step_replan/showcase_summary.txt")
    return parser
```

In `scripts/run_all.sh`, switch:

```bash
SIM_OUTPUT_DIR="${PROJECT_ROOT}/outputs/current_observer_step_replan"
SIM_VIS_DIR="${SIM_OUTPUT_DIR}/visualizations"
python -m simulation.benchmark \
  --report_multi_seed \
  --method rag_feedback \
  --n_trials 20 \
  --seeds 42 43 44 \
  --output "${SIM_OUTPUT_DIR}/simulation_benchmark_result.json"
python -m simulation.benchmark \
  --compare_multi_seed \
  --n_trials 20 \
  --seeds 42 43 44 \
  --multi_seed_methods rag rag_feedback task_heuristic fixed \
  --output_dir "${SIM_OUTPUT_DIR}"
```

- [ ] **Step 4: Re-run the current-profile tests**

Run:

```bash
./venv/bin/python -m unittest tests.test_current_profile -v
```

Expected: `OK`.

- [ ] **Step 5: Commit the current-profile default changes**

```bash
git add simulation/benchmark.py reporting/visualize_results.py reporting/generate_showcase.py scripts/run_all.sh tests/test_current_profile.py
git commit -m "Align default simulation profile with current outputs"
```

## Task 7: Regenerate simulation current outputs and sync the docs

**Files:**
- Modify: `README.md`
- Modify: `docs/overview.md`
- Modify: `docs/DESIGN.md`
- Modify: `simulation/README.md`
- Refresh: `outputs/current_observer_step_replan/*`

- [ ] **Step 1: Run the full test suite before regenerating outputs**

Run:

```bash
./venv/bin/python -m unittest discover -s tests -p 'test_*.py'
```

Expected: `OK`.

- [ ] **Step 2: Regenerate the simulation current benchmark, comparison, multi-seed, visuals, and showcase**

Run:

```bash
./venv/bin/python -m simulation.benchmark --report_multi_seed --method rag_feedback --n_trials 20 --seeds 42 43 44 --output outputs/current_observer_step_replan/simulation_benchmark_result.json
./venv/bin/python -m simulation.benchmark --compare_feedback --n_trials 20 --seed 42 --output_dir outputs/current_observer_step_replan
./venv/bin/python -m simulation.benchmark --compare_multi_seed --n_trials 20 --seeds 42 43 44 --multi_seed_methods rag rag_feedback task_heuristic fixed --output_dir outputs/current_observer_step_replan
./venv/bin/python reporting/visualize_results.py --qa_json outputs/current/qa_evaluation_detail.json --sim_json outputs/current_observer_step_replan/simulation_comparison_rag_vs_baseline.json --sim_multi_seed_json outputs/current_observer_step_replan/simulation_comparison_multi_seed.json --output_dir outputs/current_observer_step_replan/visualizations
./venv/bin/python reporting/generate_showcase.py --qa_json outputs/current/qa_evaluation_detail.json --sim_json outputs/current_observer_step_replan/simulation_comparison_rag_vs_baseline.json --sim_multi_seed_json outputs/current_observer_step_replan/simulation_comparison_multi_seed.json --sim_benchmark_json outputs/current_observer_step_replan/simulation_benchmark_result.json --output outputs/current_observer_step_replan/showcase_summary.txt
```

Expected: each command exits `0` and rewrites the simulation current artifacts.

- [ ] **Step 3: Verify the new artifacts and Schema V2 fields**

Run:

```bash
find outputs/current_observer_step_replan -maxdepth 2 \( -name 'simulation_benchmark_result.json' -o -name 'simulation_benchmark_trial_records.json' -o -name 'simulation_comparison_rag_vs_baseline.json' -o -name 'simulation_comparison_multi_seed.json' -o -name 'showcase_summary.txt' -o -path 'outputs/current_observer_step_replan/visualizations/simulation_control_plan.png' \) | sort
grep -R "\"params_used\"" outputs/current_observer_step_replan || true
```

Expected:

- `find` lists the regenerated files, including `simulation_benchmark_trial_records.json`
- `grep` returns no matches for `"params_used"`

- [ ] **Step 4: Update the docs to describe the new schema and regenerated current results**

Apply the following doc-level changes:

```markdown
- Replace prose that describes benchmark control-plan fields as `params_used`.
- Describe `seed_plan` as the initial planner proposal.
- Describe `executed_plan_stats` as task-level terminal-plan aggregates.
- Describe `simulation_benchmark_trial_records.json` as the detailed execution log artifact.
- Update the default simulation commands to `rag_feedback + outputs/current_observer_step_replan`.
- Refresh any metric bullets and file lists using the regenerated outputs from Step 2.
```

Touch these files explicitly:

- `README.md`
- `docs/overview.md`
- `docs/DESIGN.md`
- `simulation/README.md`

- [ ] **Step 5: Commit the regenerated outputs and synced docs**

```bash
git add README.md docs/overview.md docs/DESIGN.md simulation/README.md outputs/current_observer_step_replan
git commit -m "Refresh simulation current outputs and docs for schema v2"
```

## Final Verification

- [ ] **Step 1: Run the full test suite one last time**

```bash
./venv/bin/python -m unittest discover -s tests -p 'test_*.py'
```

Expected: `OK`.

- [ ] **Step 2: Check the working tree and latest commits**

```bash
git status --short --branch
git log --oneline -5
```

Expected:

- clean working tree
- recent commits match the task sequence above

- [ ] **Step 3: Sanity-check the simulation current docs against real files**

```bash
find outputs/current_observer_step_replan -maxdepth 2 -type f | sort | head -n 40
grep -n "current_observer_step_replan\\|seed_plan\\|executed_plan_stats\\|trial_records" README.md docs/overview.md docs/DESIGN.md simulation/README.md
```

Expected: the docs reference the new schema fields and real simulation-current files.
