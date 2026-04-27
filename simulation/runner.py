"""Benchmark runner split out from the CLI wrapper."""

from __future__ import annotations

from collections import Counter
import random
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from . import baseline_controller as baseline
from .env import HAS_MUJOCO, ArmSimEnv, _estimate_force_window, _estimate_motion_targets, _success_model
from .rag_controller import RAGController
from .reporting import (
    approach_height_error,
    reference_approach_height,
    reference_force_center,
    reference_force_deviation,
    task_label,
    write_json,
    write_table,
)
from .seed_utils import stable_seed_offset
from .tasks import BENCHMARK_TASKS, TaskConfig


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
    reference_force_deviation: float
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
    params_used: dict
    method: str = "rag"


def _summarize_params(params: dict) -> dict:
    kept = {}
    for key in (
        "gripper_force",
        "lift_force",
        "transfer_force",
        "transfer_alignment",
        "approach_height",
        "transport_velocity",
        "placement_velocity",
        "lift_clearance",
        "uncertainty_std",
        "confidence",
        "evidence_state_summary",
        "belief_state",
        "task_constraints",
        "uncertainty_profile",
        "stage_plan",
        "belief_state_coverage",
        "uncertainty_conservative_mode",
        "uncertainty_reasons",
        "solver_mode",
        "solver_selected_candidate",
        "solver_selected_score",
        "solver_score_breakdown",
        "solver_candidate_scores",
        "solver_seed_candidate",
        "solver_seed_score",
        "solver_local_search_iterations",
        "solver_local_search_improvement",
        "solver_local_search_trace",
        "solver_adjustment_notes",
        "evidence_rule_count",
        "evidence_support_score",
        "evidence_conflict_count",
        "force_rule_mode",
        "motion_rule_mode",
        "matched_hint_keywords",
        "pre_solve_plan",
        "available_specific_force_rules",
        "suppressed_specific_force_rules",
        "available_motion_rules",
        "suppressed_motion_rules",
        "available_support_contact_rules",
        "available_numeric_motion_rules",
        "available_alignment_rules",
        "available_lift_stage_rules",
        "used_specific_force_rules",
        "used_generic_force_rules",
        "used_motion_rules",
        "used_support_contact_rules",
        "used_alignment_rules",
        "used_lift_stage_rules",
        "belief_force_floor",
        "composite_force_floor",
        "motion_aware_force_floor",
        "thin_wall_support_force_floor",
        "rubber_material_force_floor",
        "heavy_metal_force_center_floor",
        "dynamic_heavy_metal_force_center_floor",
        "long_transfer_force_center_floor",
        "long_transfer_stage_force_floor",
        "long_transfer_dynamic_force_margin",
        "long_transfer_velocity_band",
        "long_transfer_placement_velocity_cap",
        "long_transfer_lift_force_target",
        "long_transfer_lift_force_margin",
        "long_transfer_alignment_target",
        "long_transfer_alignment_force_margin",
        "dynamic_transport_mode",
        "high_speed_transfer_force_margin",
        "high_speed_placement_velocity_target",
        "motion_aware_force_cap",
        "dynamic_smooth_metal_force_cap",
        "dynamic_smooth_metal_clearance_floor",
        "static_smooth_metal_force_cap",
        "thin_wall_support_clearance_cap",
        "rubber_material_clearance_floor",
        "long_transfer_numeric_clearance_floor",
        "long_transfer_clearance_target",
        "motion_path_force_compensation",
        "calibration_notes",
        "feedback_adjusted",
        "feedback_adjustment_type",
        "feedback_stage_adjustments",
        "feedback_replan_trace",
        "selected_evidence",
        "selected_rule_types",
    ):
        if key in params:
            kept[key] = params[key]
    return kept


def _get_param_getter(method: str, data_path: str, seed: int) -> Callable[[str], dict]:
    if method == "rag":
        rag = RAGController(data_path)
        return lambda desc: rag.get_params_for_task(desc, retrieval="single")
    if method == "rag_generic_only":
        rag = RAGController(data_path)
        return lambda desc: rag.get_params_for_task(
            desc,
            retrieval="single",
            force_rule_mode="generic_only",
        )
    if method == "rag_no_motion_rules":
        rag = RAGController(data_path)
        return lambda desc: rag.get_params_for_task(
            desc,
            retrieval="single",
            motion_rule_mode="disabled",
        )
    if method == "rag_multi":
        rag = RAGController(data_path)
        return lambda desc: rag.get_params_for_task(desc, retrieval="multi")
    if method == "rag_random":
        rag = RAGController(data_path)
        return lambda desc: rag.get_params_for_task(
            desc,
            retrieval="random",
            seed=seed + stable_seed_offset(desc),
        )
    if method == "rag_llm":
        from llm_loader import get_llm

        llm = get_llm(max_new_tokens=128)
        rag = RAGController(data_path)
        return lambda desc: rag.get_params_for_task_llm(desc, llm, retrieval="single")
    if method == "direct_llm":
        from llm_loader import get_llm

        llm = get_llm(max_new_tokens=128)
        return lambda desc: baseline.get_params_llm_direct(desc, llm)
    if method == "rag_learned":
        from .learned_controller import LearnedParamController

        learned = LearnedParamController(data_path=data_path)
        return lambda desc: learned.get_params_for_task(desc)
    if method == "rag_feedback":
        rag = RAGController(data_path)
        return lambda desc: rag.get_params_for_task(desc, retrieval="single")
    if method == "fixed":
        return lambda desc: baseline.get_params_fixed(desc, seed=seed)
    if method == "task_heuristic":
        return lambda desc: baseline.get_params_task_heuristic(desc, seed=seed)
    if method == "random":
        return lambda desc: baseline.get_params_random(desc, seed=seed)
    raise ValueError(f"未知 method: {method}")


def _get_feedback_controller(method: str, data_path: str):
    if method != "rag_feedback":
        return None
    return RAGController(data_path)


def _wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    margin = z * ((p * (1 - p) + z * z / (4 * n)) / n) ** 0.5 / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def _run_surrogate_trial(task: TaskConfig, params: dict, rng: random.Random | None = None) -> tuple[bool, float, dict]:
    profile = task.profile.__dict__ if task.profile is not None else {}
    mass_kg = float(profile.get("mass_kg", 0.05))
    surface_friction = float(profile.get("surface_friction", 0.5))
    fragility = float(profile.get("fragility", 0.7))
    velocity_scale = float(profile.get("velocity_scale", 0.8))
    preferred_height = float(profile.get("preferred_approach_height", 0.05))
    tolerance = float(profile.get("approach_height_tolerance", 0.02))
    size_xyz = tuple(profile.get("size_xyz", (0.025, 0.025, 0.025)))
    transport_velocity = float(params.get("transport_velocity", 0.3))
    lift_force = float(params.get("lift_force", params["gripper_force"]))
    transfer_force = float(params.get("transfer_force", params["gripper_force"]))
    transfer_alignment = float(params.get("transfer_alignment", 0.0))
    placement_velocity = float(params.get("placement_velocity", transport_velocity))
    lift_clearance = float(params.get("lift_clearance", 0.06))
    horizontal_distance = ((task.target_pos[0] - task.object_pos[0]) ** 2 + (task.target_pos[1] - task.object_pos[1]) ** 2) ** 0.5
    travel_distance = horizontal_distance + 2.0 * float(params.get("approach_height", 0.05)) + 2.0 * lift_clearance
    min_force_needed, max_safe_force, nominal_force = _estimate_force_window(
        mass_kg=mass_kg,
        surface_friction=surface_friction,
        fragility=fragility,
        travel_distance=travel_distance,
        size_xyz=size_xyz,
    )
    recommended_velocity, min_lift_clearance = _estimate_motion_targets(
        mass_kg=mass_kg,
        surface_friction=surface_friction,
        fragility=fragility,
        size_xyz=size_xyz,
    )
    success, diag = _success_model(
        float(params["gripper_force"]),
        min_force_needed=min_force_needed,
        max_safe_force=max_safe_force,
        nominal_force=nominal_force,
        surface_friction=surface_friction,
        mass_kg=mass_kg,
        fragility=fragility,
        travel_distance=travel_distance,
        horizontal_distance=horizontal_distance,
        approach_height=float(params.get("approach_height", 0.05)),
        transport_velocity=transport_velocity,
        lift_force=lift_force,
        transfer_force=transfer_force,
        transfer_alignment=transfer_alignment,
        placement_velocity=placement_velocity,
        lift_clearance=lift_clearance,
        recommended_velocity=recommended_velocity,
        min_lift_clearance=min_lift_clearance,
        preferred_approach_height=preferred_height,
        approach_height_tolerance=tolerance,
        rng=rng,
    )
    distance = 0.0 if success else 0.03 + 0.04 * max(diag["slip_risk"], diag["compression_risk"])
    sim_elapsed = round(0.35 + travel_distance / max(0.12, transport_velocity), 4)
    info = {
        "distance": round(distance, 4),
        "steps": 12,
        "sim_time": sim_elapsed,
        "wall_time": sim_elapsed,
        "travel_distance": round(travel_distance, 4),
        "horizontal_distance": round(horizontal_distance, 4),
        "transport_velocity": round(transport_velocity, 4),
        "lift_force": round(lift_force, 4),
        "transfer_force": round(transfer_force, 4),
        "transfer_alignment": round(transfer_alignment, 4),
        "placement_velocity": round(placement_velocity, 4),
        "lift_clearance": round(lift_clearance, 4),
        "force_gain": 0.0,
        "force_clip": 0.0,
        "static_push_estimate": 0.0,
        "approach_height_error": round(abs(float(params.get("approach_height", 0.05)) - preferred_height), 4),
        "slip_risk": diag["slip_risk"],
        "compression_risk": diag["compression_risk"],
        "velocity_risk": diag["velocity_risk"],
        "clearance_risk": diag["clearance_risk"],
        "lift_hold_risk": diag["lift_hold_risk"],
        "transfer_sway_risk": diag["transfer_sway_risk"],
        "placement_settle_risk": diag["placement_settle_risk"],
        "physics_ok": True,
        "grip_success": success,
        "failure_bucket": diag["failure_bucket"],
        "dominant_failure_mode": diag["dominant_failure_mode"],
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
        "placement_stage_control_active": diag["placement_stage_control_active"],
        "placement_transfer_carryover": diag["placement_transfer_carryover"],
        "placement_clearance_excess": diag["placement_clearance_excess"],
        "profile": profile,
    }
    return success, sim_elapsed, info


def _serialize_results(results: list[BenchmarkResult]) -> list[dict]:
    rows = []
    for result in results:
        task = next(item for item in BENCHMARK_TASKS if item.task_id == result.task_id)
        rows.append(
            {
                "task_id": result.task_id,
                "task_label": task_label(task),
                "task_description": result.task_description,
                "task_split": result.task_split,
                "challenge_tags": list(task.challenge_tags),
                "method": result.method,
                "n_trials": result.n_trials,
                "reference_force_range": list(result.reference_force_range),
                "reference_force_center": round(reference_force_center(task), 4),
                "reference_approach_height": reference_approach_height(task),
                "success_count": result.success_count,
                "success_rate": round(result.success_rate, 4),
                "avg_time_sec": round(result.avg_time, 4),
                "avg_steps": round(result.avg_steps, 2),
                "avg_distance_error": round(result.avg_distance_error, 4),
                "success_rate_ci95": [round(result.ci95_low, 4), round(result.ci95_high, 4)],
                "reference_force_deviation": round(result.reference_force_deviation, 4),
                "avg_slip_risk": round(result.avg_slip_risk, 4),
                "avg_compression_risk": round(result.avg_compression_risk, 4),
                "avg_velocity_risk": round(result.avg_velocity_risk, 4),
                "avg_clearance_risk": round(result.avg_clearance_risk, 4),
                "avg_lift_hold_risk": round(result.avg_lift_hold_risk, 4),
                "avg_transfer_sway_risk": round(result.avg_transfer_sway_risk, 4),
                "avg_placement_settle_risk": round(result.avg_placement_settle_risk, 4),
                "avg_stability_score": round(result.avg_stability_score, 4),
                "physics_fail_rate": round(result.physics_fail_rate, 4),
                "lift_hold_fail_rate": round(result.lift_hold_fail_rate, 4),
                "transfer_sway_fail_rate": round(result.transfer_sway_fail_rate, 4),
                "placement_settle_fail_rate": round(result.placement_settle_fail_rate, 4),
                "dominant_failure_mode": result.dominant_failure_mode,
                "belief_state_coverage": result.params_used.get("belief_state_coverage"),
                "uncertainty_conservative_mode": result.params_used.get("uncertainty_conservative_mode"),
                "solver_selected_candidate": result.params_used.get("solver_selected_candidate"),
                "evidence_rule_count": result.params_used.get("evidence_rule_count"),
                "evidence_support_score": result.params_used.get("evidence_support_score"),
                "evidence_conflict_count": result.params_used.get("evidence_conflict_count"),
                "force_rule_mode": result.params_used.get("force_rule_mode"),
                "motion_rule_mode": result.params_used.get("motion_rule_mode"),
                "available_specific_force_rules": result.params_used.get("available_specific_force_rules"),
                "suppressed_specific_force_rules": result.params_used.get("suppressed_specific_force_rules"),
                "available_motion_rules": result.params_used.get("available_motion_rules"),
                "suppressed_motion_rules": result.params_used.get("suppressed_motion_rules"),
                "params": result.params_used,
            }
        )
    return rows


def run_benchmark(
    data_path: str = "mechanical_data.txt",
    n_trials_per_task: int = 10,
    gui: bool = False,
    seed: int = 42,
    output_path: str | None = "outputs/current/simulation_benchmark_result.json",
    method: str = "rag",
    max_feedback_retries: int = 1,
) -> list[BenchmarkResult]:
    get_params = _get_param_getter(method, data_path, seed)
    feedback_controller = _get_feedback_controller(method, data_path)
    results: list[BenchmarkResult] = []

    for task in BENCHMARK_TASKS:
        params = get_params(task.description)
        env = ArmSimEnv(gui=gui, seed=seed + stable_seed_offset(task.task_id)) if HAS_MUJOCO else None
        surrogate_rng = None if HAS_MUJOCO else random.Random(seed + stable_seed_offset(task.task_id))
        successes = 0
        times: list[float] = []
        step_counts: list[float] = []
        distance_errors: list[float] = []
        slip_risks: list[float] = []
        compression_risks: list[float] = []
        velocity_risks: list[float] = []
        clearance_risks: list[float] = []
        lift_hold_risks: list[float] = []
        transfer_sway_risks: list[float] = []
        placement_settle_risks: list[float] = []
        stability_scores: list[float] = []
        failure_counts: Counter[str] = Counter()
        last_params_used = _summarize_params(params)

        for _ in range(n_trials_per_task):
            current_params = dict(params)
            if HAS_MUJOCO and env is not None:
                success, elapsed, info = env.execute_pick_place(
                    object_pos=task.object_pos,
                    target_pos=task.target_pos,
                    gripper_force=float(current_params["gripper_force"]),
                    approach_height=float(current_params.get("approach_height", 0.05)),
                    transport_velocity=float(current_params.get("transport_velocity", 0.3)),
                    lift_force=float(current_params.get("lift_force", current_params["gripper_force"])),
                    transfer_force=float(current_params.get("transfer_force", current_params["gripper_force"])),
                    transfer_alignment=float(current_params.get("transfer_alignment", 0.0)),
                    placement_velocity=float(current_params.get("placement_velocity", current_params.get("transport_velocity", 0.3))),
                    lift_clearance=float(current_params.get("lift_clearance", 0.06)),
                    object_profile=task.profile.__dict__ if task.profile is not None else None,
                )
            else:
                success, elapsed, info = _run_surrogate_trial(task, current_params, rng=surrogate_rng)

            retries = 0
            while not success and feedback_controller is not None and retries < max_feedback_retries:
                current_params = feedback_controller.get_params_after_feedback(
                    task.description,
                    current_params,
                    success,
                    info,
                )
                if HAS_MUJOCO and env is not None:
                    success, elapsed_retry, info = env.execute_pick_place(
                        object_pos=task.object_pos,
                        target_pos=task.target_pos,
                        gripper_force=float(current_params["gripper_force"]),
                        approach_height=float(current_params.get("approach_height", 0.05)),
                        transport_velocity=float(current_params.get("transport_velocity", 0.3)),
                        lift_force=float(current_params.get("lift_force", current_params["gripper_force"])),
                        transfer_force=float(current_params.get("transfer_force", current_params["gripper_force"])),
                        transfer_alignment=float(current_params.get("transfer_alignment", 0.0)),
                        placement_velocity=float(current_params.get("placement_velocity", current_params.get("transport_velocity", 0.3))),
                        lift_clearance=float(current_params.get("lift_clearance", 0.06)),
                        object_profile=task.profile.__dict__ if task.profile is not None else None,
                    )
                else:
                    success, elapsed_retry, info = _run_surrogate_trial(task, current_params, rng=surrogate_rng)
                elapsed += elapsed_retry
                retries += 1

            last_params_used = _summarize_params(current_params)
            successes += 1 if success else 0
            times.append(elapsed)
            step_counts.append(info.get("steps", 0))
            distance_errors.append(info.get("distance", 0.0))
            slip_risks.append(info.get("slip_risk", 0.0))
            compression_risks.append(info.get("compression_risk", 0.0))
            velocity_risks.append(info.get("velocity_risk", 0.0))
            clearance_risks.append(info.get("clearance_risk", 0.0))
            lift_hold_risks.append(info.get("lift_hold_risk", 0.0))
            transfer_sway_risks.append(info.get("transfer_sway_risk", 0.0))
            placement_settle_risks.append(info.get("placement_settle_risk", 0.0))
            stability_scores.append(info.get("stability_score", 0.0))
            failure_counts[info.get("failure_bucket", "unknown_failure")] += 1

        if env is not None:
            env.close()
        ci95_low, ci95_high = _wilson_interval(successes, n_trials_per_task)
        fail_rate_map = {
            "physics_fail": failure_counts.get("physics_fail", 0) / n_trials_per_task,
            "lift_hold_fail": failure_counts.get("lift_hold_fail", 0) / n_trials_per_task,
            "transfer_sway_fail": failure_counts.get("transfer_sway_fail", 0) / n_trials_per_task,
            "placement_settle_fail": failure_counts.get("placement_settle_fail", 0) / n_trials_per_task,
        }
        dominant_failure_mode = (
            max(fail_rate_map, key=fail_rate_map.get)
            if any(rate > 0.0 for rate in fail_rate_map.values())
            else "none"
        )
        results.append(
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
                reference_force_deviation=reference_force_deviation(
                    task,
                    float(last_params_used.get("gripper_force", 0.0)),
                ),
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
                params_used=last_params_used,
                method=method,
            )
        )

    if output_path:
        write_json(output_path, _serialize_results(results))
    return results


def run_benchmark_multi_seed_report(
    data_path: str = "mechanical_data.txt",
    n_trials_per_task: int = 10,
    seeds: list[int] | None = None,
    output_path: str | None = "outputs/current/simulation_benchmark_result.json",
    method: str = "rag",
) -> list[dict]:
    seeds = seeds or [42, 43, 44]
    per_seed_results = {
        seed: run_benchmark(
            data_path=data_path,
            n_trials_per_task=n_trials_per_task,
            gui=False,
            seed=seed,
            output_path=None,
            method=method,
        )
        for seed in seeds
    }

    summary_rows: list[dict] = []
    for task in BENCHMARK_TASKS:
        task_runs = [
            next(result for result in per_seed_results[seed] if result.task_id == task.task_id)
            for seed in seeds
        ]

        def _mean(values: list[float]) -> float:
            return round(statistics.mean(values), 4)

        def _std(values: list[float]) -> float:
            return round(statistics.stdev(values), 4) if len(values) > 1 else 0.0

        success_rates = [run.success_rate for run in task_runs]
        avg_times = [run.avg_time for run in task_runs]
        avg_steps = [run.avg_steps for run in task_runs]
        distance_errors = [run.avg_distance_error for run in task_runs]
        force_deviations = [run.reference_force_deviation for run in task_runs]
        slip_risks = [run.avg_slip_risk for run in task_runs]
        compression_risks = [run.avg_compression_risk for run in task_runs]
        velocity_risks = [run.avg_velocity_risk for run in task_runs]
        clearance_risks = [run.avg_clearance_risk for run in task_runs]
        lift_hold_risks = [run.avg_lift_hold_risk for run in task_runs]
        transfer_sway_risks = [run.avg_transfer_sway_risk for run in task_runs]
        placement_settle_risks = [run.avg_placement_settle_risk for run in task_runs]
        stability_scores = [run.avg_stability_score for run in task_runs]
        physics_fail_rates = [run.physics_fail_rate for run in task_runs]
        lift_hold_fail_rates = [run.lift_hold_fail_rate for run in task_runs]
        transfer_sway_fail_rates = [run.transfer_sway_fail_rate for run in task_runs]
        placement_settle_fail_rates = [run.placement_settle_fail_rate for run in task_runs]
        gripper_forces = [float(run.params_used.get("gripper_force", 0.0)) for run in task_runs]
        lift_forces = [
            float(run.params_used.get("lift_force", run.params_used.get("gripper_force", 0.0)))
            for run in task_runs
        ]
        transfer_forces = [
            float(run.params_used.get("transfer_force", run.params_used.get("gripper_force", 0.0)))
            for run in task_runs
        ]
        transfer_alignments = [
            float(run.params_used.get("transfer_alignment", 0.0))
            for run in task_runs
        ]
        approach_heights = [float(run.params_used.get("approach_height", 0.0)) for run in task_runs]
        transport_velocities = [float(run.params_used.get("transport_velocity", 0.0)) for run in task_runs]
        placement_velocities = [float(run.params_used.get("placement_velocity", run.params_used.get("transport_velocity", 0.0))) for run in task_runs]
        lift_clearances = [float(run.params_used.get("lift_clearance", 0.0)) for run in task_runs]
        dynamic_transport_modes = [
            str(run.params_used.get("dynamic_transport_mode", "static"))
            for run in task_runs
        ]
        failure_rate_means = {
            "physics_fail": _mean(physics_fail_rates),
            "lift_hold_fail": _mean(lift_hold_fail_rates),
            "transfer_sway_fail": _mean(transfer_sway_fail_rates),
            "placement_settle_fail": _mean(placement_settle_fail_rates),
        }

        summary_rows.append(
            {
                "task_id": task.task_id,
                "task_label": task_label(task),
                "task_description": task.description,
                "task_split": task.split,
                "challenge_tags": list(task.challenge_tags),
                "method": method,
                "seeds": seeds,
                "n_trials_per_seed": n_trials_per_task,
                "total_trials": n_trials_per_task * len(seeds),
                "reference_force_range": list(task.reference_force_range),
                "reference_force_center": round(reference_force_center(task), 4),
                "reference_approach_height": reference_approach_height(task),
                "success_rate_mean": _mean(success_rates),
                "success_rate_std": _std(success_rates),
                "avg_time_sec_mean": _mean(avg_times),
                "avg_time_sec_std": _std(avg_times),
                "avg_steps_mean": _mean(avg_steps),
                "avg_steps_std": _std(avg_steps),
                "avg_distance_error_mean": _mean(distance_errors),
                "avg_distance_error_std": _std(distance_errors),
                "reference_force_deviation_mean": _mean(force_deviations),
                "reference_force_deviation_std": _std(force_deviations),
                "avg_slip_risk_mean": _mean(slip_risks),
                "avg_slip_risk_std": _std(slip_risks),
                "avg_compression_risk_mean": _mean(compression_risks),
                "avg_compression_risk_std": _std(compression_risks),
                "avg_velocity_risk_mean": _mean(velocity_risks),
                "avg_velocity_risk_std": _std(velocity_risks),
                "avg_clearance_risk_mean": _mean(clearance_risks),
                "avg_clearance_risk_std": _std(clearance_risks),
                "avg_lift_hold_risk_mean": _mean(lift_hold_risks),
                "avg_lift_hold_risk_std": _std(lift_hold_risks),
                "avg_transfer_sway_risk_mean": _mean(transfer_sway_risks),
                "avg_transfer_sway_risk_std": _std(transfer_sway_risks),
                "avg_placement_settle_risk_mean": _mean(placement_settle_risks),
                "avg_placement_settle_risk_std": _std(placement_settle_risks),
                "avg_stability_score_mean": _mean(stability_scores),
                "avg_stability_score_std": _std(stability_scores),
                "physics_fail_rate_mean": failure_rate_means["physics_fail"],
                "physics_fail_rate_std": _std(physics_fail_rates),
                "lift_hold_fail_rate_mean": failure_rate_means["lift_hold_fail"],
                "lift_hold_fail_rate_std": _std(lift_hold_fail_rates),
                "transfer_sway_fail_rate_mean": failure_rate_means["transfer_sway_fail"],
                "transfer_sway_fail_rate_std": _std(transfer_sway_fail_rates),
                "placement_settle_fail_rate_mean": failure_rate_means["placement_settle_fail"],
                "placement_settle_fail_rate_std": _std(placement_settle_fail_rates),
                "dominant_failure_mode": (
                    max(failure_rate_means, key=failure_rate_means.get)
                    if any(value > 0.0 for value in failure_rate_means.values())
                    else "none"
                ),
                "gripper_force_mean": _mean(gripper_forces),
                "gripper_force_std": _std(gripper_forces),
                "lift_force_mean": _mean(lift_forces),
                "lift_force_std": _std(lift_forces),
                "transfer_force_mean": _mean(transfer_forces),
                "transfer_force_std": _std(transfer_forces),
                "transfer_alignment_mean": _mean(transfer_alignments),
                "transfer_alignment_std": _std(transfer_alignments),
                "approach_height_mean": _mean(approach_heights),
                "approach_height_std": _std(approach_heights),
                "transport_velocity_mean": _mean(transport_velocities),
                "transport_velocity_std": _std(transport_velocities),
                "placement_velocity_mean": _mean(placement_velocities),
                "placement_velocity_std": _std(placement_velocities),
                "lift_clearance_mean": _mean(lift_clearances),
                "lift_clearance_std": _std(lift_clearances),
                "dynamic_transport_mode": Counter(dynamic_transport_modes).most_common(1)[0][0],
                "belief_state_coverage_mean": _mean(
                    [float(run.params_used.get("belief_state_coverage", 0.0)) for run in task_runs]
                ),
                "uncertainty_conservative_mode_mean": _mean(
                    [1.0 if run.params_used.get("uncertainty_conservative_mode") else 0.0 for run in task_runs]
                ),
                "solver_selected_candidate": Counter(
                    [str(run.params_used.get("solver_selected_candidate", "rule_aggregate")) for run in task_runs]
                ).most_common(1)[0][0],
                "evidence_rule_count_mean": _mean(
                    [float(run.params_used.get("evidence_rule_count", 0.0)) for run in task_runs]
                ),
                "evidence_support_score_mean": _mean(
                    [float(run.params_used.get("evidence_support_score", 0.0)) for run in task_runs]
                ),
                "evidence_conflict_count_mean": _mean(
                    [float(run.params_used.get("evidence_conflict_count", 0.0)) for run in task_runs]
                ),
                "available_specific_force_rules_mean": _mean(
                    [1.0 if run.params_used.get("available_specific_force_rules") else 0.0 for run in task_runs]
                ),
                "suppressed_specific_force_rules_mean": _mean(
                    [1.0 if run.params_used.get("suppressed_specific_force_rules") else 0.0 for run in task_runs]
                ),
                "available_motion_rules_mean": _mean(
                    [1.0 if run.params_used.get("available_motion_rules") else 0.0 for run in task_runs]
                ),
                "suppressed_motion_rules_mean": _mean(
                    [1.0 if run.params_used.get("suppressed_motion_rules") else 0.0 for run in task_runs]
                ),
                "available_lift_stage_rules_mean": _mean(
                    [1.0 if run.params_used.get("available_lift_stage_rules") else 0.0 for run in task_runs]
                ),
                "used_lift_stage_rules_mean": _mean(
                    [1.0 if run.params_used.get("used_lift_stage_rules") else 0.0 for run in task_runs]
                ),
                "per_seed_success_rate": {str(seed): round(run.success_rate, 4) for seed, run in zip(seeds, task_runs)},
            }
        )

    if output_path:
        write_json(output_path, summary_rows)
        table_path = Path(output_path).with_suffix(".txt")
        lines = [
            f"{method} benchmark 多 seed 汇总结果",
            "=" * 108,
            f"seeds={seeds}, n_trials_per_seed={n_trials_per_task}, total_trials={n_trials_per_task * len(seeds)}",
            "",
            f"{'任务':<32} {'成功率(mean±std)':<18} {'稳定度(mean±std)':<18} {'时间(mean±std)':<18}",
            "-" * 108,
        ]
        for row in summary_rows:
            lines.append(
                f"{row['task_label']:<32} "
                f"{row['success_rate_mean']:.2%}±{row['success_rate_std']:.2%} "
                f"{row['avg_stability_score_mean']:.3f}±{row['avg_stability_score_std']:.3f} "
                f"{row['avg_time_sec_mean']:.3f}s±{row['avg_time_sec_std']:.3f}"
            )
            if row.get("belief_state_coverage_mean") is not None and row.get("solver_selected_candidate") is not None:
                lines.append(
                    " " * 4
                    + f"belief={row['belief_state_coverage_mean']:.3f}, "
                    + f"conservative_rate={row['uncertainty_conservative_mode_mean']:.2%}, "
                    + f"solver={row['solver_selected_candidate']}"
                )
        lines.append("")
        lines.append(f"JSON: {output_path}")
        write_table(table_path, lines)
    return summary_rows


def run_benchmark_comparison(
    data_path: str = "mechanical_data.txt",
    n_trials_per_task: int = 10,
    seed: int = 42,
    output_dir: str | None = None,
    methods: list[str] | None = None,
) -> dict[str, list[BenchmarkResult]]:
    methods = methods or ["rag", "task_heuristic", "fixed"]
    output_dir = Path(output_dir or "outputs/current")
    output_dir.mkdir(parents=True, exist_ok=True)
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

    comparison = []
    for task in BENCHMARK_TASKS:
        row = {
            "task_id": task.task_id,
            "task_label": task_label(task),
            "task_description": task.description,
            "task_split": task.split,
            "challenge_tags": list(task.challenge_tags),
            "reference_force_range": list(task.reference_force_range),
            "reference_force_center": round(reference_force_center(task), 4),
            "reference_approach_height": reference_approach_height(task),
        }
        for method in methods:
            result = next(r for r in all_results[method] if r.task_id == task.task_id)
            row[f"{method}_success_rate"] = round(result.success_rate, 4)
            row[f"{method}_success_count"] = result.success_count
            row[f"{method}_success_rate_ci95"] = [round(result.ci95_low, 4), round(result.ci95_high, 4)]
            row[f"{method}_gripper_force"] = result.params_used.get("gripper_force")
            row[f"{method}_lift_force"] = result.params_used.get("lift_force", result.params_used.get("gripper_force"))
            row[f"{method}_transfer_force"] = result.params_used.get("transfer_force")
            row[f"{method}_transfer_alignment"] = result.params_used.get("transfer_alignment")
            row[f"{method}_approach_height"] = result.params_used.get("approach_height")
            row[f"{method}_transport_velocity"] = result.params_used.get("transport_velocity")
            row[f"{method}_placement_velocity"] = result.params_used.get("placement_velocity")
            row[f"{method}_lift_clearance"] = result.params_used.get("lift_clearance")
            row[f"{method}_dynamic_transport_mode"] = result.params_used.get("dynamic_transport_mode")
            error = approach_height_error(task, result.params_used.get("approach_height"))
            row[f"{method}_approach_height_error"] = None if error is None else round(error, 4)
            row[f"{method}_avg_time_sec"] = round(result.avg_time, 4)
            row[f"{method}_avg_steps"] = round(result.avg_steps, 2)
            row[f"{method}_avg_distance_error"] = round(result.avg_distance_error, 4)
            row[f"{method}_reference_force_deviation"] = round(result.reference_force_deviation, 4)
            row[f"{method}_avg_velocity_risk"] = round(result.avg_velocity_risk, 4)
            row[f"{method}_avg_clearance_risk"] = round(result.avg_clearance_risk, 4)
            row[f"{method}_avg_lift_hold_risk"] = round(result.avg_lift_hold_risk, 4)
            row[f"{method}_avg_transfer_sway_risk"] = round(result.avg_transfer_sway_risk, 4)
            row[f"{method}_avg_placement_settle_risk"] = round(result.avg_placement_settle_risk, 4)
            row[f"{method}_avg_stability_score"] = round(result.avg_stability_score, 4)
            row[f"{method}_physics_fail_rate"] = round(result.physics_fail_rate, 4)
            row[f"{method}_lift_hold_fail_rate"] = round(result.lift_hold_fail_rate, 4)
            row[f"{method}_transfer_sway_fail_rate"] = round(result.transfer_sway_fail_rate, 4)
            row[f"{method}_placement_settle_fail_rate"] = round(result.placement_settle_fail_rate, 4)
            row[f"{method}_dominant_failure_mode"] = result.dominant_failure_mode
            row[f"{method}_evidence_rule_count"] = result.params_used.get("evidence_rule_count")
            row[f"{method}_evidence_support_score"] = result.params_used.get("evidence_support_score")
            row[f"{method}_evidence_conflict_count"] = result.params_used.get("evidence_conflict_count")
            row[f"{method}_force_rule_mode"] = result.params_used.get("force_rule_mode")
            row[f"{method}_motion_rule_mode"] = result.params_used.get("motion_rule_mode")
            row[f"{method}_available_specific_force_rules"] = result.params_used.get("available_specific_force_rules")
            row[f"{method}_suppressed_specific_force_rules"] = result.params_used.get("suppressed_specific_force_rules")
            row[f"{method}_available_motion_rules"] = result.params_used.get("available_motion_rules")
            row[f"{method}_suppressed_motion_rules"] = result.params_used.get("suppressed_motion_rules")
        comparison.append(row)

    comparison_path = output_dir / "simulation_comparison_rag_vs_baseline.json"
    write_json(comparison_path, comparison)
    table_path = output_dir / "simulation_comparison_rag_vs_baseline.txt"
    lines = [
        "RAG vs 基线 对比结果",
        "=" * 68,
        f"n_trials_per_task={n_trials_per_task}, seed={seed}",
        "",
        f"{'任务':<32} {'参考力范围(N)':<14} " + " ".join(f"{method}_成功率" for method in methods),
        "-" * 68,
    ]
    for task in BENCHMARK_TASKS:
        reference = f"{task.reference_force_range[0]}-{task.reference_force_range[1]}"
        rates = [f"{next(r for r in all_results[method] if r.task_id == task.task_id).success_rate:.2%}" for method in methods]
        lines.append(f"{task_label(task):<32} {reference:<14} " + " ".join(rates))
    lines.append("")
    lines.append(f"对比 JSON: {comparison_path}")
    write_table(table_path, lines)
    return all_results


def _write_split_summary(
    comparison_rows: list[dict],
    methods: list[str],
    output_dir: Path,
) -> None:
    split_rows = []
    split_order = ["train", "val", "test"]
    splits = [split for split in split_order if any(row["task_split"] == split for row in comparison_rows)]
    for split in splits:
        rows = [row for row in comparison_rows if row["task_split"] == split]
        summary = {"task_split": split, "n_tasks": len(rows)}
        for method in methods:
            values = [row.get(f"{method}_success_rate_mean") for row in rows if row.get(f"{method}_success_rate_mean") is not None]
            if values:
                summary[f"{method}_success_rate_mean"] = round(statistics.mean(values), 4)
                summary[f"{method}_success_rate_std"] = round(statistics.stdev(values), 4) if len(values) > 1 else 0.0
        split_rows.append(summary)

    write_json(output_dir / "simulation_split_summary.json", split_rows)
    lines = [
        "Simulation split 汇总（按 train/val/test 聚合）",
        "=" * 72,
        "",
        f"{'split':<12} {'n_tasks':<8} " + " ".join(f"{method}_成功率(mean±std)" for method in methods),
        "-" * 72,
    ]
    for row in split_rows:
        parts = []
        for method in methods:
            mean = row.get(f"{method}_success_rate_mean")
            std = row.get(f"{method}_success_rate_std")
            parts.append("N/A" if mean is None else f"{mean:.2%}±{std:.2%}")
        lines.append(f"{row['task_split']:<12} {row['n_tasks']:<8} " + " ".join(parts))
    write_table(output_dir / "simulation_split_summary.txt", lines)


def _write_challenge_summary(
    comparison_rows: list[dict],
    methods: list[str],
    output_dir: Path,
) -> None:
    tags = sorted(
        {
            tag
            for row in comparison_rows
            for tag in row.get("challenge_tags", [])
        }
    )
    summary_rows = []
    for tag in tags:
        rows = [row for row in comparison_rows if tag in row.get("challenge_tags", [])]
        summary = {"challenge_tag": tag, "n_tasks": len(rows)}
        for method in methods:
            values = [
                row.get(f"{method}_success_rate_mean")
                for row in rows
                if row.get(f"{method}_success_rate_mean") is not None
            ]
            if values:
                summary[f"{method}_success_rate_mean"] = round(statistics.mean(values), 4)
                summary[f"{method}_success_rate_std"] = round(statistics.stdev(values), 4) if len(values) > 1 else 0.0
        summary_rows.append(summary)

    write_json(output_dir / "simulation_challenge_summary.json", summary_rows)
    lines = [
        "Simulation challenge 汇总（按 challenge_tags 聚合）",
        "=" * 80,
        "",
        f"{'challenge':<18} {'n_tasks':<8} " + " ".join(f"{method}_成功率(mean±std)" for method in methods),
        "-" * 80,
    ]
    for row in summary_rows:
        parts = []
        for method in methods:
            mean = row.get(f"{method}_success_rate_mean")
            std = row.get(f"{method}_success_rate_std")
            parts.append("N/A" if mean is None else f"{mean:.2%}±{std:.2%}")
        lines.append(f"{row['challenge_tag']:<18} {row['n_tasks']:<8} " + " ".join(parts))
    write_table(output_dir / "simulation_challenge_summary.txt", lines)


def run_benchmark_comparison_multi_seed(
    data_path: str = "mechanical_data.txt",
    n_trials_per_task: int = 10,
    seeds: list[int] | None = None,
    output_dir: str | None = None,
    methods: list[str] | None = None,
) -> dict[tuple[str, str], list[float]]:
    seeds = seeds or [42, 43, 44]
    methods = methods or ["rag", "task_heuristic", "direct_llm", "fixed"]
    output_dir = Path(output_dir or "outputs/current")
    output_dir.mkdir(parents=True, exist_ok=True)

    rates: dict[tuple[str, str], list[float]] = {}
    results_by_seed: dict[int, dict[str, list[BenchmarkResult]]] = {}
    for seed in seeds:
        all_results = run_benchmark_comparison(
            data_path=data_path,
            n_trials_per_task=n_trials_per_task,
            seed=seed,
            output_dir=str(output_dir),
            methods=methods,
        )
        results_by_seed[seed] = all_results
        for method in methods:
            for result in all_results[method]:
                rates.setdefault((result.task_id, method), []).append(result.success_rate)

    comparison = []
    for task in BENCHMARK_TASKS:
        row = {
            "task_id": task.task_id,
            "task_label": task_label(task),
            "task_description": task.description,
            "task_split": task.split,
            "challenge_tags": list(task.challenge_tags),
            "reference_force_range": list(task.reference_force_range),
        }
        for method in methods:
            values = rates.get((task.task_id, method), [])
            row[f"{method}_success_rate_mean"] = round(statistics.mean(values), 4) if values else None
            row[f"{method}_success_rate_std"] = round(statistics.stdev(values), 4) if len(values) > 1 else 0.0
            task_runs = [
                next(result for result in results_by_seed[seed][method] if result.task_id == task.task_id)
                for seed in seeds
            ]
            evidence_scores = [
                float(run.params_used.get("evidence_support_score", 0.0))
                for run in task_runs
                if run.params_used.get("evidence_support_score") is not None
            ]
            conflict_counts = [
                float(run.params_used.get("evidence_conflict_count", 0.0))
                for run in task_runs
                if run.params_used.get("evidence_conflict_count") is not None
            ]
            rule_counts = [
                float(run.params_used.get("evidence_rule_count", 0.0))
                for run in task_runs
                if run.params_used.get("evidence_rule_count") is not None
            ]
            row[f"{method}_evidence_support_score_mean"] = (
                round(statistics.mean(evidence_scores), 4) if evidence_scores else None
            )
            row[f"{method}_evidence_conflict_count_mean"] = (
                round(statistics.mean(conflict_counts), 4) if conflict_counts else None
            )
            row[f"{method}_evidence_rule_count_mean"] = (
                round(statistics.mean(rule_counts), 4) if rule_counts else None
            )
            available_specific = [
                1.0 if run.params_used.get("available_specific_force_rules") else 0.0
                for run in task_runs
            ]
            suppressed_specific = [
                1.0 if run.params_used.get("suppressed_specific_force_rules") else 0.0
                for run in task_runs
            ]
            row[f"{method}_available_specific_force_rules_mean"] = (
                round(statistics.mean(available_specific), 4) if available_specific else None
            )
            row[f"{method}_suppressed_specific_force_rules_mean"] = (
                round(statistics.mean(suppressed_specific), 4) if suppressed_specific else None
            )
            available_motion = [
                1.0 if run.params_used.get("available_motion_rules") else 0.0
                for run in task_runs
            ]
            suppressed_motion = [
                1.0 if run.params_used.get("suppressed_motion_rules") else 0.0
                for run in task_runs
            ]
            placement_velocities = [
                float(run.params_used.get("placement_velocity", run.params_used.get("transport_velocity", 0.0)))
                for run in task_runs
            ]
            lift_forces = [
                float(run.params_used.get("lift_force", run.params_used.get("gripper_force", 0.0)))
                for run in task_runs
            ]
            transfer_forces = [
                float(run.params_used.get("transfer_force", run.params_used.get("gripper_force", 0.0)))
                for run in task_runs
            ]
            transfer_alignments = [
                float(run.params_used.get("transfer_alignment", 0.0))
                for run in task_runs
            ]
            dynamic_transport_modes = [
                str(run.params_used.get("dynamic_transport_mode", "static"))
                for run in task_runs
            ]
            belief_coverages = [
                float(run.params_used.get("belief_state_coverage", 0.0))
                for run in task_runs
                if run.params_used.get("belief_state_coverage") is not None
            ]
            conservative_modes = [
                1.0 if run.params_used.get("uncertainty_conservative_mode") else 0.0
                for run in task_runs
            ]
            solver_candidates = [
                str(run.params_used.get("solver_selected_candidate", "rule_aggregate"))
                for run in task_runs
            ]
            row[f"{method}_available_motion_rules_mean"] = (
                round(statistics.mean(available_motion), 4) if available_motion else None
            )
            row[f"{method}_suppressed_motion_rules_mean"] = (
                round(statistics.mean(suppressed_motion), 4) if suppressed_motion else None
            )
            row[f"{method}_belief_state_coverage_mean"] = (
                round(statistics.mean(belief_coverages), 4) if belief_coverages else None
            )
            row[f"{method}_uncertainty_conservative_mode_mean"] = (
                round(statistics.mean(conservative_modes), 4) if conservative_modes else None
            )
            row[f"{method}_solver_selected_candidate"] = (
                Counter(solver_candidates).most_common(1)[0][0] if solver_candidates else None
            )
            available_lift_stage = [
                1.0 if run.params_used.get("available_lift_stage_rules") else 0.0
                for run in task_runs
            ]
            used_lift_stage = [
                1.0 if run.params_used.get("used_lift_stage_rules") else 0.0
                for run in task_runs
            ]
            row[f"{method}_available_lift_stage_rules_mean"] = (
                round(statistics.mean(available_lift_stage), 4) if available_lift_stage else None
            )
            row[f"{method}_used_lift_stage_rules_mean"] = (
                round(statistics.mean(used_lift_stage), 4) if used_lift_stage else None
            )
            row[f"{method}_lift_force_mean"] = (
                round(statistics.mean(lift_forces), 4) if lift_forces else None
            )
            row[f"{method}_transfer_force_mean"] = (
                round(statistics.mean(transfer_forces), 4) if transfer_forces else None
            )
            row[f"{method}_transfer_alignment_mean"] = (
                round(statistics.mean(transfer_alignments), 4) if transfer_alignments else None
            )
            row[f"{method}_placement_velocity_mean"] = (
                round(statistics.mean(placement_velocities), 4) if placement_velocities else None
            )
            row[f"{method}_dynamic_transport_mode"] = (
                Counter(dynamic_transport_modes).most_common(1)[0][0] if dynamic_transport_modes else None
            )
            lift_hold_risks = [run.avg_lift_hold_risk for run in task_runs]
            transfer_sway_risks = [run.avg_transfer_sway_risk for run in task_runs]
            placement_settle_risks = [run.avg_placement_settle_risk for run in task_runs]
            physics_fail_rates = [run.physics_fail_rate for run in task_runs]
            lift_hold_fail_rates = [run.lift_hold_fail_rate for run in task_runs]
            transfer_sway_fail_rates = [run.transfer_sway_fail_rate for run in task_runs]
            placement_settle_fail_rates = [run.placement_settle_fail_rate for run in task_runs]
            row[f"{method}_avg_lift_hold_risk_mean"] = round(statistics.mean(lift_hold_risks), 4) if lift_hold_risks else None
            row[f"{method}_avg_transfer_sway_risk_mean"] = round(statistics.mean(transfer_sway_risks), 4) if transfer_sway_risks else None
            row[f"{method}_avg_placement_settle_risk_mean"] = (
                round(statistics.mean(placement_settle_risks), 4) if placement_settle_risks else None
            )
            row[f"{method}_physics_fail_rate_mean"] = round(statistics.mean(physics_fail_rates), 4) if physics_fail_rates else None
            row[f"{method}_lift_hold_fail_rate_mean"] = round(statistics.mean(lift_hold_fail_rates), 4) if lift_hold_fail_rates else None
            row[f"{method}_transfer_sway_fail_rate_mean"] = (
                round(statistics.mean(transfer_sway_fail_rates), 4) if transfer_sway_fail_rates else None
            )
            row[f"{method}_placement_settle_fail_rate_mean"] = (
                round(statistics.mean(placement_settle_fail_rates), 4) if placement_settle_fail_rates else None
            )
            failure_means = {
                "physics_fail": row[f"{method}_physics_fail_rate_mean"],
                "lift_hold_fail": row[f"{method}_lift_hold_fail_rate_mean"],
                "transfer_sway_fail": row[f"{method}_transfer_sway_fail_rate_mean"],
                "placement_settle_fail": row[f"{method}_placement_settle_fail_rate_mean"],
            }
            row[f"{method}_dominant_failure_mode"] = (
                max(failure_means, key=failure_means.get)
                if any((value or 0.0) > 0.0 for value in failure_means.values())
                else "none"
            )
        comparison.append(row)

    out_json = output_dir / "simulation_comparison_multi_seed.json"
    write_json(out_json, comparison)
    lines = [
        "RAG vs 基线 对比结果（多 seed mean±std）",
        "=" * 108,
        f"n_trials_per_task={n_trials_per_task}, seeds={seeds}",
        "",
        f"{'任务':<32} {'参考力范围(N)':<12} "
        + " ".join(f"{method}_成功率(mean±std)" for method in methods),
        "-" * 108,
    ]
    for task in BENCHMARK_TASKS:
        reference = f"{task.reference_force_range[0]}-{task.reference_force_range[1]}"
        parts = []
        for method in methods:
            values = rates.get((task.task_id, method), [])
            if values:
                mean = statistics.mean(values)
                std = statistics.stdev(values) if len(values) > 1 else 0.0
                parts.append(f"{mean:.2%}±{std:.2%}")
            else:
                parts.append("N/A")
        lines.append(f"{task_label(task):<32} {reference:<12} " + " ".join(parts))
        if "rag" in methods:
            rag_row = next(item for item in comparison if item["task_id"] == task.task_id)
            coverage = rag_row.get("rag_belief_state_coverage_mean")
            conservative = rag_row.get("rag_uncertainty_conservative_mode_mean")
            solver = rag_row.get("rag_solver_selected_candidate")
            if coverage is not None and conservative is not None and solver is not None:
                lines.append(
                    " " * 4
                    + f"rag belief={coverage:.3f}, conservative_rate={conservative:.2%}, solver={solver}"
                )
    lines.append("")
    lines.append(f"对比 JSON: {out_json}")
    write_table(output_dir / "simulation_comparison_multi_seed.txt", lines)
    _write_split_summary(comparison, methods, output_dir)
    _write_challenge_summary(comparison, methods, output_dir)
    return rates


def _support_bucket(score: float | None) -> str:
    if score is None:
        return "unknown_support"
    if score < 2.5:
        return "low_support"
    if score < 4.0:
        return "medium_support"
    return "high_support"


def _write_evidence_condition_summary(
    comparison_rows: list[dict],
    output_dir: Path,
) -> None:
    support_rows = []
    for bucket in ("low_support", "medium_support", "high_support"):
        rows = [row for row in comparison_rows if row.get("support_bucket") == bucket]
        if not rows:
            continue
        support_rows.append(
            {
                "group": bucket,
                "n_tasks": len(rows),
                "rag_success_rate_mean": round(statistics.mean(row["rag_success_rate_mean"] for row in rows), 4),
                "rag_generic_only_success_rate_mean": round(
                    statistics.mean(row["rag_generic_only_success_rate_mean"] for row in rows),
                    4,
                ),
                "rag_minus_generic_only_gain_mean": round(
                    statistics.mean(row["rag_minus_generic_only_success_gain"] for row in rows),
                    4,
                ),
                "rag_evidence_support_score_mean": round(
                    statistics.mean(row["rag_evidence_support_score_mean"] for row in rows),
                    4,
                ),
            }
        )

    specific_rows = []
    grouped = (
        ("specific_rule_available", True),
        ("no_specific_rule_available", False),
    )
    for label, require_specific in grouped:
        rows = [
            row
            for row in comparison_rows
            if ((row.get("rag_available_specific_force_rules_mean", 0.0) >= 0.5) is require_specific)
        ]
        if not rows:
            continue
        specific_rows.append(
            {
                "group": label,
                "n_tasks": len(rows),
                "rag_success_rate_mean": round(statistics.mean(row["rag_success_rate_mean"] for row in rows), 4),
                "rag_generic_only_success_rate_mean": round(
                    statistics.mean(row["rag_generic_only_success_rate_mean"] for row in rows),
                    4,
                ),
                "rag_minus_generic_only_gain_mean": round(
                    statistics.mean(row["rag_minus_generic_only_success_gain"] for row in rows),
                    4,
                ),
                "rag_evidence_support_score_mean": round(
                    statistics.mean(row["rag_evidence_support_score_mean"] for row in rows),
                    4,
                ),
            }
        )

    payload = {
        "by_support_bucket": support_rows,
        "by_specific_rule_availability": specific_rows,
    }
    write_json(output_dir / "simulation_evidence_dependence_summary.json", payload)

    lines = [
        "Simulation evidence dependence 汇总",
        "=" * 80,
        "",
        "按证据强度分组",
        f"{'group':<28} {'n_tasks':<8} {'rag(mean)':<12} {'generic_only(mean)':<20} {'gain':<12}",
        "-" * 80,
    ]
    for row in support_rows:
        lines.append(
            f"{row['group']:<28} {row['n_tasks']:<8} "
            f"{row['rag_success_rate_mean']:.2%} "
            f"{row['rag_generic_only_success_rate_mean']:.2%} "
            f"{row['rag_minus_generic_only_gain_mean']:.2%}"
        )
    lines.extend(
        [
            "",
            "按对象特定规则可用性分组",
            f"{'group':<28} {'n_tasks':<8} {'rag(mean)':<12} {'generic_only(mean)':<20} {'gain':<12}",
            "-" * 80,
        ]
    )
    for row in specific_rows:
        lines.append(
            f"{row['group']:<28} {row['n_tasks']:<8} "
            f"{row['rag_success_rate_mean']:.2%} "
            f"{row['rag_generic_only_success_rate_mean']:.2%} "
            f"{row['rag_minus_generic_only_gain_mean']:.2%}"
        )
    write_table(output_dir / "simulation_evidence_dependence_summary.txt", lines)


def run_evidence_ablation_multi_seed(
    data_path: str = "mechanical_data.txt",
    n_trials_per_task: int = 10,
    seeds: list[int] | None = None,
    output_dir: str | None = None,
) -> list[dict]:
    seeds = seeds or [42, 43, 44]
    methods = ["rag", "rag_generic_only"]
    output_dir = Path(output_dir or "outputs/current")
    output_dir.mkdir(parents=True, exist_ok=True)

    results_by_seed: dict[int, dict[str, list[BenchmarkResult]]] = {}
    for seed in seeds:
        results_by_seed[seed] = run_benchmark_comparison(
            data_path=data_path,
            n_trials_per_task=n_trials_per_task,
            seed=seed,
            output_dir=str(output_dir),
            methods=methods,
        )

    comparison_rows: list[dict] = []
    for task in BENCHMARK_TASKS:
        rag_runs = [
            next(result for result in results_by_seed[seed]["rag"] if result.task_id == task.task_id)
            for seed in seeds
        ]
        generic_runs = [
            next(result for result in results_by_seed[seed]["rag_generic_only"] if result.task_id == task.task_id)
            for seed in seeds
        ]
        rag_success = [run.success_rate for run in rag_runs]
        generic_success = [run.success_rate for run in generic_runs]
        support_scores = [float(run.params_used.get("evidence_support_score", 0.0)) for run in rag_runs]
        available_specific = [1.0 if run.params_used.get("available_specific_force_rules") else 0.0 for run in rag_runs]
        suppressed_specific = [1.0 if run.params_used.get("suppressed_specific_force_rules") else 0.0 for run in generic_runs]
        row = {
            "task_id": task.task_id,
            "task_label": task_label(task),
            "task_description": task.description,
            "task_split": task.split,
            "challenge_tags": list(task.challenge_tags),
            "seeds": seeds,
            "n_trials_per_seed": n_trials_per_task,
            "rag_success_rate_mean": round(statistics.mean(rag_success), 4),
            "rag_success_rate_std": round(statistics.stdev(rag_success), 4) if len(rag_success) > 1 else 0.0,
            "rag_generic_only_success_rate_mean": round(statistics.mean(generic_success), 4),
            "rag_generic_only_success_rate_std": round(statistics.stdev(generic_success), 4) if len(generic_success) > 1 else 0.0,
            "rag_minus_generic_only_success_gain": round(
                statistics.mean(rag_success) - statistics.mean(generic_success),
                4,
            ),
            "rag_evidence_support_score_mean": round(statistics.mean(support_scores), 4),
            "rag_available_specific_force_rules_mean": round(statistics.mean(available_specific), 4),
            "rag_generic_only_suppressed_specific_force_rules_mean": round(statistics.mean(suppressed_specific), 4),
            "support_bucket": _support_bucket(round(statistics.mean(support_scores), 4)),
        }
        comparison_rows.append(row)

    out_json = output_dir / "simulation_evidence_ablation.json"
    write_json(out_json, comparison_rows)
    lines = [
        "Simulation evidence ablation（RAG vs generic-only）",
        "=" * 84,
        f"n_trials_per_task={n_trials_per_task}, seeds={seeds}",
        "",
        f"{'任务':<32} {'support':<12} {'rag(mean±std)':<18} {'generic_only(mean±std)':<26} {'gain':<10}",
        "-" * 84,
    ]
    for row in comparison_rows:
        lines.append(
            f"{row['task_label']:<32} "
            f"{row['rag_evidence_support_score_mean']:.2f} "
            f"{row['rag_success_rate_mean']:.2%}±{row['rag_success_rate_std']:.2%} "
            f"{row['rag_generic_only_success_rate_mean']:.2%}±{row['rag_generic_only_success_rate_std']:.2%} "
            f"{row['rag_minus_generic_only_success_gain']:.2%}"
        )
    lines.append("")
    lines.append(f"JSON: {out_json}")
    write_table(output_dir / "simulation_evidence_ablation.txt", lines)
    _write_evidence_condition_summary(comparison_rows, output_dir)
    return comparison_rows


def _write_motion_dependence_summary(
    comparison_rows: list[dict],
    output_dir: Path,
) -> None:
    support_rows = []
    for bucket in ("low_support", "medium_support", "high_support"):
        rows = [row for row in comparison_rows if row.get("support_bucket") == bucket]
        if not rows:
            continue
        support_rows.append(
            {
                "group": bucket,
                "n_tasks": len(rows),
                "rag_success_rate_mean": round(statistics.mean(row["rag_success_rate_mean"] for row in rows), 4),
                "rag_no_motion_rules_success_rate_mean": round(
                    statistics.mean(row["rag_no_motion_rules_success_rate_mean"] for row in rows),
                    4,
                ),
                "rag_minus_no_motion_rules_gain_mean": round(
                    statistics.mean(row["rag_minus_no_motion_rules_success_gain"] for row in rows),
                    4,
                ),
                "rag_evidence_support_score_mean": round(
                    statistics.mean(row["rag_evidence_support_score_mean"] for row in rows),
                    4,
                ),
            }
        )

    motion_rows = []
    grouped = (
        ("motion_rule_available", True),
        ("no_motion_rule_available", False),
    )
    for label, require_motion in grouped:
        rows = [
            row
            for row in comparison_rows
            if ((row.get("rag_available_motion_rules_mean", 0.0) >= 0.5) is require_motion)
        ]
        if not rows:
            continue
        motion_rows.append(
            {
                "group": label,
                "n_tasks": len(rows),
                "rag_success_rate_mean": round(statistics.mean(row["rag_success_rate_mean"] for row in rows), 4),
                "rag_no_motion_rules_success_rate_mean": round(
                    statistics.mean(row["rag_no_motion_rules_success_rate_mean"] for row in rows),
                    4,
                ),
                "rag_minus_no_motion_rules_gain_mean": round(
                    statistics.mean(row["rag_minus_no_motion_rules_success_gain"] for row in rows),
                    4,
                ),
                "rag_evidence_support_score_mean": round(
                    statistics.mean(row["rag_evidence_support_score_mean"] for row in rows),
                    4,
                ),
            }
        )

    payload = {
        "by_support_bucket": support_rows,
        "by_motion_rule_availability": motion_rows,
    }
    write_json(output_dir / "simulation_motion_dependence_summary.json", payload)

    lines = [
        "Simulation motion dependence 汇总",
        "=" * 80,
        "",
        "按证据强度分组",
        f"{'group':<28} {'n_tasks':<8} {'rag(mean)':<12} {'no_motion(mean)':<18} {'gain':<12}",
        "-" * 80,
    ]
    for row in support_rows:
        lines.append(
            f"{row['group']:<28} {row['n_tasks']:<8} "
            f"{row['rag_success_rate_mean']:.2%} "
            f"{row['rag_no_motion_rules_success_rate_mean']:.2%} "
            f"{row['rag_minus_no_motion_rules_gain_mean']:.2%}"
        )
    lines.extend(
        [
            "",
            "按 motion rule 可用性分组",
            f"{'group':<28} {'n_tasks':<8} {'rag(mean)':<12} {'no_motion(mean)':<18} {'gain':<12}",
            "-" * 80,
        ]
    )
    for row in motion_rows:
        lines.append(
            f"{row['group']:<28} {row['n_tasks']:<8} "
            f"{row['rag_success_rate_mean']:.2%} "
            f"{row['rag_no_motion_rules_success_rate_mean']:.2%} "
            f"{row['rag_minus_no_motion_rules_gain_mean']:.2%}"
        )
    write_table(output_dir / "simulation_motion_dependence_summary.txt", lines)


def run_motion_ablation_multi_seed(
    data_path: str = "mechanical_data.txt",
    n_trials_per_task: int = 10,
    seeds: list[int] | None = None,
    output_dir: str | None = None,
) -> list[dict]:
    seeds = seeds or [42, 43, 44]
    methods = ["rag", "rag_no_motion_rules"]
    output_dir = Path(output_dir or "outputs/current")
    output_dir.mkdir(parents=True, exist_ok=True)

    results_by_seed: dict[int, dict[str, list[BenchmarkResult]]] = {}
    for seed in seeds:
        results_by_seed[seed] = run_benchmark_comparison(
            data_path=data_path,
            n_trials_per_task=n_trials_per_task,
            seed=seed,
            output_dir=str(output_dir),
            methods=methods,
        )

    comparison_rows: list[dict] = []
    for task in BENCHMARK_TASKS:
        rag_runs = [
            next(result for result in results_by_seed[seed]["rag"] if result.task_id == task.task_id)
            for seed in seeds
        ]
        no_motion_runs = [
            next(result for result in results_by_seed[seed]["rag_no_motion_rules"] if result.task_id == task.task_id)
            for seed in seeds
        ]
        rag_success = [run.success_rate for run in rag_runs]
        no_motion_success = [run.success_rate for run in no_motion_runs]
        support_scores = [float(run.params_used.get("evidence_support_score", 0.0)) for run in rag_runs]
        available_motion = [1.0 if run.params_used.get("available_motion_rules") else 0.0 for run in rag_runs]
        suppressed_motion = [1.0 if run.params_used.get("suppressed_motion_rules") else 0.0 for run in no_motion_runs]
        row = {
            "task_id": task.task_id,
            "task_label": task_label(task),
            "task_description": task.description,
            "task_split": task.split,
            "challenge_tags": list(task.challenge_tags),
            "seeds": seeds,
            "n_trials_per_seed": n_trials_per_task,
            "rag_success_rate_mean": round(statistics.mean(rag_success), 4),
            "rag_success_rate_std": round(statistics.stdev(rag_success), 4) if len(rag_success) > 1 else 0.0,
            "rag_no_motion_rules_success_rate_mean": round(statistics.mean(no_motion_success), 4),
            "rag_no_motion_rules_success_rate_std": round(statistics.stdev(no_motion_success), 4) if len(no_motion_success) > 1 else 0.0,
            "rag_minus_no_motion_rules_success_gain": round(
                statistics.mean(rag_success) - statistics.mean(no_motion_success),
                4,
            ),
            "rag_evidence_support_score_mean": round(statistics.mean(support_scores), 4),
            "rag_available_motion_rules_mean": round(statistics.mean(available_motion), 4),
            "rag_no_motion_rules_suppressed_motion_rules_mean": round(statistics.mean(suppressed_motion), 4),
            "support_bucket": _support_bucket(round(statistics.mean(support_scores), 4)),
        }
        comparison_rows.append(row)

    out_json = output_dir / "simulation_motion_ablation.json"
    write_json(out_json, comparison_rows)
    lines = [
        "Simulation motion ablation（RAG vs no-motion-rules）",
        "=" * 84,
        f"n_trials_per_task={n_trials_per_task}, seeds={seeds}",
        "",
        f"{'任务':<32} {'support':<12} {'rag(mean±std)':<18} {'no_motion(mean±std)':<24} {'gain':<10}",
        "-" * 84,
    ]
    for row in comparison_rows:
        lines.append(
            f"{row['task_label']:<32} "
            f"{row['rag_evidence_support_score_mean']:.2f} "
            f"{row['rag_success_rate_mean']:.2%}±{row['rag_success_rate_std']:.2%} "
            f"{row['rag_no_motion_rules_success_rate_mean']:.2%}±{row['rag_no_motion_rules_success_rate_std']:.2%} "
            f"{row['rag_minus_no_motion_rules_success_gain']:.2%}"
        )
    lines.append("")
    lines.append(f"JSON: {out_json}")
    write_table(output_dir / "simulation_motion_ablation.txt", lines)
    _write_motion_dependence_summary(comparison_rows, output_dir)
    return comparison_rows


def run_retrieval_ablation(
    data_path: str = "mechanical_data.txt",
    n_trials_per_task: int = 10,
    seed: int = 42,
    output_dir: str | None = None,
) -> dict[str, list[BenchmarkResult]]:
    methods = ["rag", "rag_multi", "rag_random", "fixed"]
    output_dir = Path(output_dir or "outputs/current")
    output_dir.mkdir(parents=True, exist_ok=True)
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

    comparison = []
    for task in BENCHMARK_TASKS:
        row = {
            "task_id": task.task_id,
            "task_label": task_label(task),
            "task_description": task.description,
            "task_split": task.split,
            "reference_force_range": list(task.reference_force_range),
        }
        for method in methods:
            result = next(r for r in all_results[method] if r.task_id == task.task_id)
            row[f"{method}_success_rate"] = round(result.success_rate, 4)
            row[f"{method}_gripper_force"] = result.params_used.get("gripper_force")
        comparison.append(row)

    out_json = output_dir / "simulation_ablation_retrieval.json"
    write_json(out_json, comparison)
    labels = {"rag": "单query", "rag_multi": "多query", "rag_random": "随机文档", "fixed": "固定25N"}
    lines = [
        "检索策略消融：单 query vs 多 query vs 随机文档 vs 固定基线",
        "=" * 80,
        f"n_trials_per_task={n_trials_per_task}, seed={seed}",
        "",
        f"{'任务':<24} {'参考力范围(N)':<14} " + " ".join(labels[method] for method in methods),
        "-" * 80,
    ]
    for task in BENCHMARK_TASKS:
        reference = f"{task.reference_force_range[0]}-{task.reference_force_range[1]}"
        rates = [f"{next(r for r in all_results[m] if r.task_id == task.task_id).success_rate:.2%}" for m in methods]
        lines.append(f"{task_label(task):<24} {reference:<14} " + " ".join(rates))
    lines.append("")
    lines.append(f"JSON: {out_json}")
    write_table(output_dir / "simulation_ablation_retrieval.txt", lines)
    return all_results
