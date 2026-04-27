import json
from pathlib import Path
import tempfile
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

    def test_multi_seed_trial_record_serializer_keeps_seed_context(self):
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
            executed_plan_stats={"mean": {"gripper_force": 12.0}},
            planner_diagnostics={"belief_state_coverage": 0.8},
            trial_records=[{"trial_index": 0, "terminal_plan": {"gripper_force": 12.0}}],
            method="rag_feedback",
        )

        rows = runner._serialize_multi_seed_trial_records(
            per_seed_results={42: [base], 43: [base]},
            method="rag_feedback",
            seeds=[42, 43],
        )

        self.assertEqual(rows[0]["trial_record_count"], 2)
        self.assertEqual(rows[0]["trial_records"][0]["seed"], 42)
        self.assertEqual(rows[0]["trial_records"][1]["seed"], 43)

    def test_multi_seed_comparison_emits_nested_method_payload(self):
        rag_seed_42 = runner.BenchmarkResult(
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
                "mean": {"gripper_force": 12.0, "placement_velocity": 0.21},
                "dynamic_transport_mode_mode": "static",
            },
            planner_diagnostics={
                "belief_state_coverage": 0.8,
                "solver_selected_candidate": "belief_seed",
                "evidence_support_score": 4.5,
            },
            trial_records=[],
            method="rag_feedback",
        )
        rag_seed_43 = runner.BenchmarkResult(
            **{
                **rag_seed_42.__dict__,
                "success_rate": 0.8,
                "executed_plan_stats": {
                    "mean": {"gripper_force": 13.0, "placement_velocity": 0.23},
                    "dynamic_transport_mode_mode": "dynamic",
                },
            }
        )
        heuristic_seed_42 = runner.BenchmarkResult(
            **{
                **rag_seed_42.__dict__,
                "success_rate": 0.4,
                "planner_diagnostics": {"belief_state_coverage": 0.2},
                "method": "task_heuristic",
            }
        )
        heuristic_seed_43 = runner.BenchmarkResult(
            **{
                **heuristic_seed_42.__dict__,
                "success_rate": 0.5,
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir, \
             patch.object(runner, "BENCHMARK_TASKS", [_task()]), \
             patch.object(
                 runner,
                 "_run_benchmark_method_results",
                 side_effect=[
                     {"rag_feedback": [rag_seed_42], "task_heuristic": [heuristic_seed_42]},
                     {"rag_feedback": [rag_seed_43], "task_heuristic": [heuristic_seed_43]},
                 ],
             ):
            rows = runner.run_benchmark_comparison_multi_seed(
                n_trials_per_task=2,
                seeds=[42, 43],
                output_dir=tmpdir,
                methods=["rag_feedback", "task_heuristic"],
            )

            split_rows = json.loads(
                (Path(tmpdir) / "simulation_split_summary.json").read_text(encoding="utf-8")
            )

        method_row = rows[0]["methods"]["rag_feedback"]
        self.assertAlmostEqual(method_row["success_rate_mean"], 0.9)
        self.assertEqual(method_row["planner_diagnostics"]["belief_state_coverage_mean"], 0.8)
        self.assertEqual(method_row["executed_plan_stats"]["mean"]["gripper_force"], 12.5)
        self.assertEqual(method_row["failure_rates"]["dominant_failure_mode"], "none")
        self.assertNotIn("rag_feedback_success_rate_mean", rows[0])
        self.assertEqual(split_rows[0]["rag_feedback_success_rate_mean"], 0.9)


if __name__ == "__main__":
    unittest.main()
