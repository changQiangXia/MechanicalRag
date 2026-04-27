import unittest
from unittest.mock import patch

from simulation.control_core import (
    EvidenceConstraintHints,
    PhaseObservation,
    build_control_belief,
    summarize_control_evidence,
    synthesize_control_seed,
)
from simulation.env import ArmSimEnv, HAS_MUJOCO, simulate_stepwise_execution


def _rule(
    *,
    score: float,
    specificity: int,
    matched_terms: list[str],
    force_candidates: list[float] | None = None,
) -> dict:
    return {
        "score": score,
        "specificity": specificity,
        "matched_terms": matched_terms,
        "force_candidates": force_candidates or [],
    }


class AdaptiveExecutionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.features = {
            "metal": True,
            "smooth_metal": True,
            "rubber": False,
            "small": False,
            "large": True,
            "heavy": True,
            "thin_wall": False,
            "high_speed": True,
            "long_transfer": True,
        }
        self.force_rule_a = _rule(
            score=4.4,
            specificity=3,
            matched_terms=["metal", "heavy", "long_transfer", "lift_stage"],
            force_candidates=[42.0],
        )
        self.force_rule_b = _rule(
            score=4.1,
            specificity=3,
            matched_terms=["metal", "heavy", "long_transfer"],
            force_candidates=[48.0],
        )
        self.motion_rule = _rule(
            score=4.0,
            specificity=2,
            matched_terms=["long_transfer", "high_speed"],
        )
        self.alignment_rule = _rule(
            score=3.9,
            specificity=2,
            matched_terms=["long_transfer", "heavy"],
        )
        self.lift_stage_rule = _rule(
            score=3.8,
            specificity=2,
            matched_terms=["long_transfer", "heavy", "lift_stage"],
        )
        self.rules = [
            self.force_rule_a,
            self.force_rule_b,
            self.motion_rule,
            self.alignment_rule,
            self.lift_stage_rule,
        ]

    def _belief(self):
        summary = summarize_control_evidence(
            features=self.features,
            rules=self.rules,
            selected_force_rules=[self.force_rule_a, self.force_rule_b],
            specific_force_rules=[self.force_rule_a, self.force_rule_b],
            motion_rules=[self.motion_rule, self.alignment_rule],
            numeric_motion_rules=[self.motion_rule, self.alignment_rule],
            alignment_rules=[self.alignment_rule],
            lift_stage_rules=[self.lift_stage_rule],
            support_contact_rules=[],
            default_force=45.0,
        )
        belief = build_control_belief(
            features=self.features,
            evidence_summary=summary,
            dynamic_transport_mode="long_transfer",
            support_score=4.2,
            conflict_count=0,
            force_rule_mode="all",
            motion_rule_mode="all",
            available_specific_force_rules=True,
            available_motion_rules=True,
            available_numeric_motion_rules=True,
            available_alignment_rules=True,
            available_lift_stage_rules=True,
            available_support_contact_rules=False,
        )
        return belief

    def test_belief_driven_seed_synthesis_exposes_trace(self):
        belief = self._belief()
        hints = EvidenceConstraintHints(
            force_floor=46.0,
            transport_velocity_cap=0.22,
            placement_velocity_cap=0.18,
            clearance_floor=0.075,
            alignment_target=0.82,
            lift_force_margin=0.8,
            transfer_force_margin=0.6,
            source_notes=["numeric_motion_cap", "alignment_support"],
        )
        seed_plan, trace = synthesize_control_seed(
            {
                "gripper_force": 38.0,
                "approach_height": 0.05,
                "transport_velocity": 0.28,
                "lift_force": 38.0,
                "transfer_force": 38.0,
                "placement_velocity": 0.26,
                "transfer_alignment": 0.2,
                "lift_clearance": 0.06,
            },
            belief,
            hints,
        )
        self.assertEqual(trace["seed_mode"], "belief_constraint_synthesis")
        self.assertLessEqual(seed_plan["transport_velocity"], 0.22)
        self.assertLessEqual(seed_plan["placement_velocity"], 0.18)
        self.assertGreaterEqual(seed_plan["lift_clearance"], 0.075)
        self.assertGreaterEqual(seed_plan["transfer_alignment"], 0.82)

    def test_stepwise_execution_emits_observer_trace(self):
        success, _, info = simulate_stepwise_execution(
            object_pos=(0.0, 0.0, 0.0),
            target_pos=(0.38, 0.0, 0.0),
            params={
                "gripper_force": 41.0,
                "approach_height": 0.05,
                "transport_velocity": 0.34,
                "lift_force": 41.0,
                "transfer_force": 41.0,
                "placement_velocity": 0.30,
                "transfer_alignment": 0.2,
                "lift_clearance": 0.06,
            },
            object_profile={
                "mass_kg": 0.36,
                "surface_friction": 0.18,
                "fragility": 0.72,
                "velocity_scale": 0.8,
                "target_tolerance": 0.04,
                "size_xyz": (0.055, 0.045, 0.03),
                "preferred_approach_height": 0.05,
                "approach_height_tolerance": 0.02,
            },
        )
        self.assertIn("observer_trace", info)
        self.assertGreater(len(info["observer_trace"]), 0)
        snapshot = info["observer_trace"][0]
        self.assertIn("stage", snapshot)
        self.assertIn("stage_progress", snapshot)
        self.assertIn("slip_indicator", snapshot)
        self.assertIn("compression_indicator", snapshot)
        self.assertIn("estimated_failure_stage", snapshot)
        self.assertIn(success, (True, False))

    def test_stepwise_execution_replans_inside_single_trial(self):
        def _step_replan(observation: dict, current_params: dict) -> dict | None:
            if observation["stage"] != "transfer":
                return None
            if observation["velocity_margin"] >= 0.0:
                return None
            updated = dict(current_params)
            updated["transport_velocity"] = max(0.12, current_params["transport_velocity"] - 0.06)
            updated["placement_velocity"] = max(0.12, current_params["placement_velocity"] - 0.06)
            updated["lift_clearance"] = min(0.14, current_params["lift_clearance"] + 0.008)
            updated["transfer_force"] = min(50.0, current_params["transfer_force"] + 1.2)
            return updated

        _, _, info = simulate_stepwise_execution(
            object_pos=(0.0, 0.0, 0.0),
            target_pos=(0.42, 0.0, 0.0),
            params={
                "gripper_force": 40.0,
                "approach_height": 0.05,
                "transport_velocity": 0.38,
                "lift_force": 40.0,
                "transfer_force": 40.0,
                "placement_velocity": 0.34,
                "transfer_alignment": 0.15,
                "lift_clearance": 0.058,
            },
            object_profile={
                "mass_kg": 0.38,
                "surface_friction": 0.16,
                "fragility": 0.74,
                "velocity_scale": 0.8,
                "target_tolerance": 0.04,
                "size_xyz": (0.06, 0.05, 0.03),
                "preferred_approach_height": 0.05,
                "approach_height_tolerance": 0.02,
            },
            step_replan_callback=_step_replan,
            max_step_replans=2,
        )
        self.assertEqual(info["execution_feedback_mode"], "suffix_counterfactual_replan")
        self.assertGreaterEqual(info["step_replan_count"], 1)
        self.assertIn("counterfactual_replan_trace", info)
        self.assertGreater(len(info["counterfactual_replan_trace"]), 0)
        self.assertEqual(info["counterfactual_replan_trace"][0]["start_phase"], "transfer")

    def test_mujoco_execute_pick_place_forwards_symbolic_control_context(self):
        if not HAS_MUJOCO:
            self.skipTest("MuJoCo unavailable in this environment")

        captured: dict = {}

        def _fake_stepwise_execution(**kwargs):
            captured.update(kwargs["params"])
            return True, 0.0, {"applied_params": dict(kwargs["params"])}

        with patch("simulation.env.simulate_stepwise_execution", side_effect=_fake_stepwise_execution):
            env = ArmSimEnv(gui=False, seed=0)
            try:
                success, _, _ = env.execute_pick_place(
                    object_pos=(0.0, 0.0, 0.0),
                    target_pos=(0.2, 0.0, 0.0),
                    gripper_force=42.0,
                    approach_height=0.05,
                    transport_velocity=0.22,
                    lift_force=42.0,
                    transfer_force=42.0,
                    placement_velocity=0.18,
                    transfer_alignment=0.0,
                    lift_clearance=0.06,
                    object_profile={
                        "mass_kg": 0.42,
                        "surface_friction": 0.18,
                        "fragility": 0.72,
                        "velocity_scale": 0.8,
                        "target_tolerance": 0.04,
                        "size_xyz": (0.055, 0.045, 0.03),
                        "preferred_approach_height": 0.05,
                        "approach_height_tolerance": 0.02,
                    },
                    step_replan_callback=lambda observation, params: None,
                    max_step_replans=1,
                    control_context={
                        "belief_state": {"mass_band": "heavy"},
                        "task_constraints": {"preferred_transport_mode": "static"},
                        "uncertainty_profile": {"conservative_mode": True},
                    },
                )
            finally:
                env.close()

        self.assertTrue(success)
        self.assertEqual(captured["belief_state"]["mass_band"], "heavy")
        self.assertEqual(captured["task_constraints"]["preferred_transport_mode"], "static")
        self.assertTrue(captured["uncertainty_profile"]["conservative_mode"])

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

    def test_stepwise_execution_rolls_posterior_into_next_phase(self):
        def _fake_phase_eval(*, phase, params, object_profile, rng=None):
            del params, object_profile, rng
            if phase == "grasp":
                return (
                    PhaseObservation(
                        phase="grasp",
                        contact_stability_obs=0.74,
                        micro_slip_obs=0.18,
                        observation_confidence=0.81,
                        trigger_reason="micro_slip_obs",
                    ),
                    True,
                    {"phase_success": True},
                )
            return (
                PhaseObservation(phase=phase, observation_confidence=0.72),
                True,
                {"phase_success": True},
            )

        with patch("simulation.env._evaluate_phase_execution", side_effect=_fake_phase_eval):
            _, _, info = simulate_stepwise_execution(
                object_pos=(0.0, 0.0, 0.0),
                target_pos=(0.30, 0.0, 0.0),
                params={
                    "gripper_force": 42.0,
                    "approach_height": 0.05,
                    "transport_velocity": 0.22,
                    "lift_force": 42.0,
                    "transfer_force": 42.0,
                    "placement_velocity": 0.18,
                    "transfer_alignment": 0.0,
                    "lift_clearance": 0.06,
                    "belief_state": {
                        "mass_band": "heavy",
                        "dynamic_load_band": "high",
                        "center_of_mass_risk": 0.72,
                        "alignment_confidence": 0.34,
                    },
                    "task_constraints": {"preferred_transport_mode": "static"},
                    "uncertainty_profile": {"conservative_mode": False},
                },
                object_profile={"mass_kg": 0.42, "surface_friction": 0.18},
            )

        grasp_update = next(row for row in info["belief_update_trace"] if row["posterior"]["phase"] == "grasp")
        lift_update = next(row for row in info["belief_update_trace"] if row["prior"]["phase"] == "lift")
        self.assertEqual(lift_update["prior"]["grip_hold_margin"], grasp_update["posterior"]["grip_hold_margin"])


if __name__ == "__main__":
    unittest.main()
