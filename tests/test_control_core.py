import unittest

from simulation.control_core import (
    build_control_belief,
    build_execution_prior,
    PhaseObservation,
    apply_phase_observation,
    replan_control_plan,
    solve_control_plan,
    summarize_control_evidence,
)
from simulation.feedback import FeedbackSignal, build_feedback_replan_request


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


class ControlCoreTest(unittest.TestCase):
    def setUp(self) -> None:
        self.features = {
            "metal": False,
            "smooth_metal": False,
            "rubber": False,
            "small": False,
            "large": True,
            "heavy": False,
            "thin_wall": False,
            "high_speed": False,
            "long_transfer": True,
        }
        self.force_rule_a = _rule(
            score=4.2,
            specificity=3,
            matched_terms=["large", "long_transfer", "lift_stage"],
            force_candidates=[32.0],
        )
        self.force_rule_b = _rule(
            score=3.9,
            specificity=2,
            matched_terms=["large", "long_transfer", "heavy"],
            force_candidates=[42.0],
        )
        self.motion_rule = _rule(
            score=4.0,
            specificity=3,
            matched_terms=["large", "long_transfer"],
        )
        self.alignment_rule = _rule(
            score=3.7,
            specificity=2,
            matched_terms=["large", "long_transfer"],
        )
        self.lift_stage_rule = _rule(
            score=3.5,
            specificity=2,
            matched_terms=["large", "long_transfer", "lift_stage"],
        )
        self.rules = [
            self.force_rule_a,
            self.force_rule_b,
            self.motion_rule,
            self.alignment_rule,
            self.lift_stage_rule,
        ]

    def _build_summary_and_belief(self):
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
            default_force=33.0,
        )
        belief = build_control_belief(
            features=self.features,
            evidence_summary=summary,
            dynamic_transport_mode=summary.preferred_transport_mode,
            support_score=3.8,
            conflict_count=1,
            force_rule_mode="all",
            motion_rule_mode="all",
            available_specific_force_rules=True,
            available_motion_rules=True,
            available_numeric_motion_rules=True,
            available_alignment_rules=True,
            available_lift_stage_rules=True,
            available_support_contact_rules=False,
        )
        return summary, belief

    def test_evidence_summary_exposes_confidence_slots(self):
        summary, _ = self._build_summary_and_belief()
        self.assertEqual(summary.preferred_transport_mode, "long_transfer")
        self.assertGreater(summary.force_center, 30.0)
        self.assertGreater(summary.force_std, 4.0)
        self.assertGreater(summary.alignment_confidence, 0.1)
        self.assertGreater(summary.numeric_motion_confidence, 0.1)

    def test_local_search_improves_seed_score(self):
        _, belief = self._build_summary_and_belief()
        solved_plan, trace = solve_control_plan(
            {
                "gripper_force": 31.5,
                "approach_height": 0.05,
                "transport_velocity": 0.24,
                "lift_force": 31.5,
                "transfer_force": 31.5,
                "placement_velocity": 0.22,
                "transfer_alignment": 0.55,
                "lift_clearance": 0.055,
            },
            belief,
        )
        self.assertEqual(trace["solver_mode"], "belief_seeded_local_search")
        self.assertLessEqual(trace["solver_selected_score"], trace["solver_seed_score"])
        self.assertGreater(trace["solver_local_search_iterations"], 0)
        self.assertGreaterEqual(trace["solver_local_search_improvement"], 0.0)
        self.assertGreaterEqual(solved_plan["transfer_alignment"], 0.55)

    def test_feedback_replan_returns_trace(self):
        summary, belief = self._build_summary_and_belief()
        _, solve_trace = solve_control_plan(
            {
                "gripper_force": 31.5,
                "approach_height": 0.05,
                "transport_velocity": 0.24,
                "lift_force": 31.5,
                "transfer_force": 31.5,
                "placement_velocity": 0.22,
                "transfer_alignment": 0.55,
                "lift_clearance": 0.055,
            },
            belief,
        )
        previous_params = {
            "gripper_force": 31.5,
            "approach_height": 0.05,
            "transport_velocity": 0.24,
            "lift_force": 31.5,
            "transfer_force": 31.5,
            "placement_velocity": 0.22,
            "transfer_alignment": 0.55,
            "lift_clearance": 0.055,
            "confidence": 0.62,
            "uncertainty_std": 0.11,
            **belief.to_trace_dict(),
            "solver_mode": solve_trace["solver_mode"],
            "solver_selected_candidate": solve_trace["solver_selected_candidate"],
        }
        signal = FeedbackSignal(
            success=False,
            gripper_force=31.5,
            distance=0.05,
            steps=12,
            transport_velocity=0.24,
            lift_clearance=0.055,
            slip_risk=0.10,
            compression_risk=0.05,
            stability_score=0.42,
            velocity_risk=0.12,
            clearance_risk=0.08,
            lift_hold_risk=0.14,
            transfer_sway_risk=0.36,
            placement_settle_risk=0.18,
            failure_bucket="transfer_sway_fail",
            dynamic_transport_mode="long_transfer",
        )
        request = build_feedback_replan_request(previous_params, signal, "increase", step=4.0)
        updated = replan_control_plan(previous_params, request)
        self.assertIn("feedback_replan_trace", updated)
        self.assertEqual(updated["feedback_stage_adjustments"], ["transfer"])
        self.assertEqual(updated["feedback_replan_trace"]["stage_bias"], "transfer")
        self.assertEqual(updated["solver_mode"], "belief_seeded_local_search")
        self.assertTrue(updated["uncertainty_conservative_mode"])
        self.assertIn("solver_local_search_trace", updated)

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


if __name__ == "__main__":
    unittest.main()
