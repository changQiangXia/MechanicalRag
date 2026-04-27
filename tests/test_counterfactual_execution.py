import unittest
from unittest.mock import patch

from simulation.control_core import PhaseObservation
from simulation.env import simulate_stepwise_execution


class CounterfactualExecutionTest(unittest.TestCase):
    def test_lift_replan_freezes_completed_prefix(self):
        phase_log: list[str] = []

        def _fake_phase_eval(*, phase, params, object_profile, rng=None):
            del params, object_profile, rng
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


if __name__ == "__main__":
    unittest.main()
