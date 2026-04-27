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

    def test_showcase_gap_helper_skips_missing_baseline(self):
        rows = [
            {
                "task_id": "pick_demo",
                "methods": {
                    "rag_feedback": {"success_rate_mean": 0.84},
                    "task_heuristic": {"success_rate_mean": 0.61},
                },
            }
        ]
        self.assertAlmostEqual(
            showcase._mean_method_gap(rows, "rag_feedback", "task_heuristic", "success_rate_mean"),
            0.23,
        )
        self.assertIsNone(
            showcase._mean_method_gap(rows, "rag_feedback", "direct_llm", "success_rate_mean"),
        )


if __name__ == "__main__":
    unittest.main()
