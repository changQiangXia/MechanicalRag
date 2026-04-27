import json
import unittest
from pathlib import Path


CURRENT_DIR = Path("outputs/current_observer_step_replan")


def _multi_seed_rows() -> list[dict]:
    return json.loads((CURRENT_DIR / "simulation_benchmark_result.json").read_text(encoding="utf-8"))


class HeavyLoadRegressionTest(unittest.TestCase):
    def test_current_profile_keeps_heavy_gain_and_online_repair_contribution(self):
        rows = _multi_seed_rows()
        heavy = next(row for row in rows if row["task_id"] == "pick_metal_heavy")
        heavy_fast = next(row for row in rows if row["task_id"] == "pick_metal_heavy_fast")
        rag = heavy["methods"]["rag_feedback"]
        observer_only = heavy["methods"]["rag_feedback_observer_only"]
        heavy_fast_rag = heavy_fast["methods"]["rag_feedback"]

        overall = sum(row["methods"]["rag_feedback"]["success_rate_mean"] for row in rows) / len(rows)

        self.assertGreaterEqual(rag["success_rate_mean"], 0.30)
        self.assertGreater(rag["online_diagnosis_count"], 0)
        self.assertGreater(rag["suffix_counterfactual_replan_count"], 0)
        self.assertGreater(rag["success_rate_mean"], observer_only["success_rate_mean"])
        self.assertGreaterEqual(heavy_fast_rag["success_rate_mean"], 0.8167)
        self.assertGreaterEqual(overall, 0.8122)


if __name__ == "__main__":
    unittest.main()
