import json
import unittest
from pathlib import Path


CURRENT_DIR = Path("outputs/current_observer_step_replan")


def _feedback_rows() -> list[dict]:
    return json.loads((CURRENT_DIR / "simulation_benchmark_rag_feedback.json").read_text(encoding="utf-8"))


class SimulationClaimsTest(unittest.TestCase):
    def test_docs_downscope_suffix_repair_when_online_metrics_are_zero(self):
        rows = _feedback_rows()
        total_suffix = sum(int(row.get("suffix_counterfactual_replan_count", 0)) for row in rows)
        total_online = sum(int(row.get("online_diagnosis_count", row.get("heavy_load_diagnosis_count", 0))) for row in rows)
        readme = Path("README.md").read_text(encoding="utf-8")
        sim_readme = Path("simulation/README.md").read_text(encoding="utf-8")

        if total_suffix == 0 or total_online == 0:
            self.assertIn("当前默认收益主要来自 belief-seeded planner / solver thickening", readme)
            self.assertIn("尚未证明 `suffix counterfactual repair` 已成为主导修复机制", readme)
            self.assertNotIn("已经不是主修复语义", readme)
            self.assertIn("observer / posterior logging 已接入主链", sim_readme)
            self.assertIn("当前结果尚未证明 online suffix repair 已成为主导修复机制", sim_readme)

    def test_docs_upgrade_suffix_repair_only_after_online_proof(self):
        rows = json.loads((CURRENT_DIR / "simulation_benchmark_result.json").read_text(encoding="utf-8"))
        heavy = next(row for row in rows if row["task_id"] == "pick_metal_heavy")
        rag = heavy["methods"]["rag_feedback"]
        observer_only = heavy["methods"]["rag_feedback_observer_only"]
        readme = Path("README.md").read_text(encoding="utf-8")

        has_online_proof = (
            rag.get("online_diagnosis_count", 0) > 0
            and rag.get("suffix_counterfactual_replan_count", 0) > 0
            and rag["success_rate_mean"] > observer_only["success_rate_mean"]
        )
        if has_online_proof:
            self.assertIn("rolling posterior + soft-trigger suffix repair 已在默认结果中实际触发", readme)


if __name__ == "__main__":
    unittest.main()
