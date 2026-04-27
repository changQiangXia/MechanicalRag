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
        self.assertIn("--multi_seed_methods rag rag_feedback rag_feedback_observer_only", text)


if __name__ == "__main__":
    unittest.main()
