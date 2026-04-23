"""CLI wrapper for simulation benchmark execution."""

from __future__ import annotations

import argparse
from pathlib import Path

from .env import HAS_MUJOCO
from .runner import (
    run_benchmark,
    run_benchmark_comparison,
    run_benchmark_comparison_multi_seed,
    run_evidence_ablation_multi_seed,
    run_motion_ablation_multi_seed,
    run_benchmark_multi_seed_report,
    run_retrieval_ablation,
)


DEFAULT_OUTPUT_DIR = Path("outputs/current")


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG 驱动机械臂仿真 Benchmark CLI")
    parser.add_argument("--data_path", default="mechanical_data.txt")
    parser.add_argument("--n_trials", type=int, default=10)
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_DIR / "simulation_benchmark_result.json"))
    parser.add_argument(
        "--method",
        choices=("rag", "rag_generic_only", "rag_no_motion_rules", "rag_multi", "rag_random", "rag_llm", "direct_llm", "rag_learned", "rag_feedback", "fixed", "task_heuristic", "random"),
        default="rag",
    )
    parser.add_argument("--max_feedback_retries", type=int, default=1)
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--compare_multi_seed", action="store_true")
    parser.add_argument("--report_multi_seed", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--multi_seed_methods", nargs="+", default=["rag", "rag_learned", "task_heuristic", "direct_llm", "fixed"])
    parser.add_argument("--ablation_retrieval", action="store_true")
    parser.add_argument("--compare_learned", action="store_true")
    parser.add_argument("--compare_llm", action="store_true")
    parser.add_argument("--compare_direct_llm", action="store_true")
    parser.add_argument("--compare_feedback", action="store_true")
    parser.add_argument("--compare_evidence_ablation", action="store_true")
    parser.add_argument("--compare_motion_ablation", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.ablation_retrieval:
        run_retrieval_ablation(
            data_path=args.data_path,
            n_trials_per_task=args.n_trials,
            seed=args.seed,
            output_dir=str(output_dir),
        )
        table_path = output_dir / "simulation_ablation_retrieval.txt"
        print("\n" + table_path.read_text(encoding="utf-8"))
        return

    if args.compare_multi_seed:
        run_benchmark_comparison_multi_seed(
            data_path=args.data_path,
            n_trials_per_task=args.n_trials,
            seeds=args.seeds,
            output_dir=str(output_dir),
            methods=args.multi_seed_methods,
        )
        table_path = output_dir / "simulation_comparison_multi_seed.txt"
        print("\n" + table_path.read_text(encoding="utf-8"))
        return

    if args.report_multi_seed:
        run_benchmark_multi_seed_report(
            data_path=args.data_path,
            n_trials_per_task=args.n_trials,
            seeds=args.seeds,
            output_path=args.output,
            method=args.method,
        )
        table_path = Path(args.output).with_suffix(".txt")
        if table_path.exists():
            print("\n" + table_path.read_text(encoding="utf-8"))
        return

    if args.compare_evidence_ablation:
        run_evidence_ablation_multi_seed(
            data_path=args.data_path,
            n_trials_per_task=args.n_trials,
            seeds=args.seeds,
            output_dir=str(output_dir),
        )
        table_path = output_dir / "simulation_evidence_ablation.txt"
        print("\n" + table_path.read_text(encoding="utf-8"))
        dependence_path = output_dir / "simulation_evidence_dependence_summary.txt"
        if dependence_path.exists():
            print("\n" + dependence_path.read_text(encoding="utf-8"))
        return

    if args.compare_motion_ablation:
        run_motion_ablation_multi_seed(
            data_path=args.data_path,
            n_trials_per_task=args.n_trials,
            seeds=args.seeds,
            output_dir=str(output_dir),
        )
        table_path = output_dir / "simulation_motion_ablation.txt"
        print("\n" + table_path.read_text(encoding="utf-8"))
        dependence_path = output_dir / "simulation_motion_dependence_summary.txt"
        if dependence_path.exists():
            print("\n" + dependence_path.read_text(encoding="utf-8"))
        return

    compare_methods = None
    if args.compare_learned:
        compare_methods = ["rag", "rag_learned", "task_heuristic", "fixed"]
    elif args.compare_llm:
        compare_methods = ["rag", "rag_llm", "task_heuristic", "fixed"]
    elif args.compare_direct_llm:
        compare_methods = ["rag", "rag_learned", "task_heuristic", "direct_llm", "fixed"]
    elif args.compare_feedback:
        compare_methods = ["rag", "rag_feedback", "task_heuristic", "fixed"]
    elif args.compare:
        compare_methods = ["rag", "rag_learned", "task_heuristic", "fixed"]

    if compare_methods is not None:
        run_benchmark_comparison(
            data_path=args.data_path,
            n_trials_per_task=args.n_trials,
            seed=args.seed,
            output_dir=str(output_dir),
            methods=compare_methods,
        )
        table_path = output_dir / "simulation_comparison_rag_vs_baseline.txt"
        print("\n" + table_path.read_text(encoding="utf-8"))
        return

    results = run_benchmark(
        data_path=args.data_path,
        n_trials_per_task=args.n_trials,
        gui=args.gui,
        output_path=args.output,
        method=args.method,
        seed=args.seed,
        max_feedback_retries=args.max_feedback_retries,
    )

    print("=" * 68)
    print(f"机械臂仿真 Benchmark 结果 (method={args.method})")
    if not HAS_MUJOCO:
        print("【降级模式】未安装 mujoco，结果基于环境代理模型（无 MuJoCo 物理仿真）")
    print("=" * 68)
    for result in results:
        print(f"\n任务: {result.task_description}")
        print(f"  Success Rate: {result.success_count}/{result.n_trials} = {result.success_rate:.2%}")
        print(f"  95%区间: [{result.ci95_low:.2%}, {result.ci95_high:.2%}]")
        print(f"  平均模拟时间: {result.avg_time:.3f}s")
        print(f"  平均步数: {result.avg_steps:.1f}")
        print(
            "  诊断: "
            f"stability={result.avg_stability_score:.3f}, "
            f"slip_risk={result.avg_slip_risk:.3f}, "
            f"compression_risk={result.avg_compression_risk:.3f}, "
            f"dominant_failure={result.dominant_failure_mode}"
        )
        print(f"  参数: {result.params_used}")
    print("\n结果已保存至:", args.output)


if __name__ == "__main__":
    main()
