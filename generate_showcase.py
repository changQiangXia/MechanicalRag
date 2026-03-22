"""基于结构化评测结果生成展示摘要。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load_json(path: str) -> list | dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _format_pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def build_summary(
    qa_path: str,
    sim_compare_path: str,
    sim_multi_seed_path: str,
    sim_benchmark_path: str,
    output_path: str,
) -> None:
    qa_detail = _load_json(qa_path)
    sim_compare = _load_json(sim_compare_path)
    sim_multi_seed = _load_json(sim_multi_seed_path)
    sim_benchmark = _load_json(sim_benchmark_path)

    summaries = qa_detail["summaries"]
    representative_case_ids = ("cylinder_pose", "smooth_metal_force", "thin_wall_handling")
    methods = ("direct_llm", "base_rag", "improved_rag", "problem_solving_rag")

    lines = [
        "机械 RAG 项目展示摘要",
        "=" * 64,
        "",
        "一、问答结果总览",
    ]
    for method in methods:
        summary = summaries[method]
        lines.append(
            f"- {method}: strict={summary['strict_accuracy']:.2f}, "
            f"weighted={summary['weighted_accuracy']:.2f}, "
            f"correct={summary['correct']}, incorrect={summary['incorrect']}"
        )

    lines.extend(
        [
            "",
            "二、代表性问答样例",
        ]
    )
    cases = {row["case_id"]: row for row in qa_detail["cases"]}
    for case_id in representative_case_ids:
        case = cases[case_id]
        lines.append(f"- 问题：{case['question']}")
        lines.append(f"  参考：{case['gold_answer']}")
        for method in methods:
            row = next(item for item in qa_detail["method_results"][method] if item["question"] == case["question"])
            eval_result = row["response_eval"]["label"]
            lines.append(f"  {method}: {eval_result} | {row['response']}")
        lines.append("")

    lines.append("三、仿真对比重点")
    avg_rag_gain = sum(
        row["rag_success_rate_mean"] - row["direct_llm_success_rate_mean"] for row in sim_multi_seed
    ) / len(sim_multi_seed)
    avg_fixed_gain = sum(
        row["rag_success_rate_mean"] - row["fixed_success_rate_mean"] for row in sim_multi_seed
    ) / len(sim_multi_seed)
    lines.append(f"- 多 seed 平均上，RAG 相对 Direct LLM 的成功率提升为 {_format_pct(avg_rag_gain)}。")
    lines.append(f"- 多 seed 平均上，RAG 相对固定基线的成功率提升为 {_format_pct(avg_fixed_gain)}。")

    top_gap_rows = sorted(
        sim_multi_seed,
        key=lambda row: row["rag_success_rate_mean"] - row["direct_llm_success_rate_mean"],
        reverse=True,
    )[:3]
    for row in top_gap_rows:
        gain = row["rag_success_rate_mean"] - row["direct_llm_success_rate_mean"]
        lines.append(
            f"- {row['task_id']}: RAG={row['rag_success_rate_mean']:.4f}, "
            f"Direct LLM={row['direct_llm_success_rate_mean']:.4f}, "
            f"提升={gain:.4f}"
        )

    lines.extend(
        [
            "",
            "四、正式汇报建议引用文件",
            "- qa_evaluation_detail.json",
            "- simulation_benchmark_result.json",
            "- simulation_comparison_multi_seed.json",
            "- visualizations/qa_method_summary.png",
            "- visualizations/qa_gain_over_direct_llm.png",
            "- visualizations/simulation_rag_gain.png",
            "- visualizations/simulation_multi_seed_success.png",
            "",
            "五、当前主结论",
            f"- improved_rag 在当前 8 道机械问答上达到 strict={summaries['improved_rag']['strict_accuracy']:.2f}。",
            f"- problem_solving_rag 在当前 8 道机械问答上达到 strict={summaries['problem_solving_rag']['strict_accuracy']:.2f}。",
        ]
    )

    hardest_task = min(sim_benchmark, key=lambda row: row["success_rate_mean"])
    easiest_task = max(sim_benchmark, key=lambda row: row["success_rate_mean"])
    lines.append(
        f"- RAG 多 seed benchmark 中，最难任务为 {hardest_task['task_id']}，平均成功率 {hardest_task['success_rate_mean']:.4f}。"
    )
    lines.append(
        f"- RAG 多 seed benchmark 中，最稳任务为 {easiest_task['task_id']}，平均成功率 {easiest_task['success_rate_mean']:.4f}。"
    )

    Path(output_path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qa_json", default="qa_evaluation_detail.json")
    parser.add_argument("--sim_json", default="simulation_comparison_rag_vs_baseline.json")
    parser.add_argument("--sim_multi_seed_json", default="simulation_comparison_multi_seed.json")
    parser.add_argument("--sim_benchmark_json", default="simulation_benchmark_result.json")
    parser.add_argument("--output", default="showcase_summary.txt")
    args = parser.parse_args()

    build_summary(
        qa_path=args.qa_json,
        sim_compare_path=args.sim_json,
        sim_multi_seed_path=args.sim_multi_seed_json,
        sim_benchmark_path=args.sim_benchmark_json,
        output_path=args.output,
    )
    print(f"展示摘要已生成到: {args.output}")


if __name__ == "__main__":
    main()
