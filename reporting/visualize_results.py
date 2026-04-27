"""生成问答评测与仿真 benchmark 的可视化图表。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


METHOD_LABELS = {
    "direct_llm": "Direct LLM",
    "base_rag": "Base RAG",
    "improved_rag": "Improved RAG",
    "problem_solving_rag": "Rule-heavy RAG",
    "rag": "RAG",
    "rag_feedback": "RAG + Feedback",
    "rag_learned": "RAG + Learned",
    "rag_llm": "RAG + LLM",
    "task_heuristic": "Task Heuristic",
    "fixed": "Fixed",
}

DEFAULT_SIM_OUTPUT_DIR = Path("outputs/current_observer_step_replan")


def _method_aliases(method: str) -> list[str]:
    aliases = [method]
    if method == "rag":
        aliases.append("rag_feedback")
    elif method == "rag_feedback":
        aliases.append("rag")
    return aliases


def _method_entry(row: dict, method: str) -> dict:
    methods = row.get("methods", {})
    for alias in _method_aliases(method):
        entry = methods.get(alias)
        if entry is not None:
            return entry
    return {}


def _method_metric(row: dict, method: str, field: str, default=np.nan):
    entry = _method_entry(row, method)
    if field in entry:
        return entry[field]
    for alias in _method_aliases(method):
        flat_key = f"{alias}_{field}"
        if flat_key in row:
            return row[flat_key]
    return default


def _plan_mean(row: dict, method: str, field: str, default=np.nan):
    entry = _method_entry(row, method)
    value = entry.get("executed_plan_stats", {}).get("mean", {}).get(field)
    if value is not None:
        return value
    for alias in _method_aliases(method):
        for flat_key in (f"{alias}_{field}", f"{alias}_{field}_mean"):
            if flat_key in row:
                return row[flat_key]
    return default


def _planner_metric(row: dict, method: str, field: str, default=np.nan):
    entry = _method_entry(row, method)
    value = entry.get("planner_diagnostics", {}).get(field)
    if value is not None:
        return value
    for alias in _method_aliases(method):
        flat_key = f"{alias}_{field}"
        if flat_key in row:
            return row[flat_key]
    return default


def _reference_force_deviation_mean(row: dict, method: str, default=np.nan):
    entry = _method_entry(row, method)
    stats = entry.get("reference_force_deviation_stats", {})
    if isinstance(stats, dict) and stats.get("mean") is not None:
        return stats["mean"]
    value = _method_metric(row, method, "reference_force_deviation", default)
    if isinstance(value, dict):
        return value.get("mean", default)
    return value


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _extract_methods_from_sim_rows(sim_rows: list[dict], suffix: str = "_success_rate") -> list[str]:
    methods = set()
    for row in sim_rows:
        if "methods" in row:
            methods.update(row.get("methods", {}).keys())
            continue
        for key in row:
            if key.endswith(suffix):
                methods.add(key[: -len(suffix)])
    return sorted(methods)


def _task_names(sim_rows: list[dict]) -> list[str]:
    return [row["task_id"] for row in sim_rows]


def plot_qa_summary(qa_detail: dict, output_dir: Path) -> None:
    summaries = qa_detail["summaries"]
    methods = list(summaries.keys())
    strict = [summaries[m]["strict_accuracy"] for m in methods]
    weighted = [summaries[m]["weighted_accuracy"] for m in methods]

    x = np.arange(len(methods))
    width = 0.34
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, strict, width, label="Strict Accuracy", color="#315c73")
    ax.bar(x + width / 2, weighted, width, label="Weighted Accuracy", color="#d98c3f")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("QA Method Summary")
    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_LABELS.get(m, m) for m in methods], rotation=12)
    ax.legend()
    _save(fig, output_dir / "qa_method_summary.png")


def plot_qa_heatmap(qa_detail: dict, output_dir: Path) -> None:
    cases = qa_detail["cases"]
    methods = list(qa_detail["method_results"].keys())
    matrix = []
    for method in methods:
        rows = qa_detail["method_results"][method]
        scores = [row["response_eval"]["weighted_score"] for row in rows]
        matrix.append(scores)
    matrix = np.array(matrix, dtype=float)

    fig, ax = plt.subplots(figsize=(12, 5))
    im = ax.imshow(matrix, cmap="YlGnBu", vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_title("QA Weighted Score by Question")
    ax.set_yticks(np.arange(len(methods)))
    ax.set_yticklabels([METHOD_LABELS.get(m, m) for m in methods])
    ax.set_xticks(np.arange(len(cases)))
    ax.set_xticklabels([f"Q{i + 1}" for i in range(len(cases))])
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:.1f}", ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    _save(fig, output_dir / "qa_question_heatmap.png")


def plot_qa_retrieval(qa_detail: dict, output_dir: Path) -> None:
    summaries = qa_detail["summaries"]
    methods = [m for m in summaries if summaries[m]["avg_retrieval_coverage"] is not None]
    coverages = [summaries[m]["avg_retrieval_coverage"] for m in methods]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(np.arange(len(methods)), coverages, color=["#6d8f72", "#9c6644", "#5b84b1"])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Average Coverage")
    ax.set_title("Retrieval Coverage")
    ax.set_xticks(np.arange(len(methods)))
    ax.set_xticklabels([METHOD_LABELS.get(m, m) for m in methods], rotation=12)
    _save(fig, output_dir / "qa_retrieval_coverage.png")


def plot_qa_breakdown(qa_detail: dict, output_dir: Path) -> None:
    summaries = qa_detail["summaries"]
    methods = list(summaries.keys())
    correct = np.array([summaries[m]["correct"] for m in methods], dtype=float)
    partial = np.array([summaries[m]["partial"] for m in methods], dtype=float)
    incorrect = np.array([summaries[m]["incorrect"] for m in methods], dtype=float)

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(methods))
    ax.bar(x, correct, label="Correct", color="#4c956c")
    ax.bar(x, partial, bottom=correct, label="Partial", color="#f4a259")
    ax.bar(x, incorrect, bottom=correct + partial, label="Incorrect", color="#bc4b51")
    ax.set_ylabel("Count")
    ax.set_title("QA Result Breakdown")
    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_LABELS.get(m, m) for m in methods], rotation=12)
    ax.legend()
    _save(fig, output_dir / "qa_result_breakdown.png")


def plot_qa_gain(qa_detail: dict, output_dir: Path) -> None:
    summaries = qa_detail["summaries"]
    baseline = summaries["direct_llm"]
    methods = [m for m in summaries if m != "direct_llm"]
    strict_gain = [summaries[m]["strict_accuracy"] - baseline["strict_accuracy"] for m in methods]
    weighted_gain = [summaries[m]["weighted_accuracy"] - baseline["weighted_accuracy"] for m in methods]

    x = np.arange(len(methods))
    width = 0.34
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.bar(x - width / 2, strict_gain, width, label="Strict Gain", color="#2a6f97")
    ax.bar(x + width / 2, weighted_gain, width, label="Weighted Gain", color="#ee6c4d")
    ax.axhline(0.0, color="#555555", linewidth=1)
    ax.set_ylabel("Gain over Direct LLM")
    ax.set_title("QA Gain over Direct LLM")
    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_LABELS.get(m, m) for m in methods], rotation=12)
    ax.legend()
    _save(fig, output_dir / "qa_gain_over_direct_llm.png")


def plot_sim_success(sim_rows: list[dict], output_dir: Path) -> None:
    methods = _extract_methods_from_sim_rows(sim_rows)
    tasks = _task_names(sim_rows)
    x = np.arange(len(tasks))
    width = 0.22

    fig, ax = plt.subplots(figsize=(12, 5))
    offsets = np.linspace(-width, width, num=len(methods))
    for offset, method in zip(offsets, methods):
        values = [float(_method_metric(row, method, "success_rate", 0.0)) for row in sim_rows]
        ax.bar(x + offset, values, width=width, label=METHOD_LABELS.get(method, method))
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Success Rate")
    ax.set_title("Simulation Success Rate Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(tasks, rotation=18)
    ax.legend()
    _save(fig, output_dir / "simulation_success_rates.png")


def plot_sim_success_ci(sim_rows: list[dict], output_dir: Path) -> None:
    methods = _extract_methods_from_sim_rows(sim_rows)
    tasks = _task_names(sim_rows)
    x = np.arange(len(tasks))
    width = 0.22

    fig, ax = plt.subplots(figsize=(12, 5))
    offsets = np.linspace(-width, width, num=len(methods))
    for offset, method in zip(offsets, methods):
        values = np.array([_method_metric(row, method, "success_rate", 0.0) for row in sim_rows], dtype=float)
        cis = [
            _method_metric(row, method, "success_rate_ci95", [value, value])
            for row, value in zip(sim_rows, values)
        ]
        yerr_low = [max(0.0, value - ci[0]) for value, ci in zip(values, cis)]
        yerr_high = [max(0.0, ci[1] - value) for value, ci in zip(values, cis)]
        ax.bar(
            x + offset,
            values,
            width=width,
            yerr=np.array([yerr_low, yerr_high], dtype=float),
            capsize=3,
            label=METHOD_LABELS.get(method, method),
        )
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Success Rate")
    ax.set_title("Simulation Success Rate with 95% CI")
    ax.set_xticks(x)
    ax.set_xticklabels(tasks, rotation=18)
    ax.legend()
    _save(fig, output_dir / "simulation_success_ci.png")


def plot_sim_time_steps(sim_rows: list[dict], output_dir: Path) -> None:
    methods = _extract_methods_from_sim_rows(sim_rows)
    tasks = _task_names(sim_rows)
    x = np.arange(len(tasks))
    width = 0.22
    offsets = np.linspace(-width, width, num=len(methods))

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    for offset, method in zip(offsets, methods):
        times = [float(_method_metric(row, method, "avg_time_sec", 0.0)) for row in sim_rows]
        steps = [float(_method_metric(row, method, "avg_steps", 0.0)) for row in sim_rows]
        axes[0].bar(x + offset, times, width=width, label=METHOD_LABELS.get(method, method))
        axes[1].bar(x + offset, steps, width=width, label=METHOD_LABELS.get(method, method))
    axes[0].set_ylabel("Avg Sim Time (s)")
    axes[0].set_title("Simulation Time")
    axes[1].set_ylabel("Avg Steps")
    axes[1].set_title("Simulation Steps")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(tasks, rotation=18)
    axes[0].legend()
    _save(fig, output_dir / "simulation_time_steps.png")


def plot_sim_force(sim_rows: list[dict], output_dir: Path) -> None:
    methods = _extract_methods_from_sim_rows(sim_rows)
    tasks = _task_names(sim_rows)
    x = np.arange(len(tasks))

    fig, ax = plt.subplots(figsize=(12, 5))
    for row_idx, row in enumerate(sim_rows):
        lo, hi = row["reference_force_range"]
        ax.vlines(row_idx, lo, hi, color="#999999", linewidth=8, alpha=0.4)
    for method in methods:
        forces = [_plan_mean(row, method, "gripper_force") for row in sim_rows]
        ax.plot(x, forces, marker="o", linewidth=2, label=METHOD_LABELS.get(method, method))
    ax.set_ylabel("Gripper Force (N)")
    ax.set_title("Predicted Force and Reference Range")
    ax.set_xticks(x)
    ax.set_xticklabels(tasks, rotation=18)
    ax.legend()
    _save(fig, output_dir / "simulation_force_ranges.png")


def plot_sim_distance_error(sim_rows: list[dict], output_dir: Path) -> None:
    methods = _extract_methods_from_sim_rows(sim_rows)
    tasks = _task_names(sim_rows)
    x = np.arange(len(tasks))
    width = 0.22

    fig, ax = plt.subplots(figsize=(12, 5))
    offsets = np.linspace(-width, width, num=len(methods))
    for offset, method in zip(offsets, methods):
        values = [float(_method_metric(row, method, "avg_distance_error", 0.0)) for row in sim_rows]
        ax.bar(x + offset, values, width=width, label=METHOD_LABELS.get(method, method))
    ax.set_ylabel("Avg Distance Error")
    ax.set_title("Simulation Distance Error")
    ax.set_xticks(x)
    ax.set_xticklabels(tasks, rotation=18)
    ax.legend()
    _save(fig, output_dir / "simulation_distance_error.png")


def plot_sim_force_deviation(sim_rows: list[dict], output_dir: Path) -> None:
    methods = _extract_methods_from_sim_rows(sim_rows)
    tasks = _task_names(sim_rows)
    x = np.arange(len(tasks))
    width = 0.22

    fig, ax = plt.subplots(figsize=(12, 5))
    offsets = np.linspace(-width, width, num=len(methods))
    for offset, method in zip(offsets, methods):
        values = [
            float(_reference_force_deviation_mean(row, method, 0.0))
            for row in sim_rows
        ]
        ax.bar(x + offset, values, width=width, label=METHOD_LABELS.get(method, method))
    ax.set_ylabel("Absolute Force Deviation (N)")
    ax.set_title("Deviation from Reference Force Center")
    ax.set_xticks(x)
    ax.set_xticklabels(tasks, rotation=18)
    ax.legend()
    _save(fig, output_dir / "simulation_force_deviation.png")


def plot_sim_approach_height(sim_rows: list[dict], output_dir: Path) -> None:
    methods = _extract_methods_from_sim_rows(sim_rows)
    tasks = _task_names(sim_rows)
    x = np.arange(len(tasks))

    fig, ax = plt.subplots(figsize=(12, 5))
    reference = [row.get("reference_approach_height", np.nan) for row in sim_rows]
    ax.plot(x, reference, marker="s", linewidth=2.2, linestyle="--", color="#444444", label="Reference")
    for method in methods:
        heights = [_plan_mean(row, method, "approach_height") for row in sim_rows]
        ax.plot(x, heights, marker="o", linewidth=2, label=METHOD_LABELS.get(method, method))
    ax.set_ylabel("Approach Height (m)")
    ax.set_title("Approach Height Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(tasks, rotation=18)
    ax.legend()
    _save(fig, output_dir / "simulation_approach_height.png")


def plot_sim_control_plan(sim_rows: list[dict], output_dir: Path) -> None:
    methods = _extract_methods_from_sim_rows(sim_rows)
    tasks = _task_names(sim_rows)
    x = np.arange(len(tasks))

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    for method in methods:
        velocities = [_plan_mean(row, method, "transport_velocity") for row in sim_rows]
        clearances = [_plan_mean(row, method, "lift_clearance") for row in sim_rows]
        axes[0].plot(x, velocities, marker="o", linewidth=2, label=METHOD_LABELS.get(method, method))
        axes[1].plot(x, clearances, marker="o", linewidth=2, label=METHOD_LABELS.get(method, method))
    axes[0].set_ylabel("Transport Velocity (m/s)")
    axes[0].set_title("Transport Velocity Comparison")
    axes[1].set_ylabel("Lift Clearance (m)")
    axes[1].set_title("Lift Clearance Comparison")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(tasks, rotation=18)
    axes[0].legend()
    _save(fig, output_dir / "simulation_control_plan.png")


def plot_sim_success_gain(sim_rows: list[dict], output_dir: Path) -> None:
    tasks = _task_names(sim_rows)
    x = np.arange(len(tasks))
    primary_method = (
        "rag_feedback"
        if any(_method_metric(row, "rag_feedback", "success_rate", np.nan) == _method_metric(row, "rag_feedback", "success_rate", np.nan) for row in sim_rows)
        else "rag"
    )
    baseline_method = (
        "rag"
        if primary_method == "rag_feedback"
        and any(_method_metric(row, "rag", "success_rate", np.nan) == _method_metric(row, "rag", "success_rate", np.nan) for row in sim_rows)
        else (
            "direct_llm"
            if any(_method_metric(row, "direct_llm", "success_rate", np.nan) == _method_metric(row, "direct_llm", "success_rate", np.nan) for row in sim_rows)
            else ("task_heuristic" if any(_method_metric(row, "task_heuristic", "success_rate", np.nan) == _method_metric(row, "task_heuristic", "success_rate", np.nan) for row in sim_rows) else "fixed")
        )
    )
    primary_label = METHOD_LABELS.get(primary_method, primary_method)
    baseline_label = METHOD_LABELS.get(baseline_method, baseline_method)
    primary_gain_vs_baseline = [
        float(_method_metric(row, primary_method, "success_rate", 0.0))
        - float(_method_metric(row, baseline_method, "success_rate", 0.0))
        for row in sim_rows
    ]
    heuristic_method = "task_heuristic" if any(_method_metric(row, "task_heuristic", "success_rate", np.nan) == _method_metric(row, "task_heuristic", "success_rate", np.nan) for row in sim_rows) else "fixed"
    primary_gain_vs_heuristic = [
        float(_method_metric(row, primary_method, "success_rate", 0.0))
        - float(_method_metric(row, heuristic_method, "success_rate", 0.0))
        for row in sim_rows
    ]
    heuristic_label = f"{primary_label} - Task Heuristic" if heuristic_method == "task_heuristic" else f"{primary_label} - Fixed"

    width = 0.34
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - width / 2, primary_gain_vs_baseline, width, label=f"{primary_label} - {baseline_label}", color="#2a9d8f")
    ax.bar(x + width / 2, primary_gain_vs_heuristic, width, label=heuristic_label, color="#e76f51")
    ax.axhline(0.0, color="#555555", linewidth=1)
    ax.set_ylabel("Success Rate Gain")
    ax.set_title(f"{primary_label} Gain over Baselines")
    ax.set_xticks(x)
    ax.set_xticklabels(tasks, rotation=18)
    ax.legend()
    _save(fig, output_dir / "simulation_rag_gain.png")


def plot_sim_split_summary(sim_rows: list[dict], output_dir: Path) -> None:
    methods = _extract_methods_from_sim_rows(sim_rows)
    split_order = ["train", "val", "test"]
    splits = [split for split in split_order if any(row.get("task_split") == split for row in sim_rows)]
    x = np.arange(len(splits))
    width = 0.22

    fig, ax = plt.subplots(figsize=(9, 4.8))
    offsets = np.linspace(-width, width, num=len(methods))
    for offset, method in zip(offsets, methods):
        values = []
        for split in splits:
            split_rows = [row for row in sim_rows if row.get("task_split") == split]
            metric = (
                np.mean([_method_metric(row, method, "success_rate", 0.0) for row in split_rows])
                if split_rows else 0.0
            )
            values.append(metric)
        ax.bar(x + offset, values, width=width, label=METHOD_LABELS.get(method, method))
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Average Success Rate")
    ax.set_title("Success Rate by Split")
    ax.set_xticks(x)
    ax.set_xticklabels(splits)
    ax.legend()
    _save(fig, output_dir / "simulation_split_summary.png")


def plot_multi_seed_success(sim_rows: list[dict], output_dir: Path) -> None:
    methods = _extract_methods_from_sim_rows(sim_rows, suffix="_success_rate_mean")
    tasks = _task_names(sim_rows)
    x = np.arange(len(tasks))
    width = 0.22

    fig, ax = plt.subplots(figsize=(12, 5))
    offsets = np.linspace(-width, width, num=len(methods))
    for offset, method in zip(offsets, methods):
        means = [float(_method_metric(row, method, "success_rate_mean", 0.0)) for row in sim_rows]
        stds = [float(_method_metric(row, method, "success_rate_std", 0.0)) for row in sim_rows]
        ax.bar(x + offset, means, width=width, yerr=stds, capsize=3, label=METHOD_LABELS.get(method, method))
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Mean Success Rate")
    ax.set_title("Multi-seed Success Rate")
    ax.set_xticks(x)
    ax.set_xticklabels(tasks, rotation=18)
    ax.legend()
    _save(fig, output_dir / "simulation_multi_seed_success.png")


def plot_sim_belief_diagnostics(sim_rows: list[dict], output_dir: Path) -> None:
    tasks = _task_names(sim_rows)
    x = np.arange(len(tasks))
    belief_coverages = [_planner_metric(row, "rag", "belief_state_coverage_mean", np.nan) for row in sim_rows]
    conservative_rates = [_planner_metric(row, "rag", "uncertainty_conservative_mode_mean", np.nan) for row in sim_rows]
    solver_labels = [str(_planner_metric(row, "rag", "solver_selected_candidate_mode", "n/a")) for row in sim_rows]

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    axes[0].bar(x, belief_coverages, color="#4c956c")
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("Belief Coverage")
    axes[0].set_title("RAG Belief Coverage by Task")
    for idx, (value, solver) in enumerate(zip(belief_coverages, solver_labels)):
        if np.isnan(value):
            continue
        axes[0].text(
            idx,
            min(1.02, value + 0.03),
            solver.replace("_", "\n"),
            ha="center",
            va="bottom",
            fontsize=8,
        )

    axes[1].bar(x, conservative_rates, color="#bc4b51")
    axes[1].set_ylim(0, 1.05)
    axes[1].set_ylabel("Conservative Rate")
    axes[1].set_title("RAG Uncertainty Conservative Mode Rate")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(tasks, rotation=18)
    _save(fig, output_dir / "simulation_belief_diagnostics.png")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qa_json", default="outputs/current/qa_evaluation_detail.json")
    parser.add_argument("--sim_json", default=str(DEFAULT_SIM_OUTPUT_DIR / "simulation_comparison_rag_vs_baseline.json"))
    parser.add_argument("--sim_multi_seed_json", default=str(DEFAULT_SIM_OUTPUT_DIR / "simulation_comparison_multi_seed.json"))
    parser.add_argument("--output_dir", default=str(DEFAULT_SIM_OUTPUT_DIR / "visualizations"))
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    qa_detail = json.loads(Path(args.qa_json).read_text(encoding="utf-8"))
    sim_rows = json.loads(Path(args.sim_json).read_text(encoding="utf-8"))

    plot_qa_summary(qa_detail, output_dir)
    plot_qa_heatmap(qa_detail, output_dir)
    plot_qa_retrieval(qa_detail, output_dir)
    plot_qa_breakdown(qa_detail, output_dir)
    plot_qa_gain(qa_detail, output_dir)
    plot_sim_success(sim_rows, output_dir)
    plot_sim_success_ci(sim_rows, output_dir)
    plot_sim_time_steps(sim_rows, output_dir)
    plot_sim_force(sim_rows, output_dir)
    plot_sim_distance_error(sim_rows, output_dir)
    plot_sim_force_deviation(sim_rows, output_dir)
    plot_sim_approach_height(sim_rows, output_dir)
    plot_sim_control_plan(sim_rows, output_dir)
    plot_sim_success_gain(sim_rows, output_dir)
    plot_sim_split_summary(sim_rows, output_dir)

    multi_seed_path = Path(args.sim_multi_seed_json)
    if multi_seed_path.exists():
        multi_seed_rows = json.loads(multi_seed_path.read_text(encoding="utf-8"))
        plot_multi_seed_success(multi_seed_rows, output_dir)
        if any(
            _planner_metric(row, "rag", "belief_state_coverage_mean", np.nan)
            == _planner_metric(row, "rag", "belief_state_coverage_mean", np.nan)
            for row in multi_seed_rows
        ):
            plot_sim_belief_diagnostics(multi_seed_rows, output_dir)

    print(f"图表已生成到: {output_dir}")


if __name__ == "__main__":
    main()
