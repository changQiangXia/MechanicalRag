"""
Benchmark 评测脚本，参考论文的评测协议。
- GRASPA 1.0: 固定场景、success rate、可重复
- FMB: 多任务、多物体类型
- REPLAB: N 次重复、取平均
- 支持 RAG vs 无 RAG 基线对比（--method rag|fixed|random，--compare）
"""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .env import ArmSimEnv, HAS_MUJOCO
from .rag_controller import RAGController
from .seed_utils import stable_seed_offset
from .tasks import BENCHMARK_TASKS, TaskConfig
from . import baseline_controller as baseline


@dataclass
class BenchmarkResult:
    """单次 benchmark 结果。"""
    task_id: str
    task_description: str
    task_split: str
    ideal_gripper_force: tuple[float, float]
    n_trials: int
    success_count: int
    success_rate: float
    avg_time: float
    avg_steps: float
    avg_distance_error: float
    ci95_low: float
    ci95_high: float
    force_deviation_from_center: float
    rag_params_used: dict
    method: str = "rag"  # "rag" | "fixed" | "random"


def _task_label(task: TaskConfig) -> str:
    return f"{task.task_id}/{task.description}"


def _ideal_force_center(task: TaskConfig) -> float:
    return (task.ideal_gripper_force[0] + task.ideal_gripper_force[1]) / 2.0


def _ideal_approach_height(task: TaskConfig) -> float | None:
    if task.profile is None:
        return None
    return task.profile.preferred_approach_height


def _force_deviation_from_center(task: TaskConfig, gripper_force: float) -> float:
    return abs(gripper_force - _ideal_force_center(task))


def _approach_height_error(task: TaskConfig, approach_height: float | None) -> float | None:
    ideal = _ideal_approach_height(task)
    if ideal is None or approach_height is None:
        return None
    return abs(approach_height - ideal)


def _summarize_params(params: dict) -> dict:
    kept = {}
    for key in ("gripper_force", "approach_height", "uncertainty_std"):
        if key in params:
            kept[key] = params[key]
    return kept


def _get_param_getter(
    method: str,
    data_path: str,
    seed: int,
) -> Callable[[str], dict]:
    """返回 (task_description -> params) 的获取函数。"""
    if method == "rag":
        rag = RAGController(data_path)
        return lambda desc: rag.get_params_for_task(desc, retrieval="single")
    if method == "rag_multi":
        rag = RAGController(data_path)
        return lambda desc: rag.get_params_for_task(desc, retrieval="multi")
    if method == "rag_random":
        rag = RAGController(data_path)
        return lambda desc: rag.get_params_for_task(
            desc,
            retrieval="random",
            seed=seed + stable_seed_offset(desc),
        )
    if method == "rag_llm":
        from llm_loader import get_llm
        llm = get_llm(max_new_tokens=128)
        rag = RAGController(data_path)
        return lambda desc: rag.get_params_for_task_llm(desc, llm, retrieval="single")
    if method == "direct_llm":
        from llm_loader import get_llm
        llm = get_llm(max_new_tokens=128)
        return lambda desc: baseline.get_params_llm_direct(desc, llm)
    if method == "rag_learned":
        from .learned_controller import LearnedParamController
        learned = LearnedParamController(data_path=data_path)
        return lambda desc: learned.get_params_for_task(desc)
    if method == "rag_feedback":
        # 带反馈环节：初参由 RAG 给出，失败时用 get_params_after_feedback 调整后重试
        rag = RAGController(data_path)
        return lambda desc: rag.get_params_for_task(desc, retrieval="single")
    if method == "fixed":
        return lambda desc: baseline.get_params_fixed(desc, seed=seed)
    if method == "random":
        return lambda desc: baseline.get_params_random(desc, seed=seed)
    raise ValueError(
        f"未知 method: {method}，应为 rag | rag_multi | rag_random | rag_llm | direct_llm | rag_learned | rag_feedback | fixed | random"
    )


def _get_feedback_getter(method: str, data_path: str):
    """
    若 method 支持反馈重试，返回 (rag_controller, get_params_after_feedback 的绑定)；
    否则返回 None。
    """
    if method != "rag_feedback":
        return None
    rag = RAGController(data_path)
    return rag


def run_benchmark(
    data_path: str = "mechanical_data.txt",
    n_trials_per_task: int = 10,
    gui: bool = False,
    seed: int = 42,
    output_path: str | None = "simulation_benchmark_result.json",
    method: str = "rag",
    max_feedback_retries: int = 1,
) -> list[BenchmarkResult]:
    """
    运行完整 benchmark：对每个任务用指定 method 获取参数，在仿真中执行 n_trials 次，统计 success rate。
    method: "rag" 使用知识库检索；"rag_feedback" 在 RAG 基础上增加失败后反馈调整并重试；"fixed" 固定 25N；"random" 在 [5,50]N 随机。
    max_feedback_retries: 仅当 method="rag_feedback" 时有效，每次 trial 失败后最多用反馈参数重试的次数（默认 1）。
    若未安装 mujoco，自动使用降级模式（仅 RAG 成功模型，无物理仿真）。
    """
    if not HAS_MUJOCO:
        import warnings
        warnings.warn("未检测到 mujoco，使用降级模式（RAG 成功模型，无物理仿真）", UserWarning)
    get_params = _get_param_getter(method, data_path, seed)
    feedback_rag = _get_feedback_getter(method, data_path)
    results = []

    def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
        if n == 0:
            return 0.0, 0.0
        p = k / n
        denom = 1 + z * z / n
        center = (p + z * z / (2 * n)) / denom
        margin = z * ((p * (1 - p) + z * z / (4 * n)) / n) ** 0.5 / denom
        return max(0.0, center - margin), min(1.0, center + margin)

    for task in BENCHMARK_TASKS:
        params = get_params(task.description)
        gripper_force = params["gripper_force"]
        ideal_range = task.ideal_gripper_force

        env = ArmSimEnv(gui=gui, seed=seed + stable_seed_offset(task.task_id))
        successes = 0
        times = []
        step_counts = []
        distance_errors = []
        last_params_used = _summarize_params(params)

        for trial in range(n_trials_per_task):
            current_params = dict(params)
            gripper_force = current_params["gripper_force"]
            success, elapsed, info = env.execute_pick_place(
                object_pos=task.object_pos,
                target_pos=task.target_pos,
                gripper_force=gripper_force,
                ideal_force_range=ideal_range,
                approach_height=current_params.get("approach_height", 0.05),
                object_profile=task.profile.__dict__ if task.profile is not None else None,
            )
            # 反馈环节：失败时根据反馈调整参数并重试（仅 rag_feedback）
            retries = 0
            while (
                not success
                and feedback_rag is not None
                and retries < max_feedback_retries
            ):
                current_params = feedback_rag.get_params_after_feedback(
                    task.description,
                    current_params,
                    success,
                    info,
                    ideal_range,
                )
                gripper_force = current_params["gripper_force"]
                success, elapsed_retry, info = env.execute_pick_place(
                    object_pos=task.object_pos,
                    target_pos=task.target_pos,
                    gripper_force=gripper_force,
                    ideal_force_range=ideal_range,
                    approach_height=current_params.get("approach_height", 0.05),
                    object_profile=task.profile.__dict__ if task.profile is not None else None,
                )
                elapsed += elapsed_retry
                retries += 1
            last_params_used = _summarize_params(current_params)
            if success:
                successes += 1
            times.append(elapsed)
            step_counts.append(info.get("steps", 0))
            distance_errors.append(info.get("distance", 0.0))

        env.close()
        ci95_low, ci95_high = wilson_interval(successes, n_trials_per_task)

        results.append(
            BenchmarkResult(
                task_id=task.task_id,
                task_description=task.description,
                task_split=task.split,
                ideal_gripper_force=task.ideal_gripper_force,
                n_trials=n_trials_per_task,
                success_count=successes,
                success_rate=successes / n_trials_per_task,
                avg_time=sum(times) / len(times),
                avg_steps=sum(step_counts) / len(step_counts),
                avg_distance_error=sum(distance_errors) / len(distance_errors),
                ci95_low=ci95_low,
                ci95_high=ci95_high,
                force_deviation_from_center=_force_deviation_from_center(
                    task,
                    float(last_params_used.get("gripper_force", 0.0)),
                ),
                rag_params_used=last_params_used,
                method=method,
            )
        )

    if output_path:
        out = []
        for r in results:
            out.append({
                "task_id": r.task_id,
                "task_label": _task_label(next(task for task in BENCHMARK_TASKS if task.task_id == r.task_id)),
                "task_description": r.task_description,
                "task_split": r.task_split,
                "method": r.method,
                "n_trials": r.n_trials,
                "ideal_gripper_force": list(r.ideal_gripper_force),
                "ideal_approach_height": _ideal_approach_height(next(task for task in BENCHMARK_TASKS if task.task_id == r.task_id)),
                "success_count": r.success_count,
                "success_rate": round(r.success_rate, 4),
                "avg_time_sec": round(r.avg_time, 4),
                "avg_steps": round(r.avg_steps, 2),
                "avg_distance_error": round(r.avg_distance_error, 4),
                "success_rate_ci95": [round(r.ci95_low, 4), round(r.ci95_high, 4)],
                "force_deviation_from_center": round(r.force_deviation_from_center, 4),
                "rag_params": r.rag_params_used,
            })
        Path(output_path).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    return results


def run_benchmark_multi_seed_report(
    data_path: str = "mechanical_data.txt",
    n_trials_per_task: int = 10,
    seeds: list[int] | None = None,
    output_path: str | None = "simulation_benchmark_result.json",
    method: str = "rag",
) -> list[dict]:
    """
    针对单一 method 做多 seed 汇总，输出更适合正式汇报的 mean±std 结果。
    """
    if seeds is None:
        seeds = [42, 43, 44]

    per_seed_results: dict[int, list[BenchmarkResult]] = {}
    for seed in seeds:
        per_seed_results[seed] = run_benchmark(
            data_path=data_path,
            n_trials_per_task=n_trials_per_task,
            gui=False,
            seed=seed,
            output_path=None,
            method=method,
        )

    summary_rows: list[dict] = []
    for task in BENCHMARK_TASKS:
        task_runs = [
            next(result for result in per_seed_results[seed] if result.task_id == task.task_id)
            for seed in seeds
        ]
        success_rates = [run.success_rate for run in task_runs]
        avg_times = [run.avg_time for run in task_runs]
        avg_steps = [run.avg_steps for run in task_runs]
        distance_errors = [run.avg_distance_error for run in task_runs]
        force_deviations = [run.force_deviation_from_center for run in task_runs]
        gripper_forces = [float(run.rag_params_used.get("gripper_force", 0.0)) for run in task_runs]
        approach_heights = [float(run.rag_params_used.get("approach_height", 0.0)) for run in task_runs]

        def _mean(values: list[float]) -> float:
            return round(statistics.mean(values), 4)

        def _std(values: list[float]) -> float:
            return round(statistics.stdev(values), 4) if len(values) > 1 else 0.0

        summary_rows.append(
            {
                "task_id": task.task_id,
                "task_label": _task_label(task),
                "task_description": task.description,
                "task_split": task.split,
                "method": method,
                "seeds": seeds,
                "n_trials_per_seed": n_trials_per_task,
                "total_trials": n_trials_per_task * len(seeds),
                "ideal_gripper_force": list(task.ideal_gripper_force),
                "ideal_force_center": round(_ideal_force_center(task), 4),
                "ideal_approach_height": _ideal_approach_height(task),
                "success_rate_mean": _mean(success_rates),
                "success_rate_std": _std(success_rates),
                "avg_time_sec_mean": _mean(avg_times),
                "avg_time_sec_std": _std(avg_times),
                "avg_steps_mean": _mean(avg_steps),
                "avg_steps_std": _std(avg_steps),
                "avg_distance_error_mean": _mean(distance_errors),
                "avg_distance_error_std": _std(distance_errors),
                "force_deviation_from_center_mean": _mean(force_deviations),
                "force_deviation_from_center_std": _std(force_deviations),
                "gripper_force_mean": _mean(gripper_forces),
                "gripper_force_std": _std(gripper_forces),
                "approach_height_mean": _mean(approach_heights),
                "approach_height_std": _std(approach_heights),
                "per_seed_success_rate": {str(seed): round(run.success_rate, 4) for seed, run in zip(seeds, task_runs)},
            }
        )

    if output_path:
        Path(output_path).write_text(json.dumps(summary_rows, ensure_ascii=False, indent=2), encoding="utf-8")
        table_path = Path(output_path).with_suffix(".txt")
        lines = [
            f"{method} benchmark 多 seed 汇总结果",
            "=" * 72,
            f"seeds={seeds}, n_trials_per_seed={n_trials_per_task}, total_trials={n_trials_per_task * len(seeds)}",
            "",
            f"{'任务':<32} {'成功率(mean±std)':<18} {'时间(mean±std)':<18} {'步数(mean±std)':<18}",
            "-" * 72,
        ]
        for row in summary_rows:
            lines.append(
                f"{row['task_label']:<32} "
                f"{row['success_rate_mean']:.2%}±{row['success_rate_std']:.2%} "
                f"{row['avg_time_sec_mean']:.3f}s±{row['avg_time_sec_std']:.3f} "
                f"{row['avg_steps_mean']:.2f}±{row['avg_steps_std']:.2f}"
            )
        lines.append("")
        lines.append(f"JSON: {output_path}")
        table_path.write_text("\n".join(lines), encoding="utf-8")

    return summary_rows


def run_benchmark_comparison(
    data_path: str = "mechanical_data.txt",
    n_trials_per_task: int = 10,
    seed: int = 42,
    output_dir: str | None = None,
    methods: list[str] | None = None,
) -> dict[str, list[BenchmarkResult]]:
    """
    运行 RAG 与基线对比实验，生成对比表与 JSON。
    默认对比 rag vs fixed；若指定 methods 则按列表依次运行。
    """
    if methods is None:
        methods = ["rag", "fixed"]
    output_dir = output_dir or "."
    all_results: dict[str, list[BenchmarkResult]] = {}

    for method in methods:
        out_file = Path(output_dir) / f"simulation_benchmark_{method}.json"
        all_results[method] = run_benchmark(
            data_path=data_path,
            n_trials_per_task=n_trials_per_task,
            gui=False,
            seed=seed,
            output_path=str(out_file),
            method=method,
        )

    # 写入对比汇总 JSON
    comparison = []
    for task in BENCHMARK_TASKS:
        row = {
            "task_id": task.task_id,
            "task_label": _task_label(task),
            "task_description": task.description,
            "task_split": task.split,
            "ideal_gripper_force": list(task.ideal_gripper_force),
            "ideal_force_center": round(_ideal_force_center(task), 4),
            "ideal_approach_height": _ideal_approach_height(task),
        }
        for method in methods:
            res = next(r for r in all_results[method] if r.task_id == task.task_id)
            row[f"{method}_success_rate"] = round(res.success_rate, 4)
            row[f"{method}_success_count"] = res.success_count
            row[f"{method}_success_rate_ci95"] = [round(res.ci95_low, 4), round(res.ci95_high, 4)]
            row[f"{method}_gripper_force"] = res.rag_params_used.get("gripper_force")
            row[f"{method}_approach_height"] = res.rag_params_used.get("approach_height")
            error = _approach_height_error(task, res.rag_params_used.get("approach_height"))
            row[f"{method}_approach_height_error"] = None if error is None else round(error, 4)
            row[f"{method}_avg_time_sec"] = round(res.avg_time, 4)
            row[f"{method}_avg_steps"] = round(res.avg_steps, 2)
            row[f"{method}_avg_distance_error"] = round(res.avg_distance_error, 4)
            row[f"{method}_force_deviation_from_center"] = round(res.force_deviation_from_center, 4)
        comparison.append(row)

    comparison_path = Path(output_dir) / "simulation_comparison_rag_vs_baseline.json"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    with open(comparison_path, "w", encoding="utf-8") as f:
        json.dump(comparison, f, ensure_ascii=False, indent=2)

    # 写入对比表格（便于论文/报告）
    table_path = Path(output_dir) / "simulation_comparison_rag_vs_baseline.txt"
    lines = [
        "RAG vs 无 RAG 基线 对比结果",
        "=" * 60,
        f"n_trials_per_task={n_trials_per_task}, seed={seed}",
        "",
        f"{'任务':<32} {'理想夹爪力(N)':<14} " + " ".join(f"{m}_成功率" for m in methods),
        "-" * 60,
    ]
    for task in BENCHMARK_TASKS:
        ideal = f"{task.ideal_gripper_force[0]}-{task.ideal_gripper_force[1]}"
        rates = []
        for method in methods:
            res = next(r for r in all_results[method] if r.task_id == task.task_id)
            rates.append(f"{res.success_rate:.2%}")
        lines.append(f"{_task_label(task):<32} {ideal:<14} " + " ".join(rates))
    lines.append("")
    lines.append(f"对比 JSON: {comparison_path}")
    Path(table_path).write_text("\n".join(lines), encoding="utf-8")

    return all_results


def run_benchmark_comparison_multi_seed(
    data_path: str = "mechanical_data.txt",
    n_trials_per_task: int = 10,
    seeds: list[int] | None = None,
    output_dir: str | None = None,
    methods: list[str] | None = None,
) -> dict[tuple[str, str], list[float]]:
    """
    多 seed 运行 RAG vs 基线对比，汇总 mean±std，便于写进论文。
    """
    if seeds is None:
        seeds = [42, 43, 44]
    if methods is None:
        methods = ["rag", "direct_llm", "fixed"]
    output_dir = output_dir or "."
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # 按 seed 收集每 (task_id, method) 的 success_rate 列表
    rates: dict[tuple[str, str], list[float]] = {}

    for seed in seeds:
        all_results = run_benchmark_comparison(
            data_path=data_path,
            n_trials_per_task=n_trials_per_task,
            seed=seed,
            output_dir=output_dir,
            methods=methods,
        )
        for method in methods:
            for r in all_results[method]:
                key = (r.task_id, method)
                if key not in rates:
                    rates[key] = []
                rates[key].append(r.success_rate)

    # 汇总 mean ± std
    comparison = []
    for task in BENCHMARK_TASKS:
        row = {
            "task_id": task.task_id,
            "task_label": _task_label(task),
            "task_description": task.description,
            "task_split": task.split,
            "ideal_gripper_force": list(task.ideal_gripper_force),
        }
        for method in methods:
            key = (task.task_id, method)
            vals = rates.get(key, [])
            if vals:
                row[f"{method}_success_rate_mean"] = round(statistics.mean(vals), 4)
                row[f"{method}_success_rate_std"] = round(statistics.stdev(vals), 4) if len(vals) > 1 else 0.0
            else:
                row[f"{method}_success_rate_mean"] = None
                row[f"{method}_success_rate_std"] = None
        comparison.append(row)

    out_json = Path(output_dir) / "simulation_comparison_multi_seed.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(comparison, f, ensure_ascii=False, indent=2)

    lines = [
        "RAG vs 无 RAG 基线 对比结果（多 seed mean±std）",
        "=" * 70,
        f"n_trials_per_task={n_trials_per_task}, seeds={seeds}",
        "",
        f"{'任务':<32} {'理想夹爪力(N)':<12} " + " ".join(f"{m}_成功率(mean±std)" for m in methods),
        "-" * 70,
    ]
    for task in BENCHMARK_TASKS:
        ideal = f"{task.ideal_gripper_force[0]}-{task.ideal_gripper_force[1]}"
        parts = []
        for method in methods:
            key = (task.task_id, method)
            vals = rates.get(key, [])
            if vals:
                m = statistics.mean(vals)
                s = statistics.stdev(vals) if len(vals) > 1 else 0.0
                parts.append(f"{m:.2%}±{s:.2%}")
            else:
                parts.append("N/A")
        lines.append(f"{_task_label(task):<32} {ideal:<12} " + " ".join(parts))
    lines.append("")
    lines.append(f"对比 JSON: {out_json}")
    table_path = Path(output_dir) / "simulation_comparison_multi_seed.txt"
    Path(table_path).write_text("\n".join(lines), encoding="utf-8")

    return rates


def run_retrieval_ablation(
    data_path: str = "mechanical_data.txt",
    n_trials_per_task: int = 10,
    seed: int = 42,
    output_dir: str | None = None,
) -> dict[str, list[BenchmarkResult]]:
    """
    检索策略消融：单 query (rag) vs 多 query (rag_multi) vs 随机文档 (rag_random) vs 固定基线 (fixed)。
    """
    methods = ["rag", "rag_multi", "rag_random", "fixed"]
    output_dir = output_dir or "."
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    all_results: dict[str, list[BenchmarkResult]] = {}

    for method in methods:
        out_file = Path(output_dir) / f"simulation_benchmark_{method}.json"
        all_results[method] = run_benchmark(
            data_path=data_path,
            n_trials_per_task=n_trials_per_task,
            gui=False,
            seed=seed,
            output_path=str(out_file),
            method=method,
        )

    comparison = []
    for task in BENCHMARK_TASKS:
        row = {
            "task_id": task.task_id,
            "task_label": _task_label(task),
            "task_description": task.description,
            "task_split": task.split,
            "ideal_gripper_force": list(task.ideal_gripper_force),
        }
        for method in methods:
            res = next(r for r in all_results[method] if r.task_id == task.task_id)
            row[f"{method}_success_rate"] = round(res.success_rate, 4)
            row[f"{method}_gripper_force"] = res.rag_params_used.get("gripper_force")
        comparison.append(row)

    out_json = Path(output_dir) / "simulation_ablation_retrieval.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(comparison, f, ensure_ascii=False, indent=2)

    labels = {"rag": "单query", "rag_multi": "多query", "rag_random": "随机文档", "fixed": "固定25N"}
    header = f"{'任务':<20} {'理想(N)':<10} " + " ".join(f"{labels[m]}" for m in methods)
    lines = [
        "检索策略消融：单 query vs 多 query vs 随机文档 vs 固定基线",
        "=" * 72,
        f"n_trials_per_task={n_trials_per_task}, seed={seed}",
        "",
        header,
        "-" * 72,
    ]
    for task in BENCHMARK_TASKS:
        ideal = f"{task.ideal_gripper_force[0]}-{task.ideal_gripper_force[1]}"
        rates = [f"{next(r for r in all_results[m] if r.task_id == task.task_id).success_rate:.2%}" for m in methods]
        lines.append(f"{_task_label(task):<20} {ideal:<10} " + " ".join(rates))
    lines.append("")
    lines.append(f"JSON: {out_json}")
    table_path = Path(output_dir) / "simulation_ablation_retrieval.txt"
    Path(table_path).write_text("\n".join(lines), encoding="utf-8")
    return all_results


def main():
    import argparse
    parser = argparse.ArgumentParser(description="RAG 驱动机械臂仿真 Benchmark，支持 RAG vs 基线对比")
    parser.add_argument("--data_path", default="mechanical_data.txt")
    parser.add_argument("--n_trials", type=int, default=10)
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--output", default="simulation_benchmark_result.json")
    parser.add_argument("--method", choices=("rag", "rag_multi", "rag_random", "rag_llm", "direct_llm", "rag_learned", "rag_feedback", "fixed", "random"), default="rag",
                        help="参数来源: rag=规则解析, direct_llm=直接LLM, rag_feedback=RAG+失败反馈调整并重试, rag_llm=RAG+LLM结构化JSON, rag_learned=轻量MLP, rag_multi=多query, rag_random=随机文档, fixed=固定25N, random=随机力")
    parser.add_argument("--max_feedback_retries", type=int, default=1,
                        help="仅 method=rag_feedback 时有效：每次 trial 失败后最多重试次数（默认 1）")
    parser.add_argument("--compare", action="store_true",
                        help="运行 RAG vs 基线对比，输出 simulation_comparison_rag_vs_baseline.json/.txt")
    parser.add_argument("--compare_multi_seed", action="store_true",
                        help="多 seed 对比，输出 mean±std 到 simulation_comparison_multi_seed.json/.txt")
    parser.add_argument("--report_multi_seed", action="store_true",
                        help="对单一 method 输出多 seed 汇总到 --output 指定文件，适合正式汇报")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44],
                        help="与 --compare_multi_seed 同用，指定 seed 列表，默认 42 43 44")
    parser.add_argument("--multi_seed_methods", nargs="+", default=["rag", "direct_llm", "fixed"],
                        help="与 --compare_multi_seed 同用，指定参与统计的 method 列表，默认 rag direct_llm fixed")
    parser.add_argument("--ablation_retrieval", action="store_true",
                        help="检索消融：rag / rag_multi / rag_random / fixed，输出 simulation_ablation_retrieval.json/.txt")
    parser.add_argument("--compare_learned", action="store_true",
                        help="RAG(规则) vs RAG+轻量学习(MLP+不确定性) vs 固定，输出到 simulation_comparison_rag_vs_baseline")
    parser.add_argument("--compare_llm", action="store_true",
                        help="RAG(规则) vs RAG+LLM(结构化JSON) vs 固定基线，输出到 simulation_comparison_rag_vs_baseline")
    parser.add_argument("--compare_direct_llm", action="store_true",
                        help="RAG(规则) vs 直接LLM vs 固定基线，输出到 simulation_comparison_rag_vs_baseline")
    parser.add_argument("--compare_feedback", action="store_true",
                        help="RAG(规则) vs RAG+反馈环节(失败调整重试) vs 固定基线，输出到 simulation_comparison_rag_vs_baseline")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.ablation_retrieval:
        print("运行检索策略消融（单 query vs 多 query vs 随机文档 vs 固定）...")
        run_retrieval_ablation(
            data_path=args.data_path,
            n_trials_per_task=args.n_trials,
            seed=args.seed,
            output_dir=".",
        )
        table_path = Path("simulation_ablation_retrieval.txt")
        print("\n" + table_path.read_text(encoding="utf-8"))
        print("消融结果已保存: simulation_ablation_retrieval.json, .txt")
        return

    if args.compare_multi_seed:
        print("运行 RAG vs 无 RAG 基线对比（多 seed mean±std）...")
        run_benchmark_comparison_multi_seed(
            data_path=args.data_path,
            n_trials_per_task=args.n_trials,
            seeds=args.seeds,
            output_dir=".",
            methods=args.multi_seed_methods,
        )
        table_path = Path("simulation_comparison_multi_seed.txt")
        print("\n" + table_path.read_text(encoding="utf-8"))
        print("多 seed 对比已保存: simulation_comparison_multi_seed.json, .txt")
        return

    if args.report_multi_seed:
        print(f"运行单方法多 seed 汇总（method={args.method}）...")
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
        print(f"多 seed 汇总已保存: {args.output}")
        return

    if args.compare_learned:
        print("运行 RAG(规则) vs RAG+轻量学习(MLP+不确定性) vs 固定 对比...")
        run_benchmark_comparison(
            data_path=args.data_path,
            n_trials_per_task=args.n_trials,
            seed=args.seed,
            output_dir=".",
            methods=["rag", "rag_learned", "fixed"],
        )
        table_path = Path("simulation_comparison_rag_vs_baseline.txt")
        print("\n" + table_path.read_text(encoding="utf-8"))
        print("对比已保存（含 rag / rag_learned / fixed）；rag_learned 结果中含 uncertainty_std 见 JSON）")
        return

    if args.compare_llm:
        print("运行 RAG(规则解析) vs RAG+LLM(结构化输出) vs 固定基线 对比...")
        run_benchmark_comparison(
            data_path=args.data_path,
            n_trials_per_task=args.n_trials,
            seed=args.seed,
            output_dir=".",
            methods=["rag", "rag_llm", "fixed"],
        )
        table_path = Path("simulation_comparison_rag_vs_baseline.txt")
        print("\n" + table_path.read_text(encoding="utf-8"))
        print("对比已保存: simulation_comparison_rag_vs_baseline.json/.txt（含 rag / rag_llm / fixed 三列）")
        return

    if args.compare_direct_llm:
        print("运行 RAG(规则解析) vs 直接LLM vs 固定基线 对比...")
        run_benchmark_comparison(
            data_path=args.data_path,
            n_trials_per_task=args.n_trials,
            seed=args.seed,
            output_dir=".",
            methods=["rag", "direct_llm", "fixed"],
        )
        table_path = Path("simulation_comparison_rag_vs_baseline.txt")
        print("\n" + table_path.read_text(encoding="utf-8"))
        print("对比已保存: simulation_comparison_rag_vs_baseline.json/.txt（含 rag / direct_llm / fixed 三列）")
        return

    if args.compare_feedback:
        print("运行 RAG(规则) vs RAG+反馈环节(失败调整重试) vs 固定基线 对比...")
        run_benchmark_comparison(
            data_path=args.data_path,
            n_trials_per_task=args.n_trials,
            seed=args.seed,
            output_dir=".",
            methods=["rag", "rag_feedback", "fixed"],
        )
        table_path = Path("simulation_comparison_rag_vs_baseline.txt")
        print("\n" + table_path.read_text(encoding="utf-8"))
        print("对比已保存: simulation_comparison_rag_vs_baseline.json/.txt（含 rag / rag_feedback / fixed）")
        return

    if args.compare:
        print("运行 RAG vs 无 RAG 基线对比实验...")
        all_results = run_benchmark_comparison(
            data_path=args.data_path,
            n_trials_per_task=args.n_trials,
            seed=args.seed,
            output_dir=".",
            methods=["rag", "fixed"],
        )
        table_path = Path("simulation_comparison_rag_vs_baseline.txt")
        print("\n" + table_path.read_text(encoding="utf-8"))
        print("对比结果已保存: simulation_comparison_rag_vs_baseline.json, .txt")
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

    print("=" * 60)
    print(f"机械臂仿真 Benchmark 结果 (method={args.method})")
    print("（参考 GRASPA / FMB / REPLAB 评测协议）")
    if not HAS_MUJOCO:
        print("【降级模式】未安装 mujoco，结果基于 RAG 成功模型（无物理仿真）")
    print("=" * 60)
    for r in results:
        print(f"\n任务: {r.task_description}")
        print(f"  Success Rate: {r.success_count}/{r.n_trials} = {r.success_rate:.2%}")
        print(f"  95%区间: [{r.ci95_low:.2%}, {r.ci95_high:.2%}]")
        print(f"  平均模拟时间: {r.avg_time:.3f}s")
        print(f"  平均步数: {r.avg_steps:.1f}")
        print(f"  参数: {r.rag_params_used}")
    print("\n结果已保存至:", args.output)


if __name__ == "__main__":
    main()
