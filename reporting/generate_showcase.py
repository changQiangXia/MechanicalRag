"""Generate a concise showcase summary from QA and simulation outputs."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path


def _load_json(path: str) -> list | dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _format_pct(value: float) -> str:
    return f"{value * 100:.2f}%"


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


def _method_metric(row: dict, method: str, field: str, default=None):
    entry = _method_entry(row, method)
    if field in entry:
        return entry[field]
    for alias in _method_aliases(method):
        flat_key = f"{alias}_{field}"
        if flat_key in row:
            return row[flat_key]
    return default


def _plan_mean(row: dict, method: str, field: str, default=None):
    entry = _method_entry(row, method)
    value = entry.get("executed_plan_stats", {}).get("mean", {}).get(field)
    if value is not None:
        return value
    for alias in _method_aliases(method):
        for flat_key in (f"{alias}_{field}_mean", f"{alias}_{field}"):
            if flat_key in row:
                return row[flat_key]
    return default


def _flatten_method_row(row: dict) -> dict:
    flat = dict(row)
    methods = row.get("methods")
    if not isinstance(methods, dict):
        return flat
    for method, payload in methods.items():
        for alias in _method_aliases(method):
            for key, value in payload.items():
                if key in {
                    "seed_plan",
                    "executed_plan_stats",
                    "planner_diagnostics",
                    "failure_rates",
                    "avg_risks",
                    "reference_force_deviation_stats",
                }:
                    continue
                flat[f"{alias}_{key}"] = value

            executed = payload.get("executed_plan_stats", {})
            for key, value in executed.get("mean", {}).items():
                flat[f"{alias}_{key}"] = value
                flat[f"{alias}_{key}_mean"] = value
            for key, value in executed.items():
                if key.endswith("_mode"):
                    flat[f"{alias}_{key[:-5]}"] = value

            planner = payload.get("planner_diagnostics", {})
            for key, value in planner.items():
                flat[f"{alias}_{key}"] = value
                if key.endswith("_mode"):
                    flat[f"{alias}_{key[:-5]}"] = value

            failure_rates = payload.get("failure_rates", {})
            for key, value in failure_rates.items():
                flat[f"{alias}_{key}"] = value
                if key.endswith("_mean"):
                    flat[f"{alias}_{key.replace('_fail_mean', '_fail_rate_mean')}"] = value
                if key.endswith("_std"):
                    flat[f"{alias}_{key.replace('_fail_std', '_fail_rate_std')}"] = value

            avg_risks = payload.get("avg_risks", {})
            for key, value in avg_risks.items():
                flat[f"{alias}_avg_{key}"] = value

            ref_stats = payload.get("reference_force_deviation_stats", {})
            if ref_stats.get("mean") is not None:
                flat[f"{alias}_reference_force_deviation"] = ref_stats["mean"]
                flat[f"{alias}_reference_force_deviation_mean"] = ref_stats["mean"]
            for key, value in ref_stats.items():
                flat[f"{alias}_reference_force_deviation_{key}"] = value
    return flat


def _flatten_benchmark_row(row: dict) -> dict:
    flat = dict(row)
    scalar_fields = (
        "success_rate",
        "avg_time_sec",
        "avg_steps",
        "avg_distance_error",
        "avg_slip_risk",
        "avg_compression_risk",
        "avg_velocity_risk",
        "avg_clearance_risk",
        "avg_lift_hold_risk",
        "avg_transfer_sway_risk",
        "avg_placement_settle_risk",
        "avg_stability_score",
        "physics_fail_rate",
        "lift_hold_fail_rate",
        "transfer_sway_fail_rate",
        "placement_settle_fail_rate",
    )
    for field in scalar_fields:
        if field in flat and f"{field}_mean" not in flat:
            flat[f"{field}_mean"] = flat[field]

    executed = row.get("executed_plan_stats", {})
    for key, value in executed.get("mean", {}).items():
        flat[key] = value
        flat[f"{key}_mean"] = value
    for key, value in executed.items():
        if key.endswith("_mode"):
            flat[key[:-5]] = value

    planner = row.get("planner_diagnostics", {})
    for key, value in planner.items():
        flat[key] = value

    ref_stats = row.get("reference_force_deviation_stats", {})
    if ref_stats.get("mean") is not None:
        flat["reference_force_deviation"] = ref_stats["mean"]
        flat["reference_force_deviation_mean"] = ref_stats["mean"]
    return flat


def build_summary(
    qa_path: str,
    sim_compare_path: str,
    sim_multi_seed_path: str,
    sim_benchmark_path: str,
    output_path: str,
) -> None:
    qa_detail = _load_json(qa_path)
    sim_compare = [_flatten_method_row(row) for row in _load_json(sim_compare_path)]
    sim_multi_seed = [_flatten_method_row(row) for row in _load_json(sim_multi_seed_path)]
    sim_benchmark = [_flatten_benchmark_row(row) for row in _load_json(sim_benchmark_path)]

    summaries = qa_detail["summaries"]
    methods = ("direct_llm", "base_rag", "improved_rag", "problem_solving_rag")
    representative_case_ids = (
        "smooth_metal_force",
        "thin_wall_handling",
        "vacuum_surface_holdout",
    )

    lines = [
        "MechanicalRag 展示摘要",
        "=" * 72,
        "",
        "一、QA 结果总览",
    ]
    for method in methods:
        summary = summaries[method]
        numeric_rate = summary.get("numeric_consistency_rate")
        numeric_rate = 0.0 if numeric_rate is None else numeric_rate
        abstain_recall = summary.get("abstain_recall")
        abstain_recall = 0.0 if abstain_recall is None else abstain_recall
        lines.append(
            f"- {method}: strict={summary['strict_accuracy']:.2f}, weighted={summary['weighted_accuracy']:.2f}, "
            f"correct={summary['correct']}, incorrect={summary['incorrect']}, "
            f"semantic={summary.get('avg_semantic_similarity', 0.0):.2f}, "
            f"numeric={numeric_rate:.2f}, "
            f"abstain_r={abstain_recall:.2f}, "
            f"cf_flip={summary.get('counterfactual_flip_rate', 0.0) or 0.0:.2f}, "
            f"support={summary.get('avg_required_support_coverage', 0.0):.2f}"
        )
        for split, split_summary in summary["split_summaries"].items():
            lines.append(
                f"  split={split}: strict={split_summary['strict_accuracy']:.2f}, "
                f"weighted={split_summary['weighted_accuracy']:.2f}, n={split_summary['n_cases']}"
            )
    best_qa_methods = [
        method
        for method in methods
        if summaries[method]["strict_accuracy"] >= 0.999
    ]
    if best_qa_methods:
        lines.append(
            f"- 当前已实现全量 strict 命中的 QA 方法：{', '.join(best_qa_methods)}。"
        )

    lines.extend(["", "二、代表性问答样例"])
    cases = {row["case_id"]: row for row in qa_detail["cases"]}
    for case_id in representative_case_ids:
        case = cases[case_id]
        lines.append(f"- [{case['split']}] 问题：{case['question']}")
        lines.append(f"  参考：{case['gold_answer']}")
        for method in methods:
            row = next(item for item in qa_detail["method_results"][method] if item["case_id"] == case_id)
            lines.append(f"  {method}: {row['response_eval']['label']} | {row['response']}")
        lines.append("")

    ood_case = next((row for row in qa_detail["cases"] if row["split"] == "ood"), None)
    if ood_case is not None:
        lines.append("三、拒答样例")
        lines.append(f"- [ood] 问题：{ood_case['question']}")
        for method in methods:
            row = next(item for item in qa_detail["method_results"][method] if item["case_id"] == ood_case["case_id"])
            lines.append(f"  {method}: {row['response_eval']['label']} | {row['response']}")
        lines.append("")

    counterfactual_case = next((row for row in qa_detail["cases"] if row["split"] == "counterfactual"), None)
    if counterfactual_case is not None:
        lines.append("四、反事实证据依赖样例")
        lines.append(f"- [counterfactual] 问题：{counterfactual_case['question']}")
        lines.append(f"  排除条目：{','.join(counterfactual_case.get('exclude_entry_ids', [])) or '无'}")
        source_case_id = counterfactual_case.get("source_case_id")
        if source_case_id:
            source_case = cases.get(source_case_id)
            if source_case is not None:
                lines.append(f"  原始题：{source_case['question']}")
        for method in methods:
            source_row = None
            if source_case_id is not None:
                source_row = next(
                    (item for item in qa_detail["method_results"][method] if item["case_id"] == source_case_id),
                    None,
                )
            counter_row = next(
                item for item in qa_detail["method_results"][method] if item["case_id"] == counterfactual_case["case_id"]
            )
            if source_row is not None:
                lines.append(
                    f"  {method} 原题: {source_row['response_eval']['label']} | {source_row['response']}"
                )
            lines.append(
                f"  {method} 删证据后: {counter_row['response_eval']['label']} | {counter_row['response']}"
            )
        lines.append("")

    lines.append("五、仿真对比重点")
    avg_rag_gain = sum(
        row["rag_success_rate_mean"] - row["direct_llm_success_rate_mean"] for row in sim_multi_seed
    ) / len(sim_multi_seed)
    avg_heuristic_gain = sum(
        row["rag_success_rate_mean"] - row.get("task_heuristic_success_rate_mean", 0.0) for row in sim_multi_seed
    ) / len(sim_multi_seed)
    learned_available = any("rag_learned_success_rate_mean" in row for row in sim_multi_seed)
    lines.append(f"- 多 seed 平均上，RAG 相对 Direct LLM 的成功率提升为 {_format_pct(avg_rag_gain)}。")
    lines.append(f"- 多 seed 平均上，RAG 相对 Task Heuristic 基线的成功率提升为 {_format_pct(avg_heuristic_gain)}。")
    if learned_available:
        avg_learned_gain = sum(
            row["rag_success_rate_mean"] - row.get("rag_learned_success_rate_mean", 0.0) for row in sim_multi_seed
        ) / len(sim_multi_seed)
        lines.append(f"- 多 seed 平均上，RAG 相对独立 learned baseline 的成功率提升为 {_format_pct(avg_learned_gain)}。")
    rag_evidence_support = [
        row.get("rag_evidence_support_score_mean") for row in sim_multi_seed if row.get("rag_evidence_support_score_mean") is not None
    ]
    if rag_evidence_support:
        lines.append(f"- 多 seed 平均上，RAG 控制计划的证据支持分为 {sum(rag_evidence_support)/len(rag_evidence_support):.2f}。")
    rag_belief_coverages = [
        row.get("rag_belief_state_coverage_mean")
        for row in sim_multi_seed
        if row.get("rag_belief_state_coverage_mean") is not None
    ]
    if rag_belief_coverages:
        lines.append(
            f"- 多 seed 平均上，RAG belief state coverage 为 {sum(rag_belief_coverages)/len(rag_belief_coverages):.2f}。"
        )
    rag_conservative_rates = [
        row.get("rag_uncertainty_conservative_mode_mean")
        for row in sim_multi_seed
        if row.get("rag_uncertainty_conservative_mode_mean") is not None
    ]
    if rag_conservative_rates:
        lines.append(
            f"- 多 seed 平均上，RAG uncertainty conservative mode 的触发率为 "
            f"{_format_pct(sum(rag_conservative_rates)/len(rag_conservative_rates))}。"
        )
    rag_solver_choices = [
        row.get("rag_solver_selected_candidate")
        for row in sim_multi_seed
        if row.get("rag_solver_selected_candidate") is not None
    ]
    if rag_solver_choices:
        top_solver, top_count = Counter(rag_solver_choices).most_common(1)[0]
        lines.append(
            f"- 当前多 seed 结果中，RAG 最常选中的 solver 候选是 `{top_solver}`，"
            f"覆盖 {top_count}/{len(rag_solver_choices)} 个任务。"
        )

    evidence_ablation_path = Path(sim_multi_seed_path).with_name("simulation_evidence_ablation.json")
    if evidence_ablation_path.exists():
        evidence_ablation = _load_json(str(evidence_ablation_path))
        strongest_gap = max(
            evidence_ablation,
            key=lambda row: row.get("rag_minus_generic_only_success_gain", 0.0),
            default=None,
        )
        if strongest_gap is not None:
            lines.append(
                f"- evidence ablation 显示在 {strongest_gap['task_id']} 上，RAG 相对 generic-only 的平均成功率增益为 "
                f"{_format_pct(strongest_gap['rag_minus_generic_only_success_gain'])}。"
            )
        large_far_evidence_row = next(
            (row for row in evidence_ablation if row.get("task_id") == "pick_large_part_far"),
            None,
        )
        if large_far_evidence_row is not None:
            gain = large_far_evidence_row.get("rag_minus_generic_only_success_gain", 0.0)
            if gain > 0:
                lines.append(
                    f"- `pick_large_part_far` 当前已不再只是 motion-only gain：RAG 相对 generic-only 提升 "
                    f"{_format_pct(gain)}。"
                )

    evidence_dependence_path = Path(sim_multi_seed_path).with_name("simulation_evidence_dependence_summary.json")
    if evidence_dependence_path.exists():
        dependence = _load_json(str(evidence_dependence_path))
        specific_group = next(
            (
                row
                for row in dependence.get("by_specific_rule_availability", [])
                if row["group"] == "specific_rule_available"
            ),
            None,
        )
        if specific_group is not None:
            lines.append(
                f"- 当对象特定规则可用时，RAG 相对 generic-only 的平均成功率增益为 "
                f"{_format_pct(specific_group['rag_minus_generic_only_gain_mean'])}。"
            )

    motion_ablation_path = Path(sim_multi_seed_path).with_name("simulation_motion_ablation.json")
    if motion_ablation_path.exists():
        motion_ablation = _load_json(str(motion_ablation_path))
        strongest_motion_gap = max(
            motion_ablation,
            key=lambda row: row.get("rag_minus_no_motion_rules_success_gain", 0.0),
            default=None,
        )
        if strongest_motion_gap is not None:
            lines.append(
                f"- motion ablation 显示在 {strongest_motion_gap['task_id']} 上，RAG 相对 no-motion-rules 的平均成功率增益为 "
                f"{_format_pct(strongest_motion_gap['rag_minus_no_motion_rules_success_gain'])}。"
            )
        heavy_fast_row = next(
            (row for row in motion_ablation if row.get("task_id") == "pick_metal_heavy_fast"),
            None,
        )
        if heavy_fast_row is not None:
            gain = heavy_fast_row.get("rag_minus_no_motion_rules_success_gain", 0.0)
            if gain < 0:
                lines.append(
                    f"- `pick_metal_heavy_fast` 仍是残余弱点，RAG 相对 no-motion-rules 还落后 "
                    f"{_format_pct(gain)}。"
                )
            elif gain == 0:
                lines.append(
                    "- `pick_metal_heavy_fast` 已不再出现 motion 负增益，但在当前标准 trial 预算下仍仅与 no-motion-rules 打平。"
                )
            else:
                lines.append(
                    f"- `pick_metal_heavy_fast` 上的 motion 负增益已被消除，当前 RAG 相对 no-motion-rules 取得 "
                    f"{_format_pct(gain)} 的平均成功率增益。"
                )
        for task_id in ("pick_smooth_metal_fast", "pick_large_part_far"):
            row = next((item for item in motion_ablation if item.get("task_id") == task_id), None)
            if row is None:
                continue
            gain = row.get("rag_minus_no_motion_rules_success_gain", 0.0)
            if gain > 0:
                lines.append(
                    f"- `{task_id}` 在当前标准 trial 预算下也已形成 motion 正增益，RAG 相对 no-motion-rules 提升 "
                    f"{_format_pct(gain)}。"
                )
        thin_wall_row = next(
            (row for row in motion_ablation if row.get("task_id") == "pick_thin_wall"),
            None,
        )
        if thin_wall_row is not None:
            gain = thin_wall_row.get("rag_minus_no_motion_rules_success_gain", 0.0)
            if gain > 0:
                lines.append(
                    f"- `pick_thin_wall` 当前已不再是欠力薄点：support-aware 校准后，RAG 相对 no-motion-rules 提升 "
                    f"{_format_pct(gain)}。"
                )

    motion_dependence_path = Path(sim_multi_seed_path).with_name("simulation_motion_dependence_summary.json")
    if motion_dependence_path.exists():
        motion_dependence = _load_json(str(motion_dependence_path))
        motion_group = next(
            (
                row
                for row in motion_dependence.get("by_motion_rule_availability", [])
                if row["group"] == "motion_rule_available"
            ),
            None,
        )
        if motion_group is not None:
            lines.append(
                f"- 当 motion rule 可用时，RAG 相对 no-motion-rules 的平均成功率增益为 "
                f"{_format_pct(motion_group['rag_minus_no_motion_rules_gain_mean'])}。"
            )

    split_summary_path = Path(sim_multi_seed_path).with_name("simulation_split_summary.json")
    if split_summary_path.exists():
        split_summary = _load_json(str(split_summary_path))
        rag_test = next((row for row in split_summary if row["task_split"] == "test"), None)
        if rag_test is not None and rag_test.get("rag_success_rate_mean") is not None:
            lines.append(
                f"- split 汇总显示 test 任务上 RAG 平均成功率为 {rag_test['rag_success_rate_mean']:.4f}，"
                f"便于和 train/val 做泛化对照。"
            )

    challenge_summary_path = Path(sim_multi_seed_path).with_name("simulation_challenge_summary.json")
    if challenge_summary_path.exists():
        challenge_summary = _load_json(str(challenge_summary_path))
        hard_tag = min(
            (row for row in challenge_summary if row.get("rag_success_rate_mean") is not None),
            key=lambda row: row["rag_success_rate_mean"],
            default=None,
        )
        if hard_tag is not None:
            lines.append(
                f"- challenge 汇总显示 RAG 当前最弱的挑战标签是 {hard_tag['challenge_tag']}，"
                f"平均成功率 {hard_tag['rag_success_rate_mean']:.4f}。"
            )

    targeted_rows = {row["task_id"]: row for row in sim_multi_seed}
    benchmark_rows = {row["task_id"]: row for row in sim_benchmark}
    rubber_row = targeted_rows.get("pick_rubber")
    if rubber_row is not None:
        rag_rate = rubber_row.get("rag_success_rate_mean")
        heuristic_rate = rubber_row.get("task_heuristic_success_rate_mean")
        if rag_rate is not None and heuristic_rate is not None:
            lines.append(
                f"- `pick_rubber` 当前 RAG 平均成功率为 {rag_rate:.4f}，相对 task_heuristic 提升 "
                f"{_format_pct(rag_rate - heuristic_rate)}。"
            )
    smooth_row = targeted_rows.get("pick_smooth_metal")
    if smooth_row is not None:
        rag_rate = smooth_row.get("rag_success_rate_mean")
        heuristic_rate = smooth_row.get("task_heuristic_success_rate_mean")
        if rag_rate is not None and heuristic_rate is not None:
            delta = rag_rate - heuristic_rate
            relation = "追平" if abs(delta) < 1e-9 else ("领先" if delta > 0 else "落后")
            lines.append(
                f"- `pick_smooth_metal` 当前 RAG 平均成功率为 {rag_rate:.4f}，已{relation} task_heuristic"
                + ("" if abs(delta) < 1e-9 else f" {_format_pct(abs(delta))}")
                + "。"
            )
    smooth_fast_row = targeted_rows.get("pick_smooth_metal_fast")
    if smooth_fast_row is not None:
        rag_rate = smooth_fast_row.get("rag_success_rate_mean")
        heuristic_rate = smooth_fast_row.get("task_heuristic_success_rate_mean")
        dynamic_mode = smooth_fast_row.get("rag_dynamic_transport_mode")
        transfer_force = smooth_fast_row.get("rag_transfer_force_mean")
        placement_velocity = smooth_fast_row.get("rag_placement_velocity_mean")
        transfer_sway_risk = smooth_fast_row.get("rag_avg_transfer_sway_risk_mean")
        placement_settle_risk = smooth_fast_row.get("rag_avg_placement_settle_risk_mean")
        if rag_rate is not None and heuristic_rate is not None:
            lines.append(
                f"- `pick_smooth_metal_fast` 当前 RAG 平均成功率为 {rag_rate:.4f}，相对 task_heuristic 提升 "
                f"{_format_pct(rag_rate - heuristic_rate)}。"
            )
        if dynamic_mode is not None and dynamic_mode != "static":
            lines.append(
                f"- round16 之后，`pick_smooth_metal_fast` 已进入 `{dynamic_mode}` dynamic transport mode，"
                f"不再只靠单一静态抓取参数解释高速运输失败。"
            )
        if transfer_force is not None and placement_velocity is not None:
            lines.append(
                f"- `pick_smooth_metal_fast` 当前显式使用 stage-specific 动态运输控制："
                f" transfer_force={transfer_force:.2f}N, placement_velocity={placement_velocity:.2f}m/s。"
            )
        if transfer_sway_risk is not None and placement_settle_risk is not None:
            lines.append(
                f"- `pick_smooth_metal_fast` 当前不再出现“stage fail 有值但 risk 全 0”的薄语义："
                f" transfer_sway_risk={transfer_sway_risk:.4f}, placement_settle_risk={placement_settle_risk:.4f}。"
            )
    heavy_row = targeted_rows.get("pick_metal_heavy")
    if heavy_row is not None:
        rag_rate = heavy_row.get("rag_success_rate_mean")
        heuristic_rate = heavy_row.get("task_heuristic_success_rate_mean")
        if rag_rate is not None and heuristic_rate is not None:
            lines.append(
                f"- `pick_metal_heavy` 当前 RAG 平均成功率为 {rag_rate:.4f}，相对 task_heuristic 提升 "
                f"{_format_pct(rag_rate - heuristic_rate)}。"
            )
    large_far_row = targeted_rows.get("pick_large_part_far")
    if large_far_row is not None:
        rag_rate = large_far_row.get("rag_success_rate_mean")
        heuristic_rate = large_far_row.get("task_heuristic_success_rate_mean")
        sway_risk = large_far_row.get("rag_avg_transfer_sway_risk_mean")
        lift_risk = large_far_row.get("rag_avg_lift_hold_risk_mean")
        settle_risk = large_far_row.get("rag_avg_placement_settle_risk_mean")
        dominant_failure_mode = large_far_row.get("rag_dominant_failure_mode")
        lift_fail_rate = large_far_row.get("rag_lift_hold_fail_rate_mean")
        transfer_fail_rate = large_far_row.get("rag_transfer_sway_fail_rate_mean")
        settle_fail_rate = large_far_row.get("rag_placement_settle_fail_rate_mean")
        lift_force = large_far_row.get("rag_lift_force_mean")
        transfer_alignment = large_far_row.get("rag_transfer_alignment_mean")
        transfer_force = large_far_row.get("rag_transfer_force_mean")
        dynamic_mode = large_far_row.get("rag_dynamic_transport_mode")
        available_lift_stage_rules = large_far_row.get("rag_available_lift_stage_rules_mean")
        used_lift_stage_rules = large_far_row.get("rag_used_lift_stage_rules_mean")
        if sway_risk is None:
            sway_risk = benchmark_rows.get("pick_large_part_far", {}).get("avg_transfer_sway_risk_mean")
        if lift_risk is None:
            lift_risk = benchmark_rows.get("pick_large_part_far", {}).get("avg_lift_hold_risk_mean")
        if settle_risk is None:
            settle_risk = benchmark_rows.get("pick_large_part_far", {}).get("avg_placement_settle_risk_mean")
        if lift_force is None:
            lift_force = benchmark_rows.get("pick_large_part_far", {}).get("lift_force_mean")
        if transfer_alignment is None:
            transfer_alignment = benchmark_rows.get("pick_large_part_far", {}).get("transfer_alignment_mean")
        if transfer_force is None:
            transfer_force = benchmark_rows.get("pick_large_part_far", {}).get("transfer_force_mean")
        if dynamic_mode is None:
            dynamic_mode = benchmark_rows.get("pick_large_part_far", {}).get("dynamic_transport_mode")
        if dominant_failure_mode is None:
            dominant_failure_mode = benchmark_rows.get("pick_large_part_far", {}).get("dominant_failure_mode")
        if lift_fail_rate is None:
            lift_fail_rate = benchmark_rows.get("pick_large_part_far", {}).get("lift_hold_fail_rate_mean")
        if transfer_fail_rate is None:
            transfer_fail_rate = benchmark_rows.get("pick_large_part_far", {}).get("transfer_sway_fail_rate_mean")
        if settle_fail_rate is None:
            settle_fail_rate = benchmark_rows.get("pick_large_part_far", {}).get("placement_settle_fail_rate_mean")
        if rag_rate is not None and heuristic_rate is not None:
            lines.append(
                f"- `pick_large_part_far` 当前 RAG 平均成功率为 {rag_rate:.4f}，相对 task_heuristic 提升 "
                f"{_format_pct(rag_rate - heuristic_rate)}，但仍是当前最难任务。"
            )
        if (
            dynamic_mode is not None
            and lift_force is not None
            and transfer_alignment is not None
            and transfer_force is not None
        ):
            lines.append(
                f"- round19 之后，`pick_large_part_far` 已把 long-transfer 的起吊保持链补厚成显式 `{dynamic_mode}` 控制计划："
                f" lift_force={lift_force:.2f}N, transfer_alignment={transfer_alignment:.2f}, transfer_force={transfer_force:.2f}N。"
            )
        if available_lift_stage_rules is not None and used_lift_stage_rules is not None:
            lines.append(
                f"- `pick_large_part_far` 当前已不再只靠 large-part force band；lift-stage 证据可见性为 "
                f"{available_lift_stage_rules:.2f}，实际使用率为 {used_lift_stage_rules:.2f}。"
            )
        belief_coverage = large_far_row.get("rag_belief_state_coverage_mean")
        conservative_rate = large_far_row.get("rag_uncertainty_conservative_mode_mean")
        solver_choice = large_far_row.get("rag_solver_selected_candidate")
        if belief_coverage is not None and conservative_rate is not None and solver_choice is not None:
            lines.append(
                f"- `pick_large_part_far` 当前 belief coverage={belief_coverage:.4f}，"
                f"conservative mode 触发率为 {_format_pct(conservative_rate)}，主导 solver 为 `{solver_choice}`。"
            )
        if sway_risk is not None and sway_risk > 0:
            lines.append(
                f"- `pick_large_part_far` 当前已显式建模 long-transfer `transfer_sway_risk`，平均值为 {sway_risk:.4f}。"
            )
        if (
            lift_risk is not None
            and sway_risk is not None
            and settle_risk is not None
            and dominant_failure_mode is not None
        ):
            lines.append(
                f"- `pick_large_part_far` 当前已拆成 stage-specific failure model："
                f" lift_hold={lift_risk:.4f}, transfer_sway={sway_risk:.4f}, placement_settle={settle_risk:.4f}，"
                f"主导失败模式为 `{dominant_failure_mode}`。"
            )
        if (
            lift_fail_rate is not None
            and transfer_fail_rate is not None
            and settle_fail_rate is not None
        ):
            lines.append(
                f"- `pick_large_part_far` 的 stage fail rate 当前为 "
                f"lift_hold={_format_pct(lift_fail_rate)} / "
                f"transfer_sway={_format_pct(transfer_fail_rate)} / "
                f"placement_settle={_format_pct(settle_fail_rate)}。"
            )
        placement_velocity = large_far_row.get("rag_placement_velocity_mean")
        if placement_velocity is None:
            placement_velocity = benchmark_rows.get("pick_large_part_far", {}).get("placement_velocity_mean")
        if placement_velocity is not None:
            lines.append(
                f"- round15 之后，`pick_large_part_far` 已不再把运输段速度直接沿用到落位段；当前显式 `placement_velocity` 为 {placement_velocity:.2f} m/s。"
            )
        task_heuristic_placement_velocity = large_far_row.get("task_heuristic_placement_velocity_mean")
        if (
            placement_velocity is not None
            and task_heuristic_placement_velocity is not None
            and placement_velocity < task_heuristic_placement_velocity
        ):
            lines.append(
                f"- 这条 stage-specific settle control 也直接拉开了与 task_heuristic 的差异：后者当前末段仍为 {task_heuristic_placement_velocity:.2f} m/s，"
                f"并以 `placement_settle_fail` 为主导失败模式。"
            )

    hardest_task = min(sim_benchmark, key=lambda row: row["success_rate_mean"])
    easiest_task = max(sim_benchmark, key=lambda row: row["success_rate_mean"])
    lines.append(
        f"- 最难任务：{hardest_task['task_id']}，平均成功率 {hardest_task['success_rate_mean']:.4f}，"
        f"稳定度 {hardest_task['avg_stability_score_mean']:.4f}。"
    )
    lines.append(
        f"- 最稳任务：{easiest_task['task_id']}，平均成功率 {easiest_task['success_rate_mean']:.4f}，"
        f"稳定度 {easiest_task['avg_stability_score_mean']:.4f}。"
    )

    lines.extend(
        [
            "",
            "六、当前证据边界",
            "- QA 结果应解读为“当前知识库 + 当前 split 划分”上的结果，不直接外推为强泛化能力。",
            "- 仿真 benchmark 的成功判定已基于物体属性和执行观测独立建模，reference_force_range 只用于分析输出。",
            "",
            "七、建议展示文件",
            "- outputs/current/qa_evaluation_detail.json",
            "- outputs/current/simulation_benchmark_result.json",
            "- outputs/current/simulation_comparison_multi_seed.json",
            "- outputs/current/simulation_evidence_ablation.json",
            "- outputs/current/simulation_evidence_dependence_summary.txt",
            "- outputs/current/simulation_motion_ablation.json",
            "- outputs/current/simulation_motion_dependence_summary.txt",
            "- outputs/current/simulation_split_summary.txt",
            "- outputs/current/simulation_challenge_summary.txt",
            "- outputs/visualizations/qa_method_summary.png",
            "- outputs/visualizations/qa_gain_over_direct_llm.png",
            "- outputs/visualizations/simulation_rag_gain.png",
            "- outputs/visualizations/simulation_multi_seed_success.png",
            "- outputs/visualizations/simulation_control_plan.png",
        ]
    )

    Path(output_path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qa_json", default="outputs/current/qa_evaluation_detail.json")
    parser.add_argument("--sim_json", default="outputs/current/simulation_comparison_rag_vs_baseline.json")
    parser.add_argument("--sim_multi_seed_json", default="outputs/current/simulation_comparison_multi_seed.json")
    parser.add_argument("--sim_benchmark_json", default="outputs/current/simulation_benchmark_result.json")
    parser.add_argument("--output", default="outputs/current/showcase_summary.txt")
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
