#!/bin/bash
# MechanicalRag 主运行脚本
# 在重构后的目录结构下，统一输出到 outputs/ 目录

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
QA_OUTPUT_DIR="${PROJECT_ROOT}/outputs/current"
SIM_OUTPUT_DIR="${PROJECT_ROOT}/outputs/current_observer_step_replan"
SIM_VIS_DIR="${SIM_OUTPUT_DIR}/visualizations"

cd "${PROJECT_ROOT}"

echo "=== 激活虚拟环境 ==="
source "${PROJECT_ROOT}/venv/bin/activate"

mkdir -p "${QA_OUTPUT_DIR}" "${SIM_OUTPUT_DIR}" "${SIM_VIS_DIR}"

echo ""
echo "=== 1. 环境测试 ==="
python scripts/env_test.py

echo ""
echo "=== 2. QA 结构化评测（含 core/paraphrase/robustness/compositional/procedure/holdout/counterfactual/ood） ==="
python -m qa.evaluation --data_path mechanical_data.txt --case_set full --output_dir "${QA_OUTPUT_DIR}"

echo ""
echo "=== 3. Rule-heavy QA 烟雾测试 ==="
python -m qa.problem_solving --data_path mechanical_data.txt >/tmp/mechanicalrag_rule_heavy.log || echo "Rule-heavy QA 烟雾测试失败"

echo ""
echo "=== 4. 仿真 Benchmark 多 seed 汇总 ==="
python -m simulation.benchmark \
  --report_multi_seed \
  --method rag_feedback \
  --n_trials 20 \
  --seeds 42 43 44 \
  --output "${SIM_OUTPUT_DIR}/simulation_benchmark_result.json" || echo "仿真 benchmark 汇总失败"

echo ""
echo "=== 5. 仿真基线对比（含独立 learned baseline 与 task_heuristic） ==="
python -m simulation.benchmark \
  --compare_feedback \
  --n_trials 20 \
  --output_dir "${SIM_OUTPUT_DIR}" || echo "仿真基线对比失败"

echo ""
echo "=== 6. Simulation evidence ablation（RAG vs generic-only） ==="
python -m simulation.benchmark \
  --compare_evidence_ablation \
  --n_trials 20 \
  --seeds 42 43 44 \
  --output_dir "${SIM_OUTPUT_DIR}" || echo "simulation evidence ablation 失败"

echo ""
echo "=== 7. Simulation motion ablation（RAG vs no-motion-rules） ==="
python -m simulation.benchmark \
  --compare_motion_ablation \
  --n_trials 20 \
  --seeds 42 43 44 \
  --output_dir "${SIM_OUTPUT_DIR}" || echo "simulation motion ablation 失败"

echo ""
echo "=== 8. 多 seed 稳定性对比 ==="
python -m simulation.benchmark \
  --compare_multi_seed \
  --n_trials 20 \
  --seeds 42 43 44 \
  --multi_seed_methods rag rag_feedback task_heuristic fixed \
  --output_dir "${SIM_OUTPUT_DIR}" || echo "多 seed 对比失败"

echo ""
echo "=== 9. 结果可视化 ==="
python reporting/visualize_results.py \
  --qa_json "${QA_OUTPUT_DIR}/qa_evaluation_detail.json" \
  --sim_json "${SIM_OUTPUT_DIR}/simulation_comparison_rag_vs_baseline.json" \
  --sim_multi_seed_json "${SIM_OUTPUT_DIR}/simulation_comparison_multi_seed.json" \
  --output_dir "${SIM_VIS_DIR}" || echo "可视化生成失败"

echo ""
echo "=== 10. 展示摘要 ==="
python reporting/generate_showcase.py \
  --qa_json "${QA_OUTPUT_DIR}/qa_evaluation_detail.json" \
  --sim_json "${SIM_OUTPUT_DIR}/simulation_comparison_rag_vs_baseline.json" \
  --sim_multi_seed_json "${SIM_OUTPUT_DIR}/simulation_comparison_multi_seed.json" \
  --sim_benchmark_json "${SIM_OUTPUT_DIR}/simulation_benchmark_result.json" \
  --output "${SIM_OUTPUT_DIR}/showcase_summary.txt" || echo "展示摘要生成失败"

echo ""
echo "=== 全部完成！生成的关键文件： ==="
ls -la \
  "${QA_OUTPUT_DIR}/rag_evaluation_report.txt" \
  "${QA_OUTPUT_DIR}/rag_problems.txt" \
  "${QA_OUTPUT_DIR}/direct_llm_result.txt" \
  "${QA_OUTPUT_DIR}/problem_solving_result.txt" \
  "${QA_OUTPUT_DIR}/qa_evaluation_detail.json" \
  "${SIM_OUTPUT_DIR}/simulation_benchmark_result.json" \
  "${SIM_OUTPUT_DIR}/simulation_comparison_rag_vs_baseline.json" \
  "${SIM_OUTPUT_DIR}/simulation_comparison_multi_seed.json" \
  "${SIM_OUTPUT_DIR}/simulation_evidence_ablation.json" \
  "${SIM_OUTPUT_DIR}/simulation_evidence_dependence_summary.txt" \
  "${SIM_OUTPUT_DIR}/simulation_motion_ablation.json" \
  "${SIM_OUTPUT_DIR}/simulation_motion_dependence_summary.txt" \
  "${SIM_OUTPUT_DIR}/simulation_split_summary.txt" \
  "${SIM_OUTPUT_DIR}/simulation_challenge_summary.txt" \
  "${SIM_OUTPUT_DIR}/showcase_summary.txt" 2>/dev/null || echo "部分文件可能尚未生成"
