#!/bin/bash
# Mechanical RAG 项目运行脚本
# 按 README 顺序依次运行所有步骤

set -e
cd "$(dirname "$0")"

echo "=== 激活虚拟环境 ==="
source venv/bin/activate

echo ""
echo "=== 1. 环境测试 ==="
python env_test.py

echo ""
echo "=== 2. RAG 评测 ==="
python rag_evaluation.py --data_path mechanical_data.txt

echo ""
echo "=== 3. 问题解决 ==="
python problem_solving.py --data_path mechanical_data.txt

echo ""
echo "=== 4. 仿真 Benchmark 多 seed 汇总 ==="
python -m simulation.benchmark --report_multi_seed --method rag --n_trials 20 --seeds 42 43 44 --output simulation_benchmark_result.json || echo "仿真 benchmark 汇总失败"

echo ""
echo "=== 5. 仿真基线对比 ==="
python -m simulation.benchmark --compare_direct_llm --n_trials 20 || echo "仿真基线对比失败"

echo ""
echo "=== 6. 多 seed 稳定性对比 ==="
python -m simulation.benchmark --compare_multi_seed --n_trials 20 --seeds 42 43 44 --multi_seed_methods rag direct_llm fixed || echo "多 seed 对比失败"

echo ""
echo "=== 7. 结果可视化 ==="
python visualize_results.py --qa_json qa_evaluation_detail.json --sim_json simulation_comparison_rag_vs_baseline.json --sim_multi_seed_json simulation_comparison_multi_seed.json --output_dir visualizations || echo "可视化生成失败"

echo ""
echo "=== 8. 展示摘要 ==="
python generate_showcase.py --qa_json qa_evaluation_detail.json --sim_json simulation_comparison_rag_vs_baseline.json --sim_multi_seed_json simulation_comparison_multi_seed.json --sim_benchmark_json simulation_benchmark_result.json --output showcase_summary.txt || echo "展示摘要生成失败"

echo ""
echo "=== 全部完成！生成的文件： ==="
ls -la rag_evaluation_report.txt rag_problems.txt direct_llm_result.txt problem_solving_result.txt qa_evaluation_detail.json simulation_benchmark_result.json simulation_comparison_rag_vs_baseline.json simulation_comparison_multi_seed.json showcase_summary.txt 2>/dev/null || echo "部分文件可能尚未生成"
