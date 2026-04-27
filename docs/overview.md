# MechanicalRag 概览

## 1. 项目目标

MechanicalRag 关注的是机械知识如何同时服务两个场景：

- 机械问答：从知识库中检索证据并生成受约束回答。
- 抓取仿真：从知识库中生成控制参数，并在独立环境中验证这些参数。

当前版本的重点不再是“堆更多模式”，而是把主线收束成一个更清晰的输入输出问题：

> 给定机械知识库与任务描述，系统能否稳定地产出可验证证据和可执行参数 / 回答。

## 2. 当前状态

截至 `2026-04-27 UTC`，当前仓库代码语义对应 `outputs/current_core_thickening/` 这次 control-core thickening 验证；`outputs/current/` 继续保留为 `round19` 历史基线，`outputs/current_round20_sim/` 保留 `round20` placement-stage 实验。

- 最新控制核目录：`outputs/current_core_thickening/`
- `round19` 历史基线：`outputs/current/`
- `round20` 实验目录：`outputs/current_round20_sim/`
- 新控制核新增显式 `belief_state`、`task_constraints`、`uncertainty_profile` 与 solver candidate 选择
- 但这层当前仍是规则派生的符号化中间层，不应被表述成观测后验估计或优化求解
- 当前关键结果：
  - `pick_large_part_far = 0.6500 ± 0.1323`
  - `pick_smooth_metal_fast = 0.8167 ± 0.0289`
  - `pick_metal_heavy_fast = 0.7500 ± 0.0866`
  - 相对 `round19` 历史基线，12 任务多 seed 平均成功率提升 `+0.0014`
- 当前已知回落任务：`pick_smooth_metal = 0.7333`、`pick_metal_heavy = 0.7167`

如果只想快速把握当前状态，应优先查看：

- `README.md`
- `simulation/README.md`
- `outputs/current_core_thickening/simulation_benchmark_result.json`
- `outputs/current_core_thickening/showcase_summary.txt`

## 3. 系统结构

### 3.1 QA 侧

- `qa/dataset.py`
  - 定义 `core / paraphrase / robustness / compositional / procedure / holdout / counterfactual / ood` 八类 QA cases

- `qa/pipeline.py`
  - 统一 QA pipeline
  - 显式包含查询理解、证据选择、约束回答三阶段

- `qa/evaluation.py`
  - 统一输出结构化评测结果
  - 输出 lexical / semantic / numeric / procedure / abstain / counterfactual flip 混合评分统计

### 3.2 仿真侧

- `simulation/rag_controller.py`
  - 基于知识库证据和任务描述生成八参数控制计划
  - 当前流程是 `rule aggregation -> belief bundle -> candidate solve -> final plan`
  - 其中 `belief bundle` 与 `candidate solve` 仍是叠加在旧规则聚合之后的后处理层
  - 支持 `rag_generic_only` evidence ablation
  - 支持 `rag_no_motion_rules` motion ablation

- `simulation/control_core.py`
  - 显式建模 `ObjectBeliefState`、`TaskConstraintSet`、`UncertaintyProfile` 与 `StageIntent`
  - 负责 candidate control plan 打分、solver 选择与结构化诊断输出
  - 当前更接近规则派生中间态和候选重打分适配层，而不是严格的状态估计器或规划求解器

- `simulation/env.py`
  - 基于物体属性推导独立力学窗口
  - 输出执行观测和阶段化风险

- `simulation/feedback.py`
  - 根据 `slip_risk / compression_risk / stability_score` 调整参数
  - 当前还显式使用 `lift_hold_risk / transfer_sway_risk / placement_settle_risk`

- `simulation/runner.py`
  - benchmark 核心执行逻辑
  - 负责 evidence ablation、motion ablation 与证据条件化汇总输出

### 3.3 结果侧

- `reporting/visualize_results.py`
- `reporting/generate_showcase.py`
- `outputs/current_core_thickening/`
- `outputs/current/`
- `outputs/current_round20_sim/`

## 4. 运行入口

### 4.1 环境与模型

```bash
python -m venv venv --system-site-packages
source venv/bin/activate
pip install -r requirements.txt
python scripts/download_models.py
python scripts/env_test.py
```

### 4.2 QA 评测

```bash
python -m qa.evaluation --data_path mechanical_data.txt --case_set full --output_dir outputs/current
```

### 4.3 仿真 benchmark

```bash
python -m simulation.benchmark --report_multi_seed --method rag --n_trials 20 --seeds 42 43 44 --output outputs/current_core_thickening/simulation_benchmark_result.json
python -m simulation.benchmark --compare_direct_llm --n_trials 20 --seed 42 --output_dir outputs/current_core_thickening
python -m simulation.benchmark --compare_evidence_ablation --n_trials 20 --seeds 42 43 44 --output_dir outputs/current
python -m simulation.benchmark --compare_motion_ablation --n_trials 20 --seeds 42 43 44 --output_dir outputs/current
python -m simulation.benchmark --compare_multi_seed --n_trials 20 --seeds 42 43 44 --multi_seed_methods rag rag_feedback task_heuristic direct_llm fixed --output_dir outputs/current_core_thickening
```

### 4.4 图表与摘要

```bash
python reporting/visualize_results.py --qa_json outputs/current/qa_evaluation_detail.json --sim_json outputs/current_core_thickening/simulation_comparison_rag_vs_baseline.json --sim_multi_seed_json outputs/current_core_thickening/simulation_comparison_multi_seed.json --output_dir outputs/current_core_thickening/visualizations
python reporting/generate_showcase.py --qa_json outputs/current/qa_evaluation_detail.json --sim_json outputs/current_core_thickening/simulation_comparison_rag_vs_baseline.json --sim_multi_seed_json outputs/current_core_thickening/simulation_comparison_multi_seed.json --sim_benchmark_json outputs/current_core_thickening/simulation_benchmark_result.json --output outputs/current_core_thickening/showcase_summary.txt
```

## 5. 关键输出

### 5.1 QA

- `outputs/current/qa_evaluation_detail.json`
- `outputs/current/rag_evaluation_report.txt`
- `outputs/current/problem_solving_result.txt`

### 5.2 Simulation

- `outputs/current_core_thickening/simulation_benchmark_result.json`
- `outputs/current_core_thickening/simulation_comparison_rag_vs_baseline.json`
- `outputs/current_core_thickening/simulation_comparison_multi_seed.json`
- `outputs/current_core_thickening/showcase_summary.txt`
- `outputs/current_core_thickening/visualizations/simulation_belief_diagnostics.png`
- `outputs/current/simulation_benchmark_result.json`
- `outputs/current_round20_sim/simulation_benchmark_result.json`
- `outputs/current/simulation_evidence_ablation.json`
- `outputs/current/simulation_motion_ablation.json`

## 6. 当前限制

- QA 结果是当前知识库和当前 case split 上的结果，不应直接表述为开放域强泛化。
- 仿真仍然是简化抓取问题，不等价于完整机械臂控制栈；但控制器和环境已不再共享同一套真值范围。
- `reference_force_range` 和 `reference_approach_height` 只用于分析输出，不参与成功判定。
- 仿真当前代码语义以 `outputs/current_core_thickening/` 为准；`round19/current` 继续保留为历史基线，`round20` 只保留为 placement-stage precision 实验归档。
- control-core thickening 只带来了 `+0.0014` 的平均成功率提升，并没有统一优于 `round19`；`pick_smooth_metal` 与 `pick_metal_heavy` 仍各回落约 5 个百分点。
- `belief_state` 当前主要由任务关键词和规则命中情况派生，`uncertainty_profile` 主要是覆盖率 / 缺口统计，`solver` 主要是少量候选模板重打分。
- simulation evidence ablation 目前只覆盖对象特定 force rule，对速度 / 净空等更细粒度规则的删减实验仍未展开。

## 7. 对比口径

- QA 评测统一在同一知识库、同一 split 定义和同一评分规则下运行，并额外输出 `retrieval`、`response_eval`、`evidence_trace` 以区分“证据命中”和“回答正确”。
- simulation 对比统一使用同一任务集、同一 `n_trials`、同一 seed 预算和同一环境判定逻辑。
- `rag`、`direct_llm`、`fixed`、`rag_feedback` 等方法都不能直接读取 `reference_force_range`；该字段只保留在结果层做分析统计。
- simulation 输出包含 95% CI、多 seed `mean±std`、`train / val / test` split 汇总、`challenge_tags` challenge 汇总、证据支持度 / 冲突统计、距离误差、稳定度、阶段化风险，以及 `rag` 对 `rag_generic_only` / `rag_no_motion_rules` 的双消融。
