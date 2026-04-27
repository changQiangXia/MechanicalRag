# MechanicalRag 概览

## 1. 项目目标

MechanicalRag 关注的是机械知识如何同时服务两个场景：

- 机械问答：从知识库中检索证据并生成受约束回答。
- 抓取仿真：从知识库中生成控制参数，并在独立环境中验证这些参数。

当前版本的重点不再是“堆更多模式”，而是把主线收束成一个更清晰的输入输出问题：

> 给定机械知识库与任务描述，系统能否稳定地产出可验证证据和可执行参数 / 回答。

## 2. 当前状态

截至 `2026-04-27 UTC`，当前仓库代码语义对应 `outputs/current_observer_step_replan/` 这次 observer / step-replan 闭环验证；`outputs/current/` 继续保留为 `round19` 历史基线，`outputs/current_round20_sim/` 保留 `round20` placement-stage 实验。

- 最新闭环目录：`outputs/current_observer_step_replan/`
- `round19` 历史基线：`outputs/current/`
- `round20` 实验目录：`outputs/current_round20_sim/`
- 当前仿真主链已经改成 `evidence -> belief -> seed synthesize -> local solve -> observer / step replan`
- `env.py` 现在稳定输出 `observer_trace`；`rag_feedback` 命中风险阈值时会在单次 execution 内留下 `step_replan_trace`
- 当前关键结果：
  - `rag_feedback` 12 任务多 seed平均成功率 `0.8361`
  - 相对 `rag` seed-only 路径，平均提升 `+0.1653`
  - `pick_large_part_far = 0.9167 ± 0.1041`
  - `pick_thin_wall_fast = 0.8833 ± 0.0577`
  - `pick_smooth_metal_fast = 0.8333 ± 0.0764`
- 当前主要问题：`pick_metal_heavy = 0.0000 ± 0.0000`

如果只想快速把握当前状态，应优先查看：

- `README.md`
- `simulation/README.md`
- `outputs/current_observer_step_replan/simulation_benchmark_result.json`
- `outputs/current_observer_step_replan/showcase_summary.txt`

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
  - 当前流程是 `evidence -> belief -> seed synthesize -> local solve -> final plan`
  - 支持 `rag_generic_only` evidence ablation
  - 支持 `rag_no_motion_rules` motion ablation

- `simulation/control_core.py`
  - 显式建模 `ObjectBeliefState`、`TaskConstraintSet`、`UncertaintyProfile` 与 `StageIntent`
  - 负责 belief 直驱的 `belief_constraint_synthesis` seed 和 local solver 选择
  - 当前仍不是完整后验滤波器或优化规划器

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
- `outputs/current_observer_step_replan/`
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
python -m simulation.benchmark --report_multi_seed --method rag_feedback --n_trials 20 --seeds 42 43 44 --output outputs/current_observer_step_replan/simulation_benchmark_result.json
python -m simulation.benchmark --compare_feedback --n_trials 20 --seed 42 --output_dir outputs/current_observer_step_replan
python -m simulation.benchmark --compare_evidence_ablation --n_trials 20 --seeds 42 43 44 --output_dir outputs/current
python -m simulation.benchmark --compare_motion_ablation --n_trials 20 --seeds 42 43 44 --output_dir outputs/current
python -m simulation.benchmark --compare_multi_seed --n_trials 20 --seeds 42 43 44 --multi_seed_methods rag rag_feedback task_heuristic fixed --output_dir outputs/current_observer_step_replan
```

### 4.4 图表与摘要

```bash
python reporting/visualize_results.py --qa_json outputs/current/qa_evaluation_detail.json --sim_json outputs/current_observer_step_replan/simulation_comparison_rag_vs_baseline.json --sim_multi_seed_json outputs/current_observer_step_replan/simulation_comparison_multi_seed.json --output_dir outputs/current_observer_step_replan/visualizations
```

## 5. 关键输出

### 5.1 QA

- `outputs/current/qa_evaluation_detail.json`
- `outputs/current/rag_evaluation_report.txt`
- `outputs/current/problem_solving_result.txt`

### 5.2 Simulation

- `outputs/current_observer_step_replan/simulation_benchmark_result.json`
- `outputs/current_observer_step_replan/simulation_comparison_rag_vs_baseline.json`
- `outputs/current_observer_step_replan/simulation_comparison_multi_seed.json`
- `outputs/current_observer_step_replan/simulation_benchmark_rag_feedback.json`
- `outputs/current_observer_step_replan/showcase_summary.txt`
- `outputs/current_observer_step_replan/visualizations/simulation_belief_diagnostics.png`
- `outputs/current/simulation_benchmark_result.json`
- `outputs/current_round20_sim/simulation_benchmark_result.json`
- `outputs/current/simulation_evidence_ablation.json`
- `outputs/current/simulation_motion_ablation.json`

## 6. 当前限制

- QA 结果是当前知识库和当前 case split 上的结果，不应直接表述为开放域强泛化。
- 仿真仍然是简化抓取问题，不等价于完整机械臂控制栈；但控制器和环境已不再共享同一套真值范围。
- `reference_force_range` 和 `reference_approach_height` 只用于分析输出，不参与成功判定。
- 仿真当前代码语义以 `outputs/current_observer_step_replan/` 为准；`round19/current` 继续保留为历史基线，`round20` 只保留为 placement-stage precision 实验归档。
- 当前 observer 已接到执行主链，step replan 已接到 `rag_feedback` 主路径，但它们仍是轻量执行期估计与局部修正，不是完整后验滤波与 MPC。
- `belief_state` 仍然首先从知识证据长出来，执行期观测主要进入 feedback replan，而不是从头替代 retrieval-belief 前置链。
- `pick_metal_heavy` 目前依然完全失败，是下一轮最明显的剩余缺口。
- simulation evidence ablation 目前只覆盖对象特定 force rule，对速度 / 净空等更细粒度规则的删减实验仍未展开。

## 7. 对比口径

- QA 评测统一在同一知识库、同一 split 定义和同一评分规则下运行，并额外输出 `retrieval`、`response_eval`、`evidence_trace` 以区分“证据命中”和“回答正确”。
- simulation 对比统一使用同一任务集、同一 `n_trials`、同一 seed 预算和同一环境判定逻辑。
- `rag`、`direct_llm`、`fixed`、`rag_feedback` 等方法都不能直接读取 `reference_force_range`；该字段只保留在结果层做分析统计。
- simulation 输出包含 95% CI、多 seed `mean±std`、`train / val / test` split 汇总、`challenge_tags` challenge 汇总、证据支持度 / 冲突统计、距离误差、稳定度、阶段化风险，以及 `rag` 对 `rag_generic_only` / `rag_no_motion_rules` 的双消融。
