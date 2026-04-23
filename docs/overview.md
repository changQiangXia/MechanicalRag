# MechanicalRag 概览

## 1. 项目目标

MechanicalRag 关注的是机械知识如何同时服务两个场景：

- 机械问答：从知识库中检索证据并生成受约束回答。
- 抓取仿真：从知识库中生成控制参数，并在独立环境中验证这些参数。

当前版本的重点不再是“堆更多模式”，而是把主线收束成一个更清晰的输入输出问题：

> 给定机械知识库与任务描述，系统能否稳定地产出可验证证据和可执行参数/回答。

## 2. 当前状态

截至 `2026-04-24 UTC`，当前 authoritative 主线是 `round19`，对应目录为 `outputs/current/`。

- `round19` 是当前有效主线。
- `round20` 是实验目录，对应 `outputs/current_round20_sim/`。
- `round20` 继续处理 `pick_large_part_far` 的 `placement_settle`。该实验的 authoritative benchmark 从 `0.6500 ± 0.1323` 变化到 `0.5833 ± 0.0577`。主线结果保持 `round19`。
- 当前主线关键结果：
  `pick_large_part_far = 0.6500 ± 0.1323`，主导失败模式为 `placement_settle_fail`
  `pick_smooth_metal_fast = 0.8167 ± 0.0289`
- 当前代码与文档口径已经对齐到 `round19`。`placement_precision` 只存在于 `round20` 实验输出。

如果只想快速把握当前有效结论，应优先查看：

- `README.md`
- `simulation/README.md`
- `outputs/current/simulation_benchmark_result.json`
- `outputs/current/showcase_summary.txt`

## 3. 系统结构

### 3.1 QA 侧

- `qa/dataset.py`
  - 定义 `core / paraphrase / robustness / compositional / procedure / holdout / counterfactual / ood` 八类 QA cases

- `qa/pipeline.py`
  - 统一 QA pipeline
  - 显式包含查询理解、证据选择、约束回答三阶段
  - 支持按 `entry_id` 排除证据，并做概念级证据充分性检查
  - 已修正诊断题误分类、泛概念误触发和 clause 污染问题

- `qa/evaluation.py`
  - 统一输出结构化评测结果
  - 提供 split-wise summary
  - 输出 lexical / semantic / numeric / procedure / abstain / counterfactual flip 混合评分统计

### 3.2 仿真侧

- `simulation/rag_controller.py`
  - 基于知识库证据和任务描述生成八参数控制计划
  - 支持 `rag_generic_only` evidence ablation，显式压制对象特定 force rule
  - 支持 `rag_no_motion_rules`，显式关闭 motion / clearance 路径

- `simulation/env.py`
  - 基于物体属性推导独立力学窗口
  - 输出执行观测和诊断指标
  - 成功判定显式使用 `transfer_force`、`transport_velocity`、`placement_velocity` 与 `lift_clearance`

- `simulation/feedback.py`
  - 根据 `slip_risk / compression_risk / stability_score` 调整参数

- `simulation/runner.py`
  - benchmark 核心执行逻辑
  - 负责 evidence ablation、motion ablation 与证据条件化汇总输出

- `simulation/benchmark.py`
  - CLI wrapper

### 3.3 结果侧

- `reporting/visualize_results.py`
- `reporting/generate_showcase.py`
- `outputs/current/`
- `outputs/visualizations/`

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
python -m simulation.benchmark --report_multi_seed --method rag --n_trials 20 --seeds 42 43 44 --output outputs/current/simulation_benchmark_result.json
python -m simulation.benchmark --compare_direct_llm --n_trials 20 --output_dir outputs/current
python -m simulation.benchmark --compare_evidence_ablation --n_trials 20 --seeds 42 43 44 --output_dir outputs/current
python -m simulation.benchmark --compare_motion_ablation --n_trials 20 --seeds 42 43 44 --output_dir outputs/current
python -m simulation.benchmark --compare_multi_seed --n_trials 20 --seeds 42 43 44 --multi_seed_methods rag rag_learned task_heuristic direct_llm fixed --output_dir outputs/current
```

### 4.4 图表与摘要

```bash
python reporting/visualize_results.py --qa_json outputs/current/qa_evaluation_detail.json --sim_json outputs/current/simulation_comparison_rag_vs_baseline.json --sim_multi_seed_json outputs/current/simulation_comparison_multi_seed.json --output_dir outputs/visualizations
python reporting/generate_showcase.py --qa_json outputs/current/qa_evaluation_detail.json --sim_json outputs/current/simulation_comparison_rag_vs_baseline.json --sim_multi_seed_json outputs/current/simulation_comparison_multi_seed.json --sim_benchmark_json outputs/current/simulation_benchmark_result.json --output outputs/current/showcase_summary.txt
```

### 4.5 一键运行

```bash
bash scripts/run_all.sh
```

## 5. 关键输出

### 5.1 QA

- `outputs/current/qa_evaluation_detail.json`
- `outputs/current/rag_evaluation_report.txt`
- `outputs/current/rag_problems.txt`
- `outputs/current/direct_llm_result.txt`
- `outputs/current/problem_solving_result.txt`

### 5.2 Simulation

- `outputs/current_round20_sim/simulation_benchmark_result.json`
- `outputs/current_round20_sim/showcase_summary.txt`
- `outputs/current/simulation_benchmark_result.json`
- `outputs/current/simulation_comparison_rag_vs_baseline.json`
- `outputs/current/simulation_comparison_multi_seed.json`
- `outputs/current/simulation_evidence_ablation.json`
- `outputs/current/simulation_evidence_dependence_summary.txt`
- `outputs/current/simulation_motion_ablation.json`
- `outputs/current/simulation_motion_dependence_summary.txt`
- `outputs/current/simulation_split_summary.txt`
- `outputs/current/simulation_challenge_summary.txt`
- `outputs/current/simulation_ablation_retrieval.json`

### 5.3 Reporting

- `outputs/current/showcase_summary.txt`
- `outputs/visualizations/`

## 6. 当前限制

- QA 结果是当前知识库和当前 case split 上的结果，不应直接表述为开放域强泛化。
- QA 新增的 `counterfactual` split 是条目级移除证据后的反事实测试，能证明证据依赖，但仍不是开放环境中的真实缺文档分布。
- QA 当前残余误判已被显著压缩，但结果仍然依赖当前知识库的覆盖边界，不等于开放机械知识问答已解决。
- `rule_heavy` 仍然比 `improved` 更依赖模板约束，但模板已经后移到证据约束层，不再单独维护平行实现。
- 仿真仍然是简化抓取问题，不等价于完整机械臂控制栈；但控制器和环境已不再共享同一套真值范围。
- `reference_force_range` 和 `reference_approach_height` 只用于分析输出，不参与成功判定。
- 仿真当前主线是 `round19`。`round20` 说明 placement-stage precision 的继续扩展没有带来更高的总体成功率。文档中的当前方法以 `round19 current` 为准。
- simulation evidence ablation 目前只覆盖对象特定 force rule，对速度/净空等更细粒度规则的删减实验仍未展开。
- simulation motion / dynamic transport ablation 现在已覆盖 `approach_height / lift_force / transfer_force / transfer_alignment / transport_velocity / placement_velocity / lift_clearance` 路径；在 round8 中把 `pick_smooth_metal_fast`、`pick_large_part_far` 分别推进到 `+15.00%` 与 `+3.33%`，在 round9 中把 `pick_thin_wall` 从 `36.67%±11.55%` 提升到 `73.33%±5.77%`，在 round10 中把 `pick_rubber` 提升到 `81.67%±10.41%`、把 `pick_smooth_metal` 提升到 `81.67%±2.89%` 并追平 `task_heuristic`，round11 继续做 dynamic force-center calibration，round12 补上 large-part specific evidence，round13 把 long-transfer motion numeric evidence 与环境动态风险打通，round14 把 `pick_large_part_far` 的失败拆成 `lift_hold / transfer_sway / placement_settle` 三段，round15 继续把该任务的末段落位速度从“隐含于 transport”补厚成显式 `placement_velocity` 控制，round16 则继续把 `pick_smooth_metal_fast` 的高速低摩擦运输链补厚成 `high_speed_low_friction` dynamic transport mode；当前该任务在 benchmark 中达到 `81.67%±2.89%`，controller 输出 `transfer_force=36N`、`placement_velocity=0.30m/s`、`lift_clearance=0.065m`，并首次可直接观测 `avg_transfer_sway_risk_mean=0.0569` 与 `avg_placement_settle_risk_mean=0.0584`。round17 进一步把 `pick_large_part_far` 的“抓取点尽量靠近重心”补厚成显式 `transfer_alignment`，controller 现在输出 `transfer_force=37.5N`、`transfer_alignment=0.90`、`placement_velocity=0.15m/s`，该任务在 benchmark 中提升到 `63.33%±7.64%`，evidence gain 提升到 `+8.33%`，motion gain 提升到 `+18.33%`。round19 再把该任务的起吊保持链补厚成显式 `lift_force=37.65N`，并让 lift-stage 证据真正作用于 long-transfer `lift_hold` 语义；该任务在 benchmark 中进一步提升到 `65.00%±13.23%`，`avg_lift_hold_risk_mean` 下降到 `0.0097`，evidence gain 提升到 `+10.00%`，motion gain 提升到 `+20.00%`。
- simulation 的 complex-task query 扩展已按动态证据优先级重排，`高速/长距离` 查询不再被材料类通用查询截断。
- surrogate benchmark 现已按 `seed + task_id` 固定随机序列，相同 multi-seed 命令的 JSON 输出应可复现。

## 7. 对比口径

- QA 评测统一在同一知识库、同一 split 定义和同一评分规则下运行，并额外输出 `retrieval`、`response_eval`、`evidence_trace` 以区分“证据命中”和“回答正确”。
- QA summary 额外输出 `avg_semantic_similarity`、`numeric_consistency_rate`、`avg_procedure_order_score`、`abstain_precision/recall/accuracy`、`avg_required_support_coverage`、`counterfactual_flip_rate`，用于衡量语义贴合、数值一致性、流程顺序、拒答行为、证据支撑以及“删支撑后是否停止回答”。
- simulation 对比统一使用同一任务集、同一 `n_trials`、同一 seed 预算和同一环境判定逻辑。
- `rag`、`direct_llm`、`fixed`、`rag_feedback`、`rag_learned` 等方法都不能直接读取 `reference_force_range`；该字段只保留在结果层做分析统计。
- `task_heuristic` 作为更强的非检索 baseline，只能使用任务文本与通用启发式，不读取知识检索结果和 reference 真值。
- `rag_learned` 现在使用环境 teacher 标签训练，因此不再是当前 RAG 的银标蒸馏副本。
- simulation 输出包含 95% CI、多 seed `mean±std`、`train/val/test` split 汇总、`challenge_tags` challenge 汇总、证据支持度/冲突统计、距离误差、稳定度、滑移/压坏/速度/净空风险，以及 `rag` 对 `rag_generic_only` / `rag_no_motion_rules` 的双消融，避免只用单一成功率叙事。
