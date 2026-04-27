# Simulation 模块说明

## 目标

仿真模块负责回答下面这个问题：

> 给定知识库和抓取任务描述，控制器生成的参数是否能在独立环境中取得更好的执行结果。

## 当前状态

截至 `2026-04-27 UTC`，simulation 当前代码语义对应 `outputs/current_observer_step_replan/` 这次 phase-observation / suffix-counterfactual-repair 闭环验证；`outputs/current/` 继续保留为 `round19` 历史基线，`outputs/current_round20_sim/` 保留 `round20` 的 placement-stage 实验。

- 最新闭环 benchmark：`outputs/current_observer_step_replan/simulation_benchmark_result.json`
- `round19` 历史 benchmark：`outputs/current/simulation_benchmark_result.json`
- `round20` placement-stage 实验：`outputs/current_round20_sim/simulation_benchmark_result.json`
- rolling posterior + soft-trigger suffix repair 已在默认结果中实际触发；这证明默认 `rag_feedback` 主线不再只是 observer logging，但不等于它已经成为总体最优主线
- `observer_trace` 现在只是兼容字段；主执行日志是 `phase_execution_trace`、`observation_trace`、`belief_update_trace`、`counterfactual_replan_trace`
- 这条语义的当前公开证据至少在 `pick_metal_heavy` 上已经成立：`online_diagnosis_count=60`、`suffix_counterfactual_replan_count=60`，且 `rag_feedback=96.67%` 高于 `rag_feedback_observer_only=90.00%`
- 当前公开结果里，`rag_feedback_observer_only` 12 任务多 seed平均成功率 `0.9055`，高于默认 `rag_feedback` 的 `0.8583`
- `rag_feedback` 12 任务多 seed平均成功率 `0.8583`
- 相对 `rag` seed-only 路径，平均成功率提升 `+0.1278`
- `pick_metal_heavy = 0.9667 ± 0.0577`
- `pick_metal_heavy_fast = 0.8833 ± 0.0289`
- `pick_large_part_far = 0.9333 ± 0.0764`
- `pick_thin_wall_fast = 0.7833 ± 0.1258`
- `pick_smooth_metal_fast = 0.7000 ± 0.0866`
- 当前重载验收口径已满足：`pick_metal_heavy >= 0.30`，`pick_metal_heavy_fast` 绝对成功率下降不超过 `0.05`，12 任务总体平均成功率不低于旧基线 `0.8222 - 0.01`
- online repair 的贡献目前应按 task 逐项审计，而不是外推成所有任务上的统一增益或默认整体最优结论
- `simulation_benchmark_result.json`、`simulation_comparison_rag_vs_baseline.json`、`simulation_comparison_multi_seed.json` 现在使用 Schema V2：主 summary 在 `methods.<method>` 下，控制计划看 `seed_plan` 和 `executed_plan_stats`
- `simulation_benchmark_trial_records.json` 提供逐 task / 逐 seed 的 execution 明细，包含 `terminal_plan`、`observer_trace`、`phase_execution_trace`、`observation_trace`、`belief_update_trace` 和 `counterfactual_replan_trace`

## 当前边界

- 控制器只看知识库与任务描述。
- 环境成功判定只看物体属性、执行轨迹和给定参数。
- `reference_force_range` 仅用于分析输出，不参与成功判定。
- 控制器当前主链路是 `evidence -> belief -> seed synthesize -> local solve -> final plan`。
- 当前控制计划包含 `gripper_force`、`lift_force`、`transfer_force`、`transfer_alignment`、`approach_height`、`transport_velocity`、`placement_velocity`、`lift_clearance` 八个参数。
- `RAGController` 会输出结构化证据轨迹，以及 `belief_state`、`task_constraints`、`uncertainty_profile`、`seed_mode`、`seed_plan`、`solver_selected_candidate`、`solver_score_breakdown` 等诊断字段；这些字段当前仍是轻量状态估计与局部求解日志，而不是完整后验滤波与优化规划日志。
- `feedback.py` 当前显式使用 `lift_hold_risk`、`transfer_sway_risk`、`placement_settle_risk` 做阶段化调参，而不是只依赖泛化 `stability_score`。
- `runner.py` 当前把 task-level 终态控制计划序列化到 `executed_plan_stats`，不再把“最后一次 trial 的参数”伪装成全任务 summary。
- `simulation_benchmark_trial_records.json` 会把单次 execution 的 terminal control plan、observer snapshot，以及 `phase_execution_trace / observation_trace / belief_update_trace / counterfactual_replan_trace` 单独落盘。

## 关键变量与阶段含义

`belief_state` 负责把检索证据整理成后续控制阶段可直接使用的状态描述，`task_constraints` 负责把任务里的阶段要求收成约束，`uncertainty_profile` 负责记录证据覆盖和保守模式，`seed_plan` 则是在这些输入基础上生成的第一版控制计划。

进入执行阶段后，`phase_execution_trace`、`observation_trace`、`belief_update_trace` 和 `counterfactual_replan_trace` 分别记录阶段推进、阶段观测、内部状态更新和后续阶段修正。它们组合起来，构成当前 `phase observation -> posterior update -> suffix repair` 这条链在结果文件中的主要阅读入口。

更完整的定义见 [控制链术语](../docs/terms_and_mechanisms.md#control-chain-terms) 和 [执行日志字段](../docs/terms_and_mechanisms.md#execution-trace-fields)。

## 结果字段含义

`observer-only ablation` 用来和默认 `rag_feedback` 对照，帮助读者区分“系统是否记录了执行期信息”和“系统是否真的根据这些信息改了计划”。`executed_plan_stats` 则用于汇总一组 trial 结束后实际落下来的控制参数分布。

`online_diagnosis_count` 记录有多少次 trial 进入了执行中的诊断或修正路径，`suffix_counterfactual_replan_count` 记录其中有多少次进一步进入了后缀重算，`post_failure_retry_count` 则补充说明还有多少恢复仍依赖 trial 级重试。

更完整的结果字段说明见 [结果统计字段](../docs/terms_and_mechanisms.md#result-stat-fields)。

## 当前结构

```text
simulation/
├── tasks.py          任务配置与对象属性
├── control_core.py   belief / uncertainty / solver
├── rag_controller.py RAG -> 控制参数
├── baseline_controller.py
├── env.py            MuJoCo / 代理环境
├── feedback.py       基于观测反馈调参
├── runner.py         benchmark 执行核心
├── reporting.py      benchmark 输出序列化
├── benchmark.py      CLI wrapper
└── train_learned_model.py
```

模块职责：

- `rag_controller.py`：规则聚合、检索解释与最终控制计划拼装。
- `control_core.py`：把聚合结果提升为显式 belief / constraint / uncertainty 表达，并在多个 candidate 计划里做 solver 选择。
- `env.py`：根据质量、摩擦、fragility、尺寸与路径推导内部力学窗口和阶段化风险。
- `feedback.py`：根据观测风险做局部回调。
- `runner.py`：统一 benchmark、消融和 multi-seed 汇总执行。
- `reporting.py`：表格、JSON 和文本结果序列化。

## 里程碑

- `round13`：把长距离搬运的数值运动条目落成可执行计划，环境侧加入 `transfer_sway_risk`。
- `round14`：把 `pick_large_part_far` 的失败拆成 `lift_hold_fail / transfer_sway_fail / placement_settle_fail`。
- `round15`：把末段落位速度补厚成显式 `placement_velocity`。
- `round16`：把 `pick_smooth_metal_fast` 补厚成 `high_speed_low_friction` dynamic transport mode。
- `round17`：把“抓取点尽量靠近重心”补厚成显式 `transfer_alignment`。
- `round19`：把起吊保持链补厚成显式 `lift_force`，形成 `outputs/current/` 这条历史主线。
- `round20`：把 `pick_large_part_far` 的末段落位扩展为 placement-stage `placement_precision` 实验，但总体成功率低于 `round19`。
- `2026-04-27 observer-step-replan`：在前一版 thickening 之上，把 control core 前移成 belief 直驱 seed synthesize，并把 `env.py` / `runner.py` / `rag_feedback` 接成单次 execution 内的 `phase observation -> posterior update -> suffix repair` 闭环。当前结果已经把 `pick_metal_heavy` 打到 `0.9667 ± 0.0577`，同时 `pick_metal_heavy_fast = 0.8833 ± 0.0289`，重载门槛已经通过；但默认 `rag_feedback` 的总体均值仍低于 `rag_feedback_observer_only`。

## 成功判定

成功不再直接读取“理想夹爪力范围”。

环境内部会根据：

- `mass_kg`
- `surface_friction`
- `fragility`
- `size_xyz`
- `approach_height`
- `travel_distance`

推导出内部力学窗口，并输出：

- `slip_risk`
- `compression_risk`
- `stability_score`
- `lift_hold_risk`
- `transfer_sway_risk`
- `placement_settle_risk`

benchmark 汇总的 `reference_force_range` 只是分析指标，方便比较“控制器输出离知识参考范围有多远”。

## 对比公平性

不同方法的 benchmark 对比遵循统一口径：

- 使用同一组 `BENCHMARK_TASKS`
- 使用相同的 `n_trials`、seed 和输出统计字段
- 使用同一环境成功判定逻辑
- 不向控制器或 feedback 模块暴露 `reference_force_range`
- `task_heuristic` baseline 只能用任务文本与通用启发式，不使用检索结果
- `rag_learned` baseline 使用环境 teacher 标签训练，不再读取 RAG 银标参数

输出除成功率外，还包含 95% CI、多 seed `mean±std`、按 `train / val / test` 聚合的 split 汇总、按 `challenge_tags` 聚合的 challenge 汇总、证据支持度 / 冲突统计、距离误差、稳定度以及阶段化风险指标。
当前 benchmark / comparison 主 summary 都在 `methods.<method>` 下暴露 `seed_plan`、`executed_plan_stats`、`planner_diagnostics`，配套明细则在 `simulation_benchmark_trial_records.json` 的 `trial_records` 中保留 seed 和 observer / step-replan 上下文。

## 运行

当前文档对应的复现实验：

```bash
python -m simulation.benchmark --report_multi_seed --method rag_feedback --n_trials 20 --seeds 42 43 44 --output outputs/current_observer_step_replan/simulation_benchmark_result.json
python -m simulation.benchmark --compare_feedback --n_trials 20 --seed 42 --output_dir outputs/current_observer_step_replan
python -m simulation.benchmark --compare_evidence_ablation --n_trials 20 --seeds 42 43 44 --output_dir outputs/current
python -m simulation.benchmark --compare_motion_ablation --n_trials 20 --seeds 42 43 44 --output_dir outputs/current
python -m simulation.benchmark --compare_multi_seed --n_trials 20 --seeds 42 43 44 --multi_seed_methods rag rag_feedback rag_feedback_observer_only task_heuristic fixed --output_dir outputs/current_observer_step_replan
python reporting/generate_showcase.py --qa_json outputs/current/qa_evaluation_detail.json --sim_json outputs/current_observer_step_replan/simulation_comparison_rag_vs_baseline.json --sim_multi_seed_json outputs/current_observer_step_replan/simulation_comparison_multi_seed.json --sim_benchmark_json outputs/current_observer_step_replan/simulation_benchmark_result.json --output outputs/current_observer_step_replan/showcase_summary.txt
```

## 当前限制

- simulation 当前代码语义以 `outputs/current_observer_step_replan/` 为准；`outputs/current/round19` 继续保留为历史基线，`outputs/current_round20_sim/` 保留 placement-stage `placement_precision` 实验。
- 当前 observer 已升级成 phase observation，`suffix_counterfactual_replan` 已进入 `rag_feedback` 主链，但这些日志仍是轻量执行期估计与局部修正，不是完整 posterior filter / MPC。
- `belief_state` 仍然主要从知识证据长出来，执行期观测主要进入 feedback replan，而不是从头替代 retrieval-belief 前置链。
- 当前最明显的剩余缺口不再只是 `pick_metal_heavy`，而是如何在不牺牲 `pick_smooth_metal`、`pick_smooth_metal_fast`、`pick_metal_heavy_fast` 等任务的前提下，继续把 local suffix repair 推进成更强的 posterior filter / planner。
- evidence ablation 目前只覆盖对象特定 force rule，对更细粒度的运动学 / 接触规则删减实验仍未展开。
- 当前控制计划仍是简化高层控制抽象，不等价于完整机器人轨迹优化与接触控制栈。

## 降级模式

若本机没有 MuJoCo：

- CLI 仍可运行
- runner 会回退到环境代理模型
- 代理模型会按 `seed + task_id` 固定随机序列，保证 multi-seed 结果可复现
- 输出文件格式保持一致

但需要明确，这种模式不是完整物理仿真，只用于保留流程闭环与方法对比。
