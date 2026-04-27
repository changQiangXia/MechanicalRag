# Simulation 模块说明

## 目标

仿真模块负责回答下面这个问题：

> 给定知识库和抓取任务描述，控制器生成的参数是否能在独立环境中取得更好的执行结果。

## 当前状态

截至 `2026-04-27 UTC`，simulation 当前代码语义对应 `outputs/current_core_thickening/` 这次 control-core thickening 验证；`outputs/current/` 继续保留为 `round19` 历史基线，`outputs/current_round20_sim/` 保留 `round20` 的 placement-stage 实验。

- 最新控制核 benchmark：`outputs/current_core_thickening/simulation_benchmark_result.json`
- `round19` 历史 benchmark：`outputs/current/simulation_benchmark_result.json`
- `round20` placement-stage 实验：`outputs/current_round20_sim/simulation_benchmark_result.json`
- 新控制核为 controller 增加了显式 `belief_state`、`task_constraints`、`uncertainty_profile`、`stage_plan` 和 solver candidate scoring
- `pick_large_part_far` 在新控制核下保持 `0.6500 ± 0.1323`，主导失败模式仍是 `placement_settle_fail`
- `pick_smooth_metal_fast` 保持 `0.8167 ± 0.0289`
- `pick_metal_heavy_fast` 提升到 `0.7500 ± 0.0866`
- 相对 `round19` 历史基线，12 任务多 seed 平均成功率提升 `+0.0014`
- 当前已知回落任务：`pick_smooth_metal = 0.7333`、`pick_metal_heavy = 0.7167`

## 当前边界

- 控制器只看知识库与任务描述。
- 环境成功判定只看物体属性、执行轨迹和给定参数。
- `reference_force_range` 仅用于分析输出，不参与成功判定。
- 控制器当前主链路已经变成 `rule aggregation -> belief bundle -> candidate solve -> final plan`，而不是只做一次平铺规则聚合。
- 当前控制计划包含 `gripper_force`、`lift_force`、`transfer_force`、`transfer_alignment`、`approach_height`、`transport_velocity`、`placement_velocity`、`lift_clearance` 八个参数。
- `RAGController` 会输出结构化证据轨迹，以及 `belief_state`、`task_constraints`、`uncertainty_profile`、`stage_plan`、`belief_state_coverage`、`uncertainty_conservative_mode`、`solver_selected_candidate`、`solver_selected_score`、`solver_score_breakdown`、`solver_candidate_scores` 等诊断字段。
- `feedback.py` 当前显式使用 `lift_hold_risk`、`transfer_sway_risk`、`placement_settle_risk` 做阶段化调参，而不是只依赖泛化 `stability_score`。

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
- `2026-04-27 control-core thickening`：新增 `simulation/control_core.py`，把 rule aggregation 之后的控制逻辑补成显式 belief-state、task-constraint、uncertainty-profile、stage-intent 和 candidate solver 选择链路；结果保住了两项核心动态任务，并把平均成功率轻微抬高到 `0.7528`，但 `pick_smooth_metal` 与 `pick_metal_heavy` 仍有回落。

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

## 运行

当前文档对应的复现实验：

```bash
python -m simulation.benchmark --report_multi_seed --method rag --n_trials 20 --seeds 42 43 44 --output outputs/current_core_thickening/simulation_benchmark_result.json
python -m simulation.benchmark --compare_direct_llm --n_trials 20 --seed 42 --output_dir outputs/current_core_thickening
python -m simulation.benchmark --compare_evidence_ablation --n_trials 20 --seeds 42 43 44 --output_dir outputs/current
python -m simulation.benchmark --compare_motion_ablation --n_trials 20 --seeds 42 43 44 --output_dir outputs/current
python -m simulation.benchmark --compare_multi_seed --n_trials 20 --seeds 42 43 44 --multi_seed_methods rag rag_feedback task_heuristic direct_llm fixed --output_dir outputs/current_core_thickening
```

## 当前限制

- simulation 当前代码语义以 `outputs/current_core_thickening/` 为准；`outputs/current/round19` 继续保留为历史基线，`outputs/current_round20_sim/` 保留 placement-stage `placement_precision` 实验。
- control-core thickening 只带来了 `+0.0014` 的平均成功率提升，并没有统一优于 `round19`；`pick_smooth_metal` 与 `pick_metal_heavy` 仍各回落约 5 个百分点。
- evidence ablation 目前只覆盖对象特定 force rule，对更细粒度的运动学 / 接触规则删减实验仍未展开。
- 当前控制计划仍是简化高层控制抽象，不等价于完整机器人轨迹优化与接触控制栈。

## 降级模式

若本机没有 MuJoCo：

- CLI 仍可运行
- runner 会回退到环境代理模型
- 代理模型会按 `seed + task_id` 固定随机序列，保证 multi-seed 结果可复现
- 输出文件格式保持一致

但需要明确，这种模式不是完整物理仿真，只用于保留流程闭环与方法对比。
