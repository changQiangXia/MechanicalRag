# Simulation 模块说明

## 目标

仿真模块负责回答下面这个问题：

> 给定知识库和抓取任务描述，控制器生成的参数是否能在独立环境中取得更好的执行结果。

## 当前主线状态

截至 `2026-04-24 UTC`，simulation 的 authoritative 结果在 `outputs/current/`，对应 `round19`。

- 主线 benchmark：`outputs/current/simulation_benchmark_result.json`
- round20 实验 benchmark：`outputs/current_round20_sim/simulation_benchmark_result.json`
- `pick_large_part_far` 的当前主线结果为 `0.6500 ± 0.1323`，主导失败模式为 `placement_settle_fail`
- `pick_smooth_metal_fast` 的当前主线结果为 `0.8167 ± 0.0289`
- `round20` 是实验轮次，目标是 `pick_large_part_far` 的 `placement_settle`。实验结果把 `placement_settle_risk` 降到了更低水平。总体成功率为 `0.5833 ± 0.0577`，主线结果继续保持 `round19`。
- 当前仓库代码语义已经对齐到 `round19` 主线。`placement_precision` 只保留在 `round20` 实验输出里。

当前版本的关键边界是：

- 控制器只看知识库与任务描述。
- 环境成功判定只看物体属性、执行轨迹和给定参数。
- `reference_force_range` 仅用于分析输出，不参与成功判定。
- 当前控制计划包含 `gripper_force`、`lift_force`、`transfer_force`、`transfer_alignment`、`approach_height`、`transport_velocity`、`placement_velocity`、`lift_clearance` 八个参数；其中 `lift_force`、`transfer_force`、`transfer_alignment` 与 `placement_velocity` 分别表示动态任务中的“起吊保持夹持余量”“运输阶段夹持余量”“抓取点与重心对中质量”和“末段落位速度”，而不是继续把全程都压成一个静态抓取参数。
- benchmark 任务现已带 `challenge_tags`，用于按挑战属性聚合结果。
- `RAGController` 现在会输出结构化证据轨迹，包括规则数、支持度、冲突数和所使用的关键证据条目。
- `RAGController` 现在还会输出 `matched_hint_keywords`、`composite_force_floor`、`motion_aware_force_floor`、`motion_aware_force_cap`、`thin_wall_support_force_floor`、`rubber_material_force_floor`、`heavy_metal_force_center_floor`、`dynamic_heavy_metal_force_center_floor`、`long_transfer_force_center_floor`、`long_transfer_dynamic_force_margin`、`long_transfer_velocity_band`、`long_transfer_numeric_clearance_floor`、`long_transfer_clearance_target`、`long_transfer_lift_force_target`、`long_transfer_lift_force_margin`、`long_transfer_alignment_target`、`long_transfer_alignment_force_margin`、`available_lift_stage_rules`、`used_lift_stage_rules`、`dynamic_smooth_metal_force_cap`、`rubber_material_clearance_floor`、`static_smooth_metal_force_cap`、`thin_wall_support_clearance_cap`、`motion_path_force_compensation` 与 `calibration_notes`，便于追踪复杂任务检索与定向 motion/force/alignment/lift-stage 校准是否触发。
- `RAGController` 现在支持 `rag_generic_only`，可在同一检索证据下压制对象特定 force rule，做 evidence ablation。
- `RAGController` 现在还支持 `rag_no_motion_rules`，可显式关闭 `approach_height / transport_velocity / lift_clearance` 路径，做 motion ablation。
- complex task 的 query 扩展现在优先保留 `高速/长距离` 等动态证据查询，避免被材料类通用查询截断。
- thin-wall 单检索路径现在也会补做 query enrichment，并把“多点支撑/真空吸附”显式解释为 support calibration，而不是继续机械翻译成更高 clearance。
- low-speed `rubber` / `smooth_metal` 现在也会补做 material-aware calibration：前者用 `rubber_material_force_floor` 解决欠力滑移，后者用 `static_smooth_metal_force_cap` 解决 broad-range force 过高。
- round11 继续补做 dynamic force-center calibration：`long_transfer_force_center_floor` 把 `pick_large_part_far` 拉回到 `36N` 附近，`dynamic_smooth_metal_force_cap` 把 `pick_smooth_metal_fast` 从 `40.5N` 压到 `34N`，`heavy_metal_force_center_floor` / `dynamic_heavy_metal_force_center_floor` 则分别把 `pick_metal_heavy` / `pick_metal_heavy_fast` 推到 `46N` / `48N`。
- round12 继续补做 large-part evidence thickening：知识库中新增了 `大型零件夹持力` 与 `长距离搬运大型零件` 两条 specific force evidence，single-retrieval 也新增了对应 query/backstop，因此 `pick_large_part_far` 现在会显式选中 `force` rule，而不再只依赖 `重心偏移 / 轨迹规划` 这类 motion evidence。
- round13 继续补做 long-transfer dynamics thickening：controller 现在可以解析 `0.18-0.22m/s`、`0.07m`、`0.08m左右` 这类 motion 数字，`pick_large_part_far` 会输出 `long_transfer_velocity_band=[0.18,0.22]`、`long_transfer_clearance_target=0.075`，环境侧也新增了 `transfer_sway_risk`，不再只用平坦的 `travel_distance` 惩罚表示长距离大件风险。
- round14 继续补做 stage-specific long-transfer failure decomposition：环境把 `pick_large_part_far` 的失败拆成 `lift_hold_fail / transfer_sway_fail / placement_settle_fail`，controller 也在高支持 numeric motion evidence 下启用了 `long_transfer_stage_force_floor`，最终计划收敛到 `37.05N / 0.06m / 0.20m/s / 0.08m`，benchmark 中该任务达到 `60.00%±5.00%`，且 `transfer_sway_fail` 成为主导失败模式。
- round15 继续补做 stage-specific settle control thickening：controller 在 high-support `long_transfer + large` 场景下新增 `long_transfer_placement_velocity_cap`，最终 `pick_large_part_far` 计划变为 `37.05N / 0.06m / 0.20m/s / 0.15m/s / 0.08m`，环境与 runner 也同步输出 `placement_velocity`、`dynamic_placement_velocity_cap` 与更新后的 `stage_plan`；该任务在 benchmark 中维持 `60.00%±5.00%`，evidence gain `+5.00%`，motion gain `+15.00%`，并把落位稳定性真正纳入独立控制参数。
- round16 继续补做 inertia-aware dynamic transfer thickening：controller 针对 `pick_smooth_metal_fast` 这类“高速 + 低摩擦 + 脆性 + 中长运输路径”任务新增 `dynamic_transport_mode=high_speed_low_friction`，并显式输出 `transfer_force=36N`、`placement_velocity=0.30m/s`、`lift_clearance=0.065m`；环境也不再把这类任务塞回 generic stage bucket，而会输出非零 `lift_hold_risk / transfer_sway_risk / placement_settle_risk`。结果上，`pick_smooth_metal_fast` 从 round15 的 `78.33%±5.77%` 提升到 `81.67%±2.89%`，相对 `task_heuristic` 的优势扩大到 `+21.67%`，相对 `no_motion` 的 gain 为 `+23.33%`。
- round17 继续补做 center-of-mass alignment thickening：controller 不再只把 `抓取点尽量靠近重心` 当作检索叙事，而是在 high-support `long_transfer + large` 场景下显式输出 `transfer_alignment=0.90` 与 `long_transfer_alignment_force_margin=0.45`，最终 `pick_large_part_far` 计划变为 `37.05N / 37.50N / 0.90 / 0.06m / 0.20m/s / 0.15m/s / 0.08m`；环境与 runner 也同步把 `transfer_alignment` 写入 benchmark / comparison / showcase。结果上，该任务从 round16 的 `60.00%±5.00%` 提升到 `63.33%±7.64%`，`avg_transfer_sway_risk_mean` 从 `0.0338` 降到 `0.0153`，evidence gain 提升到 `+8.33%`，motion gain 提升到 `+18.33%`，repeat benchmark 结果一致。
- round19 继续补做 lift-stage hold thickening：controller 在 `long_transfer + large` 高支持场景下新增显式 `lift_force`，并通过 `夹爪松紧度 / 接近-抓取-抬升-放置序列` 这类 lift-stage 证据触发 `long_transfer_lift_force_margin=0.60`，最终 `pick_large_part_far` 计划变为 `37.05N / 37.65N / 37.50N / 0.90 / 0.06m / 0.20m/s / 0.15m/s / 0.08m`；环境与 runner 也同步输出 `lift_force`、`lift_center_offset`、`dynamic_lift_force_shortfall` 以及 lift-stage evidence visibility。结果上，该任务从 round17 的 `63.33%±7.64%` 提升到 `65.00%±13.23%`，`avg_lift_hold_risk_mean` 从 `0.0235` 降到 `0.0097`，`lift_hold_fail_rate_mean` 从 `13.33%` 降到 `6.67%`，evidence gain 提升到 `+10.00%`，motion gain 提升到 `+20.00%`，repeat benchmark 结果一致。
- round20 继续把 `pick_large_part_far` 的末段落位扩展为显式 placement-stage precision 控制。实验把 `placement_settle_risk` 降到了更低水平。authoritative benchmark 从 `0.6500 ± 0.1323` 变化到 `0.5833 ± 0.0577`，repeat benchmark 也得到同样结果。当前 simulation 文档和代码继续采用 round19 主线变量。

## 结构

```text
simulation/
├── tasks.py          任务配置与对象属性
├── rag_controller.py RAG → 控制参数
├── baseline_controller.py
├── env.py            MuJoCo / 代理环境
├── feedback.py       基于观测反馈调参
├── runner.py         benchmark 执行核心
├── reporting.py      benchmark 输出序列化
├── benchmark.py      CLI wrapper
└── train_learned_model.py
```

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

benchmark 汇总的 `reference_force_range` 只是分析指标，方便比较“控制器输出离知识参考范围有多远”。
同时，环境会基于物体属性独立推导推荐搬运速度与最小抬升净空，并将 `velocity_risk / clearance_risk` 写入输出。
对长距离大件搬运，环境现在还会独立推导 long-transfer dynamic target，并输出 `lift_hold_risk`、`transfer_sway_risk`、`placement_settle_risk` 以及对应的 `failure_bucket`，用于表达长路径搬运中的起吊保持、运输摆动和落位稳定性风险。

## 对比公平性

不同方法的 benchmark 对比遵循统一口径：

- 使用同一组 `BENCHMARK_TASKS`
- 使用相同的 `n_trials`、seed 和输出统计字段
- 使用同一环境成功判定逻辑
- 不向控制器或 feedback 模块暴露 `reference_force_range`
- `task_heuristic` baseline 只能用任务文本与通用启发式，不使用检索结果
- `rag_learned` baseline 现在使用环境 teacher 标签训练，不再读取 RAG 银标参数

输出除成功率外，还包含 95% CI、多 seed `mean±std`、按 `train/val/test` 聚合的 split 汇总、按 `challenge_tags` 聚合的 challenge 汇总、证据支持度/冲突统计、距离误差、稳定度以及滑移/压坏/速度/净空风险，避免把单一高分误写成更强的方法结论。
现在还会额外输出 `simulation_evidence_ablation.*` 与 `simulation_evidence_dependence_summary.*`，用来判断 RAG 增益是否真来自对象特定规则。
同时会额外输出 `simulation_motion_ablation.*` 与 `simulation_motion_dependence_summary.*`，用来判断 motion / clearance 路径是否真正贡献成功率。

## 运行

```bash
python -m simulation.benchmark --method rag --n_trials 10 --output outputs/current/simulation_benchmark_result.json
python -m simulation.benchmark --compare_direct_llm --n_trials 10 --output_dir outputs/current
python -m simulation.benchmark --compare_evidence_ablation --n_trials 10 --seeds 42 43 44 --output_dir outputs/current
python -m simulation.benchmark --compare_motion_ablation --n_trials 10 --seeds 42 43 44 --output_dir outputs/current
python -m simulation.benchmark --compare_multi_seed --n_trials 10 --seeds 42 43 44 --multi_seed_methods rag rag_learned task_heuristic direct_llm fixed --output_dir outputs/current
python -m simulation.benchmark --ablation_retrieval --n_trials 5 --output_dir outputs/current
```

## 降级模式

若本机没有 MuJoCo：

- CLI 仍可运行
- runner 会回退到环境代理模型
- 代理模型会按 `seed + task_id` 固定随机序列，保证 multi-seed 结果可复现
- 输出文件格式保持一致

但需要明确，这种模式不是完整物理仿真，只用于保留流程闭环与方法对比。
