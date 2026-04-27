# Simulation Execution / Schema / Current Profile 设计

## 1. 背景

当前 `observer / step replan` 主线已经进入代码和文档，但在结果语义、默认复现入口和包边界上仍存在四类问题：

1. surrogate 执行路径会对同一组参数做两次随机评估，导致 `observer_trace` 与最终记账结果可能脱节。
2. benchmark 对“控制计划”的汇总语义错误，很多 JSON 和图实际上反映的是最后一条 trial 的参数，而不是 task 级统计量。
3. `simulation` 包根导入强绑定 `RAGController`，破坏最小依赖运行和测试隔离。
4. 默认脚本、CLI、可视化、摘要脚本仍指向旧的 `rag + outputs/current` 口径，与文档宣称的当前主线不一致。

本设计的目标不是新增算法功能，而是把当前 simulation 主线升级成“执行真相一致、schema 语义一致、默认复现一致、包边界可复用”的版本。

## 2. 本轮范围

本轮采用以下明确决策：

- 采用结构升级方案，而不是最小热修。
- JSON schema 做干净切换，不保留旧 `params_used` 作为主字段。
- 代码修复、默认口径、结果重跑、图表重生和文档同步绑定在同一轮完成。

本轮不做的事情：

- 不把当前 surrogate / MuJoCo 执行模型升级成完整 posterior filter、MPC 或优化求解器。
- 不扩展新的 benchmark 任务集。
- 不做与当前四个问题无关的大规模重构。

## 3. 设计目标

### 3.1 执行真相一致

对任意一次 surrogate execution：

- `success`
- `failure_bucket`
- 风险诊断
- `observer_trace`
- `step_replan_trace`

必须来自同一个最终 evaluation，不允许在“已决定返回”的参数上再次随机采样。

### 3.2 结果 schema 语义一致

summary、comparison、multi-seed、plot、showcase 需要明确区分：

- 初始控制器提案
- trial 终态执行计划统计
- 单次 trial 明细

任何 task 级结果都不能再由“最后一条 trial 留下来的参数”代表。

### 3.3 默认复现一致

仓库的默认 simulation current 必须与文档一致：

- 默认主方法：`rag_feedback`
- 默认 simulation 输出目录：`outputs/current_observer_step_replan`
- 默认 comparison / visualize / showcase 输入：同一 simulation current

### 3.4 包边界最小依赖

导入 `simulation.control_core`、`simulation.env`、`simulation.runner` 等模块时，不应因为包根初始化而强制拉起 langchain / Chroma / RAG 检索栈。

## 4. 非目标与约束

### 4.1 非目标

- 不要求旧结果 JSON 与新 schema 二进制兼容。
- 不要求旧 plotting 代码在不改输入的情况下继续跑。
- 不保留 `params_used` 作为兼容字段。

### 4.2 约束

- MuJoCo 路径与 surrogate 路径都必须维持可运行。
- 当前 README 中对 `outputs/current/` 作为 QA current 的引用可以保留，但 simulation current 必须与之明确分离。
- 当前外部 checklist 位于仓库外，本轮不纳入仓库。

## 5. 方案概览

本轮采用 `Execution Record + Schema V2 + Current Profile` 方案：

1. 在 `env.py` 中建立单次 execution 的唯一真相记录。
2. 在 `runner.py` 中把 task 级 summary、trial 明细和 seed 级统计显式拆层。
3. 在 `reporting/*`、`simulation/benchmark.py` 和 `scripts/run_all.sh` 中统一 current profile。
4. 在 `simulation/__init__.py` 中移除 eager RAG import。
5. 重新生成 `outputs/current_observer_step_replan/` 的 benchmark、comparison、图表和摘要，并同步 README / overview / simulation README。

## 6. 执行语义设计

### 6.1 新的不变量

在 surrogate 路径中，一次 top-level execution 只允许一个最终随机样本决定返回值。

允许的随机评估位置：

- 初始参数评估
- 某次 step replan 之后对新参数的重新评估

禁止的行为：

- 在决定“不再 replan，准备返回”之后，再对同一组最终参数调用一次 `_evaluate_execution_plan`

### 6.2 建议结构

`env.py` 内部形成显式的 evaluation record，至少包含：

- `success`
- `elapsed`
- `info`
- `diag`
- `params`
- `horizontal_distance`
- `profile`
- `observer_trace`

`simulate_stepwise_execution()` 的控制流调整为：

1. 对当前参数生成一份 evaluation record。
2. 基于该 record 生成 `observer_trace`。
3. 若命中 replan 条件，则生成新参数并进入下一轮。
4. 若不再 replan，则直接复用当前 record 返回。

返回时：

- `observer_trace` 来自最终一次被采纳或被观测的 evaluation 序列。
- `success / failure_bucket / 风险指标` 必须与最后一次返回的 evaluation record 完全一致。

### 6.3 预期效果

修复后，对固定 seed 的 surrogate trial：

- `observer_trace` 不再与最终 `failure_bucket` 脱节。
- `step_replan_trace` 的触发依据与 benchmark 记账依据来自同一条执行链。

## 7. Benchmark Schema V2

### 7.1 主 summary 的三层语义

每个 task 的 summary 需要显式区分三类信息：

#### A. `seed_plan`

控制器在进入 execution 之前给出的初始计划。

语义：

- `planner proposal before execution feedback`
- 对 `rag_feedback` 来说，是 step replan 和 trial retry 的起点
- 对不带 feedback 的方法，也仍然只代表初始提案，而不是 trial 统计量

#### B. `executed_plan_stats`

对每个 top-level trial 的 terminal applied plan 做聚合统计。

统计字段：

- `mean`
- `std`
- `min`
- `max`

覆盖的数值参数：

- `gripper_force`
- `lift_force`
- `transfer_force`
- `transfer_alignment`
- `approach_height`
- `transport_velocity`
- `placement_velocity`
- `lift_clearance`

离散汇总字段：

- `dynamic_transport_mode_mode`
- `solver_selected_candidate_mode`
- `execution_feedback_mode_mode`

#### C. `trial_records`

完整 trial 明细，不内嵌进主 summary，而是单独输出到明细文件。

### 7.2 明细文件设计

新增 `simulation_benchmark_trial_records.json`，按 task 输出 trial 列表。

每个 trial record 至少包含：

- `trial_index`
- `success`
- `failure_bucket`
- `elapsed_sec`
- `distance_error`
- `feedback_retry_count`
- `step_replan_count`
- `terminal_plan`
- `observer_trace`
- `step_replan_trace`
- 风险指标摘要

### 7.3 移除旧字段

以下旧语义字段退出主 schema：

- `params_used`
- 基于 `params_used` 推导的 task-level 单值控制计划字段

替代方案：

- summary / comparison / multi-seed / plot 默认改读 `executed_plan_stats.mean.*`
- 若需要初始计划，改读 `seed_plan.*`
- 若需要单次执行终态，改读 `trial_records[*].terminal_plan`

### 7.4 派生指标的重定义

以下指标改为基于 `executed_plan_stats` 或 trial 统计：

- `reference_force_deviation` -> `reference_force_deviation_stats`
- comparison 中的 `*_gripper_force`、`*_transport_velocity`、`*_lift_clearance` 等 -> task 级 executed mean
- multi-seed 中的控制计划均值 -> 先取每个 seed 的 task-level executed mean，再跨 seed 聚合

## 8. Current Profile 设计

### 8.1 current profile 定义

仓库显式采用双 current 概念：

- `QA current`: `outputs/current/`
- `Simulation current`: `outputs/current_observer_step_replan/`
- `round20 experimental`: `outputs/current_round20_sim/`

simulation current 的默认 profile 固定为：

- 主方法：`rag_feedback`
- 主 benchmark：observer / step-replan 闭环口径
- 默认图表、comparison、showcase 全部从该目录读取

### 8.2 默认入口统一

需要统一以下入口：

#### `scripts/run_all.sh`

- simulation 主输出目录切到 `outputs/current_observer_step_replan`
- 主 benchmark 方法改成 `rag_feedback`
- `--compare_multi_seed` 默认方法集带上 `rag_feedback`
- visualization 输出落到 simulation current 下的 `visualizations/`

#### `simulation/benchmark.py`

- `DEFAULT_OUTPUT_DIR` 改到 simulation current
- 默认 `--method` 改成 `rag_feedback`
- 默认 `--multi_seed_methods` 带上 `rag_feedback`

#### `reporting/visualize_results.py`

- 默认 simulation 输入改到 `outputs/current_observer_step_replan/*`
- 默认输出目录改到 `outputs/current_observer_step_replan/visualizations`

#### `reporting/generate_showcase.py`

- 默认 simulation 输入与输出改到 `outputs/current_observer_step_replan/*`

### 8.3 文档写法

README、`docs/overview.md`、`simulation/README.md` 统一采用：

- QA current
- Simulation current
- experimental archive

三层口径，避免单个 `current` 一词同时指 QA 和 simulation。

## 9. 包边界设计

### 9.1 目标

消除 `simulation` 包根对 `RAGController` 的强依赖。

### 9.2 方案

`simulation/__init__.py` 不再 eager import `RAGController`。

可选实现中优先级如下：

1. 包根只导出轻依赖对象，甚至接近空壳。
2. 如确有包根级 API 需求，再使用惰性导入。

本轮优先采用方案 1。

### 9.3 结果

修复后：

- `import simulation.control_core` 不再要求 langchain / Chroma
- 局部测试和无检索依赖脚本可最小依赖运行

## 10. 测试与验证

### 10.1 单元与回归验证

需要新增或调整测试，覆盖：

1. surrogate 执行一致性
   - 固定 seed 下，返回的 `success / failure_bucket` 与最终 evaluation 一致
   - 不再出现“同一参数二次采样导致的结果漂移”
2. summary schema
   - 输出包含 `seed_plan`、`executed_plan_stats`、trial records 文件
   - 不再输出 `params_used`
3. comparison / multi-seed 聚合
   - 控制计划字段来源于 task-level executed mean，而不是最后一条 trial
4. 包边界
   - 在缺少 langchain 依赖的场景下，导入 `simulation.control_core` 和 `simulation.env` 仍可成功

### 10.2 结果级验证

本轮完成后需要重新生成：

- `simulation_benchmark_result.json`
- `simulation_benchmark_trial_records.json`
- `simulation_comparison_rag_vs_baseline.json`
- `simulation_comparison_multi_seed.json`
- `visualizations/*`
- `showcase_summary.txt`

验证点：

- 图表能从新 schema 成功读数
- 文档引用的文件都真实存在
- `showcase_summary.txt` 能由脚本自动生成，而不是人工兜底

## 11. 迁移与兼容策略

本轮采用干净切换策略：

- 新 schema 直接替换旧 schema
- 不保留 `params_used` 兼容字段
- 旧结果文件可保留为历史归档，但不再作为 current 主线结果

这意味着：

- plotting / showcase / README 必须在同一轮同步完成
- 不能只改 runner 而不改消费端

## 12. 实施顺序

建议按以下顺序实施：

1. 修 `env.py` 执行真相模型，消除 surrogate 双重采样。
2. 改 `runner.py`，引入 Schema V2 与 trial records。
3. 改 `reporting/visualize_results.py` 与 `reporting/generate_showcase.py`，消费新 schema。
4. 改 `simulation/__init__.py`，解除 eager RAG import。
5. 改 `simulation/benchmark.py` 与 `scripts/run_all.sh`，统一 current profile。
6. 重跑 simulation current，重生成 comparison、图表和 showcase。
7. 同步 README、`docs/overview.md`、`simulation/README.md`。

## 13. 风险与取舍

### 13.1 风险

- Schema V2 会破坏旧结果消费脚本的兼容性。
- 结果重跑后，当前指标数值可能与上一轮文档不同。
- `generate_showcase.py` 可能暴露出此前被手写摘要绕开的 schema 假设问题。

### 13.2 取舍

本轮优先保证语义正确和默认复现一致，而不是保住旧 schema 的向后兼容。

理由：

- 现有 `params_used` 语义本身是错误的，继续兼容只会延长误导。
- 当前 simulation current 已经切到 observer / step-replan 主线，默认入口与结果消费必须同步收口。

## 14. 完成定义

本轮完成的标准是：

1. surrogate 路径不存在最终参数二次随机评估。
2. 主 summary 不再包含旧 `params_used`。
3. comparison / multi-seed / 图表中的控制计划字段均来自 `executed_plan_stats`。
4. `simulation` 包根不再强制引入 RAG 检索依赖。
5. 默认脚本、CLI、可视化、showcase、README 都指向同一个 simulation current。
6. `outputs/current_observer_step_replan/` 用新 schema 重新生成并能自洽展示。
