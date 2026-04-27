# Heavy Load Counterfactual Diagnosis 设计

## 1. 背景

当前 simulation 主线已经比早期版本更接近闭环，但它仍有四个结构性问题：

1. `observer` 仍然是整次执行评估结果的阶段投影，不是在线状态估计。
2. `step replan` 虽然发生在单次 execution 内，但语义上仍接近“整条计划重算”，而不是“冻结已执行前缀后的局部恢复”。
3. `belief update` 主要体现为 coverage / confidence / priority 的风险偏置切换，而不是显式执行状态的 posterior 修正。
4. `feedback` 主链仍然高度依赖手工规则映射和固定参数增量，缺少可解释的反事实诊断。

当前结果上，`pick_metal_heavy` 仍然是最清晰的失败锚点：

- `rag` 与 `rag_feedback` 在该任务上都仍为 `0.0` 成功率。
- 失败主要集中在重载静态抓取 / 抬升语义，而不是高速动态搬运。
- 与之对应，`pick_metal_heavy_fast` 当前并非主失败点，因此它更适合作为“非退化约束”，而不是主提升目标。

本设计的目标不是一次性把整个 simulation 链条升级成通用 POMDP / MPC 求解器，而是先围绕重载静态难例，把主链从“规则后处理 + 整计划 reroll”推进到“阶段观测 + posterior diagnosis + suffix repair”。

## 2. 本轮范围

### 2.1 本轮采用的明确决策

- 采用“强模型路线”，但第一版只优先服务 `pick_metal_heavy` 一类静态重载失败。
- 采用显式执行潜变量 + 阶段观测 + 最小充分反事实干预，而不是继续堆叠 priority / coverage 规则。
- 采用冻结已执行前缀、仅修剩余后缀的 `suffix repair` 语义，而不是整条计划重算。
- `pick_metal_heavy_fast` 作为非退化约束，不作为第一版主提升 target。

### 2.2 本轮不做的事情

- 不追求第一版就替换所有 `rag_controller.py` 中的规则壳。
- 不追求对所有 failure mode 给出统一因果解释。
- 不追求彻底消灭 trial-level post-failure retry。
- 不追求第一版就把动态重载、高速摆动、长距离搬运都纳入同一套最优诊断模型。

## 3. 设计目标

### 3.1 主目标

把 `pick_metal_heavy` 从当前 `0.0` 成功率拉到明确非零且可复现的区间，第一版验收目标固定为：

- `pick_metal_heavy >= 0.30`

### 3.2 约束目标

为了避免被解释成专项补丁，第一版同时绑定三个硬约束。基线固定为当前已提交的 `outputs/current_observer_step_replan/` 结果：

- `pick_metal_heavy_fast` 的绝对成功率下降不得超过 `0.05`
- 12 任务总体平均成功率下降不得超过 `0.01`
- 任一非 heavy 任务的绝对成功率下降不得超过 `0.05`

### 3.3 语义目标

除 benchmark 提分外，还必须满足三条执行语义目标：

1. `observer` 从结果投影转成阶段观测。
2. `step replan` 从整计划 reroll 转成 `suffix repair`。
3. `belief update` 从风险偏置切换转成显式潜变量修正。

## 4. 非目标与约束

### 4.1 非目标

- 第一版不要求成为通用最优闭环控制器。
- 第一版不要求把 `pick_metal_heavy_fast` 再进一步显著抬高。
- 第一版不要求完全替代现有 `feedback.py` 的 trial-level retry 逻辑。
- 第一版不要求删除所有旧 trace 字段；允许短期兼容层存在。

### 4.2 约束

- 保留 `simulate_stepwise_execution()` 这一顶层入口，不在第一版扩大调用面。
- 结果层 schema 不应再次退回“最后一条 trial 参数代表 task summary”的旧语义。
- 新增测试重量必须落在执行内核上，而不是再次主要长在 reporting / schema / defaults 上。

## 5. 方案概览

本轮采用 `Phase Observation + Execution Belief + Counterfactual Suffix Repair` 方案：

1. `env.py` 负责真实阶段推进、前缀冻结和阶段观测发射。
2. `control_core.py` 负责 prior / posterior belief、失败原因诊断和最小充分干预选择。
3. `feedback.py` 从诊断主链降级为兼容 / 适配层。
4. `rag_controller.py` 保留 seed planning 与证据检索职责，不继续承担主要在线修复逻辑。
5. `runner.py` 和 trial records 增加在线诊断与 suffix repair 的统计与明细输出。

第一版只在 heavy-static regime 启用这一套强模型主链，其余任务先保持现有控制逻辑或仅做兼容调用。

## 6. 执行语义设计

### 6.1 阶段推进模型

`simulate_stepwise_execution()` 需要从“先整体评估、再拆 trace”改成真正的阶段推进器：

- `approach`
- `grasp`
- `lift`
- `transfer`
- `place`

每个阶段独立完成三件事：

1. `phase_transition`
2. `phase_observation`
3. `phase_terminal_check`

一旦某阶段完成，对应执行前缀即被冻结。后续 replan 只能修改当前阶段之后的 suffix，不允许回滚或重放已完成阶段。

### 6.2 前缀冻结 / 后缀修复

新语义下：

- 在 `lift` 触发风险时，只允许修 `lift / transfer / place`
- 在 `transfer` 触发风险时，只允许修 `transfer / place`
- `approach / grasp` 一旦完成，不再重新执行

这条约束必须显式进入 trace 和测试，而不是只作为隐含实现细节。

### 6.3 观测语义

阶段观测必须只依赖“已执行前缀 + 当前阶段 + 当前参数”，不能偷看整条计划的终局 verdict。

观测层按阶段发射：

- `grasp`：`contact_stability_obs`、`micro_slip_obs`、`payload_ratio_obs`
- `lift`：`lift_progress_obs`、`lift_reserve_obs`、`tilt_obs`
- `transfer`：`sway_obs`、`velocity_stress_obs`
- `place`：`settle_obs`、`placement_error_obs`

旧 `observer_trace` 不再是主观测语义，只能作为兼容字段存在，或者被替换为更精确的阶段观测 trace。

## 7. 执行状态与 posterior belief

### 7.1 第一版显式潜变量

第一版只引入少量与重载失败强相关的 latent：

- `phase`
- `load_support_margin`
- `grip_hold_margin`
- `pose_alignment_error`
- `lift_reserve`
- `transfer_disturbance`

每个 latent 至少保存：

- `estimate`
- `uncertainty`
- `last_updated_phase`

### 7.2 prior / posterior 拆分

`control_core.py` 中的 belief 逻辑明确拆成两段：

- `prior belief`
  - 来源：文本证据、seed plan、规则约束
- `posterior belief`
  - 来源：阶段观测对 latent 的修正

posterior update 必须输出“哪个 latent 被修正、修正方向是什么、置信度如何变化”，而不再只体现在：

- coverage 降低
- confidence 降低
- safety / stability / precision priority 升高

priority 之后仍可存在，但只能作为 derived control bias，而不是主 posterior 语义。

## 8. 反事实诊断设计

### 8.1 第一版失败原因类

为了避免第一版过度发散，`pick_metal_heavy` 的静态重载失败只建模为三个可区分 cause class：

- `under_supported_load`
- `load_induced_slip`
- `alignment_coupled_overload`

它们分别主要对应：

- `load_support_margin < 0`
- `grip_hold_margin < 0`
- `pose_alignment_error` 高并拖累 `lift_reserve`

### 8.2 启用条件

第一版 counterfactual diagnosis 只在 heavy-static regime 启用，gating 条件固定为：

- posterior 显示 `mass/high` 或 `dynamic_load/high`
- 当前 transport mode 不是 high-speed 主导
- 当前风险集中在 `grasp` 或 `lift`
- `load_support_margin` 或 `lift_reserve` 明显不足

此设计刻意把静态重载与动态重载分开，避免第一版同时接管 `pick_metal_heavy_fast`。

### 8.3 候选干预集

第一版只允许在一个小而可解释的 intervention set 中做诊断：

- `+gripper_force`
- `+lift_force`
- `+transfer_force`
- `-transport_velocity`
- `+lift_clearance`
- `+lift_force & -transport_velocity`
- `+lift_force & +lift_clearance`
- `+gripper_force & +lift_force`

这组候选不是为了做全局最优搜索，而是为了回答：

- 当前最可能的失败原因是什么
- 哪个最小干预最可能翻转当前 suffix
- 该干预主要修复的是哪个 latent

### 8.4 选择策略

候选选择采用“最小充分干预”策略：

1. 先找能翻转关键 latent 的候选。
2. 再比较 suffix 成功增益。
3. 若多个候选增益接近，优先选改动最小者。
4. 避免无必要地同时提高多个参数。

这条策略的目的是避免诊断器退化成“大力出奇迹”的启发式参数放大器。

## 9. 模块职责与接口

### 9.1 `simulation/env.py`

职责改为：

- 阶段推进
- 前缀冻结
- 阶段观测生成
- 当前阶段终止 / replan 触发

对外仍保留 `(success, elapsed, info)` 返回形式，但 `info` 语义改成：

- `phase_execution_trace`
- `observation_trace`
- `belief_update_trace`
- `counterfactual_replan_trace`
- `frozen_prefix_plan`
- `terminal_suffix_plan`
- `execution_feedback_mode`
- `post_failure_retry_count`

### 9.2 `simulation/control_core.py`

职责改为：

- `build_execution_prior(...)`
- `apply_phase_observation(...)`
- `diagnose_failure_cause(...)`
- `repair_suffix_plan(...)`

增加的核心对象：

- `PhaseObservation`
- `ExecutionBelief`
- `CounterfactualIntervention`
- `CounterfactualDiagnosis`

### 9.3 `simulation/feedback.py`

职责降级为：

- 旧接口兼容
- trial-level retry 请求适配
- trace 格式化与序列化辅助

它不再拥有在线失败诊断的主导权。

### 9.4 `simulation/rag_controller.py`

职责保持为：

- 文本证据检索
- seed plan 生成
- prior constraint synthesis

执行中在线修复逻辑不再继续主要堆在该文件内。

### 9.5 `simulation/runner.py`

新增或强化以下统计：

- `online_replan_success_rate`
- `suffix_counterfactual_replan_count`
- `post_failure_retry_count`
- `heavy_load_diagnosis_count`
- `diagnosed_cause_distribution`
- `selected_intervention_distribution`

trial records 需要能直接看出：

- 哪些 trial 靠在线 suffix repair 翻盘
- 哪些 trial 最终仍依赖 post-failure retry
- `pick_metal_heavy` 的翻盘主要由哪些干预主导

## 10. 测试设计

### 10.1 执行内核测试优先

新增测试重心必须落在：

- `env` 级执行语义
- `control_core` 级 posterior update
- `counterfactual` 级最小充分干预

而不是再次主要长在：

- schema
- defaults
- reporting
- import boundary

### 10.2 测试层次

#### A. `env` 级

目标：证明 replan 是 prefix freeze + suffix repair，而不是整条 reroll。

关键断言：

- `approach / grasp` 完成后不会被重放
- `lift` 触发后只改 `lift / transfer / place`
- trace 中能看到冻结前缀与新 suffix 的边界

#### B. `control_core` 级

目标：证明 posterior update 真在修 latent。

关键断言：

- 重载负观测拉低 `load_support_margin` / `lift_reserve`
- 对齐误差主要影响 `pose_alignment_error`
- update 不能只体现为 priority 变化

#### C. `counterfactual` 级

目标：证明诊断器选的是最小充分干预。

关键断言：

- 对 heavy-static fixture，若 `+lift_force` 能翻转而 `+gripper_force` 不能，必须优先选前者
- 若单变量不够、二元组合刚好足够，必须选最小组合
- 不允许无原因地把无关参数一起改掉

#### D. benchmark 回归测试

目标：证明不是专项作弊导致全局退化。

关键断言：

- `pick_metal_heavy` 明显高于当前基线
- `pick_metal_heavy_fast` 的绝对成功率下降不得超过 `0.05`
- 12 任务总体平均成功率下降不得超过 `0.01`

## 11. 迁移策略

### 11.1 第 0 步：先立语义对象

先增加：

- `PhaseObservation`
- `ExecutionBelief`
- `CounterfactualDiagnosis`

并把这些字段写进 trial records，但暂不要求主 benchmark 立刻提分。

### 11.2 第 1 步：只在 heavy-static regime 打开新主链

只对静态重载失败启用：

- posterior diagnosis
- counterfactual suffix repair

其余任务继续沿用现有逻辑或兼容调用。

这样可以把实验解释压缩成：

- 这是针对最关键失败区的定点因果修复
- 不是全局乱改

### 11.3 第 2 步：验证通过后再扩到通用 regime

只有在 heavy-static 验证通过后，才考虑扩展到：

- `large_part_far`
- `smooth_metal_fast`
- 其它需要更强在线恢复的任务

在此之前，不对外声称系统已经成为“通用在线闭环控制器”。

## 12. 风险与缓解

### 12.1 最大风险：新旧语义混跑

最危险的情况不是实现失败，而是：

- 新 `observation_trace` 已加，但旧 `observer_trace` 仍在主链起决定作用
- 新 suffix repair 已接入，但多数成功仍靠旧 full-plan reroll 或 post-failure retry
- posterior latent update 已存在，但最终决策仍主要由 priority / coverage 规则驱动

缓解策略：

- 明确新主链与兼容层的权责
- trace 中显式区分 `observer_only`、`suffix_counterfactual_replan`、`post_failure_retry`
- 在 benchmark 中显式统计在线诊断与 retry 的占比

### 12.2 次要风险：把 `pick_metal_heavy_fast` 一起打坏

因为 `pick_metal_heavy_fast` 当前并非主失败点，第一版若让静态重载干预直接接管动态任务，最容易引入副作用。

缓解策略：

- 第一版严格使用 heavy-static gating
- `pick_metal_heavy_fast` 只作为非退化约束

## 13. 最终设计结论

本轮设计的核心结论是：

- 用阶段化观测、显式执行潜变量和最小充分反事实干预，把 `pick_metal_heavy` 的静态重载失败从“规则后处理 + 整计划重算”改造成“posterior diagnosis + suffix repair”，同时约束 `pick_metal_heavy_fast` 与全局 benchmark 不退化。

这一定义同时约束了：

- 执行语义
- 诊断语义
- 测试语义
- benchmark 验收语义

它是第一版最严谨、最不容易被质疑成专项规则补丁的落地方向。
