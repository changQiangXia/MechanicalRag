# 术语与机制说明

这份文档集中解释当前主文档里高频出现的控制链术语、执行日志字段和结果统计字段。正文中的短解释只负责帮助读者顺着主线读下去，这里负责把项目内定义集中放在一个位置。

<a id="control-chain-terms"></a>
## 控制链术语

### `belief_state`

`belief_state` 是控制器根据检索证据整理出来的对象状态描述。它会把任务里和质量、摩擦、脆弱性、速度偏好有关的信息收成统一表达，供后面的 `seed_plan` 和 solver 使用。在 `pick_metal_heavy` 这类任务里，它会把“重载、低摩擦、需要更稳的支撑”这类证据提前带入控制计划生成阶段。

### `task_constraints`

`task_constraints` 是控制器从任务文本和证据里抽出来的执行约束集合。它负责告诉后续阶段哪些参数更需要保守，哪些阶段需要额外注意稳定性、精度或速度。

### `uncertainty_profile`

`uncertainty_profile` 用来记录当前证据覆盖程度、冲突程度和保守模式是否开启。它不会单独产出控制参数，但会影响 `seed_plan` 和 solver 在候选方案之间的取舍方式。

### `seed_plan`

`seed_plan` 是进入局部求解前的第一版控制计划。可以把它理解成“基于证据和状态整理出来的初稿”，后面的 solver 会在这个初稿附近继续修正参数。

### `solver_selected_candidate`

`solver_selected_candidate` 表示当前 trial 最终选中了哪一种候选修正方向。它反映的是局部搜索或保守修正在当前任务上更偏向哪一类参数调整。

### `suffix_counterfactual_replan`

`suffix_counterfactual_replan` 表示执行中已经拿到新的阶段观测后，只对后续阶段的控制计划进行局部重算。它出现在 `phase observation -> posterior update -> suffix repair` 这条链的后半段，用来把新的风险信息推回到后续参数选择中。

<a id="execution-trace-fields"></a>
## 执行日志字段

### `phase_execution_trace`

`phase_execution_trace` 记录一次执行在各个阶段的推进情况。它把接近、抓取、起吊、转运、放置这些阶段按顺序展开，方便读者判断失败或修正发生在什么位置。

### `observation_trace`

`observation_trace` 记录各阶段拿到的观测摘要。它负责把执行过程中的风险和状态变化从环境层带回控制链，成为后续更新的重要输入。

### `belief_update_trace`

`belief_update_trace` 记录观测进入后，控制器内部状态描述发生了哪些更新。它连接了“环境看到了什么”和“控制器准备如何理解这些信息”这两部分。

### `counterfactual_replan_trace`

`counterfactual_replan_trace` 记录触发局部修正后，系统尝试了哪一类后续阶段修复思路。它最适合和 `observation_trace` 一起看，用来理解一次修正是由什么阶段现象引起的。

### `observer_trace`

`observer_trace` 是保留给旧结构兼容使用的观测摘要字段。当前主执行日志已经细分成多类 trace，但这个字段仍方便老结果或老工具继续读取基础观测信息。

<a id="result-stat-fields"></a>
## 结果统计字段

### `observer-only ablation`

`observer-only ablation` 是保留观测日志但关闭在线修复的对照方法。它最有用的地方是把“系统是否看到了执行期信息”和“系统是否真的根据这些信息改了计划”区分开来。

### `executed_plan_stats`

`executed_plan_stats` 是任务级汇总后的终态控制计划统计。它展示的是一组 trial 在执行结束时实际落下来的参数分布，而不是单次 trial 的瞬时参数值。

### `online_diagnosis_count`

`online_diagnosis_count` 表示有多少次 trial 进入了在线诊断或在线修正路径。这个数适合和成功率、对照方法、具体任务一起看，用来判断执行中修正是否真的参与了当前方法。

### `suffix_counterfactual_replan_count`

`suffix_counterfactual_replan_count` 表示有多少次 trial 进入了后缀重算路径。它和 `online_diagnosis_count` 一起看时，能帮助读者判断在线修正是偶发情况还是当前任务上的常见路径。

### `post_failure_retry_count`

`post_failure_retry_count` 表示有多少次 trial 在一次执行结束后，仍需要依靠 trial 级重试来兜底。这个字段有助于区分“执行中已经修好”与“执行结束后重新再来一次”这两类恢复方式。

<a id="task-examples"></a>
## 代表性任务示例

### `pick_metal_heavy`

`pick_metal_heavy` 适合用来理解重载场景里的 `belief_state`、在线诊断和后缀修正。这个任务里，系统需要把重载、低摩擦、稳定搬运这些证据更早前推到参数生成阶段，因此它是观察 `belief_state -> seed_plan -> online diagnosis` 这条链最直接的例子。

### `pick_large_part_far`

`pick_large_part_far` 适合用来理解长距离搬运中的阶段化日志。这个任务更容易在起吊、转运和放置之间暴露不同阶段的风险，因此很适合配合 `phase_execution_trace`、`observation_trace` 和 `counterfactual_replan_trace` 一起看。
