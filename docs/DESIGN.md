# MechanicalRag 设计文档

## 1. 设计目标

本次重构围绕四个目标展开：

1. 把仓库从“展示包”收束成“可维护工程骨架”。
2. 把 QA 从平行脚本堆叠改成统一证据驱动 pipeline。
3. 把仿真从“控制器与评测器共享真值”改成“控制器生成参数、环境独立判定”。
4. 把文档、脚本、输出、归档分层，降低根目录噪声。

## 2. 当前状态说明

本设计文档描述仓库主线如何整理成可维护工程骨架，也记录当前有效版本的边界。

截至 `2026-04-27 UTC`：

- 当前仓库代码语义对应 `outputs/current_observer_step_replan/` 这次 observer / step-replan 闭环验证
- `outputs/current/` 继续保留为 `round19` 历史基线
- `round20` 作为 placement-stage 实验归档保留在 `outputs/current_round20_sim/`
- README、overview、simulation 文档需要围绕“历史基线 + 最新 observer-step-replan 闭环”两层口径组织，而不是继续写成单一 `round19 current`

## 3. 顶层结构

```text
README.md              外部入口
docs/                  主文档、设计文档、归档文档
qa/                    QA 主模块
simulation/            仿真主模块
reporting/             可视化与摘要
scripts/               环境、自检、批处理脚本
outputs/               当前批次输出
archive/               历史快照与原始文档
```

设计原则：

- 根目录只保留一级入口和核心资产。
- 输出与源码分离。
- 历史材料不再和当前主线并列。
- 多版本能力尽量通过 mode / config 收束，而不是复制新脚本。

## 4. QA 设计

### 4.1 统一内核

`qa/pipeline.py` 现在是 QA 主内核。它把原先散落在 `improved_rag` 和 `problem_solving_rag` 里的逻辑收束为一个 `MechanicalQAPipeline`，并通过 `mode` 区分：

- `mode=improved`
- `mode=rule_heavy`

兼容层：

- `qa/problem_solving.py` 只是统一 pipeline 的 wrapper，不再维护独立实现。

### 4.2 显式三阶段

QA 流程被拆成三段：

1. 查询理解
   - 输出 `QueryPlan`
   - 包含问题类型、focus terms、object terms、query expansions、preferred categories
2. 证据选择
   - 统一从知识库条目向量库中检索
   - 显式产出 `evidence_trace`
3. 约束回答
   - 先尝试基于证据片段直接抽取 / 压缩
   - 只有必要时才走 LLM fallback

## 5. 仿真设计

### 5.1 边界重划

原先仿真链路的核心问题是：

- 控制器里有对象到理想参数的映射
- 任务定义里保留理想范围
- 环境成功判定直接吃理想范围
- feedback 又继续读理想范围

重构后边界变成：

- `simulation/rag_controller.py`
  - 只基于知识库证据和任务描述生成参数
  - 先做 evidence / belief 提炼，再把 seed synthesis 和 local solve 交给 `control_core`
  - 不再使用 benchmark 参考范围去 clamp 参数

- `simulation/control_core.py`
  - 把 evidence 提升为显式 `ObjectBeliefState`
  - 单独建模 `TaskConstraintSet`、`UncertaintyProfile` 与 `StageIntent`
  - 通过 `belief_constraint_synthesis` 先生成 seed，再用 lightweight solver 在多组 candidate control plan 中做选择
  - 向上层返回 `belief_state`、`seed_mode`、`seed_plan`、`belief_state_coverage`、`uncertainty_conservative_mode`、`solver_selected_candidate` 等诊断字段
  - 但它当前本质上仍是轻量 belief / uncertainty / local solve 层，不是完整观测后验估计器或真正的优化求解器

- `simulation/env.py`
  - 根据质量、摩擦、fragility、尺寸、接近高度等物体属性推导内部力学窗口
  - 成功判定不再接收外部 `ideal_force_range`
  - 向上层暴露执行期 `observer_trace`
  - 对长距离动态任务显式输出 `lift_hold_risk`、`transfer_sway_risk`、`placement_settle_risk`

- `simulation/feedback.py`
  - 根据环境反馈调参
  - 不再读取真值边界
  - 当前显式使用 stage-specific risk 做 step 级局部回调，而不是只依赖单一 `stability_score`

- `simulation/tasks.py`
  - `reference_force_range` 只用于分析输出
  - 不再作为执行逻辑输入

### 5.2 runner / reporting 分离

`simulation/benchmark.py` 现在只是 CLI wrapper。

真正的执行逻辑拆到：

- `simulation/runner.py`
- `simulation/reporting.py`

这样 benchmark 不再承担全部职责：

- runner 负责试验执行
- reporting 负责序列化和表格输出
- benchmark 负责参数解析和命令分发
- `simulation_benchmark_result.json` 现在使用 Schema V2：`seed_plan` 表示初始 planner proposal，`executed_plan_stats` 表示 task 级 terminal-plan 聚合，`planner_diagnostics` 单独暴露 belief / solver 统计
- `simulation_benchmark_trial_records.json` 作为配套明细文件，逐 task 输出 `trial_records`，并保留 seed 上下文、`observer_trace` 与 `step_replan_trace`

### 5.3 fallback

无 MuJoCo 时，runner 会回退到环境代理模型，仍保留：

- 参数生成
- 独立环境评分
- 多方法对比

但文档中会明确说明这不是完整物理仿真。

## 6. 文档与输出设计

长期文档骨架：

- `README.md`
- `docs/overview.md`
- `docs/DESIGN.md`

输出目录：

- `outputs/current_observer_step_replan/`
- `outputs/current/`
- `outputs/current_round20_sim/`

其中 `outputs/current_observer_step_replan/` 当前的核心 simulation 产物包括：

- `simulation_benchmark_result.json`
  - 多 seed task-level summary
  - 主体在 `methods.<method>` 下，避免再把终态控制计划压扁到旧 `params_used`
- `simulation_benchmark_trial_records.json`
  - 逐 task / 逐 seed 的 execution 明细
  - 每条 trial 保留 `terminal_plan`、`observer_trace`、`step_replan_trace`
- `simulation_comparison_rag_vs_baseline.json`
  - 单 seed 方法对比，使用嵌套 `methods.<method>` payload
- `simulation_comparison_multi_seed.json`
  - 多 seed 方法对比，同样使用嵌套 `methods.<method>` payload，并把 planner / executed-plan 聚合保留在 method payload 内

历史与归档：

- `archive/reproduction_3.22/`
- `archive/docs_raw/`
- `docs/archive/`

## 7. 当前取舍

- QA 仍然保留受控模板，但模板现在服务于“证据约束回答”，不再直接伪装成独立方法增益。
- 仿真仍然是简化控制问题，不是完整机械臂控制栈；但评测边界已经比之前干净。
- observer / step-replan 闭环已经补上，且 `rag_feedback` 相对 `rag` seed-only 路径有明显提升；但 static heavy-metal 任务仍完全失败。
- 更具体地说，当前 `belief_state` 主要仍是知识证据派生状态，执行期观测主要进入 feedback replan；`uncertainty_profile` 仍偏 coverage / gap 统计，solver 仍是轻量 local search。
- 为兼顾现有运行成本，仍保留 `direct_llm` / `fixed` / `rag_learned` 等基线，但它们都挂在统一 benchmark runner 上。
