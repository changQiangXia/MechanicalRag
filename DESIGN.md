**Mechanical Embodied RAG 项目设计说明**

本文档说明当前版本的设计思路，重点围绕问答链路、仿真链路、评测方式和结果组织方式展开。

**一、项目目标**

项目要解决的是一个很具体的问题：把机械知识库中的设备参数、流程知识和物理先验，转成可以直接用于问答和仿真控制的有效信息。

当前目标分成两部分：

- 问答部分：验证不同 RAG 设计对机械问题回答质量的影响
- 仿真部分：验证知识驱动的参数选择能否提升抓取任务成功率

**二、整体结构**

整体链路可以概括为下面这条路径：

```text
模型与缓存
        ├── direct_llm
        └── base_rag / improved_rag / problem_solving_rag

机械知识库 mechanical_data.txt
        └── base_rag / improved_rag / problem_solving_rag

结构化评测：qa_dataset.py + rag_evaluation.py
        ↓
仿真模块：tasks.py + env.py + rag_controller.py / baseline_controller.py
        ↓
benchmark：simulation.benchmark
        ↓
图表与摘要：visualize_results.py + generate_showcase.py
```

这样的组织方式有两个好处：

- 同一份知识库可以同时服务问答和仿真，整个项目逻辑更统一
- 结果文件、图表文件和展示摘要都来自同一套结构化输出，汇报时更稳定

**三、问答系统设计**

当前问答部分保留了四条链路，其中 `direct_llm` 是纯模型 baseline，另外三条链路会读取机械知识库。

`direct_llm`

- 直接调用模型回答机械问题
- 不经过检索，也不读取向量库
- 作用是给整套链路补上一个严谨的 baseline

`base_rag`

- 标准的分块、向量化、相似度检索、LLM 生成流程
- 作用是提供最基础的 RAG 对照版本

`improved_rag`

- 对知识库按条目建立结构化索引
- 检索阶段加入意图匹配、查询扩展、词项命中数和类别偏好
- 回答阶段优先做抽取式回答，减少跨材料范围混写和整段照搬

`problem_solving_rag`

- 在 `improved_rag` 的基础上继续强化规则与模板
- 主要用于保留定向问题修复后的规则与模板
- 当前 8 题上与 `improved_rag` 同分，更适合用于记录修复思路

四条链路一起评测的价值很直接：

- `direct_llm` 用来说明纯模型回答的波动
- `base_rag` 用来说明基础检索增强的收益和局限
- `improved_rag` 用来说明结构化检索与答案约束的收益
- `problem_solving_rag` 用来说明问题驱动优化的最终效果

**四、问答评测设计**

问答评测由 `qa_dataset.py` 和 `rag_evaluation.py` 共同完成。

每道题都带有三类信息：

- `required_groups`：回答中必须覆盖的关键信息
- `evidence_groups`：检索片段中应该命中的证据
- `forbidden_keywords`：与知识库冲突的错误表达

评分方式分成两层：

- 检索层：看证据是否命中
- 回答层：看要点是否完整、是否出现冲突词

这样做的原因是机械题很容易出现“看起来像回答了，细节其实错位”的情况。单看文本表面流畅度，很难判断回答质量。

**五、仿真系统设计**

仿真部分由五个核心模块组成。

`simulation/tasks.py`

- 定义 6 个 benchmark 任务
- 当前包含 `train / val / test` 划分
- 每个任务都给出理想夹爪力范围和推荐接近高度

`simulation/env.py`

- 负责执行 pick-and-place
- 在有 MuJoCo 时运行物理仿真
- 没有 MuJoCo 时走降级模式，保持评测流程可运行

`simulation/rag_controller.py`

- 把任务描述映射成控制参数
- 当前支持单 query、多 query、随机文档和 LLM 结构化输出等路径

`simulation/baseline_controller.py`

- 提供 `direct_llm`、`fixed`、`random` 等基线参数来源

`simulation/benchmark.py`

- 负责多任务、多次试验、多方法对比
- 当前支持单方法运行、直接 LLM 对比、多 seed 汇总、检索消融、LLM 对比、学习模块对比

**六、稳定性与可复现设计**

这一版项目专门加强了可复现性。

- 模型下载统一走 ModelScope
- `simulation/seed_utils.py` 处理跨进程稳定种子
- `simulation_benchmark_result.json` 改成多 seed 汇总格式
- `run_all.sh` 统一产出评测、benchmark、图表和展示摘要

这套设计能避免两个常见问题：

- 同一命令重复运行时结果漂移
- 汇报阶段引用的 JSON、TXT、PNG 来自不同批次运行

**七、结果组织方式**

当前结果组织成四层：

- 文本报告：`rag_evaluation_report.txt`、`rag_problems.txt`、`showcase_summary.txt`
- 结构化数据：`qa_evaluation_detail.json`、`simulation_benchmark_result.json`、`simulation_comparison_multi_seed.json`
- 图表：`visualizations/`
- 归档：`reproduction(3.22)/original` 与 `reproduction(3.22)/improved`

这样分层后，代码复现、写文档和做答辩展示都更方便。

**八、当前版本的设计重点**

当前版本最强调三件事：

- 对比链路完整，已经补齐 `direct_llm` baseline
- 仿真结果可直接汇报，已经有多 seed 汇总和图表支持
- 文档与结果同步，运行入口和输出文件保持一致
