# 仿真机械臂模块

基于 **MuJoCo** 的 RAG 驱动机械臂仿真，参考以下论文的评测方法：

## 参考论文与 Benchmark

| 论文/基准 | 参考内容 |
|-----------|----------|
| **GRASPA 1.0** | 固定场景、二值成功/失败、复合评分（算法 vs 平台约束） |
| **FMB** (Functional Manipulation Benchmark) | 多阶段操作、抓取-重定向-装配、泛化性评测 |
| **REPLAB** | 低成本可复现平台、92k 抓取尝试、success rate 统计 |
| **Multi-Object Grasping** | 多物体抓取协议、efficiency / selective grasping |

## 模块结构

```
simulation/
├── scene.xml       # MuJoCo 场景（地面、桌子、可移动方块）
├── env.py          # MuJoCo 仿真环境（桌面、物体、物理；返回失败时力偏差提示）
├── feedback.py     # 反馈环节：根据执行结果给出夹爪力调整建议并重试
├── rag_controller.py  # RAG → 控制参数（夹爪力、接近高度；含 get_params_after_feedback）
├── tasks.py        # 预定义任务（含 train/val/test 划分）
├── benchmark.py    # 评测脚本：success rate、completion time、95% CI、baseline 对比
└── README.md
```

## 成功模型

仿真中的「成功」由两部分决定：
1. **物理**：物体是否到达目标区域（MuJoCo 仿真）
2. **知识**：RAG 提供的夹爪力是否在知识库的理想范围内（30–50N 光滑金属、5–15N 橡胶等）

若参数错误（如橡胶件用 50N），则 success 概率降低，体现 RAG 对控制的指导作用。
当前版本还会记录每个任务的 95% CI、理想力中心偏差、train/val/test split 以及多 seed 的 mean±std。

## 反馈环节（RAG + 机械臂闭环）

在 RAG 给出初参并执行一次抓取后，若仿真判定失败，会进入**反馈环节**：

1. **env** 返回本次执行的 `success`、`distance` 以及相对理想力范围的偏差提示（`force_likely_low` / `force_likely_high`）。
2. **feedback** 模块根据上述信号给出调整建议（`increase` / `decrease` 夹爪力），并生成微调后的参数。
3. 使用新参数**重试**一次抓取（可配置 `--max_feedback_retries`，默认 1 次）。

流程：**RAG 初参 → 执行 → 失败 → 反馈调整 → 再执行**，形成闭环，便于对比「仅 RAG」与「RAG+反馈」的成功率。

## 运行

```bash
# 无 GUI，快速评测（RAG 参数）
python -m simulation.benchmark --n_trials 10

# 仅跑固定/随机基线（无 RAG）
python -m simulation.benchmark --method fixed --n_trials 10 --output result_fixed.json
python -m simulation.benchmark --method random --n_trials 10 --output result_random.json

# RAG vs 无 RAG 对比（论文用）
python -m simulation.benchmark --compare --n_trials 5
# RAG vs 直接 LLM vs 固定基线
python -m simulation.benchmark --compare_direct_llm --n_trials 20
# 多 seed 对比（mean±std）
python -m simulation.benchmark --compare_multi_seed --n_trials 20 --seeds 42 43 44 --multi_seed_methods rag direct_llm fixed
# 检索消融：单 query / 多 query / 随机文档 / 固定
python -m simulation.benchmark --ablation_retrieval --n_trials 5
# LLM 结构化输出对比：RAG(规则) vs RAG+LLM(JSON) vs 固定（需加载 Qwen2，较慢）
python -m simulation.benchmark --compare_llm --n_trials 3
# P3 轻量学习模块（RAG 嵌入→MLP→夹爪力，含不确定性）
python -m simulation.benchmark --method rag_learned --n_trials 5
python -m simulation.benchmark --compare_learned --n_trials 3
# 反馈环节：RAG 初参 → 执行 → 失败则根据力偏差调整参数并重试
python -m simulation.benchmark --method rag_feedback --n_trials 5
python -m simulation.benchmark --compare_feedback --n_trials 3
# 可选：单独预训练模型（首次运行 rag_learned 也会自动训练）
python -m simulation.train_learned_model --data_path mechanical_data.txt
# 生成 simulation_comparison_rag_vs_baseline.json/.txt、simulation_comparison_multi_seed.json/.txt、simulation_ablation_retrieval.json/.txt

# 带 GUI 可视化（需已安装 mujoco）
python -m simulation.benchmark --n_trials 3 --gui --output result.json
```

## 降级模式（无 MuJoCo）

若本机未安装 mujoco，仿真会自动使用**降级模式**：
- 不跑 MuJoCo 物理，仅用 RAG 成功模型（夹爪力是否在知识库理想范围内）判定 success
- 接口与完整版一致，仍会生成 `simulation_benchmark_result.json`
- 运行时会提示：`【降级模式】未检测到 mujoco，结果基于 RAG 成功模型（无物理仿真）`

## 完整仿真（含 MuJoCo 物理）

- **Linux**：PyPI 有预编译 wheel，`pip install mujoco` 后即可运行。
- **macOS**：若 `pip install mujoco` 从源码编译失败（如 MUJOCO_PATH / 无对应 wheel），可：
  - 直接运行 `python -m simulation.benchmark`，会**自动使用降级模式**（仅 RAG 成功模型，无物理仿真），仍会生成结果文件；
  - 或使用下方 Docker 在 Linux 容器中跑完整 MuJoCo 仿真。

### 使用 Docker

**先决条件**：本机已安装 [Docker](https://docs.docker.com/get-docker/)（Mac 可用 `brew install --cask docker`，安装后打开 Docker Desktop 一次）。

在项目根目录执行：

```bash
cd /path/to/mechanical_rag_project

# 构建镜像
docker build -f Dockerfile.simulation -t mech-rag-sim .

# 运行完整仿真，结果写回当前目录
docker run --rm -v "$(pwd)":/out mech-rag-sim
```

结果会出现在本机的 `simulation_benchmark_result.json`。
