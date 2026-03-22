"""
Benchmark 任务定义，参考论文的评测协议。
- GRASPA 1.0: 固定场景、成功/失败二值、可重复性
- FMB: 多阶段操作、泛化性
- REPLAB: Success rate、completion time
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from .rag_controller import GROUND_TRUTH_PARAMS


@dataclass(frozen=True)
class ObjectProfile:
    """任务物体的简化属性，用于成功率和模拟执行时间建模。"""

    name: str
    mass_kg: float
    surface_friction: float
    fragility: float
    velocity_scale: float
    target_tolerance: float
    size_xyz: tuple[float, float, float]
    preferred_approach_height: float
    approach_height_tolerance: float


@dataclass
class TaskConfig:
    """单个 benchmark 任务配置。"""
    task_id: str
    description: str          # 任务描述，用于 RAG 查询
    object_type: str         # 物体类型，对应知识库
    object_pos: tuple        # 初始位置 (x,y,z)
    target_pos: tuple        # 目标位置
    ideal_gripper_force: tuple  # (min, max) N，来自知识库
    split: str = "train"     # train / val / test，用于泛化实验
    profile: ObjectProfile | None = None


# 预定义任务，与 mechanical_data.txt 中的知识对应；含 train/val/test 划分
BENCHMARK_TASKS: List[TaskConfig] = [
    TaskConfig(
        task_id="pick_smooth_metal",
        description="抓取光滑金属零件",
        object_type="光滑金属零件",
        object_pos=(0.5, 0.05, 0.12),
        target_pos=(0.5, -0.15, 0.12),
        ideal_gripper_force=(30, 50),
        split="train",
        profile=ObjectProfile(
            name="smooth_metal",
            mass_kg=0.18,
            surface_friction=0.18,
            fragility=0.75,
            velocity_scale=0.82,
            target_tolerance=0.045,
            size_xyz=(0.025, 0.025, 0.025),
            preferred_approach_height=0.05,
            approach_height_tolerance=0.025,
        ),
    ),
    TaskConfig(
        task_id="pick_rubber",
        description="抓取橡胶零件",
        object_type="橡胶零件",
        object_pos=(0.48, 0.08, 0.12),
        target_pos=(0.52, -0.12, 0.12),
        ideal_gripper_force=(5, 15),
        split="train",
        profile=ObjectProfile(
            name="rubber",
            mass_kg=0.08,
            surface_friction=0.82,
            fragility=0.45,
            velocity_scale=0.72,
            target_tolerance=0.04,
            size_xyz=(0.022, 0.022, 0.022),
            preferred_approach_height=0.03,
            approach_height_tolerance=0.015,
        ),
    ),
    TaskConfig(
        task_id="pick_small_part",
        description="抓取小型机械零件",
        object_type="小型机械零件",
        object_pos=(0.5, 0.0, 0.12),
        target_pos=(0.5, -0.18, 0.12),
        ideal_gripper_force=(5, 15),
        split="train",
        profile=ObjectProfile(
            name="small_part",
            mass_kg=0.03,
            surface_friction=0.36,
            fragility=0.68,
            velocity_scale=0.88,
            target_tolerance=0.035,
            size_xyz=(0.018, 0.018, 0.018),
            preferred_approach_height=0.04,
            approach_height_tolerance=0.012,
        ),
    ),
    TaskConfig(
        task_id="pick_large_part",
        description="抓取大型零件",
        object_type="大型零件",
        object_pos=(0.5, 0.1, 0.12),
        target_pos=(0.5, -0.1, 0.12),
        ideal_gripper_force=(30, 50),
        split="train",
        profile=ObjectProfile(
            name="large_part",
            mass_kg=0.32,
            surface_friction=0.42,
            fragility=0.7,
            velocity_scale=0.58,
            target_tolerance=0.05,
            size_xyz=(0.032, 0.032, 0.028),
            preferred_approach_height=0.06,
            approach_height_tolerance=0.015,
        ),
    ),
    # 扩展任务（P1）：新物体类型，用于泛化 / val-test
    TaskConfig(
        task_id="pick_thin_wall",
        description="抓取薄壁件",
        object_type="薄壁件",
        object_pos=(0.49, 0.06, 0.12),
        target_pos=(0.51, -0.14, 0.12),
        ideal_gripper_force=(5, 12),
        split="val",
        profile=ObjectProfile(
            name="thin_wall",
            mass_kg=0.06,
            surface_friction=0.28,
            fragility=0.22,
            velocity_scale=0.52,
            target_tolerance=0.032,
            size_xyz=(0.03, 0.03, 0.012),
            preferred_approach_height=0.02,
            approach_height_tolerance=0.008,
        ),
    ),
    TaskConfig(
        task_id="pick_metal_heavy",
        description="抓取重型光滑金属零件",
        object_type="光滑金属零件",
        object_pos=(0.52, -0.02, 0.12),
        target_pos=(0.48, -0.18, 0.12),
        ideal_gripper_force=(30, 50),
        split="test",
        profile=ObjectProfile(
            name="metal_heavy",
            mass_kg=0.42,
            surface_friction=0.16,
            fragility=0.82,
            velocity_scale=0.48,
            target_tolerance=0.05,
            size_xyz=(0.035, 0.035, 0.03),
            preferred_approach_height=0.05,
            approach_height_tolerance=0.012,
        ),
    ),
]
