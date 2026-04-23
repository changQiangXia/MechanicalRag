"""
Benchmark 任务定义，参考论文的评测协议。
- GRASPA 1.0: 固定场景、成功/失败二值、可重复性
- FMB: 多阶段操作、泛化性
- REPLAB: Success rate、completion time
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

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
    reference_force_range: tuple  # (min, max) N，仅用于分析输出，不参与环境成功判定
    split: str = "train"     # train / val / test，用于泛化实验
    profile: ObjectProfile | None = None
    challenge_tags: tuple[str, ...] = ()


# 预定义任务，与 mechanical_data.txt 中的知识对应；含更均衡的 train/val/test 划分
BENCHMARK_TASKS: List[TaskConfig] = [
    TaskConfig(
        task_id="pick_smooth_metal",
        description="抓取光滑金属零件",
        object_type="光滑金属零件",
        object_pos=(0.5, 0.05, 0.12),
        target_pos=(0.5, -0.15, 0.12),
        reference_force_range=(30, 50),
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
        challenge_tags=("low_friction", "fragile"),
    ),
    TaskConfig(
        task_id="pick_rubber",
        description="抓取橡胶零件",
        object_type="橡胶零件",
        object_pos=(0.48, 0.08, 0.12),
        target_pos=(0.52, -0.12, 0.12),
        reference_force_range=(5, 15),
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
        challenge_tags=("high_friction", "compliant"),
    ),
    TaskConfig(
        task_id="pick_small_part",
        description="抓取小型机械零件",
        object_type="小型机械零件",
        object_pos=(0.5, 0.0, 0.12),
        target_pos=(0.5, -0.18, 0.12),
        reference_force_range=(5, 15),
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
        challenge_tags=("small", "precision"),
    ),
    TaskConfig(
        task_id="pick_large_part",
        description="抓取大型零件",
        object_type="大型零件",
        object_pos=(0.5, 0.1, 0.12),
        target_pos=(0.5, -0.1, 0.12),
        reference_force_range=(30, 50),
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
        challenge_tags=("large", "heavy"),
    ),
    TaskConfig(
        task_id="pick_smooth_metal_fast",
        description="高速搬运光滑金属零件",
        object_type="光滑金属零件",
        object_pos=(0.46, 0.11, 0.12),
        target_pos=(0.56, -0.18, 0.12),
        reference_force_range=(35, 50),
        split="train",
        profile=ObjectProfile(
            name="smooth_metal_fast",
            mass_kg=0.16,
            surface_friction=0.18,
            fragility=0.78,
            velocity_scale=1.02,
            target_tolerance=0.048,
            size_xyz=(0.025, 0.025, 0.024),
            preferred_approach_height=0.05,
            approach_height_tolerance=0.02,
        ),
        challenge_tags=("low_friction", "fragile", "high_speed"),
    ),
    TaskConfig(
        task_id="pick_small_part_fast",
        description="高速搬运小型机械零件",
        object_type="小型机械零件",
        object_pos=(0.47, 0.02, 0.12),
        target_pos=(0.56, -0.2, 0.12),
        reference_force_range=(8, 18),
        split="train",
        profile=ObjectProfile(
            name="small_part_fast",
            mass_kg=0.028,
            surface_friction=0.34,
            fragility=0.66,
            velocity_scale=1.15,
            target_tolerance=0.03,
            size_xyz=(0.018, 0.018, 0.017),
            preferred_approach_height=0.04,
            approach_height_tolerance=0.01,
        ),
        challenge_tags=("small", "precision", "high_speed"),
    ),
    TaskConfig(
        task_id="pick_thin_wall",
        description="抓取薄壁件",
        object_type="薄壁件",
        object_pos=(0.49, 0.06, 0.12),
        target_pos=(0.51, -0.14, 0.12),
        reference_force_range=(5, 12),
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
        challenge_tags=("thin_wall", "fragile"),
    ),
    TaskConfig(
        task_id="pick_rubber_fast",
        description="高速搬运橡胶零件",
        object_type="橡胶零件",
        object_pos=(0.46, 0.1, 0.12),
        target_pos=(0.57, -0.17, 0.12),
        reference_force_range=(8, 18),
        split="val",
        profile=ObjectProfile(
            name="rubber_fast",
            mass_kg=0.09,
            surface_friction=0.78,
            fragility=0.48,
            velocity_scale=0.95,
            target_tolerance=0.04,
            size_xyz=(0.023, 0.023, 0.023),
            preferred_approach_height=0.03,
            approach_height_tolerance=0.015,
        ),
        challenge_tags=("high_friction", "compliant", "high_speed"),
    ),
    TaskConfig(
        task_id="pick_large_part_far",
        description="长距离搬运大型零件",
        object_type="大型零件",
        object_pos=(0.43, 0.12, 0.12),
        target_pos=(0.59, -0.2, 0.12),
        reference_force_range=(32, 48),
        split="val",
        profile=ObjectProfile(
            name="large_part_far",
            mass_kg=0.34,
            surface_friction=0.44,
            fragility=0.72,
            velocity_scale=0.56,
            target_tolerance=0.052,
            size_xyz=(0.033, 0.033, 0.029),
            preferred_approach_height=0.06,
            approach_height_tolerance=0.015,
        ),
        challenge_tags=("large", "heavy", "long_transfer"),
    ),
    TaskConfig(
        task_id="pick_metal_heavy",
        description="抓取重型光滑金属零件",
        object_type="光滑金属零件",
        object_pos=(0.52, -0.02, 0.12),
        target_pos=(0.48, -0.18, 0.12),
        reference_force_range=(30, 50),
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
        challenge_tags=("low_friction", "heavy", "fragile"),
    ),
    TaskConfig(
        task_id="pick_thin_wall_fast",
        description="高速搬运薄壁件",
        object_type="薄壁件",
        object_pos=(0.47, 0.08, 0.12),
        target_pos=(0.57, -0.16, 0.12),
        reference_force_range=(8, 16),
        split="test",
        profile=ObjectProfile(
            name="thin_wall_fast",
            mass_kg=0.055,
            surface_friction=0.27,
            fragility=0.2,
            velocity_scale=0.72,
            target_tolerance=0.034,
            size_xyz=(0.03, 0.03, 0.012),
            preferred_approach_height=0.02,
            approach_height_tolerance=0.008,
        ),
        challenge_tags=("thin_wall", "fragile", "high_speed"),
    ),
    TaskConfig(
        task_id="pick_metal_heavy_fast",
        description="高速搬运重型光滑金属零件",
        object_type="光滑金属零件",
        object_pos=(0.44, -0.01, 0.12),
        target_pos=(0.58, -0.21, 0.12),
        reference_force_range=(38, 50),
        split="test",
        profile=ObjectProfile(
            name="metal_heavy_fast",
            mass_kg=0.45,
            surface_friction=0.15,
            fragility=0.84,
            velocity_scale=0.6,
            target_tolerance=0.052,
            size_xyz=(0.036, 0.036, 0.03),
            preferred_approach_height=0.05,
            approach_height_tolerance=0.012,
        ),
        challenge_tags=("low_friction", "heavy", "fragile", "high_speed"),
    ),
]
