"""
MuJoCo 机械臂仿真环境。
参考：GRASPA 1.0、FMB、REPLAB 等论文的 tabletop manipulation 与 success rate 评测。
"""

from __future__ import annotations

import random
import time
from pathlib import Path
from typing import Tuple

import numpy as np

try:
    import mujoco
    HAS_MUJOCO = True
except ImportError:
    HAS_MUJOCO = False


def _success_model(
    gripper_force: float,
    ideal_range: Tuple[float, float],
    *,
    surface_friction: float = 0.5,
    mass_kg: float = 0.05,
    fragility: float = 0.7,
    travel_distance: float = 0.2,
    approach_height: float = 0.05,
    preferred_approach_height: float = 0.05,
    approach_height_tolerance: float = 0.02,
    noise: float = 0.15,
) -> bool:
    """
    基于 RAG 知识的成功模型（参考论文中的 grasp quality / success 建模）。
    若 gripper_force 在理想范围内，成功率高；否则易滑脱或损坏。
    """
    lo, hi = ideal_range
    force_center = (lo + hi) / 2.0
    half_range = max(1.0, (hi - lo) / 2.0)
    center_offset = abs(gripper_force - force_center) / half_range
    if lo <= gripper_force <= hi:
        # 理想区间内部仍保留中心值偏好，边界参数稳定性更弱。
        p_success = 0.94 - random.uniform(0, noise) - 0.08 * min(1.0, center_offset)
    elif gripper_force < lo:
        p_success = 0.28 + (gripper_force / lo) * 0.22
    else:
        p_success = 0.48 - min(1.0, (gripper_force - hi) / 30) * 0.28
    p_success += max(-0.05, min(0.08, (surface_friction - 0.4) * 0.18))
    p_success -= max(0.0, (mass_kg - 0.1) * 0.20)
    p_success -= max(0.0, travel_distance - 0.18) * 0.22
    height_ratio = abs(approach_height - preferred_approach_height) / max(approach_height_tolerance, 1e-3)
    p_success -= min(0.18, max(0.0, height_ratio - 0.25) * 0.08)
    if gripper_force > hi:
        p_success -= max(0.0, (1.0 - fragility) * (gripper_force - hi) / 35.0)
    return random.random() < max(0.1, min(0.95, p_success))


def _scene_path() -> Path:
    return Path(__file__).resolve().parent / "scene.xml"


def _compute_transport_dynamics(
    mass_kg: float,
    surface_friction: float,
    fragility: float,
    velocity_scale: float,
) -> Tuple[float, float, float]:
    """
    估计平面运输阶段的控制强度。
    通过静摩擦补偿避免重物体和低速任务在固定步数内天然不可达。
    """
    static_push_estimate = mass_kg * 9.81 * (0.18 + surface_friction)
    force_gain = 48.0 * velocity_scale + 24.0 * static_push_estimate
    force_clip = 9.0 + 10.0 * velocity_scale + 8.5 * static_push_estimate + 3.5 * (1.0 - fragility)
    return force_gain, force_clip, static_push_estimate


class ArmSimEnv:
    """
    桌面抓取仿真环境（MuJoCo）。
    结合物理仿真与基于知识的成功模型，评估 RAG 驱动的参数选择。
    """

    def __init__(self, gui: bool = False, time_step: float = 1.0 / 120.0, seed: int | None = None):
        if not HAS_MUJOCO:
            raise ImportError("请安装 mujoco: pip install mujoco")
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        self.gui = gui
        self.time_step = time_step
        path = _scene_path()
        if not path.exists():
            raise FileNotFoundError(f"场景文件不存在: {path}")
        self.model = mujoco.MjModel.from_xml_path(str(path))
        self.model.opt.timestep = time_step
        self.data = mujoco.MjData(self.model)
        self._object_body_id = self.model.body("object").id
        self._object_geom_id = self.model.geom("box_geom").id
        # 自由关节的 qpos 布局: 前 3 个为位置，后 4 个为四元数 (w,x,y,z)
        self._qpos_pos_slice = slice(0, 3)
        self._qpos_quat_slice = slice(3, 7)
        self._viewer = None
        if gui:
            try:
                self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
            except Exception:
                self._viewer = None

    def _set_object_pose(self, pos: Tuple[float, float, float]) -> None:
        self.data.qpos[self._qpos_pos_slice] = np.array(pos, dtype=np.float64)
        self.data.qpos[self._qpos_quat_slice] = np.array([1, 0, 0, 0], dtype=np.float64)
        self.data.qvel[:] = 0
        mujoco.mj_forward(self.model, self.data)

    def _get_object_pos(self) -> np.ndarray:
        return self.data.qpos[self._qpos_pos_slice].copy()

    def _configure_object_profile(self, object_profile: dict | None) -> dict:
        profile = {
            "mass_kg": 0.05,
            "surface_friction": 0.5,
            "fragility": 0.7,
            "velocity_scale": 0.8,
            "target_tolerance": 0.04,
            "size_xyz": (0.025, 0.025, 0.025),
            "preferred_approach_height": 0.05,
            "approach_height_tolerance": 0.02,
        }
        if object_profile:
            profile.update(object_profile)
        self.model.body_mass[self._object_body_id] = float(profile["mass_kg"])
        self.model.geom_friction[self._object_geom_id] = np.array(
            [
                float(profile["surface_friction"]),
                max(0.03, float(profile["surface_friction"]) * 0.35),
                0.02,
            ],
            dtype=np.float64,
        )
        self.model.geom_size[self._object_geom_id] = np.array(profile["size_xyz"], dtype=np.float64)
        mujoco.mj_forward(self.model, self.data)
        return profile

    def execute_pick_place(
        self,
        object_pos: Tuple[float, float, float],
        target_pos: Tuple[float, float, float],
        gripper_force: float,
        ideal_force_range: Tuple[float, float],
        approach_height: float = 0.05,
        object_profile: dict | None = None,
        max_steps: int = 800,
    ) -> Tuple[bool, float, dict]:
        """
        执行 pick-and-place，成功与否由 RAG 参数与知识库理想范围共同决定。
        参考 FMB/REPLAB：success rate、completion time。
        """
        profile = self._configure_object_profile(object_profile)
        self._set_object_pose(object_pos)
        self.data.xfrc_applied[:] = 0
        start = time.time()
        step = 0
        travel_distance = float(np.linalg.norm(np.array(target_pos) - np.array(object_pos))) + 2.0 * approach_height
        mass_kg = float(profile["mass_kg"])
        surface_friction = float(profile["surface_friction"])
        fragility = float(profile["fragility"])
        velocity_scale = float(profile["velocity_scale"])
        target_tolerance = float(profile["target_tolerance"])
        preferred_approach_height = float(profile["preferred_approach_height"])
        approach_height_tolerance = float(profile["approach_height_tolerance"])

        # 静摩擦补偿让大件、薄壁件和重金属件都保留难度差异，同时具备完成条件。
        force_gain, force_clip, static_push_estimate = _compute_transport_dynamics(
            mass_kg,
            surface_friction,
            fragility,
            velocity_scale,
        )

        for step in range(max_steps):
            pos = self._get_object_pos()
            tgt = np.array(target_pos, dtype=np.float64)
            d = np.linalg.norm(tgt - pos)
            if d < target_tolerance:
                break
            force = (tgt - pos) * force_gain
            force = np.clip(force, -force_clip, force_clip)
            self.data.xfrc_applied[self._object_body_id, :3] = force
            mujoco.mj_step(self.model, self.data)
            self.data.xfrc_applied[self._object_body_id, :3] = 0
            if self._viewer is not None and self._viewer.is_running():
                self._viewer.sync()

        wall_elapsed = time.time() - start
        pos = self._get_object_pos()
        dist = float(np.linalg.norm(pos - np.array(target_pos)))
        physics_ok = dist < target_tolerance + 0.02
        success = physics_ok and _success_model(
            gripper_force,
            ideal_force_range,
            surface_friction=surface_friction,
            mass_kg=mass_kg,
            fragility=fragility,
            travel_distance=travel_distance,
            approach_height=approach_height,
            preferred_approach_height=preferred_approach_height,
            approach_height_tolerance=approach_height_tolerance,
        )
        # 使用模拟执行时间，避免墙钟时间过小导致各任务看起来没有差异。
        lift_speed = max(0.05, 0.22 * velocity_scale / (1.0 + mass_kg * 1.8))
        sim_elapsed = (min(step + 1, max_steps) * self.time_step) + (2.0 * approach_height / lift_speed)

        # 反馈环节：失败时给出力偏差提示，供 RAG 反馈闭环使用
        lo, hi = ideal_force_range
        force_likely_low = (not success) and (gripper_force < lo)
        force_likely_high = (not success) and (gripper_force > hi)
        info = {
            "distance": dist,
            "steps": min(step + 1, max_steps),
            "sim_time": round(sim_elapsed, 4),
            "wall_time": round(wall_elapsed, 4),
            "travel_distance": round(travel_distance, 4),
            "force_gain": round(force_gain, 4),
            "force_clip": round(force_clip, 4),
            "static_push_estimate": round(static_push_estimate, 4),
            "approach_height_error": round(abs(approach_height - preferred_approach_height), 4),
            "force_likely_low": force_likely_low,
            "force_likely_high": force_likely_high,
            "profile": profile,
        }
        return success, sim_elapsed, info

    def close(self) -> None:
        if self._viewer is not None:
            try:
                self._viewer.close()
            except Exception:
                pass
            self._viewer = None


if not HAS_MUJOCO:

    class ArmSimEnv:
        """
        降级模式：无 MuJoCo 时仅用 RAG 成功模型 + 模拟耗时，不跑物理。
        接口与完整版一致，benchmark 可照常运行。
        """

        def __init__(self, gui: bool = False, time_step: float = 1.0 / 120.0, seed: int | None = None):
            if seed is not None:
                random.seed(seed)
                try:
                    np.random.seed(seed)
                except Exception:
                    pass
            self._mock = True

        def execute_pick_place(
            self,
            object_pos: Tuple[float, float, float],
            target_pos: Tuple[float, float, float],
            gripper_force: float,
            ideal_force_range: Tuple[float, float],
            approach_height: float = 0.05,
            object_profile: dict | None = None,
            max_steps: int = 400,
        ) -> Tuple[bool, float, dict]:
            profile = {
                "mass_kg": 0.05,
                "surface_friction": 0.5,
                "fragility": 0.7,
                "velocity_scale": 0.8,
                "target_tolerance": 0.04,
                "preferred_approach_height": 0.05,
                "approach_height_tolerance": 0.02,
            }
            if object_profile:
                profile.update(object_profile)
            dist = sum((a - b) ** 2 for a, b in zip(object_pos, target_pos)) ** 0.5
            travel_distance = dist + 2.0 * approach_height
            lift_speed = max(0.05, 0.22 * profile["velocity_scale"] / (1.0 + profile["mass_kg"] * 1.8))
            elapsed = travel_distance / lift_speed
            success = _success_model(
                gripper_force,
                ideal_force_range,
                surface_friction=profile["surface_friction"],
                mass_kg=profile["mass_kg"],
                fragility=profile["fragility"],
                travel_distance=travel_distance,
                approach_height=approach_height,
                preferred_approach_height=profile["preferred_approach_height"],
                approach_height_tolerance=profile["approach_height_tolerance"],
            )
            lo, hi = ideal_force_range
            force_likely_low = (not success) and (gripper_force < lo)
            force_likely_high = (not success) and (gripper_force > hi)
            info = {
                "distance": 0.0 if success else dist,
                "steps": max_steps,
                "sim_time": round(elapsed, 4),
                "travel_distance": round(travel_distance, 4),
                "approach_height_error": round(abs(approach_height - profile["preferred_approach_height"]), 4),
                "force_likely_low": force_likely_low,
                "force_likely_high": force_likely_high,
                "profile": profile,
            }
            return success, elapsed, info

        def close(self) -> None:
            pass
