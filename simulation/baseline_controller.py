"""
无 RAG 基线：固定或随机参数，用于论文对比实验（RAG vs 无 RAG）。
"""

from __future__ import annotations

import random
from typing import Any

from .seed_utils import stable_seed_offset


# 固定基线：所有任务使用同一夹爪力（不依赖知识库）
FIXED_GRIPPER_FORCE = 25.0  # N，介于橡胶(5–15)与金属(30–50)之间，对各类任务均非最优
FIXED_APPROACH_HEIGHT = 0.05
FIXED_TRANSPORT_VELOCITY = 0.30
FIXED_LIFT_CLEARANCE = 0.06


def _clamp_plan(force: float, height: float, velocity: float, clearance: float) -> dict[str, Any]:
    return {
        "gripper_force": round(max(5.0, min(50.0, force)), 2),
        "approach_height": round(max(0.02, min(0.1, height)), 3),
        "transport_velocity": round(max(0.12, min(0.8, velocity)), 3),
        "lift_clearance": round(max(0.03, min(0.14, clearance)), 3),
        "rag_source": "",
    }


def get_params_fixed(_task_description: str, seed: int | None = None) -> dict[str, Any]:
    """固定参数基线：不检索、不区分任务，始终返回相同夹爪力。"""
    return _clamp_plan(
        FIXED_GRIPPER_FORCE,
        FIXED_APPROACH_HEIGHT,
        FIXED_TRANSPORT_VELOCITY,
        FIXED_LIFT_CLEARANCE,
    )


def get_params_random(task_description: str, seed: int | None = 42) -> dict[str, Any]:
    """随机参数基线：在 [5, 50] N 内均匀采样，用于对比「无知识」时的表现。"""
    if seed is not None:
        rng = random.Random(seed + stable_seed_offset(task_description))
    else:
        rng = random.Random()
    force = 5.0 + rng.uniform(0, 45.0)  # [5, 50] N
    return _clamp_plan(
        force,
        0.02 + rng.uniform(0.0, 0.06),
        0.12 + rng.uniform(0.0, 0.55),
        0.03 + rng.uniform(0.0, 0.08),
    )


def get_params_task_heuristic(task_description: str, seed: int | None = None) -> dict[str, Any]:
    """
    强启发式基线：仅根据任务文本做对象级控制计划猜测，不使用检索证据。
    该基线比 fixed/random 更强，但仍不接触 reference 真值。
    """
    desc = task_description
    if "高速" in desc and "薄壁" in desc:
        return _clamp_plan(11.0, 0.02, 0.22, 0.06)
    if "高速" in desc and "重型" in desc and "金属" in desc:
        return _clamp_plan(44.0, 0.055, 0.22, 0.095)
    if "高速" in desc and "金属" in desc:
        return _clamp_plan(40.0, 0.05, 0.34, 0.08)
    if "高速" in desc and "小型" in desc:
        return _clamp_plan(16.0, 0.04, 0.46, 0.055)
    if "高速" in desc and "橡胶" in desc:
        return _clamp_plan(13.0, 0.03, 0.38, 0.05)
    if "长距离" in desc and "大型" in desc:
        return _clamp_plan(32.0, 0.06, 0.2, 0.09)
    if "薄壁" in desc:
        return _clamp_plan(9.0, 0.025, 0.18, 0.05)
    if "橡胶" in desc:
        return _clamp_plan(10.0, 0.03, 0.30, 0.045)
    if "小型" in desc:
        return _clamp_plan(12.0, 0.04, 0.40, 0.05)
    if "重型" in desc and "金属" in desc:
        return _clamp_plan(38.0, 0.055, 0.18, 0.09)
    if "大型" in desc:
        return _clamp_plan(30.0, 0.06, 0.22, 0.085)
    if "金属" in desc:
        return _clamp_plan(34.0, 0.05, 0.24, 0.07)
    return _clamp_plan(
        FIXED_GRIPPER_FORCE,
        FIXED_APPROACH_HEIGHT,
        FIXED_TRANSPORT_VELOCITY,
        FIXED_LIFT_CLEARANCE,
    )


def _parse_json_params(text: str) -> dict[str, Any] | None:
    import json
    import re

    text = text.strip()
    try:
        obj = json.loads(text)
        if "gripper_force" in obj:
            return obj
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{[^{}]*\"gripper_force\"[^{}]*\}", text)
    if match:
        try:
            obj = json.loads(match.group(0))
            if "gripper_force" in obj:
                return obj
        except json.JSONDecodeError:
            return None
    return None


def get_params_llm_direct(task_description: str, llm: Any) -> dict[str, Any]:
    """
    直接 LLM 基线：不给检索上下文，仅根据任务描述生成控制参数。
    """
    prompt = f"""你现在负责机械臂抓取任务的参数设置。
任务描述：{task_description}

请直接给出夹爪力、接近高度、搬运速度、抬升净空，只输出一个 JSON：
{{"gripper_force": 数字, "approach_height": 数字, "transport_velocity": 数字, "lift_clearance": 数字}}
不要输出解释。
"""
    try:
        out = llm.invoke(prompt)
        text = out if isinstance(out, str) else str(out)
        parsed = _parse_json_params(text)
        if parsed is not None:
            force = float(parsed.get("gripper_force", FIXED_GRIPPER_FORCE))
            height = float(parsed.get("approach_height", FIXED_APPROACH_HEIGHT))
            velocity = float(parsed.get("transport_velocity", FIXED_TRANSPORT_VELOCITY))
            clearance = float(parsed.get("lift_clearance", FIXED_LIFT_CLEARANCE))
            return _clamp_plan(force, height, velocity, clearance)
    except Exception:
        pass
    return get_params_task_heuristic(task_description)
