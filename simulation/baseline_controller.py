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


def get_params_fixed(_task_description: str, seed: int | None = None) -> dict[str, Any]:
    """固定参数基线：不检索、不区分任务，始终返回相同夹爪力。"""
    return {
        "gripper_force": FIXED_GRIPPER_FORCE,
        "approach_height": FIXED_APPROACH_HEIGHT,
        "rag_source": "",
    }


def get_params_random(task_description: str, seed: int | None = 42) -> dict[str, Any]:
    """随机参数基线：在 [5, 50] N 内均匀采样，用于对比「无知识」时的表现。"""
    if seed is not None:
        rng = random.Random(seed + stable_seed_offset(task_description))
    else:
        rng = random.Random()
    force = 5.0 + rng.uniform(0, 45.0)  # [5, 50] N
    return {
        "gripper_force": round(force, 2),
        "approach_height": FIXED_APPROACH_HEIGHT,
        "rag_source": "",
    }


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

请直接给出夹爪力和接近高度，只输出一个 JSON：
{{"gripper_force": 数字, "approach_height": 数字}}
不要输出解释。
"""
    try:
        out = llm.invoke(prompt)
        text = out if isinstance(out, str) else str(out)
        parsed = _parse_json_params(text)
        if parsed is not None:
            force = float(parsed.get("gripper_force", FIXED_GRIPPER_FORCE))
            height = float(parsed.get("approach_height", FIXED_APPROACH_HEIGHT))
            return {
                "gripper_force": round(max(5.0, min(50.0, force)), 2),
                "approach_height": round(max(0.02, min(0.1, height)), 3),
                "rag_source": "",
            }
    except Exception:
        pass
    return get_params_fixed(task_description)
