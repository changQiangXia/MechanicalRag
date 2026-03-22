"""
机械臂 RAG 反馈环节：根据执行结果（成功/失败、夹爪力与理想范围）给出参数调整建议，
支持重试时自动微调夹爪力，形成「RAG 初参 → 执行 → 反馈 → 调整 → 再执行」闭环。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# 夹爪力合理范围，与 rag_controller 一致
FORCE_MIN, FORCE_MAX = 5.0, 50.0


@dataclass
class FeedbackSignal:
    """单次执行后的反馈信号。"""
    success: bool
    gripper_force: float
    ideal_force_range: tuple[float, float]
    distance: float  # 物体与目标距离，失败时可能偏大
    steps: int
    # 可选：由 env 或成功模型给出的失败原因提示
    force_likely_low: bool = False  # 疑似夹爪力不足（滑脱）
    force_likely_high: bool = False  # 疑似夹爪力过大（损坏/不稳定）


def suggest_force_adjustment(signal: FeedbackSignal) -> str:
    """
    根据反馈信号给出夹爪力调整建议。
    返回 "increase" | "decrease" | "none"。
    """
    if signal.success:
        return "none"
    lo, hi = signal.ideal_force_range
    force = signal.gripper_force
    if signal.force_likely_low or force < lo:
        return "increase"
    if signal.force_likely_high or force > hi:
        return "decrease"
    # 在范围内仍失败（可能是物理/位姿等）：小幅尝试增加力以提高抓稳概率
    if force <= (lo + hi) / 2:
        return "increase"
    return "decrease"


def adjust_params_by_feedback(
    params: dict[str, Any],
    suggestion: str,
    step: float = 5.0,
    force_min: float = FORCE_MIN,
    force_max: float = FORCE_MAX,
) -> dict[str, Any]:
    """
    根据建议微调参数字典（主要调整 gripper_force），用于重试。
    """
    out = dict(params)
    force = float(out.get("gripper_force", 25.0))
    if suggestion == "increase":
        force = min(force_max, force + step)
    elif suggestion == "decrease":
        force = max(force_min, force - step)
    out["gripper_force"] = round(force, 2)
    out["feedback_adjusted"] = True  # 标记为反馈调整后的参数
    return out


def build_feedback_signal(
    success: bool,
    gripper_force: float,
    ideal_force_range: tuple[float, float],
    info: dict,
) -> FeedbackSignal:
    """
    从 env.execute_pick_place 的返回值构建 FeedbackSignal。
    info 可包含 distance, steps, force_likely_low, force_likely_high（由 env 可选填充）。
    """
    lo, hi = ideal_force_range
    force_likely_low = info.get("force_likely_low", False)
    force_likely_high = info.get("force_likely_high", False)
    if not force_likely_low and not force_likely_high and not success:
        # 启发式：失败且力偏小则可能滑脱，力偏大则可能损坏
        if gripper_force < lo:
            force_likely_low = True
        elif gripper_force > hi:
            force_likely_high = True
    return FeedbackSignal(
        success=success,
        gripper_force=gripper_force,
        ideal_force_range=ideal_force_range,
        distance=float(info.get("distance", 0.0)),
        steps=int(info.get("steps", 0)),
        force_likely_low=force_likely_low,
        force_likely_high=force_likely_high,
    )
