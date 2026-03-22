"""问答评测数据集与结构化判分规则。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class QACase:
    case_id: str
    question: str
    gold_answer: str
    required_groups: tuple[tuple[str, ...], ...]
    evidence_groups: tuple[tuple[str, ...], ...]
    forbidden_keywords: tuple[str, ...] = ()


QA_CASES: tuple[QACase, ...] = (
    QACase(
        case_id="payload_small_part",
        question="夹爪payload最大为2kg时，适合抓取哪种类型的工件？",
        gold_answer="适合轻型工件，典型对象为小型机械零件。",
        required_groups=(("小型", "轻型"), ("机械零件", "工件")),
        evidence_groups=(("payload", "2kg"), ("小型机械零件",)),
        forbidden_keywords=("大型", "重型", "薄壁件"),
    ),
    QACase(
        case_id="calibration_reset",
        question="机械臂标定后，为什么需要进行复位操作？",
        gold_answer="标定后需要复位到初始安全位置，便于后续抓取和运行安全。",
        required_groups=(("复位",), ("初始安全位置", "安全位置")),
        evidence_groups=(("标定",), ("复位",), ("初始安全位置", "安全位置")),
    ),
    QACase(
        case_id="cylinder_pose",
        question="抓取圆柱形工件时，最佳抓取姿态是什么？",
        gold_answer="优先采用对称抓取姿态，抓取点靠近工件两端三分之一位置。",
        required_groups=(("对称抓取", "对称"), ("两端1/3", "三分之一", "1/3")),
        evidence_groups=(("圆柱形工件", "圆柱形"), ("对称抓取", "对称"), ("两端1/3", "三分之一", "1/3")),
        forbidden_keywords=("顺时针", "复位", "急停"),
    ),
    QACase(
        case_id="low_friction_force",
        question="摩擦系数低的零件，抓取时夹爪力应如何调整？",
        gold_answer="需要适当增大夹爪力；对于光滑金属零件，可参考30-50N。",
        required_groups=(("增大", "增加"), ("夹爪力",)),
        evidence_groups=(("摩擦",), ("光滑金属", "摩擦系数低"), ("30-50N", "30到50N")),
        forbidden_keywords=("减小", "降低"),
    ),
    QACase(
        case_id="repeatability_small_part",
        question="机械臂重复定位精度±0.05mm，能满足小型零件的抓取需求吗？",
        gold_answer="可以满足小型零件抓取需求。",
        required_groups=(("满足", "可以"), ("小型零件", "小型机械零件")),
        evidence_groups=(("重复定位精度±0.05mm", "±0.05mm"), ("小型机械零件", "小型零件")),
        forbidden_keywords=("不能", "无法"),
    ),
    QACase(
        case_id="smooth_metal_force",
        question="抓取光滑金属零件时，夹爪力应控制在什么范围？",
        gold_answer="建议控制在30-50N。",
        required_groups=(("30-50N", "30到50N"),),
        evidence_groups=(("光滑金属零件", "光滑金属"), ("30-50N", "30到50N")),
        forbidden_keywords=("5-15N", "5到15N"),
    ),
    QACase(
        case_id="rubber_force",
        question="抓取橡胶零件时，夹爪力应控制在什么范围？",
        gold_answer="建议控制在5-15N。",
        required_groups=(("5-15N", "5到15N"),),
        evidence_groups=(("橡胶零件", "橡胶"), ("5-15N", "5到15N")),
        forbidden_keywords=("30-50N", "30到50N"),
    ),
    QACase(
        case_id="thin_wall_handling",
        question="薄壁件抓取时需要注意什么？",
        gold_answer="要避免径向夹持导致变形，优先采用多点分散支撑或真空吸附。",
        required_groups=(("避免",), ("径向",), ("多点", "分散支撑", "真空吸附", "真空")),
        evidence_groups=(("薄壁件",), ("径向",), ("多点", "分散支撑", "真空吸附", "真空")),
        forbidden_keywords=("大力夹紧", "单点硬夹"),
    ),
)
