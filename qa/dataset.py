"""QA benchmark cases with core, stress, counterfactual and OOD splits."""

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
    split: str = "core"
    variant: str = "original"
    source_case_id: str | None = None
    expected_behavior: str = "answer"
    abstain_keywords: tuple[str, ...] = ()
    exclude_entry_ids: tuple[str, ...] = ()


CORE_QA_CASES: tuple[QACase, ...] = (
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


PARAPHRASE_QA_CASES: tuple[QACase, ...] = (
    QACase(
        case_id="payload_small_part_paraphrase",
        source_case_id="payload_small_part",
        question="当夹爪额定payload只有2kg时，更适合处理什么类型的工件？",
        gold_answer="更适合轻型工件，典型对象仍是小型机械零件。",
        required_groups=(("轻型", "小型"), ("机械零件", "工件")),
        evidence_groups=(("payload", "2kg"), ("小型机械零件",)),
        forbidden_keywords=("大型", "重型"),
        split="paraphrase",
        variant="paraphrase",
    ),
    QACase(
        case_id="calibration_reset_paraphrase",
        source_case_id="calibration_reset",
        question="完成机械臂标定后，为什么还要让它回到初始安全位？",
        gold_answer="因为标定后需要复位到初始安全位置，便于后续抓取和运行安全。",
        required_groups=(("复位",), ("初始安全位置", "安全位", "安全位置")),
        evidence_groups=(("标定",), ("复位",), ("初始安全位置", "安全位置")),
        split="paraphrase",
        variant="paraphrase",
    ),
    QACase(
        case_id="smooth_metal_force_paraphrase",
        source_case_id="smooth_metal_force",
        question="面对表面很滑的金属件，夹持力大致该设到哪个区间？",
        gold_answer="可参考30-50N。",
        required_groups=(("30-50N", "30到50N"),),
        evidence_groups=(("光滑金属零件", "光滑金属"), ("30-50N", "30到50N")),
        forbidden_keywords=("5-15N", "5到15N"),
        split="paraphrase",
        variant="paraphrase",
    ),
    QACase(
        case_id="thin_wall_handling_paraphrase",
        source_case_id="thin_wall_handling",
        question="抓薄壁零件时，夹持方式上最该避免什么，又推荐什么方案？",
        gold_answer="应避免径向夹持，优先采用多点分散支撑或真空吸附。",
        required_groups=(("避免",), ("径向",), ("多点", "真空吸附", "分散支撑")),
        evidence_groups=(("薄壁件",), ("径向",), ("多点", "真空吸附", "分散支撑")),
        forbidden_keywords=("单点硬夹",),
        split="paraphrase",
        variant="paraphrase",
    ),
)


ROBUSTNESS_QA_CASES: tuple[QACase, ...] = (
    QACase(
        case_id="smooth_metal_force_robustness",
        source_case_id="smooth_metal_force",
        question="知识库里同时出现了5-15N和30-50N两个范围。若任务是抓取光滑金属零件，最终该引用哪个范围？",
        gold_answer="应引用30-50N，而不是橡胶零件对应的5-15N。",
        required_groups=(("30-50N", "30到50N"),),
        evidence_groups=(("光滑金属零件", "光滑金属"), ("30-50N", "30到50N")),
        forbidden_keywords=("5-15N", "5到15N"),
        split="robustness",
        variant="distractor",
    ),
    QACase(
        case_id="calibration_reset_robustness",
        source_case_id="calibration_reset",
        question="急停释放需要顺时针旋转，但这里问的是标定结束后的复位目标位置。应该复位到哪里？",
        gold_answer="应复位到初始安全位置，而不是急停释放动作。",
        required_groups=(("复位",), ("初始安全位置", "安全位置")),
        evidence_groups=(("标定",), ("复位",), ("初始安全位置", "安全位置")),
        forbidden_keywords=("顺时针",),
        split="robustness",
        variant="distractor",
    ),
)


COMPOSITIONAL_QA_CASES: tuple[QACase, ...] = (
    QACase(
        case_id="thin_wall_vacuum_composition",
        question="如果工件既是薄壁件又要求适配光滑表面，优先推荐什么末端方案？",
        gold_answer="优先采用真空吸附/真空吸盘方案，可同时兼顾薄壁件受力与光滑表面适配。",
        required_groups=(("真空吸附", "真空吸盘", "真空"), ("薄壁件",), ("光滑表面", "光滑")),
        evidence_groups=(("薄壁件", "真空吸附"), ("真空吸盘", "光滑表面")),
        forbidden_keywords=("径向夹持",),
        split="compositional",
        variant="cross_entry",
    ),
    QACase(
        case_id="high_speed_margin_composition",
        question="高速搬运工件时，夹持策略应如何调整，并需要额外注意什么？",
        gold_answer="应增大夹持力余量，并注意急停时的惯性前冲。",
        required_groups=(("增大", "增加"), ("夹持力余量", "夹爪力余量", "夹持力"), ("惯性", "前冲")),
        evidence_groups=(("高速运动", "高速搬运"), ("夹持力余量",), ("惯性前冲", "惯性")),
        split="compositional",
        variant="cross_entry",
    ),
    QACase(
        case_id="grasp_sequence_composition",
        question="机械臂抓取规划通常包含哪些阶段？",
        gold_answer="通常包括接近、抓取、抬升和放置等阶段。",
        required_groups=(("接近",), ("抓取",), ("抬升",), ("放置",)),
        evidence_groups=(("抓取规划",), ("接近", "抓取", "抬升", "放置")),
        split="compositional",
        variant="sequence",
    ),
)


PROCEDURE_QA_CASES: tuple[QACase, ...] = (
    QACase(
        case_id="power_on_sequence_procedure",
        question="机械臂上电的正确顺序是什么？",
        gold_answer="先接通控制柜电源，等待伺服就绪，再启动示教器，最后使能机械臂。",
        required_groups=(("控制柜电源",), ("伺服就绪",), ("示教器",), ("使能",)),
        evidence_groups=(("上电顺序",), ("控制柜电源",), ("伺服就绪",), ("示教器",), ("使能",)),
        split="procedure",
        variant="holdout_procedure",
    ),
    QACase(
        case_id="changeover_sequence_procedure",
        question="换产时需要执行哪些关键步骤？",
        gold_answer="应先备份当前程序，再加载新程序、更换末端执行器，重新标定并试运行验证。",
        required_groups=(("备份当前程序", "备份"), ("加载新程序", "加载"), ("更换末端执行器", "更换"), ("重新标定", "标定"), ("试运行",)),
        evidence_groups=(("换产流程",), ("备份当前程序",), ("加载新程序",), ("更换末端执行器",), ("试运行",)),
        split="procedure",
        variant="holdout_procedure",
    ),
)


HOLDOUT_QA_CASES: tuple[QACase, ...] = (
    QACase(
        case_id="vacuum_surface_holdout",
        question="真空吸盘更适合哪类表面？",
        gold_answer="更适合光滑表面。",
        required_groups=(("光滑表面", "光滑"),),
        evidence_groups=(("真空吸盘",), ("光滑表面", "光滑")),
        forbidden_keywords=("粗糙表面",),
        split="holdout",
        variant="holdout",
    ),
    QACase(
        case_id="servo_alarm_holdout",
        question="机械臂出现伺服报警时，通常先检查哪几类原因？",
        gold_answer="先检查编码器连接、驱动器状态和过载原因。",
        required_groups=(("编码器",), ("驱动器",), ("过载",)),
        evidence_groups=(("伺服报警",), ("编码器",), ("驱动器",), ("过载",)),
        split="holdout",
        variant="holdout",
    ),
    QACase(
        case_id="collision_recovery_holdout",
        question="机械臂发生碰撞后，安全恢复时通常先做哪些处理？",
        gold_answer="先检查编码器和机械限位，再慢速回零并校准工具坐标系。",
        required_groups=(("编码器",), ("机械限位",), ("慢速回零", "回零"), ("工具坐标系", "校准")),
        evidence_groups=(("碰撞后",), ("编码器",), ("机械限位",), ("慢速回零", "回零"), ("工具坐标系",)),
        split="holdout",
        variant="holdout",
    ),
    QACase(
        case_id="gripper_fault_holdout",
        question="如果夹爪不动作，通常优先检查哪些环节？",
        gold_answer="优先检查气路或电路、电磁阀和限位开关，并做手动单步测试。",
        required_groups=(("气路", "电路"), ("电磁阀",), ("限位开关",)),
        evidence_groups=(("夹爪不动作",), ("气路", "电路"), ("电磁阀",), ("限位开关",)),
        split="holdout",
        variant="holdout",
    ),
)


COUNTERFACTUAL_QA_CASES: tuple[QACase, ...] = (
    QACase(
        case_id="calibration_reset_counterfactual",
        source_case_id="calibration_reset",
        question="机械臂标定后，为什么需要进行复位操作？",
        gold_answer="当前知识库已移除该问题所需的关键复位证据，无法可靠回答。",
        required_groups=(),
        evidence_groups=(),
        split="counterfactual",
        variant="support_removed",
        expected_behavior="abstain",
        abstain_keywords=("知识库未提供", "无法可靠回答", "证据不足", "关键证据已移除"),
        exclude_entry_ids=("13",),
    ),
    QACase(
        case_id="smooth_metal_force_counterfactual",
        source_case_id="smooth_metal_force",
        question="抓取光滑金属零件时，夹爪力应控制在什么范围？",
        gold_answer="当前知识库已移除光滑金属零件的对象特定力值证据，无法可靠回答。",
        required_groups=(),
        evidence_groups=(),
        split="counterfactual",
        variant="support_removed",
        expected_behavior="abstain",
        abstain_keywords=("知识库未提供", "无法可靠回答", "证据不足", "关键证据已移除"),
        exclude_entry_ids=("22",),
    ),
    QACase(
        case_id="rubber_force_counterfactual",
        source_case_id="rubber_force",
        question="抓取橡胶零件时，夹爪力应控制在什么范围？",
        gold_answer="当前知识库已移除橡胶零件的对象特定力值证据，无法可靠回答。",
        required_groups=(),
        evidence_groups=(),
        split="counterfactual",
        variant="support_removed",
        expected_behavior="abstain",
        abstain_keywords=("知识库未提供", "无法可靠回答", "证据不足", "关键证据已移除"),
        exclude_entry_ids=("22",),
    ),
    QACase(
        case_id="thin_wall_handling_counterfactual",
        source_case_id="thin_wall_handling",
        question="薄壁件抓取时需要注意什么？",
        gold_answer="当前知识库已移除薄壁件抓取约束证据，无法可靠回答。",
        required_groups=(),
        evidence_groups=(),
        split="counterfactual",
        variant="support_removed",
        expected_behavior="abstain",
        abstain_keywords=("知识库未提供", "无法可靠回答", "证据不足", "关键证据已移除"),
        exclude_entry_ids=("24",),
    ),
    QACase(
        case_id="vacuum_surface_counterfactual",
        source_case_id="vacuum_surface_holdout",
        question="真空吸盘更适合哪类表面？",
        gold_answer="当前知识库已移除真空吸盘适配表面的关键证据，无法可靠回答。",
        required_groups=(),
        evidence_groups=(),
        split="counterfactual",
        variant="support_removed",
        expected_behavior="abstain",
        abstain_keywords=("知识库未提供", "无法可靠回答", "证据不足", "关键证据已移除"),
        exclude_entry_ids=("11",),
    ),
    QACase(
        case_id="servo_alarm_counterfactual",
        source_case_id="servo_alarm_holdout",
        question="机械臂出现伺服报警时，通常先检查哪几类原因？",
        gold_answer="当前知识库已移除伺服报警排查条目，无法可靠回答。",
        required_groups=(),
        evidence_groups=(),
        split="counterfactual",
        variant="support_removed",
        expected_behavior="abstain",
        abstain_keywords=("知识库未提供", "无法可靠回答", "证据不足", "关键证据已移除"),
        exclude_entry_ids=("41",),
    ),
    QACase(
        case_id="collision_recovery_counterfactual",
        source_case_id="collision_recovery_holdout",
        question="机械臂发生碰撞后，安全恢复时通常先做哪些处理？",
        gold_answer="当前知识库已移除碰撞恢复流程证据，无法可靠回答。",
        required_groups=(),
        evidence_groups=(),
        split="counterfactual",
        variant="support_removed",
        expected_behavior="abstain",
        abstain_keywords=("知识库未提供", "无法可靠回答", "证据不足", "关键证据已移除"),
        exclude_entry_ids=("44",),
    ),
    QACase(
        case_id="power_on_sequence_counterfactual",
        source_case_id="power_on_sequence_procedure",
        question="机械臂上电的正确顺序是什么？",
        gold_answer="当前知识库已移除上电顺序条目，无法可靠回答。",
        required_groups=(),
        evidence_groups=(),
        split="counterfactual",
        variant="support_removed",
        expected_behavior="abstain",
        abstain_keywords=("知识库未提供", "无法可靠回答", "证据不足", "关键证据已移除"),
        exclude_entry_ids=("15",),
    ),
)


OOD_QA_CASES: tuple[QACase, ...] = (
    QACase(
        case_id="welding_current_ood",
        question="机械臂焊接铝板时推荐焊接电流是多少？",
        gold_answer="当前知识库未提供焊接电流相关证据，无法可靠回答。",
        required_groups=(),
        evidence_groups=(),
        split="ood",
        variant="unsupported",
        expected_behavior="abstain",
        abstain_keywords=("知识库未提供", "无法可靠回答", "证据不足", "无法判断"),
    ),
    QACase(
        case_id="camera_fps_ood",
        question="视觉相机用于高速抓取时，采集帧率应该设置为多少fps？",
        gold_answer="当前知识库未提供相机帧率设置相关证据，无法可靠回答。",
        required_groups=(),
        evidence_groups=(),
        split="ood",
        variant="unsupported",
        expected_behavior="abstain",
        abstain_keywords=("知识库未提供", "无法可靠回答", "证据不足", "无法判断"),
    ),
    QACase(
        case_id="plc_protocol_ood",
        question="PLC 和机械臂之间推荐使用哪种工业以太网通信协议？",
        gold_answer="当前知识库未提供具体工业以太网协议选择依据，无法可靠回答。",
        required_groups=(),
        evidence_groups=(),
        split="ood",
        variant="unsupported",
        expected_behavior="abstain",
        abstain_keywords=("知识库未提供", "无法可靠回答", "证据不足", "无法判断"),
    ),
    QACase(
        case_id="agv_battery_ood",
        question="AGV 小车电池通常多久需要更换一次？",
        gold_answer="当前知识库未提供 AGV 电池维护周期相关证据，无法可靠回答。",
        required_groups=(),
        evidence_groups=(),
        split="ood",
        variant="unsupported",
        expected_behavior="abstain",
        abstain_keywords=("知识库未提供", "无法可靠回答", "证据不足", "无法判断"),
    ),
)


QA_CASES: tuple[QACase, ...] = (
    CORE_QA_CASES
    + PARAPHRASE_QA_CASES
    + ROBUSTNESS_QA_CASES
    + COMPOSITIONAL_QA_CASES
    + PROCEDURE_QA_CASES
    + HOLDOUT_QA_CASES
    + COUNTERFACTUAL_QA_CASES
    + OOD_QA_CASES
)

SPLIT_ORDER: tuple[str, ...] = (
    "core",
    "paraphrase",
    "robustness",
    "compositional",
    "procedure",
    "holdout",
    "counterfactual",
    "ood",
)


def get_cases(case_set: str = "full") -> tuple[QACase, ...]:
    if case_set == "core":
        return CORE_QA_CASES
    if case_set == "stress":
        return (
            PARAPHRASE_QA_CASES
            + ROBUSTNESS_QA_CASES
            + COMPOSITIONAL_QA_CASES
            + PROCEDURE_QA_CASES
            + HOLDOUT_QA_CASES
            + COUNTERFACTUAL_QA_CASES
            + OOD_QA_CASES
        )
    if case_set == "full":
        return QA_CASES
    raise ValueError(f"未知 case_set: {case_set}")
