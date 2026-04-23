"""Unified evidence-driven QA pipeline for the MechanicalRag project."""

from __future__ import annotations

import argparse
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from chroma_compat import get_chroma_client_settings
from llm_loader import get_llm
from model_provider import resolve_model_path


DEFAULT_MODEL = "Qwen/Qwen2-0.5B-Instruct"
DEFAULT_EMBEDDING = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_DB_ROOT = Path(".cache/chroma")
DEFAULT_DB_DIRS = {
    "improved": DEFAULT_DB_ROOT / "qa_improved",
    "rule_heavy": DEFAULT_DB_ROOT / "qa_rule_heavy",
}
COLLECTION_PREFIX = "mechanical_qa"
DEFAULT_ABSTAIN_RESPONSE = "当前知识库未提供足够证据，无法可靠回答该问题。"
QUESTION_NORMALIZATIONS = (
    ("薄壁零件", "薄壁件"),
    ("表面很滑的金属件", "光滑金属零件"),
    ("表面很滑的金属", "光滑金属"),
    ("初始安全位", "初始安全位置"),
    ("安全位", "安全位置"),
    ("夹持方式", "夹持方式"),
)
MECHANICAL_KEYWORDS = (
    "夹爪力",
    "payload",
    "负载",
    "标定",
    "复位",
    "摩擦",
    "抓取姿态",
    "重复定位精度",
    "真空吸附",
    "真空吸盘",
    "伺服报警",
)
TERM_VOCAB = (
    "payload",
    "2kg",
    "轻型工件",
    "小型机械零件",
    "小型零件",
    "标定",
    "复位",
    "初始安全位置",
    "安全位置",
    "圆柱形工件",
    "圆柱形",
    "对称抓取",
    "两端1/3",
    "三分之一",
    "摩擦",
    "摩擦系数低",
    "光滑金属零件",
    "光滑金属",
    "橡胶零件",
    "橡胶",
    "30-50N",
    "5-15N",
    "重复定位精度",
    "±0.05mm",
    "薄壁件",
    "径向夹持",
    "多点分散支撑",
    "真空吸附",
    "真空吸盘",
    "光滑表面",
    "伺服报警",
    "编码器",
    "驱动器",
    "过载",
)
OBJECT_TERMS = (
    "光滑金属零件",
    "光滑金属",
    "橡胶零件",
    "橡胶",
    "小型机械零件",
    "小型零件",
    "薄壁件",
    "圆柱形工件",
    "圆柱形",
    "真空吸盘",
)
ACTION_MARKERS = (
    "先",
    "再",
    "最后",
    "优先",
    "避免",
    "检查",
    "复位",
    "释放",
    "启动",
    "关闭",
    "加载",
    "更换",
)
DIAGNOSTIC_TERMS = ("编码器", "驱动器", "过载", "气路", "电路", "电磁阀", "限位开关", "通信")
SURFACE_TERMS = ("光滑表面", "光滑金属零件", "光滑金属", "薄壁件", "橡胶零件", "圆柱形工件")
STAGE_TERMS = ("接近", "抓取", "抬升", "搬运", "放置")
CONCEPT_CATALOG = {
    "payload_limit": {
        "aliases": ("payload", "负载", "2kg", "轻型工件", "小型机械零件"),
        "expansion_terms": ("payload 2kg 轻型工件 小型机械零件",),
        "categories": ("一、设备参数",),
    },
    "reset_safety": {
        "aliases": ("标定", "复位", "初始安全位置", "安全位置", "安全位"),
        "expansion_terms": ("标定 复位 初始安全位置",),
        "categories": ("二、流程知识",),
    },
    "grasp_pose": {
        "aliases": ("圆柱形工件", "圆柱形", "对称抓取", "两端1/3", "三分之一"),
        "expansion_terms": ("圆柱形工件 对称抓取 两端1/3 重心稳定",),
        "categories": ("三、物理先验知识",),
    },
    "force_selection": {
        "aliases": ("夹爪力", "夹持力", "30-50N", "5-15N", "摩擦系数低"),
        "expansion_terms": ("夹爪力 范围 抓取 力",),
        "categories": ("三、物理先验知识", "一、设备参数"),
    },
    "thin_wall": {
        "aliases": ("薄壁件", "径向夹持", "多点分散支撑", "真空吸附"),
        "expansion_terms": ("薄壁件 径向夹持 多点分散支撑 真空吸附",),
        "categories": ("三、物理先验知识",),
    },
    "vacuum_surface": {
        "aliases": ("真空吸盘", "真空吸附", "光滑表面", "真空度"),
        "expansion_terms": ("真空吸盘 光滑表面 真空度",),
        "categories": ("一、设备参数", "四、传感器与感知"),
    },
    "servo_alarm": {
        "aliases": ("伺服报警", "编码器", "驱动器", "过载"),
        "expansion_terms": ("伺服报警 编码器 驱动器 过载",),
        "categories": ("六、故障与维护",),
    },
    "power_on": {
        "aliases": ("上电", "控制柜电源", "伺服就绪", "示教器", "使能"),
        "expansion_terms": ("上电顺序 控制柜电源 伺服就绪 示教器 使能",),
        "categories": ("二、流程知识",),
    },
    "grasp_sequence": {
        "aliases": ("抓取规划", "接近", "抓取", "抬升", "放置"),
        "expansion_terms": ("抓取规划 接近 抓取 抬升 放置 序列",),
        "categories": ("二、流程知识",),
    },
    "collision_recovery": {
        "aliases": ("碰撞后", "机械限位", "慢速回零", "工具坐标系"),
        "expansion_terms": ("碰撞后 编码器 机械限位 慢速回零 校准 工具坐标系",),
        "categories": ("六、故障与维护",),
    },
    "changeover": {
        "aliases": ("换产", "备份当前程序", "加载新程序", "更换末端执行器", "试运行"),
        "expansion_terms": ("换产流程 备份当前程序 加载新程序 更换末端执行器 试运行",),
        "categories": ("二、流程知识",),
    },
    "inertia_margin": {
        "aliases": ("高速运动", "高速搬运", "夹持力余量", "惯性前冲", "急停"),
        "expansion_terms": ("高速运动 增大夹持力余量 急停 惯性前冲",),
        "categories": ("三、物理先验知识",),
    },
    "gripper_fault": {
        "aliases": ("夹爪不动作", "气路", "电路", "电磁阀", "限位开关"),
        "expansion_terms": ("夹爪不动作 气路 电路 电磁阀 限位开关",),
        "categories": ("六、故障与维护",),
    },
}
WEAK_CONCEPT_ALIASES = {
    "grasp_sequence": ("抓取",),
}
QUERY_LITERAL_TERMS = (
    "fps",
    "帧率",
    "焊接电流",
    "协议",
    "plc",
    "agv",
    "电池",
)
CONCEPT_SUFFICIENCY_RULES = {
    "reset_safety": {
        "required_groups": (("安全位置", "安全位"),),
        "min_required_groups": 1,
    },
    "thin_wall": {
        "required_groups": (("薄壁", "薄壁件"), ("径向夹持", "多点分散支撑", "真空吸附")),
        "min_required_groups": 2,
    },
    "vacuum_surface": {
        "required_groups": (("真空吸盘", "真空吸附"), ("光滑表面", "光滑")),
        "min_required_groups": 2,
    },
    "servo_alarm": {
        "required_groups": (("伺服报警",), ("编码器",), ("驱动器",), ("过载",)),
        "min_required_groups": 3,
    },
    "power_on": {
        "required_groups": (("控制柜电源",), ("伺服就绪",), ("示教器",), ("使能",)),
        "min_required_groups": 3,
    },
    "collision_recovery": {
        "required_groups": (("碰撞后",), ("编码器",), ("机械限位",), ("慢速回零", "回零"), ("工具坐标系",)),
        "min_required_groups": 4,
    },
    "gripper_fault": {
        "required_groups": (("夹爪不动作",), ("气路", "电路"), ("电磁阀",), ("限位开关",)),
        "min_required_groups": 3,
    },
}
QUESTION_PATTERNS = (
    {
        "name": "payload_small_part",
        "triggers": ("payload", "2kg"),
        "expansions": ("payload 2kg 轻型工件 小型机械零件",),
        "focus_terms": ("payload", "2kg", "轻型工件", "小型机械零件"),
        "preferred_categories": ("一、设备参数",),
    },
    {
        "name": "calibration_reset",
        "triggers": ("标定", "复位"),
        "expansions": ("标定 复位 初始安全位置",),
        "focus_terms": ("标定", "复位", "初始安全位置", "安全位置"),
        "preferred_categories": ("二、流程知识",),
    },
    {
        "name": "cylinder_pose",
        "triggers": ("圆柱形", "姿态"),
        "expansions": ("圆柱形工件 对称抓取 两端1/3 重心稳定",),
        "focus_terms": ("圆柱形工件", "对称抓取", "两端1/3", "三分之一", "重心稳定"),
        "preferred_categories": ("三、物理先验知识",),
    },
    {
        "name": "low_friction_force",
        "triggers": ("摩擦系数低", "夹爪力"),
        "expansions": ("摩擦系数低 光滑金属零件 夹爪力增大 30-50N",),
        "focus_terms": ("摩擦", "光滑金属", "夹爪力", "增大", "30-50N"),
        "preferred_categories": ("三、物理先验知识",),
    },
    {
        "name": "repeatability_small_part",
        "triggers": ("重复定位精度", "小型零件"),
        "expansions": ("重复定位精度±0.05mm 小型机械零件 满足抓取需求",),
        "focus_terms": ("重复定位精度", "±0.05mm", "小型机械零件", "小型零件"),
        "preferred_categories": ("一、设备参数",),
    },
    {
        "name": "smooth_metal_force",
        "triggers": ("光滑金属", "夹爪力"),
        "expansions": ("光滑金属零件 30-50N 夹爪力",),
        "focus_terms": ("光滑金属零件", "光滑金属", "夹爪力", "30-50N"),
        "preferred_categories": ("三、物理先验知识",),
    },
    {
        "name": "rubber_force",
        "triggers": ("橡胶", "夹爪力"),
        "expansions": ("橡胶零件 5-15N 夹爪力",),
        "focus_terms": ("橡胶零件", "橡胶", "夹爪力", "5-15N"),
        "preferred_categories": ("三、物理先验知识",),
    },
    {
        "name": "thin_wall_handling",
        "triggers": ("薄壁件",),
        "expansions": ("薄壁件 径向夹持 多点分散支撑 真空吸附",),
        "focus_terms": ("薄壁件", "径向", "多点", "分散支撑", "真空吸附"),
        "preferred_categories": ("三、物理先验知识",),
    },
    {
        "name": "vacuum_surface",
        "triggers": ("真空吸盘",),
        "expansions": ("真空吸盘 光滑表面 真空度",),
        "focus_terms": ("真空吸盘", "光滑表面", "真空度"),
        "preferred_categories": ("一、设备参数", "四、传感器与感知"),
    },
    {
        "name": "servo_alarm",
        "triggers": ("伺服报警",),
        "expansions": ("伺服报警 编码器 驱动器 过载原因",),
        "focus_terms": ("伺服报警", "编码器", "驱动器", "过载"),
        "preferred_categories": ("六、故障与维护",),
    },
)
MODE_SETTINGS = {
    "improved": {
        "retrieval_k": 4,
        "candidate_k": 7,
        "top_clauses": 2,
        "llm_fallback": True,
    },
    "rule_heavy": {
        "retrieval_k": 5,
        "candidate_k": 8,
        "top_clauses": 3,
        "llm_fallback": False,
    },
}


@dataclass(frozen=True)
class QueryPlan:
    question: str
    normalized_question: str
    question_type: str
    focus_terms: tuple[str, ...]
    object_terms: tuple[str, ...]
    expansions: tuple[str, ...]
    preferred_categories: tuple[str, ...]
    matched_profile: str | None
    canonical_concepts: tuple[str, ...]


@dataclass(frozen=True)
class EvidenceRow:
    entry_id: str
    category: str
    excerpt: str
    score: float
    support_terms: tuple[str, ...]
    evidence_types: tuple[str, ...]
    numeric_spans: tuple[str, ...]


def _extract_force_ranges(text: str) -> tuple[str, ...]:
    return tuple(dict.fromkeys(re.findall(r"\d+(?:\.\d+)?-\d+(?:\.\d+)?N", text)))


def _split_clauses(text: str) -> list[str]:
    clauses = []
    for part in re.split(r"[；。]", text):
        cleaned = part.strip()
        if cleaned:
            clauses.append(cleaned)
    return clauses


def _trim_prefix(text: str) -> str:
    text = text.strip()
    if "——" in text:
        text = text.split("——", 1)[1].strip()
    return text


def _contains_any(text: str, keywords: tuple[str, ...]) -> bool:
    return any(keyword in text for keyword in keywords)


def _normalize_question(question: str) -> str:
    normalized = question.strip()
    for source, target in QUESTION_NORMALIZATIONS:
        normalized = normalized.replace(source, target)
    return normalized


def _question_type(question: str) -> str:
    if "范围" in question or "区间" in question or "多少N" in question:
        return "range"
    if "顺序" in question or "哪些阶段" in question or "阶段" in question:
        return "procedure"
    if "为什么" in question or "原因" in question:
        return "rationale"
    if "流程" in question or "步骤" in question:
        return "procedure"
    if "检查" in question or "排查" in question:
        return "diagnosis"
    if "注意" in question or "避免" in question:
        return "caution"
    if "推荐" in question or "优先" in question or "方案" in question:
        return "selection"
    if "满足" in question or "适合" in question or "哪种" in question:
        return "fit"
    return "general"


def _match_profile(question: str) -> dict[str, Any] | None:
    for profile in QUESTION_PATTERNS:
        if all(token in question for token in profile["triggers"]):
            return profile
    return None


def _collect_focus_terms(
    question: str,
    profile: dict[str, Any] | None,
    concepts: tuple[str, ...] = (),
) -> tuple[str, ...]:
    terms: list[str] = []
    for term in TERM_VOCAB:
        if term in question and term not in terms:
            terms.append(term)
    for term in re.findall(r"\d+(?:\.\d+)?(?:-\d+(?:\.\d+)?)?(?:N|mm|kg|°)", question):
        if term not in terms:
            terms.append(term)
    if profile is not None:
        for term in profile["focus_terms"]:
            if term not in terms:
                terms.append(term)
    for concept in concepts:
        for alias in CONCEPT_CATALOG.get(concept, {}).get("aliases", ()):
            if alias in question and alias not in terms:
                terms.append(alias)
    return tuple(terms)


def _collect_object_terms(question: str) -> tuple[str, ...]:
    terms = [term for term in OBJECT_TERMS if term in question]
    return tuple(dict.fromkeys(terms))


def _extract_numeric_spans(text: str) -> tuple[str, ...]:
    matches = re.findall(r"\d+(?:\.\d+)?(?:-\d+(?:\.\d+)?)?(?:N|mm|kg|kPa|°|g|ms|h)", text)
    return tuple(dict.fromkeys(matches))


def _extract_evidence_types(text: str) -> tuple[str, ...]:
    labels: list[str] = []
    if _extract_force_ranges(text):
        labels.append("force_range")
    if any(term in text for term in ("payload", "负载", "kg")):
        labels.append("payload")
    if any(term in text for term in ACTION_MARKERS):
        labels.append("action")
    if any(term in text for term in ("避免", "防止", "优先", "注意")):
        labels.append("constraint")
    if any(term in text for term in DIAGNOSTIC_TERMS):
        labels.append("diagnostic")
    if any(term in text for term in STAGE_TERMS):
        labels.append("stage")
    if any(term in text for term in SURFACE_TERMS):
        labels.append("object_surface")
    return tuple(labels)


def _matched_concept_aliases(question: str, concept: str) -> tuple[str, ...]:
    aliases = CONCEPT_CATALOG.get(concept, {}).get("aliases", ())
    matched = tuple(alias for alias in aliases if alias in question)
    weak_aliases = WEAK_CONCEPT_ALIASES.get(concept, ())
    strong = tuple(alias for alias in matched if alias not in weak_aliases)
    if matched and strong:
        return matched
    if matched and not weak_aliases:
        return matched
    return ()


def _collect_canonical_concepts(question: str, profile: dict[str, Any] | None) -> tuple[str, ...]:
    concepts: list[str] = []
    for name, spec in CONCEPT_CATALOG.items():
        if _matched_concept_aliases(question, name):
            concepts.append(name)
    if profile is not None:
        profile_to_concept = {
            "payload_small_part": "payload_limit",
            "calibration_reset": "reset_safety",
            "cylinder_pose": "grasp_pose",
            "low_friction_force": "force_selection",
            "repeatability_small_part": "payload_limit",
            "smooth_metal_force": "force_selection",
            "rubber_force": "force_selection",
            "thin_wall_handling": "thin_wall",
            "vacuum_surface": "vacuum_surface",
            "servo_alarm": "servo_alarm",
        }
        concept_name = profile_to_concept.get(profile["name"])
        if concept_name and concept_name not in concepts:
            concepts.append(concept_name)
    return tuple(concepts)


def _extract_query_literals(question: str) -> tuple[str, ...]:
    literals: list[str] = []
    for token in re.findall(r"[A-Za-z]{2,}", question):
        lowered = token.lower()
        if lowered not in literals:
            literals.append(lowered)
    lowered_question = question.lower()
    for token in QUERY_LITERAL_TERMS:
        if token in lowered_question or token in question:
            normalized = token.lower()
            if normalized not in literals:
                literals.append(normalized)
    return tuple(literals)


def _expansions_from_concepts(concepts: tuple[str, ...], fallback_profile: dict[str, Any] | None) -> tuple[str, ...]:
    expansions: list[str] = []
    for concept in concepts:
        for term in CONCEPT_CATALOG.get(concept, {}).get("expansion_terms", ()):
            if term not in expansions:
                expansions.append(term)
    if fallback_profile is not None:
        for term in fallback_profile["expansions"]:
            if term not in expansions:
                expansions.append(term)
    return tuple(expansions)


def _preferred_categories_from_concepts(
    concepts: tuple[str, ...],
    fallback_profile: dict[str, Any] | None,
) -> tuple[str, ...]:
    categories: list[str] = []
    for concept in concepts:
        for category in CONCEPT_CATALOG.get(concept, {}).get("categories", ()):
            if category not in categories:
                categories.append(category)
    if fallback_profile is not None:
        for category in fallback_profile["preferred_categories"]:
            if category not in categories:
                categories.append(category)
    return tuple(categories)


def _build_query_plan(question: str) -> QueryPlan:
    normalized = _normalize_question(question)
    profile = _match_profile(normalized)
    concepts = _collect_canonical_concepts(normalized, profile)
    return QueryPlan(
        question=question,
        normalized_question=normalized,
        question_type=_question_type(normalized),
        focus_terms=_collect_focus_terms(normalized, profile, concepts),
        object_terms=_collect_object_terms(normalized),
        expansions=_expansions_from_concepts(concepts, profile),
        preferred_categories=_preferred_categories_from_concepts(concepts, profile),
        matched_profile=profile["name"] if profile is not None else None,
        canonical_concepts=concepts,
    )


def _parse_entries(data_path: str) -> list[Document]:
    text = Path(data_path).read_text(encoding="utf-8")
    category = "未分类"
    documents: list[Document] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("## "):
            category = line[3:].strip()
            continue
        match = re.match(r"^(\d+)\.\s*(.+)$", line)
        if not match:
            continue
        entry_id, content = match.groups()
        clauses = _split_clauses(content)
        documents.append(
            Document(
                page_content=content,
                metadata={
                    "entry_id": entry_id,
                    "category": category,
                    "force_range_text": "|".join(_extract_force_ranges(content)),
                    "clause_count": len(clauses),
                    "weight": sum(2 for keyword in MECHANICAL_KEYWORDS if keyword in content),
                },
            )
        )
    return documents


def _display_object_term(plan: QueryPlan) -> str | None:
    if not plan.object_terms:
        return None
    term = plan.object_terms[0]
    if term == "光滑金属":
        return "光滑金属零件"
    if term == "橡胶":
        return "橡胶零件"
    if term == "圆柱形":
        return "圆柱形工件"
    return term


def _support_terms_for_doc(doc: Document, plan: QueryPlan) -> tuple[str, ...]:
    content = doc.page_content
    supported: list[str] = [term for term in plan.focus_terms if term in content]
    for concept in plan.canonical_concepts:
        for alias in CONCEPT_CATALOG.get(concept, {}).get("aliases", ()):
            if alias in content and alias not in supported:
                supported.append(alias)
    return tuple(supported)


def _score_doc(doc: Document, plan: QueryPlan, semantic_score: float, semantic_rank: int) -> float:
    content = doc.page_content
    evidence_types = _extract_evidence_types(content)
    lexical_hits = sum(1 for term in plan.focus_terms if term in content)
    object_hits = sum(2 for term in plan.object_terms if term in content)
    concept_hits = sum(
        2
        for concept in plan.canonical_concepts
        if any(alias in content for alias in CONCEPT_CATALOG.get(concept, {}).get("aliases", ()))
    )
    category_bonus = 2 if doc.metadata.get("category") in plan.preferred_categories else 0
    range_bonus = 2 if plan.question_type == "range" and _extract_force_ranges(content) else 0
    procedure_bonus = 2 if plan.question_type == "procedure" and "stage" in evidence_types else 0
    diagnosis_bonus = 2 if plan.question_type == "diagnosis" and "diagnostic" in evidence_types else 0
    constraint_bonus = 2 if plan.question_type in {"caution", "selection"} and "constraint" in evidence_types else 0
    order_bonus = 2 if plan.question_type == "procedure" and any(marker in content for marker in ("先", "再", "最后")) else 0
    distractor_penalty = 0
    if "顺时针" in content and "初始安全位置" not in content and "复位目标位置" in plan.question:
        distractor_penalty = 3
    return (
        lexical_hits * 2
        + object_hits
        + concept_hits
        + category_bonus
        + range_bonus
        + procedure_bonus
        + diagnosis_bonus
        + constraint_bonus
        + order_bonus
        + float(doc.metadata.get("weight", 0)) * 0.2
        + semantic_score
        - semantic_rank * 0.05
        - distractor_penalty
    )


def _score_clause(clause: str, plan: QueryPlan) -> float:
    evidence_types = _extract_evidence_types(clause)
    overlap = sum(2 for term in plan.focus_terms if term in clause)
    object_bonus = sum(2 for term in plan.object_terms if term in clause)
    concept_bonus = sum(
        1
        for concept in plan.canonical_concepts
        if any(alias in clause for alias in CONCEPT_CATALOG.get(concept, {}).get("aliases", ()))
    )
    range_bonus = 2 if plan.question_type == "range" and _extract_force_ranges(clause) else 0
    action_bonus = 0
    if plan.question_type in {"caution", "procedure"}:
        action_bonus += sum(1 for marker in ("避免", "优先", "检查", "先", "再", "最后") if marker in clause)
    if plan.question_type == "diagnosis":
        action_bonus += sum(1 for marker in ("检查", "编码器", "驱动器", "过载") if marker in clause)
    if plan.question_type == "procedure" and "stage" in evidence_types:
        action_bonus += 2
    if plan.question_type == "selection" and "constraint" in evidence_types:
        action_bonus += 2
    return overlap + object_bonus + concept_bonus + range_bonus + action_bonus


def _range_answer(plan: QueryPlan, clauses: list[str]) -> str | None:
    best_clause = None
    best_score = -1.0
    for clause in clauses:
        ranges = _extract_force_ranges(clause)
        if not ranges:
            continue
        score = _score_clause(clause, plan)
        if score > best_score:
            best_score = score
            best_clause = clause
    if best_clause is None:
        return None
    range_text = _extract_force_ranges(best_clause)[0]
    subject = _display_object_term(plan)
    if subject is not None:
        return f"抓取{subject}时，夹爪力建议控制在{range_text}。"
    if "摩擦系数低" in plan.question:
        return f"夹爪力需要适当增大，可参考{range_text}。"
    return f"建议控制在{range_text}。"


def _best_clauses(plan: QueryPlan, docs: list[Document], limit: int) -> list[str]:
    scored: list[dict[str, Any]] = []
    for doc in docs:
        for clause in _split_clauses(doc.page_content):
            if not clause:
                continue
            score = _score_clause(clause, plan)
            if score <= 0:
                continue
            scored.append(
                {
                    "score": score,
                    "clause": clause,
                    "concept_hits": tuple(
                        concept
                        for concept in plan.canonical_concepts
                        if any(
                            alias in clause
                            for alias in CONCEPT_CATALOG.get(concept, {}).get("aliases", ())
                            if alias not in WEAK_CONCEPT_ALIASES.get(concept, ())
                        )
                    ),
                    "evidence_types": _extract_evidence_types(clause),
                }
            )
    scored.sort(key=lambda item: (len(item["concept_hits"]), item["score"]), reverse=True)
    selected: list[str] = []
    for concept in plan.canonical_concepts:
        candidate = next(
            (
                item
                for item in scored
                if concept in item["concept_hits"] and item["clause"] not in selected
            ),
            None,
        )
        if candidate is not None:
            selected.append(candidate["clause"])
        if len(selected) >= limit:
            return selected
    if plan.question_type == "procedure":
        candidate = next(
            (
                item
                for item in scored
                if "stage" in item["evidence_types"] and item["clause"] not in selected
            ),
            None,
        )
        if candidate is not None:
            selected.append(candidate["clause"])
    if plan.question_type == "diagnosis":
        candidate = next(
            (
                item
                for item in scored
                if "diagnostic" in item["evidence_types"] and item["clause"] not in selected
            ),
            None,
        )
        if candidate is not None and len(selected) < limit:
            selected.append(candidate["clause"])
    for item in scored:
        clause = item["clause"]
        if clause not in selected:
            selected.append(clause)
        if len(selected) >= limit:
            break
    return selected


def _concept_sufficiency(plan: QueryPlan, clauses: list[str]) -> dict[str, Any]:
    clause_text = "\n".join(clauses)
    details: list[dict[str, Any]] = []
    for concept in plan.canonical_concepts:
        spec = CONCEPT_SUFFICIENCY_RULES.get(concept)
        if spec is None:
            continue
        groups = tuple(spec["required_groups"])
        matched = ["/".join(group) for group in groups if _contains_any(clause_text, group)]
        required_count = min(int(spec["min_required_groups"]), len(groups))
        details.append(
            {
                "concept": concept,
                "required_groups": ["/".join(group) for group in groups],
                "matched_groups": matched,
                "min_required_groups": required_count,
                "sufficient": len(matched) >= required_count,
            }
        )
    return {
        "checked": bool(details),
        "all_sufficient": all(detail["sufficient"] for detail in details) if details else True,
        "details": details,
    }


def _has_object_specific_range_support(plan: QueryPlan, clauses: list[str]) -> bool:
    if plan.question_type != "range" or not plan.object_terms:
        return True
    return any(
        _extract_force_ranges(clause) and any(term in clause for term in plan.object_terms)
        for clause in clauses
    )


def _support_summary(plan: QueryPlan, evidence_rows: list[EvidenceRow], clauses: list[str]) -> dict[str, Any]:
    focus_terms = tuple(dict.fromkeys((*plan.focus_terms, *plan.object_terms)))
    matched_focus = {
        term
        for row in evidence_rows
        for term in row.support_terms
        if term in focus_terms
    }
    best_doc_score = evidence_rows[0].score if evidence_rows else 0.0
    best_clause_score = max((_score_clause(clause, plan) for clause in clauses), default=0.0)
    literal_tokens = _extract_query_literals(plan.question)
    matched_literal_tokens = tuple(
        token
        for token in literal_tokens
        if any(token in clause.lower() for clause in clauses)
    )
    concept_sufficiency = _concept_sufficiency(plan, clauses)
    return {
        "focus_terms": focus_terms,
        "matched_focus_terms": tuple(sorted(matched_focus)),
        "focus_coverage": round(len(matched_focus) / len(focus_terms), 4) if focus_terms else None,
        "best_doc_score": round(best_doc_score, 4),
        "best_clause_score": round(best_clause_score, 4),
        "evidence_count": len(evidence_rows),
        "active_entry_ids": tuple(row.entry_id for row in evidence_rows),
        "matched_concepts": len(plan.canonical_concepts),
        "matched_profile": plan.matched_profile,
        "literal_tokens": literal_tokens,
        "matched_literal_tokens": matched_literal_tokens,
        "object_specific_range_support": _has_object_specific_range_support(plan, clauses),
        "concept_sufficiency": concept_sufficiency,
    }


def _abstain_reason(plan: QueryPlan, evidence_rows: list[EvidenceRow], clauses: list[str]) -> str | None:
    summary = _support_summary(plan, evidence_rows, clauses)
    focus_coverage = summary["focus_coverage"]
    best_doc_score = summary["best_doc_score"]
    best_clause_score = summary["best_clause_score"]
    no_schema_match = not plan.canonical_concepts and plan.matched_profile is None
    no_focus = len(summary["focus_terms"]) == 0
    unmatched_literals = [
        token for token in summary["literal_tokens"] if token not in summary["matched_literal_tokens"]
    ]

    if no_schema_match and no_focus and best_doc_score < 4.5:
        return "query_out_of_knowledge_base"
    if unmatched_literals and no_schema_match:
        return "unmatched_literal_term"
    if not summary["object_specific_range_support"]:
        return "missing_object_specific_range_evidence"
    if not summary["concept_sufficiency"]["all_sufficient"]:
        failing = next(
            (
                detail["concept"]
                for detail in summary["concept_sufficiency"]["details"]
                if not detail["sufficient"]
            ),
            "unknown",
        )
        return f"insufficient_concept_evidence:{failing}"
    if focus_coverage is not None and focus_coverage < 0.2 and best_clause_score < 2.5 and best_doc_score < 5.0:
        return "insufficient_focus_coverage"
    if plan.question_type == "range" and not any(_extract_force_ranges(clause) for clause in clauses) and best_clause_score < 4.0:
        return "missing_numeric_evidence"
    if plan.question_type in {"procedure", "diagnosis"} and best_clause_score < 3.0 and best_doc_score < 5.0:
        return "weak_procedure_support"
    if not clauses and best_doc_score < 4.0:
        return "no_reliable_clause"
    return None


def _ordered_stage_terms(clauses: list[str]) -> list[str]:
    seen: list[str] = []
    for stage in STAGE_TERMS:
        if any(stage in clause for clause in clauses):
            seen.append(stage)
    return seen


def _diagnostic_terms_from_clauses(clauses: list[str]) -> list[str]:
    seen: list[str] = []
    for term in DIAGNOSTIC_TERMS:
        if any(term in clause for clause in clauses):
            seen.append(term)
    return seen


def _constraint_answer(plan: QueryPlan, docs: list[Document], mode: str) -> tuple[str | None, list[str]]:
    limit = MODE_SETTINGS[mode]["top_clauses"]
    clauses = _best_clauses(plan, docs, limit)
    if not clauses:
        return None, []

    if "grasp_pose" in plan.canonical_concepts:
        clause = next(
            (
                c
                for c in clauses
                if "对称抓取" in c or "两端1/3" in c or "三分之一" in c
            ),
            clauses[0],
        )
        return _trim_prefix(clause) + "。", clauses

    if "collision_recovery" in plan.canonical_concepts:
        clause = next(
            (
                c
                for c in clauses
                if "碰撞后" in c or "机械限位" in c or "工具坐标系" in c
            ),
            clauses[0],
        )
        return _trim_prefix(clause) + "。", clauses

    if "gripper_fault" in plan.canonical_concepts:
        clause = next((c for c in clauses if "夹爪不动作" in c or "电磁阀" in c), clauses[0])
        fault_terms = []
        for term in ("气路", "电路", "电磁阀", "限位开关"):
            if term in clause and term not in fault_terms:
                fault_terms.append(term)
        if fault_terms:
            return f"优先检查{'、'.join(fault_terms)}。", clauses
        return _trim_prefix(clause) + "。", clauses

    if "reset_safety" in plan.canonical_concepts and any("安全位置" in clause or "安全位" in clause for clause in clauses):
        clause = next((c for c in clauses if "安全位置" in c or "安全位" in c), clauses[0])
        return _trim_prefix(clause) + "。", clauses

    if plan.question_type == "range":
        answer = _range_answer(plan, clauses)
        if answer is not None:
            return answer, clauses

    merged = "；".join(_trim_prefix(clause) for clause in clauses)
    if plan.question_type == "caution":
        if "inertia_margin" in plan.canonical_concepts:
            clause = next(
                (c for c in clauses if "夹持力余量" in c or "惯性前冲" in c or "惯性" in c),
                clauses[0],
            )
            return _trim_prefix(clause) + "。", clauses
        avoid_part = next((c for c in clauses if "避免" in c), clauses[0])
        prefer_part = next((c for c in clauses if "优先" in c or "真空吸附" in c), clauses[-1])
        if avoid_part != prefer_part:
            return f"{_trim_prefix(avoid_part)}；{_trim_prefix(prefer_part)}。", clauses
        return _trim_prefix(avoid_part) + "。", clauses

    if plan.question_type == "selection":
        preferred = next(
            (c for c in clauses if "优先" in c or "适配" in c or "适合" in c or "真空吸盘" in c),
            clauses[0],
        )
        support = next(
            (c for c in clauses if c != preferred and ("光滑表面" in c or "真空吸盘" in c)),
            None,
        )
        if support is None:
            support = next(
                (c for c in clauses if c != preferred and ("薄壁件" in c or "真空" in c)),
                None,
            )
        preferred_text = _trim_prefix(preferred)
        if "优先" in preferred_text:
            preferred_text = preferred_text[preferred_text.index("优先") :]
        if "thin_wall" in plan.canonical_concepts and "薄壁件" not in preferred_text:
            preferred_text = "针对薄壁件，" + preferred_text
        if support is not None and "光滑表面" in support and "光滑表面" not in preferred_text:
            preferred_text = preferred_text.rstrip("。") + "，适配光滑表面"
        if support is not None:
            return f"{preferred_text}；{_trim_prefix(support)}。", clauses
        return preferred_text + "。", clauses

    if plan.question_type == "rationale":
        clause = next((c for c in clauses if "复位" in c and ("安全位置" in c or "安全位" in c)), clauses[0])
        return _trim_prefix(clause) + "。", clauses

    if plan.question_type == "fit":
        if "重复定位精度" in plan.question and any(
            token in "\n".join(clauses) for token in ("小型机械零件", "小型零件")
        ):
            return "可以满足小型零件抓取需求。", clauses
        clause = clauses[0]
        if any(token in clause for token in ("适合", "适用于", "满足", "小型机械零件", "光滑表面")):
            return _trim_prefix(clause) + "。", clauses
        return _trim_prefix(clause) + "。", clauses

    if plan.question_type == "diagnosis":
        terms = _diagnostic_terms_from_clauses(clauses)
        if terms:
            return f"先检查{'、'.join(terms)}。", clauses
        clause = next((c for c in clauses if "检查" in c), clauses[0])
        return _trim_prefix(clause) + "。", clauses

    if plan.question_type == "procedure":
        if "changeover" in plan.canonical_concepts:
            clause = next(
                (c for c in clauses if "备份当前程序" in c or "换产流程" in c or "更换末端执行器" in c),
                clauses[0],
            )
            return _trim_prefix(clause) + "。", clauses
        if "grasp_sequence" in plan.canonical_concepts:
            stages = _ordered_stage_terms(clauses)
            if stages:
                return f"流程通常包括{'-'.join(stages)}。", clauses
        ordered = [clause for clause in clauses if any(marker in clause for marker in ("先", "再", "最后"))]
        if ordered:
            return _trim_prefix(ordered[0]) + "。", clauses
        stages = _ordered_stage_terms(clauses)
        if stages:
            return f"流程通常包括{'-'.join(stages)}。", clauses
        return _trim_prefix(clauses[0]) + "。", clauses

    return merged + "。", clauses


def _llm_fallback(llm: Any, plan: QueryPlan, clauses: list[str]) -> str | None:
    if llm is None or not clauses:
        return None
    context = "\n".join(f"- {clause}" for clause in clauses)
    prompt = (
        "你是机械工程问答助手。请只根据下面证据回答，不要补充证据里没有的信息。\n"
        "回答要求：优先保留数值范围、抓取姿态、安全步骤等关键约束，长度控制在60字内。\n\n"
        f"证据：\n{context}\n\n问题：{plan.question}\n回答："
    )
    response = llm.invoke(prompt)
    if not isinstance(response, str):
        response = str(response)
    text = response.strip()
    if sum(1 for char in text if "\u4e00" <= char <= "\u9fff") >= 2:
        return text
    return None


class MechanicalQAPipeline:
    """Evidence-driven QA pipeline with explicit query understanding and evidence selection."""

    def __init__(
        self,
        data_path: str,
        model_name: str = DEFAULT_MODEL,
        embedding_model_name: str = DEFAULT_EMBEDDING,
        db_dir: str | Path | None = None,
        mode: str = "improved",
    ) -> None:
        if mode not in MODE_SETTINGS:
            raise ValueError(f"未知 mode: {mode}")
        self.data_path = data_path
        self.mode = mode
        db_dir = Path(db_dir) if db_dir is not None else DEFAULT_DB_DIRS[mode]
        db_dir.parent.mkdir(parents=True, exist_ok=True)
        documents = _parse_entries(data_path)
        self.documents = documents
        embedding_model_path = resolve_model_path(embedding_model_name)
        embeddings = HuggingFaceEmbeddings(model_name=embedding_model_path)
        self.vector_db = Chroma.from_documents(
            documents,
            embeddings,
            ids=[f"entry_{doc.metadata['entry_id']}" for doc in documents],
            collection_name=f"{COLLECTION_PREFIX}_{mode}",
            persist_directory=str(db_dir),
            client_settings=get_chroma_client_settings(str(db_dir)),
        )
        self.vector_db.persist()
        self.llm = get_llm(model_name=model_name, max_new_tokens=128)

    def understand_query(self, question: str) -> QueryPlan:
        return _build_query_plan(question)

    def select_evidence(
        self,
        plan: QueryPlan,
        k: int | None = None,
        exclude_entry_ids: tuple[str, ...] | None = None,
    ) -> tuple[list[Document], list[EvidenceRow]]:
        settings = MODE_SETTINGS[self.mode]
        top_k = k or settings["retrieval_k"]
        excluded = set(exclude_entry_ids or ())
        candidate_queries = [plan.normalized_question, *plan.expansions]
        for concept in plan.canonical_concepts:
            aliases = CONCEPT_CATALOG.get(concept, {}).get("aliases", ())
            if aliases:
                candidate_queries.append(" ".join(aliases[:4]))
        dedup: dict[str, Document] = {}
        scores: dict[str, float] = {}
        ranks: dict[str, int] = {}

        for candidate_query in candidate_queries:
            retrieved = self.vector_db.similarity_search_with_score(candidate_query, k=settings["candidate_k"])
            for rank, (doc, distance) in enumerate(retrieved, 1):
                if str(doc.metadata.get("entry_id")) in excluded:
                    continue
                key = doc.page_content
                semantic_score = 1.0 / (1.0 + distance)
                score = _score_doc(doc, plan, semantic_score, rank)
                if key not in dedup or score > scores[key]:
                    dedup[key] = doc
                    scores[key] = score
                    ranks[key] = rank

        lexical_terms = list(plan.focus_terms) + list(plan.object_terms)
        for concept in plan.canonical_concepts:
            lexical_terms.extend(CONCEPT_CATALOG.get(concept, {}).get("aliases", ()))
        lexical_terms = list(dict.fromkeys(term for term in lexical_terms if term))
        lexical_rank = settings["candidate_k"] + 1
        for doc in self.documents:
            if str(doc.metadata.get("entry_id")) in excluded:
                continue
            content = doc.page_content
            if not any(term in content for term in lexical_terms):
                continue
            key = content
            score = _score_doc(doc, plan, semantic_score=0.0, semantic_rank=lexical_rank)
            if key not in dedup or score > scores[key]:
                dedup[key] = doc
                scores[key] = score
                ranks[key] = lexical_rank

        docs = sorted(dedup.values(), key=lambda item: scores[item.page_content], reverse=True)[:top_k]
        evidence_rows = [
            EvidenceRow(
                entry_id=str(doc.metadata.get("entry_id")),
                category=str(doc.metadata.get("category")),
                excerpt=doc.page_content,
                score=round(scores[doc.page_content], 4),
                support_terms=_support_terms_for_doc(doc, plan),
                evidence_types=_extract_evidence_types(doc.page_content),
                numeric_spans=_extract_numeric_spans(doc.page_content),
            )
            for doc in docs
        ]
        return docs, evidence_rows

    def answer(
        self,
        question: str,
        k: int | None = None,
        exclude_entry_ids: tuple[str, ...] | None = None,
    ) -> tuple[str, list[Document], dict[str, Any]]:
        plan = self.understand_query(question)
        docs, evidence_rows = self.select_evidence(plan, k=k, exclude_entry_ids=exclude_entry_ids)
        constrained, clauses = _constraint_answer(plan, docs, self.mode)
        support_summary = _support_summary(plan, evidence_rows, clauses)
        abstain_reason = _abstain_reason(plan, evidence_rows, clauses)
        answer = constrained
        if abstain_reason is not None:
            answer = DEFAULT_ABSTAIN_RESPONSE
        elif answer is None and MODE_SETTINGS[self.mode]["llm_fallback"]:
            answer = _llm_fallback(self.llm, plan, clauses)
        if answer is None:
            answer = _trim_prefix(clauses[0]) + "。" if clauses else ""
        debug = {
            "query_plan": asdict(plan),
            "evidence_trace": [asdict(row) for row in evidence_rows],
            "selected_clauses": clauses,
            "mode": self.mode,
            "excluded_entry_ids": list(exclude_entry_ids or ()),
            "support_summary": support_summary,
            "abstained": abstain_reason is not None,
            "abstain_reason": abstain_reason,
        }
        return answer, docs, debug


def build_components(
    data_path: str,
    model_name: str = DEFAULT_MODEL,
    embedding_model_name: str = DEFAULT_EMBEDDING,
    db_dir: str | Path | None = None,
    mode: str = "improved",
) -> MechanicalQAPipeline:
    return MechanicalQAPipeline(
        data_path=data_path,
        model_name=model_name,
        embedding_model_name=embedding_model_name,
        db_dir=db_dir,
        mode=mode,
    )


def answer_question(
    pipeline: MechanicalQAPipeline,
    question: str,
    k: int | None = None,
    exclude_entry_ids: tuple[str, ...] | None = None,
) -> tuple[str, list[Document], dict[str, Any]]:
    return pipeline.answer(question, k=k, exclude_entry_ids=exclude_entry_ids)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="mechanical_data.txt")
    parser.add_argument("--model_name", default=DEFAULT_MODEL)
    parser.add_argument("--embedding_model_name", default=DEFAULT_EMBEDDING)
    parser.add_argument("--db_dir", default=None)
    parser.add_argument("--mode", choices=tuple(MODE_SETTINGS.keys()), default="improved")
    args = parser.parse_args()

    if not Path(args.data_path).exists():
        raise FileNotFoundError(f"未找到数据文件: {args.data_path}")

    pipeline = build_components(
        data_path=args.data_path,
        model_name=args.model_name,
        embedding_model_name=args.embedding_model_name,
        db_dir=args.db_dir,
        mode=args.mode,
    )
    questions = [
        "机械臂标定后，为什么需要进行复位操作？",
        "抓取圆柱形工件时，最佳抓取姿态是什么？",
        "抓取光滑金属零件时，夹爪力应控制在什么范围？",
        "真空吸盘更适合哪类表面？",
    ]

    print("=" * 50)
    print(f"统一 QA Pipeline 测试结果 ({args.mode})")
    print("=" * 50)
    for idx, question in enumerate(questions, 1):
        answer, docs, debug = pipeline.answer(question)
        print(f"\n第{idx}个问题：{question}")
        print("回答：", answer)
        print("证据：")
        for row in debug["evidence_trace"]:
            print(f"- [{row['category']}] score={row['score']}: {row['excerpt']}")
        print(f"命中的约束片段：{debug['selected_clauses']}")


if __name__ == "__main__":
    main()
