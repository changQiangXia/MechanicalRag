"""RAG-driven controller with structured evidence traces and rule aggregation."""

from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any

from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from chroma_compat import get_chroma_client_settings
from model_provider import resolve_model_path
from .control_core import (
    EvidenceConstraintHints,
    build_control_belief,
    replan_control_plan,
    solve_control_plan,
    summarize_control_evidence,
    synthesize_control_seed,
)


TASK_DESCRIPTION_HINTS = (
    {"keywords": ("高速", "薄壁"), "force": 11.0, "approach_height": 0.02, "transport_velocity": 0.22, "lift_clearance": 0.06},
    {"keywords": ("高速", "重型", "金属"), "force": 45.0, "approach_height": 0.05, "transport_velocity": 0.30, "lift_clearance": 0.072},
    {"keywords": ("高速", "光滑金属"), "force": 40.0, "approach_height": 0.05, "transport_velocity": 0.34, "lift_clearance": 0.061},
    {"keywords": ("高速", "小型"), "force": 16.0, "approach_height": 0.04, "transport_velocity": 0.46, "lift_clearance": 0.055},
    {"keywords": ("高速", "橡胶"), "force": 13.0, "approach_height": 0.03, "transport_velocity": 0.38, "lift_clearance": 0.05},
    {"keywords": ("长距离", "大型"), "force": 33.0, "approach_height": 0.055, "transport_velocity": 0.2, "lift_clearance": 0.07},
    {"keywords": ("薄壁",), "force": 9.0, "approach_height": 0.02, "transport_velocity": 0.18, "lift_clearance": 0.05},
    {"keywords": ("橡胶",), "force": 10.0, "approach_height": 0.03, "transport_velocity": 0.30, "lift_clearance": 0.045},
    {"keywords": ("小型",), "force": 12.0, "approach_height": 0.04, "transport_velocity": 0.40, "lift_clearance": 0.05},
    {"keywords": ("大型",), "force": 34.0, "approach_height": 0.06, "transport_velocity": 0.22, "lift_clearance": 0.085},
    {"keywords": ("重型", "金属"), "force": 42.0, "approach_height": 0.05, "transport_velocity": 0.18, "lift_clearance": 0.07},
    {"keywords": ("光滑金属",), "force": 36.0, "approach_height": 0.05, "transport_velocity": 0.24, "lift_clearance": 0.07},
)
GENERIC_FORCE_MARKERS = ("夹爪力范围", "气动夹爪", "伺服夹爪", "设备参数")
NEUTRAL_TRANSPORT_VELOCITY = 0.30
NEUTRAL_LIFT_CLEARANCE = 0.06


def _parse_force_candidates(text: str) -> list[float]:
    candidates: list[float] = []
    for match in re.finditer(r"(\d+)-(\d+)\s*N", text):
        lo, hi = match.groups()
        candidates.append((float(lo) + float(hi)) / 2.0)
    for match in re.finditer(r"夹爪力\s*(\d+)\s*N?", text):
        candidates.append(float(match.group(1)))
    return candidates


def _parse_velocity_band(text: str) -> tuple[float, float] | None:
    match = re.search(r"(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\s*m/s", text)
    if match:
        lo, hi = match.groups()
        lo_f = float(lo)
        hi_f = float(hi)
        if lo_f <= hi_f:
            return lo_f, hi_f
    return None


def _parse_clearance_floor(text: str) -> float | None:
    match = re.search(r"(?:不低于|不少于|至少|不应低于)\s*(0?\.\d+)\s*m(?!/s)", text)
    if match:
        return float(match.group(1))
    return None


def _parse_clearance_target(text: str) -> float | None:
    if "净空" not in text and "抬升" not in text:
        return None
    match = re.search(r"(?:保持|宜保持|维持)\s*(0?\.\d+)\s*m(?:左右)?", text)
    if match:
        return float(match.group(1))
    match = re.search(r"(0?\.\d+)\s*m左右", text)
    if match:
        return float(match.group(1))
    return None


def _split_clauses(text: str) -> list[str]:
    clauses = []
    for part in re.split(r"[；。]", text):
        cleaned = part.strip()
        if cleaned:
            clauses.append(cleaned)
    return clauses


def _heuristic_defaults(task_description: str) -> tuple[float, float, float, float]:
    for hint in TASK_DESCRIPTION_HINTS:
        if all(keyword in task_description for keyword in hint["keywords"]):
            return (
                hint["force"],
                hint["approach_height"],
                hint["transport_velocity"],
                hint["lift_clearance"],
            )
    return 25.0, 0.05, 0.30, 0.06


def _matched_hint(task_description: str) -> dict[str, Any] | None:
    for hint in TASK_DESCRIPTION_HINTS:
        if all(keyword in task_description for keyword in hint["keywords"]):
            return hint
    return None


def _parse_json_params(text: str) -> dict[str, Any] | None:
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
            pass
    return None


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
        documents.append(
            Document(
                page_content=content,
                metadata={
                    "entry_id": entry_id,
                    "category": category,
                },
            )
        )
    return documents


def _task_features(task_description: str) -> dict[str, bool]:
    return {
        "metal": "金属" in task_description,
        "smooth_metal": "光滑金属" in task_description,
        "rubber": "橡胶" in task_description,
        "small": "小型" in task_description,
        "large": "大型" in task_description,
        "heavy": "重型" in task_description,
        "thin_wall": "薄壁" in task_description,
        "high_speed": "高速" in task_description,
        "long_transfer": "长距离" in task_description,
    }


def _rule_from_clause(task_description: str, clause: str, category: str, entry_id: str) -> dict[str, Any] | None:
    features = _task_features(task_description)
    matched_terms: list[str] = []
    score = 0.0
    raw_support_contact = "真空吸附" in clause or "多点分散支撑" in clause
    lift_stage_evidence = False
    if features["thin_wall"] and "薄壁" in clause:
        matched_terms.append("thin_wall")
        score += 4.0
    if (features["smooth_metal"] or features["metal"]) and "金属" in clause:
        matched_terms.append("metal")
        score += 3.0
    if features["rubber"] and "橡胶" in clause:
        matched_terms.append("rubber")
        score += 3.0
    if features["small"] and "小型" in clause:
        matched_terms.append("small")
        score += 2.5
    if features["large"] and "大型" in clause:
        matched_terms.append("large")
        score += 2.5
    if features["heavy"] and ("重心" in clause or "负载" in clause or "重型" in clause or "payload" in clause.lower() or "扭矩" in clause):
        matched_terms.append("heavy")
        score += 2.0
    if features["high_speed"] and ("高速运动" in clause or "惯性前冲" in clause or "节拍" in clause):
        matched_terms.append("high_speed")
        score += 3.0
    if features["long_transfer"] and ("重心偏移" in clause or "轨迹规划" in clause or "搬运" in clause):
        matched_terms.append("long_transfer")
        score += 2.0
    if (
        (features["long_transfer"] or features["large"] or features["heavy"])
        and any(token in clause for token in ("松紧度", "脱落", "接近-抓取-抬升-放置序列", "确认工件放置稳定"))
    ):
        lift_stage_evidence = True
        matched_terms.append("lift_stage")
        score += 2.0
    elif (
        features["long_transfer"]
        and "稳定" in clause
        and any(token in clause for token in ("抓取", "工件", "重心"))
    ):
        lift_stage_evidence = True
        matched_terms.append("lift_stage")
        score += 1.5
    support_contact = raw_support_contact and bool(matched_terms)
    if any(token in clause for token in ("夹爪力", "夹持力", "抓取")):
        score += 1.5
    if support_contact:
        score += 1.0

    force_candidates = _parse_force_candidates(clause)
    force_margin = 0.0
    if features["high_speed"] and ("夹持力余量" in clause or "惯性前冲" in clause):
        force_margin += 2.5 if "夹持力余量" in clause else 1.5

    velocity_floor = None
    velocity_band = None
    if features["high_speed"] and ("节拍" in clause or "高速分拣" in clause or "高速运动" in clause):
        velocity_floor = 0.36 if not features["thin_wall"] else 0.22

    velocity_cap = None
    if "惯性前冲" in clause:
        velocity_cap = 0.34 if not features["thin_wall"] else 0.24
    if features["thin_wall"] and ("薄壁" in clause or "易碎" in clause):
        velocity_cap = min(velocity_cap or 1.0, 0.24 if features["high_speed"] else 0.20)

    if ("搬运速度" in clause or "速度" in clause) and (features["long_transfer"] or features["large"]):
        velocity_band = _parse_velocity_band(clause)
        if velocity_band is not None:
            lo, hi = velocity_band
            velocity_floor = max(velocity_floor or lo, lo)
            velocity_cap = min(velocity_cap or hi, hi)

    clearance_delta = 0.0
    if support_contact and not features["thin_wall"]:
        clearance_delta += 0.012
    clearance_floor = None
    clearance_target = None
    if features["long_transfer"] or features["large"]:
        clearance_floor = _parse_clearance_floor(clause)
        clearance_target = _parse_clearance_target(clause)

    approach_height_target = None
    if features["thin_wall"] and "薄壁" in clause:
        approach_height_target = 0.02
    elif features["long_transfer"] and ("轨迹规划" in clause or "重心偏移" in clause):
        approach_height_target = 0.06
    elif features["heavy"] and ("重心" in clause or "重型" in clause):
        approach_height_target = 0.055

    generic_force = any(marker in clause for marker in GENERIC_FORCE_MARKERS)
    numeric_motion_evidence = velocity_band is not None or clearance_floor is not None or clearance_target is not None
    rule_type = "support"
    if force_candidates:
        rule_type = "force"
    elif (
        velocity_floor is not None
        or velocity_cap is not None
        or clearance_delta > 0.0
        or clearance_floor is not None
        or clearance_target is not None
        or approach_height_target is not None
    ):
        rule_type = "motion"

    if (
        score <= 0.0
        and not force_candidates
        and velocity_floor is None
        and velocity_cap is None
        and clearance_delta == 0.0
        and clearance_floor is None
        and clearance_target is None
        and approach_height_target is None
        and not lift_stage_evidence
    ):
        return None

    return {
        "entry_id": entry_id,
        "category": category,
        "clause": clause,
        "score": round(score, 4),
        "matched_terms": matched_terms,
        "specificity": len(set(matched_terms)),
        "force_candidates": force_candidates,
        "force_margin": round(force_margin, 4),
        "velocity_floor": velocity_floor,
        "velocity_cap": velocity_cap,
        "velocity_band": list(velocity_band) if velocity_band is not None else None,
        "clearance_delta": round(clearance_delta, 4),
        "clearance_floor": clearance_floor,
        "clearance_target": clearance_target,
        "approach_height_target": approach_height_target,
        "support_contact": support_contact,
        "lift_stage_evidence": lift_stage_evidence,
        "generic_force": generic_force,
        "numeric_motion_evidence": numeric_motion_evidence,
        "rule_type": rule_type,
    }


def _build_rule_trace(task_description: str, docs: list[Document]) -> list[dict[str, Any]]:
    rules: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for doc in docs:
        entry_id = str(doc.metadata.get("entry_id", ""))
        category = str(doc.metadata.get("category", ""))
        for clause in _split_clauses(doc.page_content):
            key = (entry_id, clause)
            if key in seen:
                continue
            seen.add(key)
            rule = _rule_from_clause(task_description, clause, category, entry_id)
            if rule is not None:
                rules.append(rule)
    rules.sort(key=lambda item: (item["specificity"], item["score"]), reverse=True)
    return rules


def _aggregate_plan(
    task_description: str,
    rules: list[dict[str, Any]],
    force_rule_mode: str = "all",
    motion_rule_mode: str = "all",
) -> tuple[float, float, float, float, float, float, float, float, dict[str, Any]]:
    if force_rule_mode not in {"all", "generic_only"}:
        raise ValueError(f"未知 force_rule_mode: {force_rule_mode}")
    if motion_rule_mode not in {"all", "disabled"}:
        raise ValueError(f"未知 motion_rule_mode: {motion_rule_mode}")
    default_force, default_height, default_velocity, default_clearance = _heuristic_defaults(task_description)
    matched_hint = _matched_hint(task_description)
    features = _task_features(task_description)
    calibration_notes: list[str] = []

    specific_force_rules = [rule for rule in rules if rule["force_candidates"] and rule["specificity"] > 0]
    generic_force_rules = [rule for rule in rules if rule["force_candidates"] and rule["specificity"] == 0]
    motion_rules = [
        rule
        for rule in rules
        if rule["velocity_floor"] is not None
        or rule["velocity_cap"] is not None
        or rule["clearance_delta"] > 0.0
        or rule.get("clearance_floor") is not None
        or rule.get("clearance_target") is not None
        or rule["approach_height_target"] is not None
    ]
    available_specific_force_rules = bool(specific_force_rules)
    suppressed_specific_force_rules = force_rule_mode == "generic_only" and available_specific_force_rules
    available_motion_rules = bool(motion_rules)
    suppressed_motion_rules = motion_rule_mode == "disabled" and available_motion_rules
    support_contact_rules = [rule for rule in rules if rule["support_contact"]]
    available_support_contact_rules = bool(support_contact_rules)
    numeric_motion_rules = [rule for rule in motion_rules if rule.get("numeric_motion_evidence")]
    available_numeric_motion_rules = bool(numeric_motion_rules)
    alignment_rules = [
        rule
        for rule in motion_rules
        if features["long_transfer"]
        and ("重心" in rule["clause"] or "力矩" in rule["clause"] or "轨迹规划" in rule["clause"])
    ]
    available_alignment_rules = bool(alignment_rules)
    lift_stage_rules = [
        rule
        for rule in rules
        if features["long_transfer"]
        and features["large"]
        and rule.get("lift_stage_evidence")
    ]
    available_lift_stage_rules = bool(lift_stage_rules)
    if force_rule_mode == "generic_only":
        selected_force_rules = generic_force_rules[:3]
    else:
        selected_force_rules = specific_force_rules[:3] if specific_force_rules else generic_force_rules[:3]

    selected_rules = rules[:3]
    support_score = round(sum(rule["score"] for rule in selected_rules) / len(selected_rules), 4) if selected_rules else 0.0
    force_conflict = 0
    if specific_force_rules and generic_force_rules:
        specific_center = sum(sum(rule["force_candidates"]) / len(rule["force_candidates"]) for rule in specific_force_rules[:2]) / min(2, len(specific_force_rules))
        generic_center = sum(sum(rule["force_candidates"]) / len(rule["force_candidates"]) for rule in generic_force_rules[:2]) / min(2, len(generic_force_rules))
        if abs(specific_center - generic_center) >= 6.0:
            force_conflict += 1
    if len(selected_force_rules) >= 2:
        centers = [sum(rule["force_candidates"]) / len(rule["force_candidates"]) for rule in selected_force_rules]
        if max(centers) - min(centers) >= 8.0:
            force_conflict += 1

    evidence_summary = summarize_control_evidence(
        features=features,
        rules=rules,
        selected_force_rules=selected_force_rules,
        specific_force_rules=specific_force_rules,
        motion_rules=motion_rules,
        numeric_motion_rules=numeric_motion_rules,
        alignment_rules=alignment_rules,
        lift_stage_rules=lift_stage_rules,
        support_contact_rules=support_contact_rules,
        default_force=default_force,
    )
    belief = build_control_belief(
        features=features,
        evidence_summary=evidence_summary,
        dynamic_transport_mode=evidence_summary.preferred_transport_mode,
        support_score=support_score,
        conflict_count=force_conflict,
        force_rule_mode=force_rule_mode,
        motion_rule_mode=motion_rule_mode,
        available_specific_force_rules=available_specific_force_rules,
        available_motion_rules=available_motion_rules,
        available_numeric_motion_rules=available_numeric_motion_rules,
        available_alignment_rules=available_alignment_rules,
        available_lift_stage_rules=available_lift_stage_rules,
        available_support_contact_rules=available_support_contact_rules,
    )

    velocity_floors = [float(rule["velocity_floor"]) for rule in rules if rule["velocity_floor"] is not None]
    velocity_caps = [float(rule["velocity_cap"]) for rule in rules if rule["velocity_cap"] is not None]
    velocity_bands = [tuple(rule["velocity_band"]) for rule in numeric_motion_rules if rule.get("velocity_band") is not None]
    clearance_floors = [float(rule["clearance_floor"]) for rule in numeric_motion_rules if rule.get("clearance_floor") is not None]
    clearance_targets = [float(rule["clearance_target"]) for rule in numeric_motion_rules if rule.get("clearance_target") is not None]
    approach_targets = [float(rule["approach_height_target"]) for rule in motion_rules if rule["approach_height_target"] is not None]

    belief_force_floor = None
    if belief.uncertainty.conservative_mode:
        belief_force_floor = max(
            float(default_force),
            float(evidence_summary.force_center) - 0.35 * float(evidence_summary.force_std),
        )

    composite_force_floor = None
    if (
        matched_hint is not None
        and len(matched_hint["keywords"]) >= 2
        and (features["heavy"] or features["long_transfer"])
        and force_rule_mode == "all"
    ):
        composite_force_floor = float(default_force)

    motion_aware_force_floor = None
    if (
        motion_rule_mode == "all"
        and available_motion_rules
        and features["heavy"]
        and features["high_speed"]
        and features["metal"]
    ):
        motion_aware_force_floor = float(default_force) + 1.0
    elif motion_rule_mode == "all" and available_motion_rules and features["long_transfer"] and features["large"]:
        motion_aware_force_floor = float(default_force) + 1.5

    rubber_material_force_floor = None
    if features["rubber"] and not features["high_speed"]:
        rubber_material_force_floor = float(default_force) + 3.5

    thin_wall_support_force_floor = None
    if motion_rule_mode == "all" and features["thin_wall"] and available_support_contact_rules and not bool(specific_force_rules):
        thin_wall_support_force_floor = float(default_force) + 4.0

    heavy_metal_force_center_floor = None
    if force_rule_mode == "all" and bool(specific_force_rules) and features["heavy"] and features["metal"] and not features["high_speed"]:
        heavy_metal_force_center_floor = float(default_force) + 4.0

    dynamic_heavy_metal_force_center_floor = None
    if motion_rule_mode == "all" and available_motion_rules and force_rule_mode == "all" and bool(specific_force_rules) and features["heavy"] and features["high_speed"] and features["metal"]:
        dynamic_heavy_metal_force_center_floor = float(default_force) + 3.0

    static_smooth_metal_force_cap = None
    dynamic_smooth_metal_force_cap = None
    if features["smooth_metal"] and not features["heavy"]:
        if features["high_speed"] and available_motion_rules and bool(specific_force_rules):
            dynamic_smooth_metal_force_cap = float(default_force) - 6.0
        elif not features["high_speed"]:
            static_smooth_metal_force_cap = float(default_force) - 1.5

    force_floor_candidates = [
        candidate
        for candidate in (
            belief_force_floor,
            composite_force_floor,
            motion_aware_force_floor,
            rubber_material_force_floor,
            thin_wall_support_force_floor,
            heavy_metal_force_center_floor,
            dynamic_heavy_metal_force_center_floor,
        )
        if candidate is not None
    ]
    force_cap_candidates = [
        candidate
        for candidate in (
            dynamic_smooth_metal_force_cap,
            static_smooth_metal_force_cap,
        )
        if candidate is not None
    ]

    long_transfer_velocity_band = None
    transport_velocity_floor = max(velocity_floors) if velocity_floors else None
    transport_velocity_cap = min(velocity_caps) if velocity_caps else None
    if features["heavy"] and features["high_speed"] and available_motion_rules:
        transport_velocity_cap = min(float(default_velocity), transport_velocity_cap or float(default_velocity))
    if velocity_bands:
        lo = max(band[0] for band in velocity_bands)
        hi = min(band[1] for band in velocity_bands)
        if lo <= hi:
            long_transfer_velocity_band = [round(lo, 4), round(hi, 4)]
            transport_velocity_floor = max(transport_velocity_floor or lo, lo)
            transport_velocity_cap = min(transport_velocity_cap or hi, hi)

    long_transfer_placement_velocity_cap = None
    if features["long_transfer"] and features["large"] and available_numeric_motion_rules:
        precision_cap = 0.15 if belief.task_constraints.precision_priority < 0.85 else 0.14
        long_transfer_placement_velocity_cap = round(max(0.12, min(transport_velocity_cap or default_velocity, precision_cap)), 4)

    dynamic_smooth_metal_clearance_floor = None
    if features["smooth_metal"] and features["high_speed"] and not features["heavy"] and available_motion_rules:
        dynamic_smooth_metal_clearance_floor = 0.065

    rubber_material_clearance_floor = None
    if features["rubber"] and not features["high_speed"]:
        rubber_material_clearance_floor = 0.06

    thin_wall_support_clearance_cap = float(default_clearance) if (features["thin_wall"] and available_support_contact_rules and not features["high_speed"]) else None
    long_transfer_numeric_clearance_floor = max(clearance_floors) if clearance_floors else None
    long_transfer_clearance_target = None
    if clearance_targets:
        weighted_target = sum(clearance_targets) / len(clearance_targets)
        long_transfer_clearance_target = weighted_target
        if long_transfer_numeric_clearance_floor is not None:
            long_transfer_clearance_target = max(
                long_transfer_numeric_clearance_floor,
                min(weighted_target, long_transfer_numeric_clearance_floor + 0.01),
            )

    clearance_floor_candidates = [
        candidate
        for candidate in (
            0.062 if belief.task_constraints.stability_priority >= 0.82 and belief.object_state.dynamic_load_band == "high" else None,
            0.055 if belief.task_constraints.stability_priority >= 0.82 and belief.object_state.dynamic_load_band != "high" else None,
            rubber_material_clearance_floor,
            dynamic_smooth_metal_clearance_floor,
            long_transfer_numeric_clearance_floor,
        )
        if candidate is not None
    ]
    clearance_floor = max(clearance_floor_candidates) if clearance_floor_candidates else None
    clearance_target = long_transfer_clearance_target

    approach_height_target = None
    if approach_targets:
        approach_height_target = sum(approach_targets) / len(approach_targets)
    elif belief.task_constraints.safety_priority >= 0.85 and belief.object_state.fragility_band == "high":
        approach_height_target = max(default_height, 0.05)

    long_transfer_alignment_target = None
    if belief.task_constraints.required_alignment:
        long_transfer_alignment_target = min(
            0.95,
            0.65
            + 0.05 * min(2, len(alignment_rules))
            + (0.05 if available_numeric_motion_rules else 0.0)
            + (0.05 if bool(specific_force_rules) else 0.0),
        )

    long_transfer_lift_force_margin = 0.0
    if belief.task_constraints.required_lift_margin:
        long_transfer_lift_force_margin = min(
            0.8,
            0.25
            + 0.10 * min(2, len(lift_stage_rules))
            + (0.05 if available_alignment_rules else 0.0)
            + (0.05 if available_numeric_motion_rules else 0.0),
        )

    long_transfer_alignment_force_margin = 0.0
    if belief.task_constraints.required_alignment:
        long_transfer_alignment_force_margin = min(
            0.6,
            0.25 + 0.5 * max(0.0, (long_transfer_alignment_target or 0.55) - 0.5),
        )

    motion_path_force_compensation = min(
        2.0,
        max(0.0, 10.0 * max(0.0, (clearance_target or default_clearance) - default_clearance)),
    )
    if approach_height_target is not None:
        motion_path_force_compensation += min(0.8, 6.0 * max(0.0, approach_height_target - default_height))

    hints = EvidenceConstraintHints(
        force_floor=max(force_floor_candidates) if force_floor_candidates else None,
        force_cap=min(force_cap_candidates) if force_cap_candidates else None,
        transport_velocity_floor=transport_velocity_floor,
        transport_velocity_cap=transport_velocity_cap,
        placement_velocity_cap=long_transfer_placement_velocity_cap,
        clearance_floor=clearance_floor,
        clearance_target=clearance_target,
        approach_height_target=approach_height_target,
        alignment_target=long_transfer_alignment_target,
        lift_force_margin=long_transfer_lift_force_margin,
        transfer_force_margin=long_transfer_alignment_force_margin or (2.0 if features["smooth_metal"] and features["high_speed"] and not features["heavy"] else 0.0),
        gripper_force_bias=motion_path_force_compensation,
        source_notes=[
            "belief_constraint_seed",
            *(["numeric_motion_band"] if long_transfer_velocity_band is not None else []),
            *(["specific_force_center"] if selected_force_rules else []),
        ],
    )
    pre_solve_plan, seed_trace = synthesize_control_seed(
        {
            "gripper_force": default_force,
            "approach_height": default_height,
            "transport_velocity": default_velocity,
            "lift_force": default_force,
            "transfer_force": default_force,
            "placement_velocity": default_velocity,
            "transfer_alignment": 0.0,
            "lift_clearance": default_clearance,
        },
        belief,
        hints,
    )
    solved_plan, solver_trace = solve_control_plan(pre_solve_plan, belief)
    calibration_notes = list(seed_trace["seed_notes"])
    for note in solver_trace["solver_adjustment_notes"]:
        if note not in calibration_notes:
            calibration_notes.append(note)

    dynamic_transport_mode = belief.task_constraints.preferred_transport_mode
    high_speed_transfer_force_margin = 2.0 if features["smooth_metal"] and features["high_speed"] and not features["heavy"] else None
    high_speed_placement_velocity_target = long_transfer_placement_velocity_cap if dynamic_transport_mode == "high_speed_low_friction" else None
    long_transfer_lift_force_target = round(float(pre_solve_plan["lift_force"]), 4) if long_transfer_lift_force_margin > 0.0 else None

    trace = {
        "selected_rules": [
            {
                "entry_id": rule["entry_id"],
                "category": rule["category"],
                "clause": rule["clause"],
                "score": rule["score"],
                "matched_terms": rule["matched_terms"],
                "force_candidates": rule["force_candidates"],
                "rule_type": rule["rule_type"],
            }
            for rule in selected_rules
        ],
        "rule_count": len(rules),
        "support_score": support_score,
        "conflict_count": force_conflict,
        "force_rule_mode": force_rule_mode,
        "motion_rule_mode": motion_rule_mode,
        **belief.to_trace_dict(),
        "seed_mode": seed_trace["seed_mode"],
        "seed_notes": seed_trace["seed_notes"],
        "seed_hints": seed_trace["hints"],
        "seed_plan": seed_trace["seed_plan"],
        "matched_hint_keywords": list(matched_hint["keywords"]) if matched_hint is not None else [],
        "available_specific_force_rules": available_specific_force_rules,
        "suppressed_specific_force_rules": suppressed_specific_force_rules,
        "available_motion_rules": available_motion_rules,
        "suppressed_motion_rules": suppressed_motion_rules,
        "available_support_contact_rules": available_support_contact_rules,
        "available_numeric_motion_rules": available_numeric_motion_rules,
        "available_alignment_rules": available_alignment_rules,
        "available_lift_stage_rules": available_lift_stage_rules,
        "used_specific_force_rules": bool(selected_force_rules) and not suppressed_specific_force_rules and bool(specific_force_rules),
        "used_generic_force_rules": bool(generic_force_rules),
        "used_motion_rules": available_motion_rules and not suppressed_motion_rules,
        "used_support_contact_rules": available_support_contact_rules and not suppressed_motion_rules,
        "used_alignment_rules": available_alignment_rules and not suppressed_motion_rules and dynamic_transport_mode == "long_transfer",
        "used_lift_stage_rules": available_lift_stage_rules and not suppressed_motion_rules and long_transfer_lift_force_margin > 0.0,
        "pre_solve_plan": pre_solve_plan,
        "belief_force_floor": belief_force_floor,
        "composite_force_floor": composite_force_floor,
        "motion_aware_force_floor": motion_aware_force_floor,
        "thin_wall_support_force_floor": thin_wall_support_force_floor,
        "rubber_material_force_floor": rubber_material_force_floor,
        "heavy_metal_force_center_floor": heavy_metal_force_center_floor,
        "dynamic_heavy_metal_force_center_floor": dynamic_heavy_metal_force_center_floor,
        "long_transfer_force_center_floor": max(force_floor_candidates) if features["long_transfer"] and features["large"] and force_floor_candidates else None,
        "long_transfer_stage_force_floor": max(force_floor_candidates) if features["long_transfer"] and features["large"] and long_transfer_clearance_target is not None and force_floor_candidates else None,
        "long_transfer_dynamic_force_margin": round(hints.gripper_force_bias, 4) if hints.gripper_force_bias > 0.0 else None,
        "long_transfer_velocity_band": long_transfer_velocity_band,
        "long_transfer_placement_velocity_cap": long_transfer_placement_velocity_cap,
        "dynamic_transport_mode": dynamic_transport_mode,
        "lift_force": round(float(solved_plan["lift_force"]), 4),
        "long_transfer_lift_force_target": long_transfer_lift_force_target,
        "long_transfer_lift_force_margin": round(long_transfer_lift_force_margin, 4) if long_transfer_lift_force_margin > 0.0 else None,
        "transfer_force": round(float(solved_plan["transfer_force"]), 4),
        "transfer_alignment": round(float(solved_plan["transfer_alignment"]), 4),
        "long_transfer_alignment_target": long_transfer_alignment_target,
        "long_transfer_alignment_force_margin": round(long_transfer_alignment_force_margin, 4) if long_transfer_alignment_force_margin > 0.0 else None,
        "high_speed_transfer_force_margin": high_speed_transfer_force_margin,
        "high_speed_placement_velocity_target": high_speed_placement_velocity_target,
        "motion_aware_force_cap": min(force_cap_candidates) if force_cap_candidates else None,
        "dynamic_smooth_metal_force_cap": dynamic_smooth_metal_force_cap,
        "dynamic_smooth_metal_clearance_floor": dynamic_smooth_metal_clearance_floor,
        "static_smooth_metal_force_cap": static_smooth_metal_force_cap,
        "thin_wall_support_clearance_cap": thin_wall_support_clearance_cap,
        "rubber_material_clearance_floor": rubber_material_clearance_floor,
        "long_transfer_numeric_clearance_floor": long_transfer_numeric_clearance_floor,
        "long_transfer_clearance_target": long_transfer_clearance_target,
        "motion_path_force_compensation": round(motion_path_force_compensation, 4),
        "solver_mode": solver_trace["solver_mode"],
        "solver_selected_candidate": solver_trace["solver_selected_candidate"],
        "solver_selected_score": solver_trace["solver_selected_score"],
        "solver_score_breakdown": solver_trace["solver_score_breakdown"],
        "solver_candidate_scores": solver_trace["solver_candidate_scores"],
        "solver_seed_candidate": solver_trace["solver_seed_candidate"],
        "solver_seed_score": solver_trace["solver_seed_score"],
        "solver_local_search_iterations": solver_trace["solver_local_search_iterations"],
        "solver_local_search_improvement": solver_trace["solver_local_search_improvement"],
        "solver_local_search_trace": solver_trace["solver_local_search_trace"],
        "solver_adjustment_notes": solver_trace["solver_adjustment_notes"],
        "calibration_notes": calibration_notes,
    }
    return (
        max(5.0, min(50.0, solved_plan["gripper_force"])),
        max(0.02, min(0.08, solved_plan["approach_height"])),
        max(0.12, min(0.8, solved_plan["transport_velocity"])),
        max(0.03, min(0.14, solved_plan["lift_clearance"])),
        max(solved_plan["gripper_force"], min(50.0, solved_plan["lift_force"])),
        max(solved_plan["gripper_force"], min(50.0, solved_plan["transfer_force"])),
        max(0.12, min(0.8, solved_plan["placement_velocity"])),
        max(0.0, min(1.0, solved_plan["transfer_alignment"])),
        trace,
    )

    force = default_force
    if selected_force_rules:
        weights = [max(0.5, rule["score"] + 0.8 * rule["specificity"]) for rule in selected_force_rules]
        candidates = [sum(rule["force_candidates"]) / len(rule["force_candidates"]) for rule in selected_force_rules]
        force = sum(weight * candidate for weight, candidate in zip(weights, candidates)) / sum(weights)
    if evidence_summary.specific_force_confidence >= 0.2:
        force = 0.65 * force + 0.35 * float(evidence_summary.force_center)

    force_margin = max((rule["force_margin"] for rule in rules), default=0.0)
    force += force_margin
    composite_force_floor = None
    if (
        matched_hint is not None
        and len(matched_hint["keywords"]) >= 2
        and (features["heavy"] or features["long_transfer"])
        and force_rule_mode == "all"
    ):
        composite_force_floor = float(default_force)
        if force < composite_force_floor:
            force = composite_force_floor
            calibration_notes.append("composite_force_floor")

    belief_force_floor = None
    if belief.uncertainty.conservative_mode:
        belief_force_floor = max(
            float(default_force),
            float(evidence_summary.force_center) - 0.35 * float(evidence_summary.force_std),
        )
        if force < belief_force_floor:
            force = belief_force_floor
            calibration_notes.append("belief_force_floor")

    motion_aware_force_floor = None
    thin_wall_support_force_floor = None
    rubber_material_force_floor = None
    heavy_metal_force_center_floor = None
    dynamic_heavy_metal_force_center_floor = None
    long_transfer_force_center_floor = None
    long_transfer_stage_force_floor = None
    if (
        motion_rule_mode == "all"
        and available_motion_rules
        and force_rule_mode == "all"
        and bool(specific_force_rules)
        and features["heavy"]
        and features["high_speed"]
        and features["metal"]
    ):
        # Heavy high-speed transfer should enforce a stronger force floor only when
        # the motion plan actually chooses the dynamic transport path.
        motion_aware_force_floor = float(default_force) + 1.0
        if force < motion_aware_force_floor:
            force = motion_aware_force_floor
            calibration_notes.append("motion_aware_force_floor")
    elif (
        features["rubber"]
        and not features["high_speed"]
    ):
        # Low-speed rubber should use the upper half of the compliant-material
        # band; the naive 10N midpoint leaves too much slip margin.
        rubber_material_force_floor = float(default_force) + 3.5
        if force < rubber_material_force_floor:
            force = rubber_material_force_floor
            calibration_notes.append("rubber_material_force_floor")
    elif (
        motion_rule_mode == "all"
        and features["thin_wall"]
        and not features["high_speed"]
        and available_support_contact_rules
        and not bool(specific_force_rules)
    ):
        # Thin-wall support evidence implies distributed contact rather than a
        # narrow radial pinch, so the fragile default force can be safely lifted.
        thin_wall_support_force_floor = float(default_force) + 4.0
        if force < thin_wall_support_force_floor:
            force = thin_wall_support_force_floor
            calibration_notes.append("thin_wall_support_force_floor")
    elif (
        motion_rule_mode == "all"
        and available_motion_rules
        and features["long_transfer"]
        and features["large"]
    ):
        motion_aware_force_floor = float(default_force) + 1.5
        if force < motion_aware_force_floor:
            force = motion_aware_force_floor
            calibration_notes.append("motion_aware_force_floor")

    height = default_height if motion_rule_mode == "all" else 0.05
    if motion_rule_mode == "all":
        height_targets = [rule for rule in motion_rules if rule["approach_height_target"] is not None]
        if height_targets:
            weights = [max(0.5, rule["score"] + 0.5 * rule["specificity"]) for rule in height_targets]
            targets = [float(rule["approach_height_target"]) for rule in height_targets]
            height = sum(weight * target for weight, target in zip(weights, targets)) / sum(weights)
    if belief.task_constraints.safety_priority >= 0.85 and belief.object_state.fragility_band == "high":
        height = min(0.08, max(height, default_height))

    velocity = default_velocity if motion_rule_mode == "all" else NEUTRAL_TRANSPORT_VELOCITY
    if motion_rule_mode == "all":
        velocity_floors = [rule["velocity_floor"] for rule in rules if rule["velocity_floor"] is not None]
        velocity_caps = [rule["velocity_cap"] for rule in rules if rule["velocity_cap"] is not None]
        if velocity_floors:
            velocity = max(velocity, max(velocity_floors))
        if velocity_caps:
            velocity = min(velocity, min(velocity_caps))
        if features["heavy"] and features["high_speed"] and available_motion_rules:
            # For heavy metal transfer, keep the speed within the composite prior instead of
            # inheriting the generic fast-transfer cap.
            capped_velocity = min(velocity, default_velocity)
            if capped_velocity != velocity:
                calibration_notes.append("composite_velocity_cap")
            velocity = capped_velocity
    if belief.uncertainty.conservative_mode:
        velocity = max(0.12, velocity - 0.015)
        calibration_notes.append("belief_velocity_guard")
    long_transfer_velocity_band = None
    if motion_rule_mode == "all" and features["long_transfer"] and features["large"]:
        velocity_bands = [tuple(rule["velocity_band"]) for rule in numeric_motion_rules if rule.get("velocity_band") is not None]
        if velocity_bands:
            lo = max(band[0] for band in velocity_bands)
            hi = min(band[1] for band in velocity_bands)
            if lo <= hi:
                long_transfer_velocity_band = [round(lo, 4), round(hi, 4)]
                bounded_velocity = min(max(velocity, lo), hi)
                if bounded_velocity != velocity:
                    calibration_notes.append("long_transfer_velocity_band")
                velocity = bounded_velocity

    placement_velocity = velocity
    long_transfer_placement_velocity_cap = None
    if motion_rule_mode == "all" and features["long_transfer"] and features["large"] and available_numeric_motion_rules:
        precision_cap = 0.15 if belief.task_constraints.precision_priority < 0.85 else 0.14
        long_transfer_placement_velocity_cap = round(max(0.12, min(velocity, precision_cap)), 4)
        placement_velocity = long_transfer_placement_velocity_cap
        if placement_velocity < velocity:
            calibration_notes.append("long_transfer_placement_velocity_cap")

    clearance = default_clearance if motion_rule_mode == "all" else NEUTRAL_LIFT_CLEARANCE
    if motion_rule_mode == "all" and rules:
        clearance += max((rule["clearance_delta"] for rule in rules), default=0.0)
    belief_clearance_floor = None
    if belief.task_constraints.stability_priority >= 0.82:
        belief_clearance_floor = 0.062 if belief.object_state.dynamic_load_band == "high" else 0.055
        if clearance < belief_clearance_floor:
            clearance = belief_clearance_floor
            calibration_notes.append("belief_clearance_floor")
    thin_wall_support_clearance_cap = None
    rubber_material_clearance_floor = None
    long_transfer_numeric_clearance_floor = None
    long_transfer_clearance_target = None
    if (
        motion_rule_mode == "all"
        and features["thin_wall"]
        and not features["high_speed"]
        and available_support_contact_rules
    ):
        # Distributed support / vacuum evidence should reduce the need for extra
        # vertical clearance, not inflate the motion path on thin-wall objects.
        thin_wall_support_clearance_cap = float(default_clearance)
        if clearance > thin_wall_support_clearance_cap:
            clearance = thin_wall_support_clearance_cap
            calibration_notes.append("thin_wall_support_clearance_cap")
    elif (
        features["rubber"]
        and not features["high_speed"]
    ):
        # Compliant rubber pickup needs a slightly higher lift floor to avoid
        # dragging / recontact during transport.
        rubber_material_clearance_floor = 0.06
        if clearance < rubber_material_clearance_floor:
            clearance = rubber_material_clearance_floor
            calibration_notes.append("rubber_material_clearance_floor")
    if motion_rule_mode == "all":
        explicit_clearance_floors = [float(rule["clearance_floor"]) for rule in numeric_motion_rules if rule.get("clearance_floor") is not None]
        explicit_clearance_targets = [float(rule["clearance_target"]) for rule in numeric_motion_rules if rule.get("clearance_target") is not None]
        if explicit_clearance_floors:
            floor_value = max(explicit_clearance_floors)
            if features["long_transfer"] and features["large"]:
                long_transfer_numeric_clearance_floor = floor_value
            if clearance < floor_value:
                clearance = floor_value
                calibration_notes.append("explicit_clearance_floor")
        if features["long_transfer"] and features["large"] and explicit_clearance_targets:
            weighted_target = sum(explicit_clearance_targets) / len(explicit_clearance_targets)
            target_value = weighted_target
            if explicit_clearance_floors:
                target_value = max(max(explicit_clearance_floors), min(weighted_target, max(explicit_clearance_floors) + 0.01))
            long_transfer_clearance_target = round(target_value, 4)
            if clearance < target_value:
                clearance = target_value
                calibration_notes.append("long_transfer_clearance_target")

    neutral_motion_path = 2.0 * 0.05 + 2.0 * NEUTRAL_LIFT_CLEARANCE
    motion_path_delta = max(
        0.0,
        (2.0 * float(height) + 2.0 * float(clearance)) - neutral_motion_path,
    )
    motion_path_force_compensation = 0.0
    if motion_rule_mode == "all" and motion_path_delta > 0.0:
        # Longer vertical path raises slip risk on heavy objects; compensate force accordingly.
        motion_path_force_compensation = min(2.0, 10.0 * motion_path_delta)
        force += motion_path_force_compensation
        calibration_notes.append("motion_path_force_compensation")

    # Round11: after motion-path compensation, recenter broad force bands toward
    # the task-specific nominal zone instead of leaving hard tasks at conservative
    # lower bounds or overly high caps.
    if (
        motion_rule_mode == "all"
        and available_motion_rules
        and features["long_transfer"]
        and features["large"]
        and force_rule_mode == "all"
        and bool(specific_force_rules)
    ):
        # Round12: the long-transfer center correction should come from
        # large-part specific force evidence rather than motion-only rules.
        long_transfer_force_center_floor = float(default_force) + 3.0
        if force < long_transfer_force_center_floor:
            force = long_transfer_force_center_floor
            calibration_notes.append("long_transfer_force_center_floor")
    if (
        motion_rule_mode == "all"
        and force_rule_mode == "all"
        and features["long_transfer"]
        and features["large"]
        and bool(specific_force_rules)
        and available_numeric_motion_rules
        and long_transfer_clearance_target is not None
        and long_transfer_clearance_target >= 0.08
    ):
        long_transfer_stage_force_floor = float(default_force) + 4.0
        if force < long_transfer_stage_force_floor:
            force = long_transfer_stage_force_floor
            calibration_notes.append("long_transfer_stage_force_floor")
    long_transfer_dynamic_force_margin = None
    if (
        motion_rule_mode == "all"
        and features["long_transfer"]
        and features["large"]
        and force_rule_mode == "all"
        and bool(specific_force_rules)
        and available_numeric_motion_rules
    ):
        long_transfer_dynamic_force_margin = 0.05
        force += long_transfer_dynamic_force_margin
        calibration_notes.append("long_transfer_dynamic_force_margin")

    if (
        force_rule_mode == "all"
        and bool(specific_force_rules)
        and features["heavy"]
        and features["metal"]
        and not features["high_speed"]
    ):
        heavy_metal_force_center_floor = float(default_force) + 4.0
        if force < heavy_metal_force_center_floor:
            force = heavy_metal_force_center_floor
            calibration_notes.append("heavy_metal_force_center_floor")

    if (
        motion_rule_mode == "all"
        and available_motion_rules
        and force_rule_mode == "all"
        and bool(specific_force_rules)
        and features["heavy"]
        and features["high_speed"]
        and features["metal"]
    ):
        dynamic_heavy_metal_force_center_floor = float(default_force) + 3.0
        if force < dynamic_heavy_metal_force_center_floor:
            force = dynamic_heavy_metal_force_center_floor
            calibration_notes.append("dynamic_heavy_metal_force_center_floor")

    motion_aware_force_cap = None
    static_smooth_metal_force_cap = None
    dynamic_smooth_metal_force_cap = None
    if (
        motion_rule_mode == "all"
        and available_motion_rules
        and force_rule_mode == "all"
        and bool(specific_force_rules)
        and features["smooth_metal"]
        and features["high_speed"]
        and not features["heavy"]
    ):
        # Fast smooth-metal transfer still needs inertia margin, but the broad
        # 30-50N band should be recentered well below its upper half.
        dynamic_smooth_metal_force_cap = float(default_force) - 6.0
        motion_aware_force_cap = dynamic_smooth_metal_force_cap
        if force > dynamic_smooth_metal_force_cap:
            force = dynamic_smooth_metal_force_cap
            calibration_notes.append("dynamic_smooth_metal_force_cap")
    elif (
        features["smooth_metal"]
        and not features["high_speed"]
        and not features["heavy"]
    ):
        # Static smooth-metal pickup should stay in the lower-middle of the
        # broad 30-50N band unless heavy/high-speed evidence pushes it upward.
        static_smooth_metal_force_cap = float(default_force) - 1.5
        if force > static_smooth_metal_force_cap:
            force = static_smooth_metal_force_cap
            calibration_notes.append("static_smooth_metal_force_cap")

    lift_force = force
    transfer_force = force
    transfer_alignment = 0.0 if not belief.task_constraints.required_alignment else max(0.0, min(1.0, 0.55 + 0.25 * evidence_summary.alignment_confidence))
    dynamic_transport_mode = evidence_summary.preferred_transport_mode
    high_speed_transfer_force_margin = None
    high_speed_placement_velocity_target = None
    dynamic_smooth_metal_clearance_floor = None
    long_transfer_alignment_target = None
    long_transfer_alignment_force_margin = None
    long_transfer_lift_force_target = None
    long_transfer_lift_force_margin = None
    if (
        motion_rule_mode == "all"
        and features["long_transfer"]
        and features["large"]
        and available_numeric_motion_rules
    ):
        dynamic_transport_mode = "long_transfer"
        if available_lift_stage_rules and force_rule_mode == "all" and bool(specific_force_rules):
            long_transfer_lift_force_margin = round(
                min(
                    0.8,
                    0.25
                    + 0.10 * min(2, len(lift_stage_rules))
                    + (0.05 if available_alignment_rules else 0.0)
                    + (0.05 if available_numeric_motion_rules else 0.0)
                    + (0.05 if bool(specific_force_rules) else 0.0),
                ),
                4,
            )
            lift_force = max(force, min(50.0, force + long_transfer_lift_force_margin))
            long_transfer_lift_force_target = round(float(lift_force), 4)
            if lift_force > force:
                calibration_notes.append("long_transfer_lift_force_margin")
        if available_alignment_rules:
            long_transfer_alignment_target = min(
                0.95,
                0.65
                + 0.05 * min(2, len(alignment_rules))
                + (0.05 if len(alignment_rules) >= 2 else 0.0)
                + (0.05 if available_numeric_motion_rules else 0.0)
                + (0.05 if bool(specific_force_rules) else 0.0),
            )
            transfer_alignment = long_transfer_alignment_target
            calibration_notes.append("long_transfer_alignment_target")
            if force_rule_mode == "all" and bool(specific_force_rules):
                long_transfer_alignment_force_margin = round(
                    min(0.6, 0.25 + 0.5 * max(0.0, transfer_alignment - 0.5)),
                    4,
                )
                transfer_force = max(
                    force,
                    min(50.0, force + long_transfer_alignment_force_margin),
                )
                if transfer_force > force:
                    calibration_notes.append("long_transfer_alignment_force_margin")
    elif (
        motion_rule_mode == "all"
        and available_motion_rules
        and force_rule_mode == "all"
        and bool(specific_force_rules)
        and features["smooth_metal"]
        and features["high_speed"]
        and not features["heavy"]
    ):
        dynamic_transport_mode = "high_speed_low_friction"
        dynamic_smooth_metal_clearance_floor = 0.065
        if clearance < dynamic_smooth_metal_clearance_floor:
            clearance = dynamic_smooth_metal_clearance_floor
            calibration_notes.append("dynamic_smooth_metal_clearance_floor")
        high_speed_placement_velocity_target = round(max(0.24, min(velocity, 0.30)), 4)
        placement_velocity = high_speed_placement_velocity_target
        if placement_velocity < velocity:
            calibration_notes.append("high_speed_placement_velocity_target")
        high_speed_transfer_force_margin = 2.0
        transfer_force = max(force, min(float(default_force) - 4.0, force + high_speed_transfer_force_margin))
        if transfer_force > force:
            calibration_notes.append("high_speed_transfer_force_margin")

    pre_solve_plan = {
        "gripper_force": round(float(force), 4),
        "approach_height": round(float(height), 4),
        "transport_velocity": round(float(velocity), 4),
        "lift_clearance": round(float(clearance), 4),
        "lift_force": round(float(lift_force), 4),
        "transfer_force": round(float(transfer_force), 4),
        "placement_velocity": round(float(placement_velocity), 4),
        "transfer_alignment": round(float(transfer_alignment), 4),
    }
    belief = build_control_belief(
        features=features,
        evidence_summary=evidence_summary,
        dynamic_transport_mode=dynamic_transport_mode,
        support_score=support_score,
        conflict_count=force_conflict,
        force_rule_mode=force_rule_mode,
        motion_rule_mode=motion_rule_mode,
        available_specific_force_rules=available_specific_force_rules,
        available_motion_rules=available_motion_rules,
        available_numeric_motion_rules=available_numeric_motion_rules,
        available_alignment_rules=available_alignment_rules,
        available_lift_stage_rules=available_lift_stage_rules,
        available_support_contact_rules=available_support_contact_rules,
    )
    solved_plan, solver_trace = solve_control_plan(
        pre_solve_plan,
        belief,
    )
    force = solved_plan["gripper_force"]
    height = solved_plan["approach_height"]
    velocity = solved_plan["transport_velocity"]
    clearance = solved_plan["lift_clearance"]
    lift_force = solved_plan["lift_force"]
    transfer_force = solved_plan["transfer_force"]
    placement_velocity = solved_plan["placement_velocity"]
    transfer_alignment = solved_plan["transfer_alignment"]
    for note in solver_trace["solver_adjustment_notes"]:
        if note not in calibration_notes:
            calibration_notes.append(note)
    trace = {
        "selected_rules": [
            {
                "entry_id": rule["entry_id"],
                "category": rule["category"],
                "clause": rule["clause"],
                "score": rule["score"],
                "matched_terms": rule["matched_terms"],
                "force_candidates": rule["force_candidates"],
                "rule_type": rule["rule_type"],
            }
            for rule in selected_rules
        ],
        "rule_count": len(rules),
        "support_score": support_score,
        "conflict_count": force_conflict,
        "force_rule_mode": force_rule_mode,
        "motion_rule_mode": motion_rule_mode,
        **belief.to_trace_dict(),
        "matched_hint_keywords": list(matched_hint["keywords"]) if matched_hint is not None else [],
        "available_specific_force_rules": available_specific_force_rules,
        "suppressed_specific_force_rules": suppressed_specific_force_rules,
        "available_motion_rules": available_motion_rules,
        "suppressed_motion_rules": suppressed_motion_rules,
        "available_support_contact_rules": available_support_contact_rules,
        "available_numeric_motion_rules": available_numeric_motion_rules,
        "available_alignment_rules": available_alignment_rules,
        "available_lift_stage_rules": available_lift_stage_rules,
        "used_specific_force_rules": bool(selected_force_rules) and not suppressed_specific_force_rules and bool(specific_force_rules),
        "used_generic_force_rules": bool(generic_force_rules),
        "used_motion_rules": available_motion_rules and not suppressed_motion_rules,
        "used_support_contact_rules": available_support_contact_rules and not suppressed_motion_rules,
        "used_alignment_rules": available_alignment_rules and not suppressed_motion_rules and dynamic_transport_mode == "long_transfer",
        "used_lift_stage_rules": (
            available_lift_stage_rules
            and not suppressed_motion_rules
            and dynamic_transport_mode == "long_transfer"
            and long_transfer_lift_force_margin is not None
        ),
        "pre_solve_plan": pre_solve_plan,
        "belief_force_floor": belief_force_floor,
        "composite_force_floor": composite_force_floor,
        "motion_aware_force_floor": motion_aware_force_floor,
        "thin_wall_support_force_floor": thin_wall_support_force_floor,
        "rubber_material_force_floor": rubber_material_force_floor,
        "heavy_metal_force_center_floor": heavy_metal_force_center_floor,
        "dynamic_heavy_metal_force_center_floor": dynamic_heavy_metal_force_center_floor,
        "long_transfer_force_center_floor": long_transfer_force_center_floor,
        "long_transfer_stage_force_floor": long_transfer_stage_force_floor,
        "long_transfer_dynamic_force_margin": long_transfer_dynamic_force_margin,
        "long_transfer_velocity_band": long_transfer_velocity_band,
        "long_transfer_placement_velocity_cap": long_transfer_placement_velocity_cap,
        "dynamic_transport_mode": dynamic_transport_mode,
        "lift_force": round(float(lift_force), 4),
        "long_transfer_lift_force_target": long_transfer_lift_force_target,
        "long_transfer_lift_force_margin": long_transfer_lift_force_margin,
        "transfer_force": round(float(transfer_force), 4),
        "transfer_alignment": round(float(transfer_alignment), 4),
        "long_transfer_alignment_target": long_transfer_alignment_target,
        "long_transfer_alignment_force_margin": long_transfer_alignment_force_margin,
        "high_speed_transfer_force_margin": high_speed_transfer_force_margin,
        "high_speed_placement_velocity_target": high_speed_placement_velocity_target,
        "motion_aware_force_cap": motion_aware_force_cap,
        "dynamic_smooth_metal_force_cap": dynamic_smooth_metal_force_cap,
        "dynamic_smooth_metal_clearance_floor": dynamic_smooth_metal_clearance_floor,
        "static_smooth_metal_force_cap": static_smooth_metal_force_cap,
        "thin_wall_support_clearance_cap": thin_wall_support_clearance_cap,
        "rubber_material_clearance_floor": rubber_material_clearance_floor,
        "long_transfer_numeric_clearance_floor": long_transfer_numeric_clearance_floor,
        "long_transfer_clearance_target": long_transfer_clearance_target,
        "motion_path_force_compensation": round(motion_path_force_compensation, 4),
        "solver_mode": solver_trace["solver_mode"],
        "solver_selected_candidate": solver_trace["solver_selected_candidate"],
        "solver_selected_score": solver_trace["solver_selected_score"],
        "solver_score_breakdown": solver_trace["solver_score_breakdown"],
        "solver_candidate_scores": solver_trace["solver_candidate_scores"],
        "solver_seed_candidate": solver_trace["solver_seed_candidate"],
        "solver_seed_score": solver_trace["solver_seed_score"],
        "solver_local_search_iterations": solver_trace["solver_local_search_iterations"],
        "solver_local_search_improvement": solver_trace["solver_local_search_improvement"],
        "solver_local_search_trace": solver_trace["solver_local_search_trace"],
        "solver_adjustment_notes": solver_trace["solver_adjustment_notes"],
        "calibration_notes": calibration_notes,
    }
    return (
        max(5.0, min(50.0, force)),
        max(0.02, min(0.08, height)),
        max(0.12, min(0.8, velocity)),
        max(0.03, min(0.14, clearance)),
        max(force, min(50.0, lift_force)),
        max(force, min(50.0, transfer_force)),
        max(0.12, min(0.8, placement_velocity)),
        max(0.0, min(1.0, transfer_alignment)),
        trace,
    )


class RAGController:
    """Retrieve structured evidence from the knowledge base and aggregate it into control parameters."""

    def __init__(self, data_path: str = "mechanical_data.txt"):
        self.documents = _parse_entries(data_path)
        embedding_model_path = resolve_model_path("sentence-transformers/all-MiniLM-L6-v2")
        embeddings = HuggingFaceEmbeddings(model_name=embedding_model_path)
        self.vector_db = Chroma.from_documents(
            self.documents,
            embeddings,
            ids=[f"entry_{doc.metadata['entry_id']}" for doc in self.documents],
            client_settings=get_chroma_client_settings(),
        )
        self.data_path = data_path

    def _queries_multi(self, task_description: str) -> list[str]:
        parts = [task_description]
        priority_queries: list[str] = []
        material_queries: list[str] = []
        if "金属" in task_description:
            material_queries.extend(["光滑金属 夹爪力", "金属零件 抓取 力"])
        if "橡胶" in task_description:
            material_queries.extend(["橡胶零件 夹爪力", "橡胶 抓取 力"])
        if "小型" in task_description:
            material_queries.extend(["小型机械零件 夹爪力", "小型零件 抓取"])
        if "高速" in task_description:
            priority_queries.extend(["高速运动 夹持力余量", "惯性前冲 搬运"])
        if "长距离" in task_description:
            priority_queries.extend(["重心偏移 搬运", "轨迹规划 速度 加速度"])
        if "长距离" in task_description and "大型" in task_description:
            material_queries.extend(["长距离搬运大型零件 夹爪力", "大型零件 长距离 搬运 夹爪力"])
            priority_queries.extend(
                [
                    "安全流程 夹爪松紧度 脱落",
                    "抓取规划 抬升 放置",
                ]
            )
        if "大型" in task_description or "重型" in task_description:
            priority_queries.extend(["重型零件 负载 稳定", "大型零件 抓取 稳定"])
            material_queries.extend(["大型零件 夹爪力", "重型零件 抓取 力"])
        if "薄壁" in task_description:
            priority_queries.extend(["薄壁件 抓取 真空吸附"])
            material_queries.extend(["薄壁件 夹爪力"])

        queries: list[str] = []
        seen: set[str] = set()
        for query in [*parts, *priority_queries, *material_queries]:
            if query in seen:
                continue
            seen.add(query)
            queries.append(query)
        return queries[:6]

    def _select_docs(self, task_description: str, retrieval: str, seed: int | None = None) -> list[Document]:
        if retrieval == "random":
            rng = random.Random(seed)
            return rng.sample(self.documents, min(3, len(self.documents)))
        if retrieval == "multi":
            seen = set()
            docs = []
            for query in self._queries_multi(task_description):
                for doc in self.vector_db.similarity_search(query, k=2):
                    key = str(doc.metadata.get("entry_id", doc.page_content[:100]))
                    if key not in seen:
                        seen.add(key)
                        docs.append(doc)
            return docs[:5]
        docs = self.vector_db.similarity_search(task_description, k=3)
        if (
            "高速" in task_description
            or "长距离" in task_description
            or "薄壁" in task_description
            or ("橡胶" in task_description and "高速" not in task_description)
            or ("光滑金属" in task_description and "高速" not in task_description and "重型" not in task_description)
        ):
            seen = {str(doc.metadata.get("entry_id", doc.page_content[:100])) for doc in docs}
            for query in self._queries_multi(task_description)[1:]:
                for doc in self.vector_db.similarity_search(query, k=1):
                    key = str(doc.metadata.get("entry_id", doc.page_content[:100]))
                    if key not in seen:
                        seen.add(key)
                        docs.append(doc)
            material_backstops: list[tuple[str, str]] = []
            if "橡胶" in task_description and "高速" not in task_description:
                material_backstops.append(("橡胶零件", "夹爪力"))
            if "光滑金属" in task_description and "高速" not in task_description and "重型" not in task_description:
                material_backstops.append(("光滑金属零件", "夹爪力"))
            if "长距离" in task_description and "大型" in task_description:
                material_backstops.append(("长距离搬运大型零件", "夹爪力"))
            elif "大型" in task_description:
                material_backstops.append(("抓取大型零件", "夹爪力"))
            for keyword, anchor in material_backstops:
                if any(keyword in doc.page_content and anchor in doc.page_content for doc in docs):
                    continue
                fallback = next(
                    (
                        doc
                        for doc in self.documents
                        if keyword in doc.page_content and anchor in doc.page_content
                    ),
                    None,
                )
                if fallback is not None:
                    key = str(fallback.metadata.get("entry_id", fallback.page_content[:100]))
                    if key not in seen:
                        seen.add(key)
                        docs.append(fallback)
            stage_backstops: list[tuple[str, str]] = []
            if "长距离" in task_description and "大型" in task_description:
                stage_backstops.extend(
                    [
                        ("夹爪松紧度", "脱落"),
                        ("接近-抓取-抬升-放置序列", ""),
                    ]
                )
            for required, anchor in stage_backstops:
                if any(required in doc.page_content and (not anchor or anchor in doc.page_content) for doc in docs):
                    continue
                fallback = next(
                    (
                        doc
                        for doc in self.documents
                        if required in doc.page_content and (not anchor or anchor in doc.page_content)
                    ),
                    None,
                )
                if fallback is not None:
                    key = str(fallback.metadata.get("entry_id", fallback.page_content[:100]))
                    if key not in seen:
                        seen.add(key)
                        docs.append(fallback)
            if "长距离" in task_description and "大型" in task_description:
                prioritized_docs: list[Document] = []
                prioritized_seen: set[str] = set()
                priority_tokens = (
                    "长距离搬运大型零件",
                    "大型零件夹持力",
                    "重心偏移",
                    "夹爪松紧度",
                    "接近-抓取-抬升-放置序列",
                )
                for token in priority_tokens:
                    for doc in docs:
                        key = str(doc.metadata.get("entry_id", doc.page_content[:100]))
                        if token in doc.page_content and key not in prioritized_seen:
                            prioritized_seen.add(key)
                            prioritized_docs.append(doc)
                            break
                for doc in docs:
                    key = str(doc.metadata.get("entry_id", doc.page_content[:100]))
                    if key not in prioritized_seen:
                        prioritized_seen.add(key)
                        prioritized_docs.append(doc)
                docs = prioritized_docs[:5]
            else:
                docs = docs[:5]
        return docs

    def get_params_for_task(
        self,
        task_description: str,
        retrieval: str = "single",
        seed: int | None = None,
        force_rule_mode: str = "all",
        motion_rule_mode: str = "all",
    ) -> dict[str, Any]:
        docs = self._select_docs(task_description, retrieval=retrieval, seed=seed)
        context = "\n".join(doc.page_content for doc in docs)
        rules = _build_rule_trace(task_description, docs)
        force, height, velocity, clearance, lift_force, transfer_force, placement_velocity, transfer_alignment, trace = _aggregate_plan(
            task_description,
            rules,
            force_rule_mode=force_rule_mode,
            motion_rule_mode=motion_rule_mode,
        )
        confidence = min(
            1.0,
            0.3
            + 0.12 * len(docs)
            + 0.08 * min(4, trace["support_score"])
            + (0.12 if trace["used_specific_force_rules"] else 0.0)
            - 0.08 * trace["conflict_count"],
        )
        confidence = max(
            0.05,
            min(
                1.0,
                confidence
                + 0.10 * float(trace["belief_state_coverage"])
                - (0.08 if trace["uncertainty_conservative_mode"] else 0.0),
            ),
        )
        uncertainty_std = min(
            0.35,
            0.04
            + 0.12 * max(0.0, 1.0 - float(trace["belief_state_coverage"]))
            + 0.03 * float(trace["conflict_count"])
            + (0.04 if trace["uncertainty_conservative_mode"] else 0.0),
        )
        return {
            "gripper_force": round(float(force), 2),
            "approach_height": round(float(height), 3),
            "transport_velocity": round(float(velocity), 3),
            "lift_force": round(float(lift_force), 3),
            "transfer_force": round(float(transfer_force), 3),
            "placement_velocity": round(float(placement_velocity), 3),
            "transfer_alignment": round(float(transfer_alignment), 3),
            "lift_clearance": round(float(clearance), 3),
            "rag_source": context[:300] if docs else "",
            "confidence": round(confidence, 3),
            "uncertainty_std": round(uncertainty_std, 4),
            "evidence_rule_count": trace["rule_count"],
            "evidence_support_score": trace["support_score"],
            "evidence_conflict_count": trace["conflict_count"],
            "force_rule_mode": trace["force_rule_mode"],
            "motion_rule_mode": trace["motion_rule_mode"],
            "evidence_state_summary": trace["evidence_state_summary"],
            "belief_state": trace["belief_state"],
            "task_constraints": trace["task_constraints"],
            "uncertainty_profile": trace["uncertainty_profile"],
            "stage_plan": trace["stage_plan"],
            "belief_state_coverage": trace["belief_state_coverage"],
            "uncertainty_conservative_mode": trace["uncertainty_conservative_mode"],
            "uncertainty_reasons": trace["uncertainty_reasons"],
            "seed_mode": trace["seed_mode"],
            "seed_notes": trace["seed_notes"],
            "seed_hints": trace["seed_hints"],
            "seed_plan": trace["seed_plan"],
            "solver_mode": trace["solver_mode"],
            "solver_selected_candidate": trace["solver_selected_candidate"],
            "solver_selected_score": trace["solver_selected_score"],
            "solver_score_breakdown": trace["solver_score_breakdown"],
            "solver_candidate_scores": trace["solver_candidate_scores"],
            "solver_seed_candidate": trace["solver_seed_candidate"],
            "solver_seed_score": trace["solver_seed_score"],
            "solver_local_search_iterations": trace["solver_local_search_iterations"],
            "solver_local_search_improvement": trace["solver_local_search_improvement"],
            "solver_local_search_trace": trace["solver_local_search_trace"],
            "solver_adjustment_notes": trace["solver_adjustment_notes"],
            "matched_hint_keywords": trace["matched_hint_keywords"],
            "selected_evidence": [
                f"{rule['category']}#{rule['entry_id']}:{rule['clause']}"
                for rule in trace["selected_rules"]
            ],
            "selected_rule_types": [rule["rule_type"] for rule in trace["selected_rules"]],
            "available_specific_force_rules": trace["available_specific_force_rules"],
            "suppressed_specific_force_rules": trace["suppressed_specific_force_rules"],
            "available_motion_rules": trace["available_motion_rules"],
            "suppressed_motion_rules": trace["suppressed_motion_rules"],
            "available_support_contact_rules": trace["available_support_contact_rules"],
            "available_numeric_motion_rules": trace["available_numeric_motion_rules"],
            "available_alignment_rules": trace["available_alignment_rules"],
            "available_lift_stage_rules": trace["available_lift_stage_rules"],
            "used_specific_force_rules": trace["used_specific_force_rules"],
            "used_generic_force_rules": trace["used_generic_force_rules"],
            "used_motion_rules": trace["used_motion_rules"],
            "used_support_contact_rules": trace["used_support_contact_rules"],
            "used_alignment_rules": trace["used_alignment_rules"],
            "used_lift_stage_rules": trace["used_lift_stage_rules"],
            "composite_force_floor": trace["composite_force_floor"],
            "motion_aware_force_floor": trace["motion_aware_force_floor"],
            "thin_wall_support_force_floor": trace["thin_wall_support_force_floor"],
            "rubber_material_force_floor": trace["rubber_material_force_floor"],
            "heavy_metal_force_center_floor": trace["heavy_metal_force_center_floor"],
            "dynamic_heavy_metal_force_center_floor": trace["dynamic_heavy_metal_force_center_floor"],
            "long_transfer_force_center_floor": trace["long_transfer_force_center_floor"],
            "long_transfer_stage_force_floor": trace["long_transfer_stage_force_floor"],
            "long_transfer_dynamic_force_margin": trace["long_transfer_dynamic_force_margin"],
            "long_transfer_velocity_band": trace["long_transfer_velocity_band"],
            "long_transfer_placement_velocity_cap": trace["long_transfer_placement_velocity_cap"],
            "dynamic_transport_mode": trace["dynamic_transport_mode"],
            "long_transfer_lift_force_target": trace["long_transfer_lift_force_target"],
            "long_transfer_lift_force_margin": trace["long_transfer_lift_force_margin"],
            "long_transfer_alignment_target": trace["long_transfer_alignment_target"],
            "long_transfer_alignment_force_margin": trace["long_transfer_alignment_force_margin"],
            "high_speed_transfer_force_margin": trace["high_speed_transfer_force_margin"],
            "high_speed_placement_velocity_target": trace["high_speed_placement_velocity_target"],
            "motion_aware_force_cap": trace["motion_aware_force_cap"],
            "dynamic_smooth_metal_force_cap": trace["dynamic_smooth_metal_force_cap"],
            "dynamic_smooth_metal_clearance_floor": trace["dynamic_smooth_metal_clearance_floor"],
            "static_smooth_metal_force_cap": trace["static_smooth_metal_force_cap"],
            "thin_wall_support_clearance_cap": trace["thin_wall_support_clearance_cap"],
            "rubber_material_clearance_floor": trace["rubber_material_clearance_floor"],
            "long_transfer_numeric_clearance_floor": trace["long_transfer_numeric_clearance_floor"],
            "long_transfer_clearance_target": trace["long_transfer_clearance_target"],
            "motion_path_force_compensation": trace["motion_path_force_compensation"],
            "pre_solve_plan": trace["pre_solve_plan"],
            "calibration_notes": trace["calibration_notes"],
        }

    def get_params_for_task_llm(
        self,
        task_description: str,
        llm: Any,
        retrieval: str = "single",
    ) -> dict[str, Any]:
        docs = self._select_docs(task_description, retrieval=retrieval)
        context = "\n".join(doc.page_content for doc in docs)[:800]
        prompt = f"""根据以下知识库片段，为“{task_description}”任务给出夹爪力（N）、接近高度（m）、搬运速度（m/s）和抬升净空（m）。
只输出一个 JSON，格式严格为：{{"gripper_force": 数字, "approach_height": 数字, "transport_velocity": 数字, "lift_clearance": 数字}}
不要输出解释。

知识库片段：
{context}
"""
        try:
            out = llm.invoke(prompt)
            text = out if isinstance(out, str) else str(out)
            parsed = _parse_json_params(text)
            if parsed is not None:
                force = max(5.0, min(50.0, float(parsed.get("gripper_force", 25.0))))
                height = max(0.02, min(0.1, float(parsed.get("approach_height", 0.05))))
                velocity = max(0.12, min(0.8, float(parsed.get("transport_velocity", 0.30))))
                clearance = max(0.03, min(0.14, float(parsed.get("lift_clearance", 0.06))))
                rules = _build_rule_trace(task_description, docs)
                trace = _aggregate_plan(task_description, rules)[8]
                transfer_force = max(force, float(trace["transfer_force"] or force))
                lift_force = max(force, float(trace["lift_force"] or force))
                placement_velocity = float(
                    trace["high_speed_placement_velocity_target"]
                    or trace["long_transfer_placement_velocity_cap"]
                    or velocity
                )
                transfer_alignment = float(trace["transfer_alignment"] or 0.0)
                uncertainty_std = min(
                    0.35,
                    0.04
                    + 0.12 * max(0.0, 1.0 - float(trace["belief_state_coverage"]))
                    + 0.03 * float(trace["conflict_count"])
                    + (0.04 if trace["uncertainty_conservative_mode"] else 0.0),
                )
                confidence = max(
                    0.05,
                    min(
                        1.0,
                        0.55
                        + 0.08 * float(trace["belief_state_coverage"])
                        - (0.06 if trace["uncertainty_conservative_mode"] else 0.0),
                    ),
                )
                return {
                    "gripper_force": round(force, 2),
                    "approach_height": round(height, 3),
                    "transport_velocity": round(velocity, 3),
                    "lift_force": round(lift_force, 3),
                    "transfer_force": round(transfer_force, 3),
                    "placement_velocity": round(placement_velocity, 3),
                    "transfer_alignment": round(transfer_alignment, 3),
                    "lift_clearance": round(clearance, 3),
                    "rag_source": context[:300],
                    "confidence": round(confidence, 3),
                    "uncertainty_std": round(uncertainty_std, 4),
                    "evidence_rule_count": trace["rule_count"],
                    "evidence_support_score": trace["support_score"],
                    "evidence_conflict_count": trace["conflict_count"],
                    "force_rule_mode": trace["force_rule_mode"],
                    "motion_rule_mode": trace["motion_rule_mode"],
                    "evidence_state_summary": trace["evidence_state_summary"],
                    "belief_state": trace["belief_state"],
                    "task_constraints": trace["task_constraints"],
                    "uncertainty_profile": trace["uncertainty_profile"],
                    "stage_plan": trace["stage_plan"],
                    "belief_state_coverage": trace["belief_state_coverage"],
                    "uncertainty_conservative_mode": trace["uncertainty_conservative_mode"],
                    "uncertainty_reasons": trace["uncertainty_reasons"],
                    "seed_mode": trace["seed_mode"],
                    "seed_notes": trace["seed_notes"],
                    "seed_hints": trace["seed_hints"],
                    "seed_plan": trace["seed_plan"],
                    "solver_mode": trace["solver_mode"],
                    "solver_selected_candidate": trace["solver_selected_candidate"],
                    "solver_selected_score": trace["solver_selected_score"],
                    "solver_score_breakdown": trace["solver_score_breakdown"],
                    "solver_candidate_scores": trace["solver_candidate_scores"],
                    "solver_seed_candidate": trace["solver_seed_candidate"],
                    "solver_seed_score": trace["solver_seed_score"],
                    "solver_local_search_iterations": trace["solver_local_search_iterations"],
                    "solver_local_search_improvement": trace["solver_local_search_improvement"],
                    "solver_local_search_trace": trace["solver_local_search_trace"],
                    "solver_adjustment_notes": trace["solver_adjustment_notes"],
                    "matched_hint_keywords": trace["matched_hint_keywords"],
                    "selected_evidence": [
                        f"{rule['category']}#{rule['entry_id']}:{rule['clause']}"
                        for rule in trace["selected_rules"]
                    ],
                    "selected_rule_types": [rule["rule_type"] for rule in trace["selected_rules"]],
                    "available_specific_force_rules": trace["available_specific_force_rules"],
                    "suppressed_specific_force_rules": trace["suppressed_specific_force_rules"],
                    "available_motion_rules": trace["available_motion_rules"],
                    "suppressed_motion_rules": trace["suppressed_motion_rules"],
                    "available_support_contact_rules": trace["available_support_contact_rules"],
                    "available_numeric_motion_rules": trace["available_numeric_motion_rules"],
                    "available_alignment_rules": trace["available_alignment_rules"],
                    "available_lift_stage_rules": trace["available_lift_stage_rules"],
                    "used_motion_rules": trace["used_motion_rules"],
                    "used_support_contact_rules": trace["used_support_contact_rules"],
                    "used_alignment_rules": trace["used_alignment_rules"],
                    "used_lift_stage_rules": trace["used_lift_stage_rules"],
                    "composite_force_floor": trace["composite_force_floor"],
                    "motion_aware_force_floor": trace["motion_aware_force_floor"],
                    "thin_wall_support_force_floor": trace["thin_wall_support_force_floor"],
                    "rubber_material_force_floor": trace["rubber_material_force_floor"],
                    "heavy_metal_force_center_floor": trace["heavy_metal_force_center_floor"],
                    "dynamic_heavy_metal_force_center_floor": trace["dynamic_heavy_metal_force_center_floor"],
                    "long_transfer_force_center_floor": trace["long_transfer_force_center_floor"],
                    "long_transfer_stage_force_floor": trace["long_transfer_stage_force_floor"],
                    "long_transfer_dynamic_force_margin": trace["long_transfer_dynamic_force_margin"],
                    "long_transfer_velocity_band": trace["long_transfer_velocity_band"],
                    "long_transfer_placement_velocity_cap": trace["long_transfer_placement_velocity_cap"],
                    "dynamic_transport_mode": trace["dynamic_transport_mode"],
                    "long_transfer_lift_force_target": trace["long_transfer_lift_force_target"],
                    "long_transfer_lift_force_margin": trace["long_transfer_lift_force_margin"],
                    "long_transfer_alignment_target": trace["long_transfer_alignment_target"],
                    "long_transfer_alignment_force_margin": trace["long_transfer_alignment_force_margin"],
                    "high_speed_transfer_force_margin": trace["high_speed_transfer_force_margin"],
                    "high_speed_placement_velocity_target": trace["high_speed_placement_velocity_target"],
                    "motion_aware_force_cap": trace["motion_aware_force_cap"],
                    "dynamic_smooth_metal_force_cap": trace["dynamic_smooth_metal_force_cap"],
                    "dynamic_smooth_metal_clearance_floor": trace["dynamic_smooth_metal_clearance_floor"],
                    "static_smooth_metal_force_cap": trace["static_smooth_metal_force_cap"],
                    "thin_wall_support_clearance_cap": trace["thin_wall_support_clearance_cap"],
                    "rubber_material_clearance_floor": trace["rubber_material_clearance_floor"],
                    "long_transfer_numeric_clearance_floor": trace["long_transfer_numeric_clearance_floor"],
                    "long_transfer_clearance_target": trace["long_transfer_clearance_target"],
                    "motion_path_force_compensation": trace["motion_path_force_compensation"],
                    "pre_solve_plan": trace["pre_solve_plan"],
                    "calibration_notes": trace["calibration_notes"],
                }
        except Exception:
            pass
        return self.get_params_for_task(task_description, retrieval=retrieval)

    def get_params_after_feedback(
        self,
        task_description: str,
        previous_params: dict[str, Any],
        success: bool,
        info: dict[str, Any],
        adjustment_step: float = 4.0,
    ) -> dict[str, Any]:
        from .feedback import (
            adjust_params_by_feedback,
            build_feedback_replan_request,
            build_feedback_signal,
            suggest_force_adjustment,
        )

        if success:
            return dict(previous_params)
        signal = build_feedback_signal(
            success=success,
            gripper_force=previous_params["gripper_force"],
            info=info,
        )
        suggestion = suggest_force_adjustment(signal)
        replan_ready = all(
            key in previous_params
            for key in ("belief_state", "task_constraints", "uncertainty_profile", "stage_plan")
        )
        if replan_ready:
            replan_request = build_feedback_replan_request(
                previous_params,
                signal,
                suggestion,
                step=adjustment_step,
            )
            updated = replan_control_plan(previous_params, replan_request)
            if replan_request.get("requested_suffix_start") == "lift" and "counterfactual_replan_trace" in updated:
                updated["execution_feedback_mode"] = "suffix_counterfactual_replan"
            return updated
        return adjust_params_by_feedback(
            previous_params,
            suggestion,
            signal=signal,
            step=adjustment_step,
        )

    def get_params_after_observation(
        self,
        task_description: str,
        previous_params: dict[str, Any],
        observation: dict[str, Any],
        adjustment_step: float = 4.0,
    ) -> dict[str, Any]:
        from .feedback import (
            build_feedback_replan_request,
            build_feedback_signal_from_observation,
            suggest_force_adjustment,
        )

        signal = build_feedback_signal_from_observation(previous_params, observation)
        suggestion = suggest_force_adjustment(signal)
        replan_request = build_feedback_replan_request(
            previous_params,
            signal,
            suggestion,
            step=adjustment_step,
        )
        replan_request["observation_index"] = observation.get("observation_index")
        replan_request["trigger_reason"] = observation.get("trigger_reason")
        replan_request["observation_stage"] = observation.get("stage")
        updated = replan_control_plan(previous_params, replan_request)
        if replan_request.get("requested_suffix_start") == "lift" and "counterfactual_replan_trace" in updated:
            updated["execution_feedback_mode"] = "suffix_counterfactual_replan"
        return updated
