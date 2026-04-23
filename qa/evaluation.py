"""Structured QA evaluation with split-wise robustness reporting."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

from langchain_community.embeddings import HuggingFaceEmbeddings

from llm_loader import DEFAULT_CN_MODEL, get_llm
from model_provider import resolve_model_path

from .base import build_chain
from .dataset import QACase, SPLIT_ORDER, get_cases
from .pipeline import DEFAULT_DB_DIRS, DEFAULT_EMBEDDING, build_components
from .problem_solving import build_system


DEFAULT_OUTPUT_DIR = Path("outputs/current")
DETAIL_FILENAME = "qa_evaluation_detail.json"
REPORT_FILENAME = "rag_evaluation_report.txt"
PROBLEM_FILENAME = "rag_problems.txt"
DIRECT_RESULT_FILENAME = "direct_llm_result.txt"
RULE_HEAVY_RESULT_FILENAME = "problem_solving_result.txt"
NUMERIC_PATTERN = re.compile(
    r"(?:\d+/\d+)|(?:±\d+(?:\.\d+)?(?:mm|kg|N|kPa|g)?)|(?:\d+(?:\.\d+)?(?:\s*(?:-|到)\s*\d+(?:\.\d+)?)?(?:mm|kg|N|kPa|g)?)"
)
DEFAULT_ABSTAIN_MARKERS = (
    "知识库未提供",
    "无法可靠回答",
    "证据不足",
    "无法判断",
    "缺少足够证据",
)


class SemanticScorer:
    """Embedding-based semantic scorer with a tiny in-memory cache."""

    def __init__(self, embedding_model_name: str = DEFAULT_EMBEDDING):
        model_path = resolve_model_path(embedding_model_name)
        self.embedding = HuggingFaceEmbeddings(model_name=model_path)
        self._cache: dict[str, list[float]] = {}

    def _encode(self, text: str) -> list[float]:
        normalized = text.strip() or "<EMPTY>"
        if normalized not in self._cache:
            self._cache[normalized] = self.embedding.embed_query(normalized)
        return self._cache[normalized]

    def similarity(self, left: str, right: str) -> float:
        left_vec = self._encode(left)
        right_vec = self._encode(right)
        dot = sum(a * b for a, b in zip(left_vec, right_vec))
        left_norm = math.sqrt(sum(value * value for value in left_vec))
        right_norm = math.sqrt(sum(value * value for value in right_vec))
        if left_norm == 0.0 or right_norm == 0.0:
            return 0.0
        return max(0.0, min(1.0, dot / (left_norm * right_norm)))


def _contains_any(text: str, keywords: tuple[str, ...]) -> bool:
    return any(keyword in text for keyword in keywords)


def _is_abstention_response(case: QACase, response: str) -> bool:
    markers = tuple(dict.fromkeys((*DEFAULT_ABSTAIN_MARKERS, *case.abstain_keywords)))
    return any(marker in response for marker in markers)


def _normalize_numeric_span(span: str) -> str:
    return re.sub(r"\s+", "", span).replace("到", "-").lower()


def _extract_numeric_spans(text: str) -> tuple[str, ...]:
    spans = [_normalize_numeric_span(match.group(0)) for match in NUMERIC_PATTERN.finditer(text)]
    return tuple(dict.fromkeys(span for span in spans if span))


def _gold_numeric_spans(case: QACase) -> tuple[str, ...]:
    spans = list(_extract_numeric_spans(case.gold_answer))
    for group in case.required_groups:
        spans.extend(
            _normalize_numeric_span(token)
            for token in group
            if any(char.isdigit() for char in token)
        )
    return tuple(dict.fromkeys(span for span in spans if span))


def _is_procedure_case(case: QACase) -> bool:
    return case.split == "procedure" or any(
        token in case.question for token in ("顺序", "步骤", "流程", "阶段")
    )


def _strip_or_fallback(response: str, source_docs) -> str:
    response = response.strip()
    if sum(1 for char in response if "\u4e00" <= char <= "\u9fff") >= 2:
        return response
    if source_docs:
        return source_docs[0].page_content[:200]
    return response


def score_retrieval(case: QACase, source_docs) -> dict[str, Any]:
    docs = source_docs or []
    if not docs:
        return {
            "coverage": None,
            "matched_evidence": [],
            "missing_evidence": ["/".join(group) for group in case.evidence_groups],
        }

    joined = "\n".join(doc.page_content for doc in docs)
    matched = ["/".join(group) for group in case.evidence_groups if _contains_any(joined, group)]
    missing = ["/".join(group) for group in case.evidence_groups if not _contains_any(joined, group)]
    coverage = len(matched) / len(case.evidence_groups) if case.evidence_groups else None
    return {
        "coverage": coverage,
        "matched_evidence": matched,
        "missing_evidence": missing,
    }


def _numeric_consistency(case: QACase, response: str) -> dict[str, Any]:
    gold_spans = _gold_numeric_spans(case)
    response_spans = _extract_numeric_spans(response)
    if not gold_spans:
        return {
            "expected": False,
            "gold_spans": [],
            "response_spans": list(response_spans),
            "matched_spans": [],
            "consistency": None,
            "hard_mismatch": False,
        }
    matched = [span for span in response_spans if span in gold_spans]
    consistency = 1.0 if matched else 0.0
    hard_mismatch = bool(response_spans) and not matched
    return {
        "expected": True,
        "gold_spans": list(gold_spans),
        "response_spans": list(response_spans),
        "matched_spans": matched,
        "consistency": consistency,
        "hard_mismatch": hard_mismatch,
    }


def _procedure_order(case: QACase, response: str) -> dict[str, Any]:
    if not _is_procedure_case(case):
        return {
            "expected": False,
            "matched_steps": [],
            "missing_steps": [],
            "order_score": None,
        }
    matched_steps: list[str] = []
    missing_steps: list[str] = []
    last_position = -1
    in_order_count = 0
    for group in case.required_groups:
        positions = [response.find(token) for token in group if token in response]
        if not positions:
            missing_steps.append("/".join(group))
            continue
        position = min(pos for pos in positions if pos >= 0)
        matched_steps.append("/".join(group))
        if position > last_position:
            in_order_count += 1
            last_position = position
    order_score = in_order_count / len(case.required_groups) if case.required_groups else None
    return {
        "expected": True,
        "matched_steps": matched_steps,
        "missing_steps": missing_steps,
        "order_score": None if order_score is None else round(order_score, 4),
    }


def score_response(case: QACase, response: str, semantic_scorer: SemanticScorer | None = None) -> dict[str, Any]:
    matched = ["/".join(group) for group in case.required_groups if _contains_any(response, group)]
    missing = ["/".join(group) for group in case.required_groups if not _contains_any(response, group)]
    contradictions = [keyword for keyword in case.forbidden_keywords if keyword in response]
    abstained = _is_abstention_response(case, response)
    lexical_score = len(matched) / len(case.required_groups) if case.required_groups else 0.0
    semantic_similarity = (
        round(semantic_scorer.similarity(response, case.gold_answer), 4)
        if semantic_scorer is not None
        else round(lexical_score, 4)
    )
    numeric_eval = _numeric_consistency(case, response)
    procedure_eval = _procedure_order(case, response)

    weighted_parts: list[tuple[float, float]] = [(0.45, lexical_score), (0.35, semantic_similarity)]
    if numeric_eval["consistency"] is not None:
        weighted_parts.append((0.12, float(numeric_eval["consistency"])))
    if procedure_eval["order_score"] is not None:
        weighted_parts.append((0.08, float(procedure_eval["order_score"])))
    total_weight = sum(weight for weight, _ in weighted_parts)
    hybrid_score = sum(weight * value for weight, value in weighted_parts) / total_weight if total_weight else 0.0
    if case.expected_behavior == "abstain":
        abstain_correct = abstained
        if abstained:
            label = "correct"
        else:
            label = "incorrect"
    else:
        abstain_correct = not abstained
        if abstained or contradictions or numeric_eval["hard_mismatch"]:
            label = "incorrect"
        elif hybrid_score >= 0.82 and lexical_score >= 0.5:
            label = "correct"
        elif hybrid_score >= 0.45 or (semantic_similarity >= 0.72 and lexical_score > 0.0):
            label = "partial"
        else:
            label = "incorrect"
    weighted = {"correct": 1.0, "partial": 0.5, "incorrect": 0.0}[label]
    return {
        "label": label,
        "score": round(hybrid_score, 4),
        "weighted_score": weighted,
        "lexical_score": round(lexical_score, 4),
        "semantic_similarity": round(semantic_similarity, 4),
        "hybrid_score": round(hybrid_score, 4),
        "matched_required": matched,
        "missing_required": missing,
        "contradictions": contradictions,
        "expected_behavior": case.expected_behavior,
        "abstained": abstained,
        "abstain_correct": abstain_correct,
        "numeric_expected": numeric_eval["expected"],
        "numeric_gold_spans": numeric_eval["gold_spans"],
        "numeric_response_spans": numeric_eval["response_spans"],
        "numeric_matched_spans": numeric_eval["matched_spans"],
        "numeric_consistency": numeric_eval["consistency"],
        "numeric_hard_mismatch": numeric_eval["hard_mismatch"],
        "procedure_expected": procedure_eval["expected"],
        "procedure_matched_steps": procedure_eval["matched_steps"],
        "procedure_missing_steps": procedure_eval["missing_steps"],
        "procedure_order_score": procedure_eval["order_score"],
    }


def score_support(
    case: QACase,
    response: str,
    source_docs,
    debug: dict[str, Any] | None = None,
) -> dict[str, Any]:
    evidence_chunks = []
    if debug is not None and debug.get("selected_clauses"):
        evidence_chunks.extend(debug["selected_clauses"])
    elif source_docs:
        evidence_chunks.extend(doc.page_content for doc in source_docs)
    evidence_text = "\n".join(evidence_chunks)
    response_matched = ["/".join(group) for group in case.required_groups if _contains_any(response, group)]
    supported = [
        "/".join(group)
        for group in case.required_groups
        if _contains_any(response, group) and _contains_any(evidence_text, group)
    ]
    unsupported = [
        "/".join(group)
        for group in case.required_groups
        if _contains_any(response, group) and not _contains_any(evidence_text, group)
    ]
    required_supported = ["/".join(group) for group in case.required_groups if _contains_any(evidence_text, group)]
    required_support_coverage = (
        len(required_supported) / len(case.required_groups) if case.required_groups else None
    )
    answer_support_ratio = len(supported) / len(response_matched) if response_matched else None
    return {
        "supported_answer_groups": supported,
        "unsupported_answer_groups": unsupported,
        "required_supported_groups": required_supported,
        "required_support_coverage": None if required_support_coverage is None else round(required_support_coverage, 4),
        "answer_support_ratio": None if answer_support_ratio is None else round(answer_support_ratio, 4),
    }


def answer_direct_llm(llm, question: str) -> str:
    prompt = (
        "请直接根据已有的机械工程常识回答问题。"
        "回答要求：优先给出结论，长度控制在60字内；若涉及数值范围，直接给出范围。\n\n"
        f"问题：{question}\n回答："
    )
    result = llm.invoke(prompt)
    if not isinstance(result, str):
        result = str(result)
    return result.strip()


def _row_from_case(
    case: QACase,
    response: str,
    docs,
    debug: dict[str, Any] | None = None,
    semantic_scorer: SemanticScorer | None = None,
) -> dict[str, Any]:
    return {
        "case_id": case.case_id,
        "split": case.split,
        "variant": case.variant,
        "source_case_id": case.source_case_id,
        "expected_behavior": case.expected_behavior,
        "exclude_entry_ids": list(case.exclude_entry_ids),
        "question": case.question,
        "response": response,
        "retrieval": score_retrieval(case, docs),
        "response_eval": score_response(case, response, semantic_scorer=semantic_scorer),
        "support_eval": score_support(case, response, docs, debug=debug),
        "evidence_trace": [] if debug is None else debug.get("evidence_trace", []),
        "query_plan": None if debug is None else debug.get("query_plan"),
        "selected_clauses": [] if debug is None else debug.get("selected_clauses", []),
        "support_summary": None if debug is None else debug.get("support_summary"),
        "excluded_entry_ids": [] if debug is None else debug.get("excluded_entry_ids", []),
        "abstained": False if debug is None else debug.get("abstained", False),
        "abstain_reason": None if debug is None else debug.get("abstain_reason"),
    }


def evaluate_direct_llm(
    llm,
    cases: tuple[QACase, ...],
    semantic_scorer: SemanticScorer | None = None,
) -> list[dict[str, Any]]:
    rows = []
    for case in cases:
        response = answer_direct_llm(llm, case.question)
        rows.append(_row_from_case(case, response, [], debug=None, semantic_scorer=semantic_scorer))
    return rows


def evaluate_base(
    chain,
    cases: tuple[QACase, ...],
    semantic_scorer: SemanticScorer | None = None,
) -> list[dict[str, Any]]:
    rows = []
    for case in cases:
        result = chain.invoke({"query": case.question})
        response = _strip_or_fallback(result["result"], result["source_documents"])
        rows.append(
            _row_from_case(
                case,
                response,
                result["source_documents"],
                debug=None,
                semantic_scorer=semantic_scorer,
            )
        )
    return rows


def evaluate_pipeline(
    pipeline,
    cases: tuple[QACase, ...],
    semantic_scorer: SemanticScorer | None = None,
) -> list[dict[str, Any]]:
    rows = []
    for case in cases:
        response, docs, debug = pipeline.answer(
            case.question,
            exclude_entry_ids=case.exclude_entry_ids or None,
        )
        rows.append(_row_from_case(case, response, docs, debug=debug, semantic_scorer=semantic_scorer))
    return rows


def _summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    correct = sum(1 for row in rows if row["response_eval"]["label"] == "correct")
    partial = sum(1 for row in rows if row["response_eval"]["label"] == "partial")
    incorrect = sum(1 for row in rows if row["response_eval"]["label"] == "incorrect")
    weighted_score = sum(row["response_eval"]["weighted_score"] for row in rows) / n if n else 0.0
    retrieval_coverages = [
        row["retrieval"]["coverage"]
        for row in rows
        if row["retrieval"]["coverage"] is not None
    ]
    retrieval_coverage = (
        sum(retrieval_coverages) / len(retrieval_coverages) if retrieval_coverages else None
    )
    support_coverages = [
        row["support_eval"]["required_support_coverage"]
        for row in rows
        if row["support_eval"]["required_support_coverage"] is not None
    ]
    answer_support = [
        row["support_eval"]["answer_support_ratio"]
        for row in rows
        if row["support_eval"]["answer_support_ratio"] is not None
    ]
    semantic_scores = [row["response_eval"]["semantic_similarity"] for row in rows]
    hybrid_scores = [row["response_eval"]["hybrid_score"] for row in rows]
    numeric_scores = [
        row["response_eval"]["numeric_consistency"]
        for row in rows
        if row["response_eval"]["numeric_consistency"] is not None
    ]
    procedure_scores = [
        row["response_eval"]["procedure_order_score"]
        for row in rows
        if row["response_eval"]["procedure_order_score"] is not None
    ]
    predicted_abstain = sum(1 for row in rows if row["response_eval"]["abstained"])
    gold_abstain = sum(1 for row in rows if row["expected_behavior"] == "abstain")
    abstain_true_positive = sum(
        1
        for row in rows
        if row["expected_behavior"] == "abstain" and row["response_eval"]["abstained"]
    )
    abstain_behavior_correct = sum(1 for row in rows if row["response_eval"]["abstain_correct"])
    return {
        "n_cases": n,
        "correct": correct,
        "partial": partial,
        "incorrect": incorrect,
        "strict_accuracy": round(correct / n, 4) if n else 0.0,
        "weighted_accuracy": round(weighted_score, 4),
        "avg_hybrid_score": round(sum(hybrid_scores) / len(hybrid_scores), 4) if hybrid_scores else None,
        "avg_semantic_similarity": round(sum(semantic_scores) / len(semantic_scores), 4) if semantic_scores else None,
        "numeric_consistency_rate": round(sum(numeric_scores) / len(numeric_scores), 4) if numeric_scores else None,
        "avg_procedure_order_score": round(sum(procedure_scores) / len(procedure_scores), 4) if procedure_scores else None,
        "gold_abstain_cases": gold_abstain,
        "predicted_abstain_cases": predicted_abstain,
        "abstain_precision": round(abstain_true_positive / predicted_abstain, 4) if predicted_abstain else None,
        "abstain_recall": round(abstain_true_positive / gold_abstain, 4) if gold_abstain else None,
        "abstain_accuracy": round(abstain_behavior_correct / n, 4) if n else 0.0,
        "avg_retrieval_coverage": round(retrieval_coverage, 4) if retrieval_coverage is not None else None,
        "avg_required_support_coverage": round(sum(support_coverages) / len(support_coverages), 4) if support_coverages else None,
        "avg_answer_support_ratio": round(sum(answer_support) / len(answer_support), 4) if answer_support else None,
    }


def _summarize_counterfactual(rows: list[dict[str, Any]]) -> dict[str, Any]:
    row_by_case = {row["case_id"]: row for row in rows}
    counterfactual_rows = [
        row
        for row in rows
        if row["split"] == "counterfactual" and row.get("source_case_id")
    ]
    paired = 0
    source_answered = 0
    counterfactual_abstained = 0
    flip_correct = 0
    for row in counterfactual_rows:
        source_row = row_by_case.get(row["source_case_id"])
        if source_row is None:
            continue
        paired += 1
        source_is_answered = not source_row["response_eval"]["abstained"]
        counterfactual_is_abstained = row["response_eval"]["abstained"]
        source_answered += 1 if source_is_answered else 0
        counterfactual_abstained += 1 if counterfactual_is_abstained else 0
        flip_correct += 1 if source_is_answered and counterfactual_is_abstained else 0
    return {
        "paired_counterfactual_cases": paired,
        "counterfactual_source_answer_rate": round(source_answered / paired, 4) if paired else None,
        "counterfactual_abstain_rate": round(counterfactual_abstained / paired, 4) if paired else None,
        "counterfactual_flip_rate": round(flip_correct / paired, 4) if paired else None,
    }


def summarize_method(rows: list[dict[str, Any]]) -> dict[str, Any]:
    overall = _summarize_rows(rows)
    split_summaries = {}
    for split in SPLIT_ORDER:
        split_rows = [row for row in rows if row["split"] == split]
        if split_rows:
            split_summaries[split] = _summarize_rows(split_rows)
    overall.update(_summarize_counterfactual(rows))
    overall["split_summaries"] = split_summaries
    return overall


def build_problem_list(method_results: dict[str, list[dict[str, Any]]]) -> list[str]:
    problems: list[str] = []
    for method_name, rows in method_results.items():
        split_failures: dict[str, list[str]] = {}
        for row in rows:
            if row["response_eval"]["label"] == "incorrect":
                split_failures.setdefault(row["split"], []).append(row["question"])
        for split, questions in split_failures.items():
            problems.append(f"{method_name} 在 {split} split 仍有明显错误：{'；'.join(questions)}。")

        contradiction_rows = [row for row in rows if row["response_eval"]["contradictions"]]
        if contradiction_rows:
            detail = "；".join(
                f"{row['case_id']} -> {'/'.join(row['response_eval']['contradictions'])}"
                for row in contradiction_rows[:5]
            )
            problems.append(f"{method_name} 出现与知识库冲突的关键词：{detail}。")

        weak_retrieval = [
            row for row in rows
            if row["retrieval"]["coverage"] is not None and row["retrieval"]["coverage"] < 0.5
        ]
        if weak_retrieval:
            detail = "；".join(
                f"{row['case_id']}({row['split']}) -> {','.join(row['retrieval']['missing_evidence'][:2])}"
                for row in weak_retrieval[:4]
            )
            problems.append(f"{method_name} 在部分题目上证据覆盖不足：{detail}。")

        weak_support = [
            row for row in rows
            if row["support_eval"]["required_support_coverage"] is not None
            and row["support_eval"]["required_support_coverage"] < 0.5
        ]
        if weak_support:
            detail = "；".join(
                f"{row['case_id']}({row['split']}) -> {','.join(row['support_eval']['required_supported_groups'][:2]) or '无'}"
                for row in weak_support[:4]
            )
            problems.append(f"{method_name} 在部分题目上回答支撑不足：{detail}。")

        numeric_mismatch = [
            row for row in rows if row["response_eval"]["numeric_hard_mismatch"]
        ]
        if numeric_mismatch:
            detail = "；".join(
                f"{row['case_id']} -> gold={','.join(row['response_eval']['numeric_gold_spans']) or '无'}, "
                f"pred={','.join(row['response_eval']['numeric_response_spans']) or '无'}"
                for row in numeric_mismatch[:4]
            )
            problems.append(f"{method_name} 在数值题上出现硬性不一致：{detail}。")

        weak_procedure = [
            row
            for row in rows
            if row["response_eval"]["procedure_order_score"] is not None
            and row["response_eval"]["procedure_order_score"] < 0.75
        ]
        if weak_procedure:
            detail = "；".join(
                f"{row['case_id']} -> 缺失 {','.join(row['response_eval']['procedure_missing_steps'][:2]) or '无'}"
                for row in weak_procedure[:4]
            )
            problems.append(f"{method_name} 在流程型题目上的步骤顺序/覆盖仍偏弱：{detail}。")

        missed_abstain = [
            row
            for row in rows
            if row["expected_behavior"] == "abstain" and not row["response_eval"]["abstained"]
        ]
        if missed_abstain:
            detail = "；".join(f"{row['case_id']} -> {row['response']}" for row in missed_abstain[:4])
            problems.append(f"{method_name} 在 OOD/unsupported 题上仍出现无根据硬答：{detail}。")
    return problems


def _report_lines(
    cases: tuple[QACase, ...],
    method_results: dict[str, list[dict[str, Any]]],
    summaries: dict[str, dict[str, Any]],
) -> list[str]:
    lines = [
        "问答链路结构化评测报告",
        "=" * 72,
        "",
        "说明：当前结果应被解读为“当前知识库 + 当前数据划分”上的评测结果，不能直接外推为强泛化能力。",
        "",
        "一、总体汇总",
    ]
    for method_name, summary in summaries.items():
        retrieval_cov = summary["avg_retrieval_coverage"]
        retrieval_text = f"{retrieval_cov:.2f}" if retrieval_cov is not None else "N/A"
        support_cov = summary["avg_required_support_coverage"]
        support_text = f"{support_cov:.2f}" if support_cov is not None else "N/A"
        semantic_text = f"{summary['avg_semantic_similarity']:.2f}" if summary["avg_semantic_similarity"] is not None else "N/A"
        numeric_text = f"{summary['numeric_consistency_rate']:.2f}" if summary["numeric_consistency_rate"] is not None else "N/A"
        procedure_text = f"{summary['avg_procedure_order_score']:.2f}" if summary["avg_procedure_order_score"] is not None else "N/A"
        abstain_precision = summary["abstain_precision"]
        abstain_recall = summary["abstain_recall"]
        abstain_precision_text = f"{abstain_precision:.2f}" if abstain_precision is not None else "N/A"
        abstain_recall_text = f"{abstain_recall:.2f}" if abstain_recall is not None else "N/A"
        counterfactual_flip = summary.get("counterfactual_flip_rate")
        counterfactual_flip_text = f"{counterfactual_flip:.2f}" if counterfactual_flip is not None else "N/A"
        lines.append(
            f"- {method_name}: strict={summary['strict_accuracy']:.2f}, weighted={summary['weighted_accuracy']:.2f}, "
            f"correct={summary['correct']}, partial={summary['partial']}, incorrect={summary['incorrect']}, "
            f"semantic={semantic_text}, numeric={numeric_text}, procedure={procedure_text}, "
            f"abstain_p={abstain_precision_text}, abstain_r={abstain_recall_text}, cf_flip={counterfactual_flip_text}, "
            f"retrieval={retrieval_text}, support={support_text}"
        )
        for split in SPLIT_ORDER:
            split_summary = summary["split_summaries"].get(split)
            if split_summary is not None:
                split_support = split_summary["avg_required_support_coverage"]
                split_support_text = f"{split_support:.2f}" if split_support is not None else "N/A"
                split_semantic = split_summary["avg_semantic_similarity"]
                split_semantic_text = f"{split_semantic:.2f}" if split_semantic is not None else "N/A"
                split_numeric = split_summary["numeric_consistency_rate"]
                split_numeric_text = f"{split_numeric:.2f}" if split_numeric is not None else "N/A"
                split_abstain = split_summary["abstain_recall"]
                split_abstain_text = f"{split_abstain:.2f}" if split_abstain is not None else "N/A"
                lines.append(
                    f"  split={split}: strict={split_summary['strict_accuracy']:.2f}, "
                    f"weighted={split_summary['weighted_accuracy']:.2f}, semantic={split_semantic_text}, "
                    f"numeric={split_numeric_text}, abstain_r={split_abstain_text}, support={split_support_text}, n={split_summary['n_cases']}"
                )

    lines.extend(["", "二、逐题明细"])
    for idx, case in enumerate(cases, 1):
        lines.append(f"第{idx}题 [{case.split}/{case.variant}] {case.question}")
        lines.append(f"参考答案：{case.gold_answer}")
        lines.append(f"预期行为：{case.expected_behavior}")
        if case.exclude_entry_ids:
            lines.append(f"排除条目：{', '.join(case.exclude_entry_ids)}")
        for method_name, rows in method_results.items():
            row = next(item for item in rows if item["case_id"] == case.case_id)
            retrieval = row["retrieval"]
            retrieval_text = "N/A" if retrieval["coverage"] is None else f"{retrieval['coverage']:.2f}"
            semantic_text = f"{row['response_eval']['semantic_similarity']:.2f}"
            numeric_value = row["response_eval"]["numeric_consistency"]
            numeric_text = "N/A" if numeric_value is None else f"{numeric_value:.2f}"
            procedure_value = row["response_eval"]["procedure_order_score"]
            procedure_text = "N/A" if procedure_value is None else f"{procedure_value:.2f}"
            lines.append(
                f"{method_name}: label={row['response_eval']['label']}, "
                f"hybrid={row['response_eval']['hybrid_score']:.2f}, lexical={row['response_eval']['lexical_score']:.2f}, "
                f"semantic={semantic_text}, numeric={numeric_text}, procedure={procedure_text}, retrieval={retrieval_text}"
            )
            lines.append(f"回答：{row['response']}")
            if row["selected_clauses"]:
                lines.append(f"证据片段：{' | '.join(row['selected_clauses'])}")
            if row["support_summary"] is not None:
                active_entries = row["support_summary"].get("active_entry_ids", [])
                lines.append(
                    f"证据条目：{','.join(active_entries) if active_entries else '无'}"
                )
            if row["abstained"]:
                lines.append(f"拒答原因：{row['abstain_reason'] or '评测识别为拒答'}")
            if row["support_eval"]["unsupported_answer_groups"]:
                lines.append(f"未被证据支撑的回答要点：{', '.join(row['support_eval']['unsupported_answer_groups'])}")
            if row["response_eval"]["missing_required"]:
                lines.append(f"缺失要点：{', '.join(row['response_eval']['missing_required'])}")
            if row["response_eval"]["numeric_hard_mismatch"]:
                lines.append(
                    f"数值不一致：gold={','.join(row['response_eval']['numeric_gold_spans']) or '无'}; "
                    f"pred={','.join(row['response_eval']['numeric_response_spans']) or '无'}"
                )
            if row["response_eval"]["procedure_missing_steps"]:
                lines.append(f"流程缺失：{', '.join(row['response_eval']['procedure_missing_steps'])}")
            if row["response_eval"]["contradictions"]:
                lines.append(f"冲突词：{', '.join(row['response_eval']['contradictions'])}")
        lines.append("-" * 40)
    return [line + "\n" for line in lines]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="mechanical_data.txt")
    parser.add_argument("--case_set", choices=("core", "stress", "full"), default="full")
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()

    if not Path(args.data_path).exists():
        raise FileNotFoundError(f"未找到数据文件: {args.data_path}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cases = get_cases(args.case_set)
    llm = get_llm(model_name=DEFAULT_CN_MODEL, max_new_tokens=128)
    semantic_scorer = SemanticScorer(DEFAULT_EMBEDDING)
    base_chain = build_chain(
        args.data_path,
        DEFAULT_CN_MODEL,
        DEFAULT_EMBEDDING,
        ".cache/chroma/base_rag",
    )
    improved_pipeline = build_components(
        data_path=args.data_path,
        model_name=DEFAULT_CN_MODEL,
        embedding_model_name=DEFAULT_EMBEDDING,
        db_dir=DEFAULT_DB_DIRS["improved"],
        mode="improved",
    )
    rule_heavy_pipeline = build_system(args.data_path, DEFAULT_CN_MODEL)

    method_results = {
        "direct_llm": evaluate_direct_llm(llm, cases, semantic_scorer=semantic_scorer),
        "base_rag": evaluate_base(base_chain, cases, semantic_scorer=semantic_scorer),
        "improved_rag": evaluate_pipeline(improved_pipeline, cases, semantic_scorer=semantic_scorer),
        "problem_solving_rag": evaluate_pipeline(rule_heavy_pipeline, cases, semantic_scorer=semantic_scorer),
    }
    summaries = {name: summarize_method(rows) for name, rows in method_results.items()}

    report_lines = _report_lines(cases, method_results, summaries)
    (output_dir / REPORT_FILENAME).write_text("".join(report_lines), encoding="utf-8")
    (output_dir / DETAIL_FILENAME).write_text(
        json.dumps(
            {
                "case_set": args.case_set,
                "cases": [
                    {
                        "case_id": case.case_id,
                        "question": case.question,
                        "gold_answer": case.gold_answer,
                        "split": case.split,
                        "variant": case.variant,
                        "source_case_id": case.source_case_id,
                        "expected_behavior": case.expected_behavior,
                        "exclude_entry_ids": list(case.exclude_entry_ids),
                    }
                    for case in cases
                ],
                "summaries": summaries,
                "method_results": method_results,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    with (output_dir / DIRECT_RESULT_FILENAME).open("w", encoding="utf-8") as handle:
        handle.write("直接LLM基线结果\n")
        handle.write("=" * 60 + "\n")
        for case, row in zip(cases, method_results["direct_llm"]):
            handle.write(f"[{case.split}/{case.variant}] 问题：{case.question}\n")
            handle.write(f"回答：{row['response']}\n")
            handle.write(f"判定：{row['response_eval']['label']}\n")
            handle.write("-" * 40 + "\n")

    with (output_dir / RULE_HEAVY_RESULT_FILENAME).open("w", encoding="utf-8") as handle:
        handle.write("Rule-heavy QA 结果\n")
        handle.write("=" * 60 + "\n")
        for case, row in zip(cases, method_results["problem_solving_rag"]):
            handle.write(f"[{case.split}/{case.variant}] 问题：{case.question}\n")
            handle.write(f"回答：{row['response']}\n")
            handle.write(f"判定：{row['response_eval']['label']}\n")
            if row["selected_clauses"]:
                handle.write(f"证据：{' | '.join(row['selected_clauses'])}\n")
            handle.write("-" * 40 + "\n")

    problems = build_problem_list(method_results)
    (output_dir / PROBLEM_FILENAME).write_text("\n".join(problems) + "\n", encoding="utf-8")

    print("=" * 72)
    print(f"QA 结构化评测完成 (case_set={args.case_set})")
    print("=" * 72)
    for method_name, summary in summaries.items():
        retrieval_cov = summary["avg_retrieval_coverage"]
        retrieval_text = f"{retrieval_cov:.2f}" if retrieval_cov is not None else "N/A"
        support_cov = summary["avg_required_support_coverage"]
        support_text = f"{support_cov:.2f}" if support_cov is not None else "N/A"
        semantic_text = f"{summary['avg_semantic_similarity']:.2f}" if summary["avg_semantic_similarity"] is not None else "N/A"
        numeric_text = f"{summary['numeric_consistency_rate']:.2f}" if summary["numeric_consistency_rate"] is not None else "N/A"
        abstain_recall = summary["abstain_recall"]
        abstain_text = f"{abstain_recall:.2f}" if abstain_recall is not None else "N/A"
        counterfactual_flip = summary.get("counterfactual_flip_rate")
        counterfactual_text = f"{counterfactual_flip:.2f}" if counterfactual_flip is not None else "N/A"
        print(
            f"{method_name}: strict={summary['strict_accuracy']:.2f}, "
            f"weighted={summary['weighted_accuracy']:.2f}, semantic={semantic_text}, "
            f"numeric={numeric_text}, abstain_r={abstain_text}, cf_flip={counterfactual_text}, "
            f"retrieval={retrieval_text}, support={support_text}"
        )
        for split in SPLIT_ORDER:
            split_summary = summary["split_summaries"].get(split)
            if split_summary is not None:
                split_semantic = split_summary["avg_semantic_similarity"]
                split_semantic_text = f"{split_semantic:.2f}" if split_semantic is not None else "N/A"
                print(
                    f"  split={split}: strict={split_summary['strict_accuracy']:.2f}, "
                    f"weighted={split_summary['weighted_accuracy']:.2f}, semantic={split_semantic_text}, "
                    f"n={split_summary['n_cases']}"
                )
    print(f"\n结果已保存到: {output_dir}")


if __name__ == "__main__":
    main()
