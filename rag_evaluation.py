"""直接 LLM、基础 RAG、改进 RAG、问题解决型 RAG 的结构化对比评测。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from base_rag import build_chain
from improved_rag import build_components, answer_question, MECHANICAL_KEYWORDS
from llm_loader import DEFAULT_CN_MODEL, get_llm
from problem_solving import build_system, answer as answer_problem_solving
from qa_dataset import QA_CASES, QACase


def _contains_any(text: str, keywords: tuple[str, ...]) -> bool:
    return any(keyword in text for keyword in keywords)


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


def score_response(case: QACase, response: str) -> dict[str, Any]:
    matched = ["/".join(group) for group in case.required_groups if _contains_any(response, group)]
    missing = ["/".join(group) for group in case.required_groups if not _contains_any(response, group)]
    contradictions = [keyword for keyword in case.forbidden_keywords if keyword in response]
    score = len(matched) / len(case.required_groups) if case.required_groups else 0.0
    if contradictions:
        label = "incorrect"
    elif score >= 0.999:
        label = "correct"
    elif score >= 0.5:
        label = "partial"
    else:
        label = "incorrect"
    weighted = {"correct": 1.0, "partial": 0.5, "incorrect": 0.0}[label]
    return {
        "label": label,
        "score": round(score, 4),
        "weighted_score": weighted,
        "matched_required": matched,
        "missing_required": missing,
        "contradictions": contradictions,
    }


def _ensure_meaningful_response(response: str, source_docs) -> str:
    """若模型输出无意义（缺少中文），则回退为检索到的文档内容"""
    chinese_count = sum(1 for c in response if "\u4e00" <= c <= "\u9fff")
    if chinese_count >= 2 and len(response.strip()) >= 4:
        return response
    if source_docs:
        return source_docs[0].page_content[:200] + ("..." if len(source_docs[0].page_content) > 200 else "")
    return response.strip()


def answer_direct_llm(llm, question: str) -> str:
    prompt = (
        "请直接根据你已有的机械工程与机械臂常识回答问题。"
        "回答要求：优先给出结论，长度控制在60字内，涉及数值时直接写范围。\n\n"
        f"问题：{question}\n回答："
    )
    result = llm.invoke(prompt)
    if not isinstance(result, str):
        result = str(result)
    return result.strip()


def evaluate_direct_llm(llm, cases: List[QACase]) -> List[Dict]:
    results = []
    for case in cases:
        response = answer_direct_llm(llm, case.question)
        results.append(
            {
                "question": case.question,
                "response": response,
                "retrieval": score_retrieval(case, []),
                "response_eval": score_response(case, response),
            }
        )
    return results


def evaluate_base(chain, cases: List[QACase]) -> List[Dict]:
    results = []
    for case in cases:
        result = chain.invoke({"query": case.question})
        response = _ensure_meaningful_response(result["result"], result["source_documents"])
        results.append(
            {
                "question": case.question,
                "response": response,
                "retrieval": score_retrieval(case, result["source_documents"]),
                "response_eval": score_response(case, response),
            }
        )
    return results


def evaluate_improved(llm, vector_db, cases: List[QACase]) -> List[Dict]:
    results = []
    for case in cases:
        response, docs = answer_question(llm, vector_db, case.question, k=2)
        results.append(
            {
                "question": case.question,
                "response": response,
                "retrieval": score_retrieval(case, docs),
                "response_eval": score_response(case, response),
            }
        )
    return results


def evaluate_problem_solving(llm, vector_db, cases: List[QACase]) -> List[Dict]:
    results = []
    for case in cases:
        response, docs = answer_problem_solving(llm, vector_db, case.question)
        results.append(
            {
                "question": case.question,
                "response": response,
                "retrieval": score_retrieval(case, docs),
                "response_eval": score_response(case, response),
            }
        )
    return results


def summarize_method(results: List[Dict]) -> Dict[str, Any]:
    n = len(results)
    correct = sum(1 for row in results if row["response_eval"]["label"] == "correct")
    partial = sum(1 for row in results if row["response_eval"]["label"] == "partial")
    incorrect = sum(1 for row in results if row["response_eval"]["label"] == "incorrect")
    weighted_score = sum(row["response_eval"]["weighted_score"] for row in results) / n if n else 0.0
    retrieval_coverages = [
        row["retrieval"]["coverage"]
        for row in results
        if row["retrieval"]["coverage"] is not None
    ]
    retrieval_coverage = (
        sum(retrieval_coverages) / len(retrieval_coverages) if retrieval_coverages else None
    )
    return {
        "n_cases": n,
        "correct": correct,
        "partial": partial,
        "incorrect": incorrect,
        "strict_accuracy": round(correct / n, 4) if n else 0.0,
        "weighted_accuracy": round(weighted_score, 4),
        "avg_retrieval_coverage": round(retrieval_coverage, 4) if retrieval_coverage is not None else None,
    }


def build_problem_list(method_results: dict[str, List[Dict]]) -> list[str]:
    problems: list[str] = []
    for method_name, rows in method_results.items():
        incorrect_questions = [
            row["question"]
            for row in rows
            if row["response_eval"]["label"] == "incorrect"
        ]
        contradiction_rows = [
            row for row in rows if row["response_eval"]["contradictions"]
        ]
        missing_rows = [
            row for row in rows if row["response_eval"]["missing_required"]
        ]

        if incorrect_questions:
            problems.append(
                f"{method_name} 在以下题目上仍有明显错误："
                + "；".join(incorrect_questions)
                + "。"
            )
        if contradiction_rows:
            detail = "；".join(
                f"{row['question']} -> {'/'.join(row['response_eval']['contradictions'])}"
                for row in contradiction_rows
            )
            problems.append(f"{method_name} 出现与知识库冲突的关键词：{detail}。")
        if missing_rows:
            detail = "；".join(
                f"{row['question']} -> {','.join(row['response_eval']['missing_required'][:2])}"
                for row in missing_rows[:3]
            )
            problems.append(f"{method_name} 在部分题目上关键信息覆盖不完整：{detail}。")
    return problems


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="mechanical_data.txt")
    args = parser.parse_args()

    if not Path(args.data_path).exists():
        raise FileNotFoundError(f"未找到数据文件: {args.data_path}")

    cases = list(QA_CASES)
    llm = get_llm(model_name=DEFAULT_CN_MODEL, max_new_tokens=128)
    base_chain = build_chain(args.data_path, DEFAULT_CN_MODEL, "sentence-transformers/all-MiniLM-L6-v2", "./chroma_db")
    improved_llm, improved_db = build_components(
        args.data_path, DEFAULT_CN_MODEL, "sentence-transformers/all-MiniLM-L6-v2", "./improved_chroma_db"
    )
    solved_llm, solved_db = build_system(args.data_path, DEFAULT_CN_MODEL)

    method_results = {
        "direct_llm": evaluate_direct_llm(llm, cases),
        "base_rag": evaluate_base(base_chain, cases),
        "improved_rag": evaluate_improved(improved_llm, improved_db, cases),
        "problem_solving_rag": evaluate_problem_solving(solved_llm, solved_db, cases),
    }
    summaries = {name: summarize_method(rows) for name, rows in method_results.items()}

    lines = []
    print("=" * 60)
    print("直接 LLM、基础 RAG、改进 RAG、问题解决型 RAG 对比评测")
    print("=" * 60)
    for method_name, summary in summaries.items():
        retrieval_cov = summary["avg_retrieval_coverage"]
        retrieval_text = f"{retrieval_cov:.2f}" if retrieval_cov is not None else "N/A"
        header = (
            f"{method_name}: strict={summary['strict_accuracy']:.2f}, "
            f"weighted={summary['weighted_accuracy']:.2f}, "
            f"correct={summary['correct']}, partial={summary['partial']}, "
            f"incorrect={summary['incorrect']}, retrieval={retrieval_text}"
        )
        print(header)
        lines.append(header + "\n")

    lines.append("\n" + "=" * 60 + "\n")
    for idx, case in enumerate(cases, 1):
        lines.append(f"第{idx}题：{case.question}\n")
        lines.append(f"参考答案：{case.gold_answer}\n")
        for method_name, rows in method_results.items():
            row = next(item for item in rows if item["question"] == case.question)
            response_eval = row["response_eval"]
            retrieval = row["retrieval"]
            retrieval_text = (
                "N/A" if retrieval["coverage"] is None else f"{retrieval['coverage']:.2f}"
            )
            lines.append(
                f"{method_name}: label={response_eval['label']}, "
                f"score={response_eval['score']:.2f}, retrieval={retrieval_text}\n"
            )
            lines.append(f"回答：{row['response']}\n")
            if response_eval["missing_required"]:
                lines.append(f"缺失要点：{', '.join(response_eval['missing_required'])}\n")
            if response_eval["contradictions"]:
                lines.append(f"冲突词：{', '.join(response_eval['contradictions'])}\n")
        lines.append("-" * 40 + "\n")

    with open("rag_evaluation_report.txt", "w", encoding="utf-8") as f:
        f.write("问答链路结构化评测报告\n")
        f.write("=" * 60 + "\n")
        f.writelines(lines)

    with open("qa_evaluation_detail.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "cases": [
                    {
                        "case_id": case.case_id,
                        "question": case.question,
                        "gold_answer": case.gold_answer,
                    }
                    for case in cases
                ],
                "summaries": summaries,
                "method_results": method_results,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    with open("direct_llm_result.txt", "w", encoding="utf-8") as f:
        f.write("直接LLM基线结果\n")
        f.write("=" * 60 + "\n")
        for case, row in zip(cases, method_results["direct_llm"]):
            f.write(f"问题：{case.question}\n")
            f.write(f"回答：{row['response']}\n")
            f.write(f"判定：{row['response_eval']['label']}\n")
            f.write("-" * 40 + "\n")

    problems = build_problem_list(method_results)
    with open("rag_problems.txt", "w", encoding="utf-8") as f:
        for p in problems:
            print(p)
            f.write(p + "\n")


if __name__ == "__main__":
    main()
