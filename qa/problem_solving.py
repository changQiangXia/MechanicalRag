"""Compatibility wrapper for the unified QA pipeline in rule-heavy mode."""

from __future__ import annotations

import argparse
from pathlib import Path

from .pipeline import (
    DEFAULT_DB_DIRS,
    DEFAULT_EMBEDDING,
    DEFAULT_MODEL,
    MechanicalQAPipeline,
    build_components,
)


def build_system(data_path: str, model_name: str = DEFAULT_MODEL) -> MechanicalQAPipeline:
    return build_components(
        data_path=data_path,
        model_name=model_name,
        embedding_model_name=DEFAULT_EMBEDDING,
        db_dir=DEFAULT_DB_DIRS["rule_heavy"],
        mode="rule_heavy",
    )


def answer(pipeline: MechanicalQAPipeline, question: str):
    return pipeline.answer(question)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="mechanical_data.txt")
    parser.add_argument("--model_name", default=DEFAULT_MODEL)
    args = parser.parse_args()

    if not Path(args.data_path).exists():
        raise FileNotFoundError(f"未找到数据文件: {args.data_path}")

    pipeline = build_system(args.data_path, args.model_name)
    questions = [
        "机械臂标定后，为什么需要进行复位操作？",
        "抓取圆柱形工件时，最佳抓取姿态是什么？",
        "抓取光滑金属零件时，夹爪力应控制在什么范围？",
        "薄壁件抓取时需要注意什么？",
        "视觉相机用于高速抓取时，采集帧率应该设置为多少fps？",
    ]
    print("=" * 50)
    print("Rule-heavy QA 模式测试结果")
    print("=" * 50)
    for idx, question in enumerate(questions, 1):
        response, docs, debug = answer(pipeline, question)
        print(f"\n第{idx}个问题：{question}")
        print("回答：", response)
        print("证据：")
        for row in debug["evidence_trace"]:
            print(f"- [{row['category']}] {row['excerpt']}")


if __name__ == "__main__":
    main()
