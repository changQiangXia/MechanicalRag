"""
P3 轻量学习模块：独立训练脚本。
在项目根目录执行：python -m simulation.train_learned_model [--data_path mechanical_data.txt]
训练完成后，benchmark --method rag_learned 将自动加载已保存模型。
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="训练 RAG→夹爪力 轻量 MLP 模型（含不确定性 ensemble）")
    parser.add_argument("--data_path", default="mechanical_data.txt")
    parser.add_argument("--model_dir", default=None, help="模型保存目录，默认 simulation/learned_models")
    args = parser.parse_args()

    from simulation.learned_controller import train_models, DEFAULT_MODEL_DIR

    model_dir = args.model_dir or DEFAULT_MODEL_DIR
    print("训练轻量学习模块（RAG 嵌入 → MLP ensemble）...")
    train_models(data_path=args.data_path, model_dir=model_dir)
    print(f"模型已保存至: {model_dir}")
    print("运行 benchmark 使用: python -m simulation.benchmark --method rag_learned --n_trials 5")


if __name__ == "__main__":
    main()
