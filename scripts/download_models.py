"""手动预下载项目所需模型到本地 ModelScope 缓存。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from llm_loader import DEFAULT_CN_MODEL, DEFAULT_EN_MODEL
from model_provider import DEFAULT_CACHE_DIR, resolve_model_path


DEFAULT_EMBEDDING = "sentence-transformers/all-MiniLM-L6-v2"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", default="modelscope", choices=("modelscope", "huggingface", "hf"))
    parser.add_argument("--with_flan", action="store_true", help="额外下载 google/flan-t5-base")
    args = parser.parse_args()

    models = [DEFAULT_CN_MODEL, DEFAULT_EMBEDDING]
    if args.with_flan:
        models.append(DEFAULT_EN_MODEL)

    print(f"模型缓存目录: {DEFAULT_CACHE_DIR}")
    for model_name in models:
        local_path = resolve_model_path(model_name, provider=args.provider)
        print(f"{model_name} -> {local_path}")


if __name__ == "__main__":
    main()
