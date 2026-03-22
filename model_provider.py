"""模型下载与路径解析：默认优先从 ModelScope 拉取并缓存到项目目录。"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_PROVIDER = os.environ.get("MODEL_PROVIDER", "modelscope").strip().lower()
DEFAULT_CACHE_DIR = Path(
    os.environ.get("MODEL_CACHE_DIR", PROJECT_ROOT / ".modelscope_cache")
).expanduser().resolve()

MODELSCOPE_MODEL_MAP = {
    "Qwen/Qwen2-0.5B-Instruct": "qwen/Qwen2-0.5B-Instruct",
    "qwen/Qwen2-0.5B-Instruct": "qwen/Qwen2-0.5B-Instruct",
    "sentence-transformers/all-MiniLM-L6-v2": "AI-ModelScope/all-MiniLM-L6-v2",
    "AI-ModelScope/all-MiniLM-L6-v2": "AI-ModelScope/all-MiniLM-L6-v2",
    "google/flan-t5-base": "google/flan-t5-base",
}


def _normalize_provider(provider: str | None) -> str:
    normalized = (provider or DEFAULT_PROVIDER).strip().lower()
    if normalized in {"hf", "huggingface"}:
        return "huggingface"
    if normalized == "modelscope":
        return "modelscope"
    raise ValueError(f"不支持的模型来源: {provider}")


def _existing_local_path(model_name: str) -> str | None:
    path = Path(model_name).expanduser()
    if path.exists():
        return str(path.resolve())
    return None


def resolve_modelscope_model_id(model_name: str) -> str:
    """将原始模型名映射到 ModelScope 仓库名。未命中的名称按原样透传。"""
    return MODELSCOPE_MODEL_MAP.get(model_name, model_name)


@lru_cache(maxsize=None)
def resolve_model_path(model_name: str, provider: str | None = None) -> str:
    """
    返回可供 transformers / sentence-transformers 使用的模型路径。
    - 本地路径: 直接返回
    - ModelScope: 下载到项目内缓存并返回本地目录
    - Hugging Face: 原样返回模型名，维持原有行为
    """
    local_path = _existing_local_path(model_name)
    if local_path is not None:
        return local_path

    resolved_provider = _normalize_provider(provider)
    if resolved_provider == "huggingface":
        return model_name

    try:
        from modelscope import snapshot_download
    except Exception as exc:  # pragma: no cover - 导入失败时直接给用户明确提示
        raise RuntimeError(
            "无法导入 modelscope。请确认已安装兼容版本（当前项目固定为 modelscope==1.23.2）。"
        ) from exc

    DEFAULT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    modelscope_model_id = resolve_modelscope_model_id(model_name)
    try:
        return snapshot_download(modelscope_model_id, cache_dir=str(DEFAULT_CACHE_DIR))
    except Exception as exc:  # pragma: no cover - 真实下载失败取决于网络与权限
        raise RuntimeError(
            f"通过 ModelScope 下载模型失败: {modelscope_model_id}。"
            "若需回退到原始 Hugging Face，可设置环境变量 MODEL_PROVIDER=huggingface。"
        ) from exc
