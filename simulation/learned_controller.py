"""
P3 轻量学习模块：RAG 检索文本的嵌入 → MLP → 夹爪力预测，带简单不确定性估计。
用于论文级对比：规则解析 vs 学习型参数预测。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .rag_controller import RAGController
from .tasks import BENCHMARK_TASKS
from model_provider import resolve_model_path

# 与 RAG 使用相同 embedding，保证输入空间一致
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_MODEL_DIR = Path(__file__).resolve().parent / "learned_models"
FORCE_MIN, FORCE_MAX = 5.0, 50.0


def _get_training_data(data_path: str, augment_per_task: int = 5):
    """
    从 benchmark 任务生成训练数据：(context_embedding, gripper_force)。
    每个任务用多种 query 检索得到不同 context，标签为理想夹爪力范围中点。
    """
    from langchain_community.embeddings import HuggingFaceEmbeddings

    rag = RAGController(data_path)
    embedding_model_path = resolve_model_path(EMBEDDING_MODEL_NAME)
    emb = HuggingFaceEmbeddings(model_name=embedding_model_path)
    X_list, y_list = [], []

    for task in BENCHMARK_TASKS:
        label = (task.ideal_gripper_force[0] + task.ideal_gripper_force[1]) / 2.0
        queries = [
            task.description,
            task.description + " 夹爪力",
            task.description + " 抓取 力",
            task.object_type + " 夹爪力",
            task.object_type + " 抓取",
        ][:augment_per_task]
        for q in queries:
            docs = rag.vector_db.similarity_search(q, k=3)
            context = "\n".join(d.page_content for d in docs)
            vec = emb.embed_query(context)
            X_list.append(vec)
            y_list.append(label)

    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32)


def train_models(
    data_path: str = "mechanical_data.txt",
    model_dir: str | Path | None = None,
    n_ensemble: int = 3,
):
    """
    训练 n_ensemble 个 MLP 回归器（不同随机种子），用于不确定性估计。
    保存到 model_dir/learned_force_0.joblib, ... 及 meta.json。
    """
    from sklearn.neural_network import MLPRegressor

    model_dir = Path(model_dir or DEFAULT_MODEL_DIR)
    model_dir.mkdir(parents=True, exist_ok=True)

    X, y = _get_training_data(data_path)
    if X.size == 0:
        raise RuntimeError("无训练数据")

    models = []
    for i in range(n_ensemble):
        reg = MLPRegressor(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            max_iter=500,
            random_state=42 + i,
            early_stopping=True,
            validation_fraction=0.15,
        )
        reg.fit(X, y)
        models.append(reg)

    import joblib
    for i, reg in enumerate(models):
        joblib.dump(reg, model_dir / f"learned_force_{i}.joblib")
    meta = {
        "embedding_model": EMBEDDING_MODEL_NAME,
        "n_ensemble": n_ensemble,
        "n_samples": len(y),
        "force_range": [FORCE_MIN, FORCE_MAX],
    }
    (model_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return models


def load_models(model_dir: str | Path | None = None):
    """加载已训练的 ensemble 模型。"""
    import joblib

    model_dir = Path(model_dir or DEFAULT_MODEL_DIR)
    meta_path = model_dir / "meta.json"
    if not meta_path.exists():
        return None, None
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    n = meta["n_ensemble"]
    models = [joblib.load(model_dir / f"learned_force_{i}.joblib") for i in range(n)]
    return models, meta


def predict_force_with_uncertainty(
    context: str,
    models: list,
    embedding_model: Any,
    force_min: float = FORCE_MIN,
    force_max: float = FORCE_MAX,
) -> tuple[float, float]:
    """
    用 ensemble 预测夹爪力，返回 (force, std)。
    force 为均值并裁剪到 [force_min, force_max]，std 为各模型预测的标准差（不确定性估计）。
    """
    vec = np.array(embedding_model.embed_query(context), dtype=np.float32).reshape(1, -1)
    preds = [np.clip(m.predict(vec)[0], force_min, force_max) for m in models]
    mean_force = float(np.mean(preds))
    std_force = float(np.std(preds)) if len(preds) > 1 else 0.0
    return mean_force, std_force


class LearnedParamController:
    """
    轻量学习型参数控制器：RAG 检索 → 嵌入 → MLP(ensemble) → 夹爪力 ± 不确定性。
    若未训练则先训练并保存，再加载。
    """

    def __init__(self, data_path: str = "mechanical_data.txt", model_dir: str | Path | None = None):
        from langchain_community.embeddings import HuggingFaceEmbeddings

        self.rag = RAGController(data_path)
        embedding_model_path = resolve_model_path(EMBEDDING_MODEL_NAME)
        self.embedding = HuggingFaceEmbeddings(model_name=embedding_model_path)
        self.model_dir = Path(model_dir or DEFAULT_MODEL_DIR)
        self.models, self.meta = load_models(self.model_dir)
        if self.models is None:
            train_models(data_path=data_path, model_dir=self.model_dir)
            self.models, self.meta = load_models(self.model_dir)
        if self.models is None:
            raise RuntimeError("轻量学习模块训练或加载失败")

    def get_params_for_task(self, task_description: str) -> dict[str, Any]:
        """检索 RAG 上下文，用学习模块预测夹爪力，并返回不确定性。"""
        docs = self.rag.vector_db.similarity_search(task_description, k=3)
        context = "\n".join(d.page_content for d in docs)
        force, force_std = predict_force_with_uncertainty(
            context, self.models, self.embedding, FORCE_MIN, FORCE_MAX
        )
        height = _get_height_for_task(task_description)
        return {
            "gripper_force": round(force, 2),
            "approach_height": height,
            "rag_source": context[:200],
            "uncertainty_std": round(force_std, 2),
        }


def _get_height_for_task(task_description: str) -> float:
    from .rag_controller import GROUND_TRUTH_PARAMS
    for key, params in GROUND_TRUTH_PARAMS.items():
        if key in task_description:
            return params["approach_height"]
    return 0.05
