"""
独立 learned baseline：
- 训练标签不再来自 RAGController 银标
- 监督信号由任务 profile + 环境独立推导函数生成
- 一次性预测四参数控制计划，并保留 ensemble 不确定性估计
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from . import baseline_controller as baseline
from .env import _estimate_force_window, _estimate_motion_targets
from .tasks import BENCHMARK_TASKS, TaskConfig
from model_provider import resolve_model_path

# 与 RAG 使用相同 embedding，保证文本输入空间一致
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_MODEL_DIR = Path(__file__).resolve().parent / "learned_models"
MODEL_VARIANT = "residual_task_heuristic_v2"

FORCE_MIN, FORCE_MAX = 5.0, 50.0
HEIGHT_MIN, HEIGHT_MAX = 0.02, 0.10
VELOCITY_MIN, VELOCITY_MAX = 0.12, 0.80
CLEARANCE_MIN, CLEARANCE_MAX = 0.03, 0.14
PLAN_KEYS = ("gripper_force", "approach_height", "transport_velocity", "lift_clearance")


def _clip_plan(force: float, height: float, velocity: float, clearance: float) -> tuple[float, float, float, float]:
    return (
        float(np.clip(force, FORCE_MIN, FORCE_MAX)),
        float(np.clip(height, HEIGHT_MIN, HEIGHT_MAX)),
        float(np.clip(velocity, VELOCITY_MIN, VELOCITY_MAX)),
        float(np.clip(clearance, CLEARANCE_MIN, CLEARANCE_MAX)),
    )


def _keyword_features(task_text: str) -> np.ndarray:
    return np.array(
        [
            1.0 if "高速" in task_text else 0.0,
            1.0 if "金属" in task_text else 0.0,
            1.0 if "橡胶" in task_text else 0.0,
            1.0 if "小型" in task_text else 0.0,
            1.0 if "大型" in task_text else 0.0,
            1.0 if "重型" in task_text else 0.0,
            1.0 if "薄壁" in task_text else 0.0,
            1.0 if "长距离" in task_text else 0.0,
        ],
        dtype=np.float32,
    )


def _encode_task_text(embedding_model: Any, task_text: str) -> np.ndarray:
    dense = np.array(embedding_model.embed_query(task_text), dtype=np.float32)
    return np.concatenate([dense, _keyword_features(task_text)]).astype(np.float32)


def _environment_teacher_plan(task: TaskConfig) -> tuple[float, float, float, float]:
    """
    用环境内部独立可见的物体属性构造 teacher plan。
    该标签来源不读取当前 RAG 输出，因此可作为独立监督信号。
    """
    if task.profile is None:
        return _clip_plan(25.0, 0.05, 0.30, 0.06)

    profile = task.profile
    horizontal_distance = float(
        ((task.target_pos[0] - task.object_pos[0]) ** 2 + (task.target_pos[1] - task.object_pos[1]) ** 2) ** 0.5
    )
    recommended_velocity, min_lift_clearance = _estimate_motion_targets(
        mass_kg=profile.mass_kg,
        surface_friction=profile.surface_friction,
        fragility=profile.fragility,
        size_xyz=profile.size_xyz,
    )
    staged_distance = 2.0 * profile.preferred_approach_height + 2.0 * min_lift_clearance + horizontal_distance
    min_force_needed, max_safe_force, nominal_force = _estimate_force_window(
        mass_kg=profile.mass_kg,
        surface_friction=profile.surface_friction,
        fragility=profile.fragility,
        travel_distance=staged_distance,
        size_xyz=profile.size_xyz,
    )

    description = task.description
    force = nominal_force
    velocity = recommended_velocity
    clearance = min_lift_clearance
    approach_height = profile.preferred_approach_height

    if "高速" in description:
        force = min(max_safe_force - 0.8, nominal_force + 0.18 * (max_safe_force - nominal_force) + 1.2)
        velocity = min(0.62, recommended_velocity + 0.05 + 0.04 * max(0.0, profile.velocity_scale - 0.8))
        clearance += 0.008
    if "长距离" in description:
        force = min(max_safe_force - 0.8, force + 1.5)
        clearance += 0.005
    if "薄壁" in description:
        force = min(force, nominal_force)
        velocity = min(velocity, 0.24 if "高速" in description else 0.20)
        clearance += 0.006
    if "小型" in description and "高速" not in description:
        velocity = max(velocity, 0.34)
    if "重型" in description or profile.mass_kg >= 0.35:
        velocity = min(velocity, 0.24 if "高速" not in description else velocity)
        clearance += 0.004
    if "橡胶" in description:
        force = max(min_force_needed + 0.5, min(force, nominal_force + 1.0))

    return _clip_plan(force, approach_height, velocity, clearance)


def _query_variants(task: TaskConfig, augment_per_task: int) -> list[str]:
    candidates = [
        task.description,
        f"{task.description} 控制参数",
        f"{task.object_type} 抓取",
        f"{task.object_type} 夹爪力",
        f"{task.description} 速度 净空",
        f"{task.description} 稳定抓取",
    ]
    return candidates[:augment_per_task]


def _get_training_data(augment_per_task: int = 6):
    """
    生成训练数据：(task_text_features, 4D residual plan)。
    标签来自 environment teacher 与 task_heuristic 的残差，不依赖当前 RAG 输出。
    """
    from langchain_community.embeddings import HuggingFaceEmbeddings

    embedding_model_path = resolve_model_path(EMBEDDING_MODEL_NAME)
    emb = HuggingFaceEmbeddings(model_name=embedding_model_path)
    X_list, y_list = [], []

    for task in BENCHMARK_TASKS:
        teacher = np.array(_environment_teacher_plan(task), dtype=np.float32)
        heuristic = baseline.get_params_task_heuristic(task.description)
        heuristic_vec = np.array(
            [
                heuristic["gripper_force"],
                heuristic["approach_height"],
                heuristic["transport_velocity"],
                heuristic["lift_clearance"],
            ],
            dtype=np.float32,
        )
        residual = teacher - heuristic_vec
        for query in _query_variants(task, augment_per_task):
            X_list.append(_encode_task_text(emb, query))
            y_list.append(residual)

    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32)


def train_models(
    data_path: str = "mechanical_data.txt",
    model_dir: str | Path | None = None,
    n_ensemble: int = 3,
):
    """
    训练 n_ensemble 个多输出 MLP 回归器（相对 task_heuristic 的四参数残差回归）。
    `data_path` 仅保留接口兼容性，本实现不再以 RAG 银标作为训练来源。
    """
    from sklearn.neural_network import MLPRegressor

    del data_path
    model_dir = Path(model_dir or DEFAULT_MODEL_DIR)
    model_dir.mkdir(parents=True, exist_ok=True)

    X, y = _get_training_data()
    if X.size == 0:
        raise RuntimeError("无训练数据")

    models = []
    for i in range(n_ensemble):
        reg = MLPRegressor(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            max_iter=700,
            random_state=42 + i,
            early_stopping=True,
            validation_fraction=0.2,
        )
        reg.fit(X, y)
        models.append(reg)

    import joblib

    for i, reg in enumerate(models):
        joblib.dump(reg, model_dir / f"learned_plan_{i}.joblib")
    meta = {
        "embedding_model": EMBEDDING_MODEL_NAME,
        "n_ensemble": n_ensemble,
        "n_samples": int(len(y)),
        "model_variant": MODEL_VARIANT,
        "label_source": "environment_profile_teacher",
        "base_policy": "task_heuristic",
        "plan_keys": list(PLAN_KEYS),
        "plan_ranges": {
            "gripper_force": [FORCE_MIN, FORCE_MAX],
            "approach_height": [HEIGHT_MIN, HEIGHT_MAX],
            "transport_velocity": [VELOCITY_MIN, VELOCITY_MAX],
            "lift_clearance": [CLEARANCE_MIN, CLEARANCE_MAX],
        },
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
    if meta.get("model_variant") != MODEL_VARIANT:
        return None, None
    n = meta["n_ensemble"]
    models = [joblib.load(model_dir / f"learned_plan_{i}.joblib") for i in range(n)]
    return models, meta


def predict_plan_with_uncertainty(
    task_text: str,
    models: list,
    embedding_model: Any,
) -> tuple[dict[str, float], dict[str, float]]:
    """
    用 ensemble 预测相对 task_heuristic 的残差，再恢复四参数计划。
    """
    vec = _encode_task_text(embedding_model, task_text).reshape(1, -1)
    preds = np.array([np.asarray(model.predict(vec)[0], dtype=np.float32) for model in models], dtype=np.float32)
    mean_residual = preds.mean(axis=0)
    std_plan = preds.std(axis=0) if len(preds) > 1 else np.zeros_like(mean_residual)
    heuristic = baseline.get_params_task_heuristic(task_text)
    base_vec = np.array(
        [
            heuristic["gripper_force"],
            heuristic["approach_height"],
            heuristic["transport_velocity"],
            heuristic["lift_clearance"],
        ],
        dtype=np.float32,
    )
    mean_plan = base_vec + mean_residual
    clipped = _clip_plan(*mean_plan.tolist())
    mean_dict = {
        "gripper_force": round(clipped[0], 2),
        "approach_height": round(clipped[1], 3),
        "transport_velocity": round(clipped[2], 3),
        "lift_clearance": round(clipped[3], 3),
    }
    std_dict = {
        "gripper_force": round(float(std_plan[0]), 3),
        "approach_height": round(float(std_plan[1]), 4),
        "transport_velocity": round(float(std_plan[2]), 4),
        "lift_clearance": round(float(std_plan[3]), 4),
    }
    return mean_dict, std_dict


class LearnedParamController:
    """
    轻量学习型控制器：
    文本描述 -> embedding -> MLP ensemble -> 四参数控制计划 + 不确定性。
    标签来自环境 teacher，因此不再是 RAG 的蒸馏副本。
    """

    def __init__(self, data_path: str = "mechanical_data.txt", model_dir: str | Path | None = None):
        from langchain_community.embeddings import HuggingFaceEmbeddings

        del data_path
        embedding_model_path = resolve_model_path(EMBEDDING_MODEL_NAME)
        self.embedding = HuggingFaceEmbeddings(model_name=embedding_model_path)
        self.model_dir = Path(model_dir or DEFAULT_MODEL_DIR)
        self.models, self.meta = load_models(self.model_dir)
        if self.models is None:
            train_models(model_dir=self.model_dir)
            self.models, self.meta = load_models(self.model_dir)
        if self.models is None:
            raise RuntimeError("learned baseline 训练或加载失败")

    def get_params_for_task(self, task_description: str) -> dict[str, Any]:
        """基于任务描述直接预测四参数计划，并返回参数级不确定性。"""
        mean_plan, std_plan = predict_plan_with_uncertainty(task_description, self.models, self.embedding)
        return {
            **mean_plan,
            "rag_source": task_description[:200],
            "uncertainty_std": round(float(np.mean(list(std_plan.values()))), 4),
            "uncertainty_by_param": std_plan,
            "label_source": "environment_profile_teacher",
            "base_policy": "task_heuristic",
        }
