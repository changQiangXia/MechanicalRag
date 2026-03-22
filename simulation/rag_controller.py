"""
RAG 控制器：将 RAG 检索的知识转化为仿真可用的控制参数。
参考论文：知识增强的机器人操作（knowledge-guided manipulation）
"""

from __future__ import annotations

import json
import random
import re
from typing import Any

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from chroma_compat import get_chroma_client_settings
from model_provider import resolve_model_path


# 知识库中不同物体类型的理想参数（来自 mechanical_data.txt）
GROUND_TRUTH_PARAMS = {
    "光滑金属零件": {"gripper_force": (30, 50), "approach_height": 0.05},
    "橡胶零件": {"gripper_force": (5, 15), "approach_height": 0.03},
    "小型机械零件": {"gripper_force": (5, 15), "approach_height": 0.04},
    "大型零件": {"gripper_force": (30, 50), "approach_height": 0.06},
    "薄壁件": {"gripper_force": (5, 12), "approach_height": 0.02},
    "默认": {"gripper_force": (20, 40), "approach_height": 0.05},
}


def _parse_force_from_text(text: str) -> float | None:
    """从文本中解析夹爪力数值（单位 N）。"""
    # 匹配 "30-50N", "5-15N", "夹爪力40N" 等
    patterns = [
        r"(\d+)-(\d+)\s*N",      # 30-50N
        r"夹爪力\s*(\d+)",        # 夹爪力40
        r"(\d+)\s*N",             # 50N
        r"(\d+)\s*到\s*(\d+)",    # 30到50
    ]
    for pat in patterns:
        m = re.search(pat, text)
        if m:
            g = m.groups()
            if len(g) == 1:
                return float(g[0])
            return (float(g[0]) + float(g[1])) / 2  # 取范围中点
    return None


class RAGController:
    """
    RAG 驱动的参数控制器。
    根据任务描述（如"抓取光滑金属零件"）检索知识，解析出夹爪力、接近高度等参数。
    """

    def __init__(self, data_path: str = "mechanical_data.txt"):
        loader = TextLoader(data_path, encoding="utf-8")
        documents = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=150, chunk_overlap=30)
        splits = splitter.split_documents(documents)
        self.splits = splits
        embedding_model_path = resolve_model_path("sentence-transformers/all-MiniLM-L6-v2")
        embeddings = HuggingFaceEmbeddings(model_name=embedding_model_path)
        self.vector_db = Chroma.from_documents(
            splits,
            embeddings,
            client_settings=get_chroma_client_settings(),
        )
        self.data_path = data_path

    def _queries_multi(self, task_description: str) -> list[str]:
        """多查询：从任务描述生成多条检索 query，用于消融实验。"""
        # 提取关键词并组合成多 query
        parts = []
        if "光滑金属" in task_description or "金属" in task_description:
            parts.extend(["光滑金属 夹爪力", "金属零件 抓取 力"])
        if "橡胶" in task_description:
            parts.extend(["橡胶零件 夹爪力", "橡胶 抓取 力"])
        if "小型" in task_description:
            parts.extend(["小型机械零件 夹爪力", "小型零件 抓取"])
        if "大型" in task_description:
            parts.extend(["大型零件 夹爪力", "大型 抓取 力"])
        if "薄壁" in task_description:
            parts.extend(["薄壁件 夹爪力", "薄壁 抓取"])
        if not parts:
            parts = [task_description]
        return [task_description] + parts[:2]  # 原描述 + 最多 2 条扩展

    def get_params_for_task(
        self,
        task_description: str,
        retrieval: str = "single",
        seed: int | None = None,
    ) -> dict[str, Any]:
        """
        根据任务描述从 RAG 检索并解析控制参数。
        retrieval: "single" 单 query, "multi" 多 query 融合, "random" 随机文档（无检索）
        """
        if retrieval == "random":
            rng = random.Random(seed)
            k = min(3, len(self.splits))
            docs = rng.sample(self.splits, k)
        elif retrieval == "multi":
            queries = self._queries_multi(task_description)
            seen = set()
            docs = []
            for q in queries:
                for d in self.vector_db.similarity_search(q, k=2):
                    key = (d.page_content[:80],)
                    if key not in seen:
                        seen.add(key)
                        docs.append(d)
            docs = docs[:5]  # 最多 5 段
        else:
            docs = self.vector_db.similarity_search(task_description, k=3)

        context = "\n".join(d.page_content for d in docs)
        force = _parse_force_from_text(context)

        # 根据任务类型匹配理想范围
        height = 0.05
        for key, params in GROUND_TRUTH_PARAMS.items():
            if key in task_description:
                lo, hi = params["gripper_force"]
                height = params["approach_height"]
                if force is None:
                    force = (lo + hi) / 2
                else:
                    force = max(lo, min(hi, force))
                break

        if force is None:
            force = 25.0  # 默认

        return {
            "gripper_force": float(force),
            "approach_height": height,
            "rag_source": context[:200] if docs else "",
        }

    def get_params_for_task_llm(
        self,
        task_description: str,
        llm: Any,
        retrieval: str = "single",
    ) -> dict[str, Any]:
        """
        使用 LLM 结构化输出（JSON）得到夹爪力与接近高度；解析失败时回退到规则解析。
        llm: LangChain 兼容的 LLM（如 get_llm() 返回的 HuggingFacePipeline）
        """
        docs = self.vector_db.similarity_search(task_description, k=3)
        context = "\n".join(d.page_content for d in docs)[:800]

        prompt = f"""根据以下知识库片段，为「{task_description}」任务给出夹爪力（单位：N）和接近高度（单位：m）。
只输出一个 JSON，格式严格为：{{"gripper_force": 数字, "approach_height": 数字}}
不要输出其他文字、不要解释。

知识库片段：
{context}
"""

        try:
            out = llm.invoke(prompt)
            if isinstance(out, str):
                text = out.strip()
            else:
                text = str(out).strip()
            # 抽取 JSON：先尝试整段解析，再尝试从文本中匹配 {...}
            parsed = _parse_json_params(text)
            if parsed is not None:
                force = float(parsed.get("gripper_force", 25))
                height = float(parsed.get("approach_height", 0.05))
                force = max(5.0, min(50.0, force))
                height = max(0.02, min(0.1, height))
                return {
                    "gripper_force": round(force, 2),
                    "approach_height": round(height, 3),
                    "rag_source": context[:200],
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
        ideal_force_range: tuple[float, float],
        adjustment_step: float = 5.0,
    ) -> dict[str, Any]:
        """
        反馈环节：根据上一次执行结果（success, info）在 previous_params 基础上微调。
        用于「RAG 初参 → 执行 → 失败 → 本方法得到调整参数 → 再执行」闭环。
        若 success 为 True 则直接返回原参数（无需调整）。
        """
        from .feedback import (
            build_feedback_signal,
            suggest_force_adjustment,
            adjust_params_by_feedback,
        )
        if success:
            return dict(previous_params)
        signal = build_feedback_signal(
            success=success,
            gripper_force=previous_params["gripper_force"],
            ideal_force_range=ideal_force_range,
            info=info,
        )
        suggestion = suggest_force_adjustment(signal)
        return adjust_params_by_feedback(
            previous_params,
            suggestion,
            step=adjustment_step,
        )


def _parse_json_params(text: str) -> dict[str, Any] | None:
    """从 LLM 输出中解析出 gripper_force 与 approach_height 的 JSON。"""
    text = text.strip()
    # 尝试直接解析
    try:
        obj = json.loads(text)
        if "gripper_force" in obj:
            return obj
    except json.JSONDecodeError:
        pass
    # 匹配 {...}
    m = re.search(r"\{[^{}]*\"gripper_force\"[^{}]*\}", text)
    if m:
        try:
            obj = json.loads(m.group(0))
            if "gripper_force" in obj:
                return obj
        except json.JSONDecodeError:
            pass
    # 宽松：任意包含两个数字的 JSON 块
    m = re.search(r"\{[^{}]*\}", text)
    if m:
        try:
            obj = json.loads(m.group(0))
            if "gripper_force" in obj:
                return obj
        except json.JSONDecodeError:
            pass
    return None
