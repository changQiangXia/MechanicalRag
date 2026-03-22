"""Chroma 兼容层：关闭产品遥测，避免 py38 下 posthog 导入链报错。"""

from __future__ import annotations

from chromadb.config import Settings
from chromadb.telemetry.product import ProductTelemetryClient, ProductTelemetryEvent
from overrides import override


class NoOpProductTelemetry(ProductTelemetryClient):
    """空实现：保留 Chroma 接口，但不发送任何遥测。"""

    @override
    def capture(self, event: ProductTelemetryEvent) -> None:
        return None


def get_chroma_client_settings(persist_directory: str | None = None) -> Settings:
    kwargs = {
        "anonymized_telemetry": False,
        "chroma_product_telemetry_impl": "chroma_compat.NoOpProductTelemetry",
        "chroma_telemetry_impl": "chroma_compat.NoOpProductTelemetry",
    }
    if persist_directory is not None:
        kwargs["is_persistent"] = True
        kwargs["persist_directory"] = persist_directory
    return Settings(**kwargs)
