"""与进程无关的稳定种子工具。"""

from __future__ import annotations

import zlib


def stable_seed_offset(text: str, modulo: int = 10000) -> int:
    """基于文本生成稳定偏移量，避免 Python 内置 hash 的跨进程漂移。"""
    return zlib.crc32(text.encode("utf-8")) % modulo
