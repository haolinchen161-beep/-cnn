"""统一方向参数化，消除模态 FRF 全链路中的硬编码 Z 向。

提供方向字符串（"X", "Y", "Z"）与数组索引（0, 1, 2）之间的规范映射，
替代全链路中的 ``[..., 2]`` 硬编码。

铣削应用默认配置：
    RESPONSE_DIRECTION = "Y"
    FORCE_DIRECTION = "Y"

对应 H_YY FRF（Y 向激励，Y 向响应）。
"""
from __future__ import annotations

from typing import List

# 方向标签 → 笛卡尔轴索引 的规范映射
DIRECTION_TO_INDEX = {"X": 0, "Y": 1, "Z": 2}
INDEX_TO_DIRECTION = {0: "X", 1: "Y", 2: "Z"}

# 默认训练 / 数据生成方向（薄板面外颤振主方向）
DEFAULT_RESPONSE_DIRECTION = "Z"
DEFAULT_FORCE_DIRECTION = "Z"


def direction_to_index(direction: str) -> int:
    """将方向字符串转换为笛卡尔轴索引。

    Args:
        direction: ``"X"``、``"Y"`` 或 ``"Z"``（区分大小写）。

    Returns:
        0 表示 X，1 表示 Y，2 表示 Z。

    Raises:
        ValueError: 方向字符串不合法时抛出。
    """
    direction = normalize_direction(direction)
    if direction not in DIRECTION_TO_INDEX:
        raise ValueError(
            f"未知方向 '{direction}'，期望 X、Y、Z 之一。"
        )
    return DIRECTION_TO_INDEX[direction]


def normalize_direction(direction: str) -> str:
    """将方向字符串规范化为大写单字符。

    接受常见变体：
        ``"x"`` → ``"X"``
        ``"  Y "`` → ``"Y"``
        ``"UX"`` → ``"X"``（去掉 ANSYS 位移自由度前缀 U）
    """
    s = direction.strip().upper()
    # 去掉 ANSYS 可能附加的前缀 U / R / D（UX → X）
    while s and s[0] in ("U", "R", "D"):
        s = s[1:]
    # 处理多字符情况
    if len(s) > 1:
        if s in DIRECTION_TO_INDEX:
            return s
        # 从末尾向前寻找第一个看起来像轴标签的字符
        for ch in reversed(s):
            if ch in DIRECTION_TO_INDEX:
                return ch
    return s


def direction_to_onehot(direction: str) -> List[float]:
    """返回给定方向的 one-hot 3 维向量。

    Example:
        direction_to_onehot("Y") → [0.0, 1.0, 0.0]
    """
    idx = direction_to_index(direction)
    vec = [0.0, 0.0, 0.0]
    vec[idx] = 1.0
    return vec


def direction_to_frf_label(response_dir: str, force_dir: str) -> str:
    """构建人类可读的 FRF 标签，如 ``"H_YY"``。

    Args:
        response_dir: 响应（测量）方向。
        force_dir: 力激励方向。

    Returns:
        格式为 ``"H_ab"`` 的字符串，其中第一个下标是响应方向，
        第二个下标是激励方向。
    """
    a = normalize_direction(response_dir)
    b = normalize_direction(force_dir)
    return f"H_{a}{b}"
