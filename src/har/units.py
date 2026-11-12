"""
har/units.py - Đổi đơn vị vật lý và căn chỉnh trục cảm biến.

Giữ nguyên logic từ har_preprocess.py:
  to_physical_units, align_axes, gravity_rotation.
"""
from __future__ import annotations

import numpy as np

from har.config import DatasetConfig


def to_physical_units(rec: dict, cfg: DatasetConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """
    Đổi acc (-> m/s²), gyro (-> rad/s), baro (-> hPa) theo hệ số trong DatasetConfig.
    Trả (acc, gyro, baro); baro là None nếu cfg.baro_to_hpa là None.
    """
    acc  = rec["acc"]  * cfg.acc_to_ms2
    gyro = rec["gyro"] * cfg.gyro_to_rads
    baro = rec["baro"] * cfg.baro_to_hpa if cfg.baro_to_hpa is not None else None
    return acc, gyro, baro


def align_axes(acc: np.ndarray, gyro: np.ndarray, R: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Áp ma trận xoay/đổi dấu R (3×3) lên cả acc và gyro (cùng khung tham chiếu)."""
    return acc @ R.T, gyro @ R.T


def gravity_rotation(acc_static: np.ndarray, target: tuple[float, float, float] = (0.0, 0.0, 1.0)) -> np.ndarray:
    """
    Tính R xoay vector trọng lực trung bình (lúc đứng yên) về trục `target`.
    Dùng khi không biết chắc hướng gắn cảm biến. Công thức Rodrigues.
    """
    g = acc_static.mean(axis=0)
    g = g / np.linalg.norm(g)
    t = np.asarray(target, float)
    v, c = np.cross(g, t), float(g @ t)
    if np.isclose(c, -1.0):                      # ngược 180 độ
        return np.diag([1.0, -1.0, -1.0])
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx / (1 + c)
