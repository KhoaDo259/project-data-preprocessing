"""
har/resampling.py - Resampling IMU và barometer về tần số chung.

Giữ nguyên logic từ har_preprocess.py:
  resample_imu (polyphase, 238->50 Hz), resample_baro (nội suy tuyến tính),
  sync_imu_baro.
"""
from __future__ import annotations

from fractions import Fraction

import numpy as np
from scipy import signal

from har.config import DatasetConfig, TARGET_FS


def resample_imu(x: np.ndarray, fs_in: float, fs_out: float = TARGET_FS) -> np.ndarray:
    """
    Polyphase resample (có anti-alias FIR sẵn). 238 -> 50 Hz = up 25 / down 119.
    Dùng Fraction để tránh sai số float khi tính tỉ lệ up/down.
    """
    frac = Fraction(fs_out / fs_in).limit_denominator(1000)
    return signal.resample_poly(x, frac.numerator, frac.denominator, axis=0)


def resample_baro(p: np.ndarray, fs_in: float, n_out: int, fs_out: float = TARGET_FS) -> np.ndarray:
    """
    Nội suy tuyến tính 10 -> 50 Hz theo timestamp (không dùng spline/FFT: tránh ringing).
    n_out: số mẫu IMU sau resample (để đồng bộ trục thời gian).
    """
    t_in  = np.arange(len(p)) / fs_in
    t_out = np.arange(n_out) / fs_out
    return np.interp(t_out, t_in, p)   # ngoài khoảng -> giữ giá trị biên


def sync_imu_baro(
    acc: np.ndarray,
    gyro: np.ndarray,
    baro: np.ndarray | None,
    cfg: DatasetConfig,
    fs_out: float = TARGET_FS,
) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Trả (imu (N,6), baro_resampled (N,) hoặc None) cùng độ dài, cùng trục thời gian.
    Giả định IMU và baro cùng t=0 (đã xác nhận với FallAllD).
    """
    imu   = np.hstack([acc, gyro])
    imu_r = resample_imu(imu, cfg.fs_imu, fs_out)
    if baro is None:
        return imu_r, None
    return imu_r, resample_baro(baro, cfg.fs_baro, len(imu_r), fs_out)
