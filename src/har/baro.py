"""
har/baro.py - Xử lý tín hiệu barometer thành đặc trưng áp suất.

Giữ nguyên logic từ har_preprocess.py:
  pressure_to_height, baro_channels (median -> SavGol -> dpdt).
"""
from __future__ import annotations

import numpy as np
from scipy import signal

from har.config import TARGET_FS


def pressure_to_height(p_hpa: np.ndarray, p0: float | None = None) -> np.ndarray:
    """
    Độ cao tương đối (m) theo công thức khí áp chuẩn.
    p0 mặc định = trung vị 1 giây đầu (loại trừ spike khởi động).
    """
    p0 = np.median(p_hpa[:int(TARGET_FS)]) if p0 is None else p0
    return 44330.0 * (1.0 - (p_hpa / p0) ** (1 / 5.255))


def baro_channels(p_hpa: np.ndarray, fs: float = TARGET_FS) -> np.ndarray:
    """
    Trả (N,2): [dp_smooth (hPa), dpdt (hPa/s)].
      - Làm mượt: median kernel=5 (khử spike) -> Savitzky-Golay window ~0.5s, poly=2
      - dpdt: đạo hàm Savitzky-Golay (đã tích hợp làm mượt, ổn hơn np.diff)

    Lưu ý: Té ngã ~0.5-1 m => ~0.06-0.12 hPa trong <1 s, sát mức nhiễu của LPS25HB.
    Nên kiểm tra bằng ablation (IMU vs IMU+baro) chứ đừng mặc định là có ích.
    """
    p   = signal.medfilt(p_hpa, kernel_size=5)
    w   = int(0.5 * fs) | 1                              # cửa sổ lẻ ~0.5 s
    p_s  = signal.savgol_filter(p, w, polyorder=2)
    dpdt = signal.savgol_filter(p, w, polyorder=2, deriv=1, delta=1 / fs)
    return np.stack([p_s, dpdt], axis=1)
