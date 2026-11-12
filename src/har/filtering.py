"""
har/filtering.py - Lọc thông thấp Butterworth.

Giữ nguyên logic từ har_preprocess.py:
  butter_lowpass_sos, apply_lowpass.
Mặc định: bậc 4, cutoff 20 Hz, causal (zero_phase=False) khớp firmware ESP32.
"""
from __future__ import annotations

import numpy as np
from scipy import signal


def butter_lowpass_sos(cutoff: float, fs: float, order: int = 4):
    """Tạo SOS coefficients cho bộ lọc Butterworth thông thấp."""
    assert cutoff < fs / 2, f"cutoff {cutoff} Hz phải < Nyquist {fs/2} Hz"
    return signal.butter(order, cutoff, btype="low", fs=fs, output="sos")


def apply_lowpass(
    x: np.ndarray,
    fs: float,
    cutoff: float = 20.0,
    order: int = 4,
    zero_phase: bool = False,
) -> np.ndarray:
    """
    Lọc thông thấp Butterworth.

    cutoff 20 Hz: giữ động học người (đi/chạy < 10 Hz, xung va chạm khi ngã đến ~15-20 Hz).
    zero_phase=False (sosfilt, causal) để khớp với bộ lọc chạy thời gian thực trên ESP32;
    True (sosfiltfilt) chỉ dùng khi phân tích offline.
    """
    sos = butter_lowpass_sos(cutoff, fs, order)
    f = signal.sosfiltfilt if zero_phase else signal.sosfilt
    return f(sos, x, axis=0)
