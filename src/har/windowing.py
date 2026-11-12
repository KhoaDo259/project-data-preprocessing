"""
har/windowing.py - Cắt cửa sổ trượt từ chuỗi thời gian.

Giữ nguyên logic từ har_preprocess.py: make_windows.
"""
from __future__ import annotations

import numpy as np

from har.config import TARGET_FS


def make_windows(
    x: np.ndarray,
    fs: float = TARGET_FS,
    win_s: float = 2.0,
    overlap: float = 0.5,
    relative_cols: tuple[int, ...] = (),
) -> np.ndarray:
    """
    Cắt (N,C) -> (n_win, win_len, C).

    relative_cols: chỉ số cột trừ trung vị 20% đầu cửa sổ làm baseline
    (dùng cho áp suất tuyệt đối để loại offset thời tiết/thiết bị).
    """
    win  = int(win_s * fs)
    step = int(win_s * fs * (1 - overlap))
    idx  = range(0, len(x) - win + 1, step)
    if len(x) < win:
        return np.empty((0, win, x.shape[1]))
    out = np.stack([x[i:i + win].copy() for i in idx])
    for c in relative_cols:
        out[:, :, c] -= np.median(out[:, :int(0.2 * fs), c], axis=1, keepdims=True)
    return out


def find_peak_window(windows: np.ndarray, acc_cols: tuple[int, ...] = (0, 1, 2)) -> int:
    """
    Tìm chỉ số cửa sổ có đỉnh |acc| lớn nhất (dùng cho fall_mode='peak').
    acc_cols: cột nào trong windows là acc x,y,z.
    """
    acc_mag = np.linalg.norm(windows[:, :, list(acc_cols)], axis=-1)  # (n_win, win_len)
    peak_per_win = acc_mag.max(axis=1)                                  # (n_win,)
    return int(np.argmax(peak_per_win))
