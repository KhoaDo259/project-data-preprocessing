"""
har/normalization.py - Fit và transform scaler cho từng fold.

Giữ nguyên logic từ har_preprocess.py:
  fit_scaler, transform_windows, save_scaler.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def fit_scaler(X_train: np.ndarray, method: str = "zscore"):
    """
    Fit scaler theo từng kênh trên TẬP TRAIN của fold (tránh rò rỉ dữ liệu).
    X_train: (n_samples, win_len, n_channels).
    method: 'zscore' (StandardScaler) | 'minmax' (MinMaxScaler).
    """
    sc = StandardScaler() if method == "zscore" else MinMaxScaler()
    sc.fit(X_train.reshape(-1, X_train.shape[-1]))
    return sc


def transform_windows(X: np.ndarray, scaler) -> np.ndarray:
    """Áp scaler đã fit lên tập X (n, win, C), trả float32."""
    n, w, c = X.shape
    return scaler.transform(X.reshape(-1, c)).reshape(n, w, c).astype(np.float32)


def save_scaler(scaler, path: str | Path, verbose: bool = True) -> None:
    """
    Lưu scaler bằng joblib và in mean/scale để nhúng vào firmware.
    z-score firmware: output = (input - mean) / scale.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, path)
    if verbose and isinstance(scaler, StandardScaler):
        print(f"[scaler] Lưu: {path}")
        print("  mean :", np.array2string(scaler.mean_,  precision=6, separator=", "))
        print("  scale:", np.array2string(scaler.scale_, precision=6, separator=", "))
