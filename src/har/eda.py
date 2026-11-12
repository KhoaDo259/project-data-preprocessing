"""
har/eda.py - Phân tích dữ liệu khám phá (EDA): vẽ phân phối nhãn và so sánh té ngã vs ADL.

Giữ nguyên logic từ har_preprocess.py: plot_label_distribution, plot_fall_vs_adl.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from har.config import DatasetConfig
from har.units import to_physical_units


def plot_label_distribution(df: pd.DataFrame, by: str = "activity") -> plt.Figure:
    """
    Vẽ 2 biểu đồ cột: số bản ghi theo activity và theo subject.
    df: DataFrame từ build_index (cần cột 'activity' và 'subject').
    """
    fig, ax = plt.subplots(1, 2, figsize=(13, 4))
    df[by].value_counts().sort_index().plot.bar(ax=ax[0], title=f"Số bản ghi theo {by}")
    df.groupby("subject").size().plot.bar(ax=ax[1], title="Số bản ghi theo người")
    plt.tight_layout()
    return fig


def plot_fall_vs_adl(fall: dict, adl: dict, cfg: DatasetConfig) -> plt.Figure:
    """
    So sánh tín hiệu thô (đã đổi đơn vị) của 1 bản ghi té ngã và 1 ADL.
    Vẽ 3 hàng: acc (m/s²), gyro (rad/s), baro (hPa).
    """
    fig, axes = plt.subplots(3, 2, figsize=(13, 8), sharex="col")
    for j, (name, rec) in enumerate([("Té ngã", fall), ("ADL", adl)]):
        t_imu = np.arange(len(rec["acc"])) / cfg.fs_imu
        t_bar = np.arange(len(rec["baro"])) / cfg.fs_baro if cfg.fs_baro else None
        acc, gyro, baro = to_physical_units(rec, cfg)
        axes[0, j].plot(t_imu, acc)
        axes[0, j].set_ylabel("Acc (m/s²)")
        axes[1, j].plot(t_imu, gyro)
        axes[1, j].set_ylabel("Gyro (rad/s)")
        if baro is not None and t_bar is not None:
            axes[2, j].plot(t_bar, baro, ".-")
            axes[2, j].set_ylabel("Áp suất (hPa)")
        else:
            axes[2, j].text(0.5, 0.5, "Không có baro", ha="center", transform=axes[2, j].transAxes)
        axes[0, j].set_title(name)
        axes[2, j].set_xlabel("Thời gian (s)")
    plt.tight_layout()
    return fig
