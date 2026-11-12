"""
har/pipeline.py - Pipeline xử lý 1 bản ghi từ đầu đến cuối.

Giữ nguyên logic từ har_preprocess.py: process_record.
Gọi theo thứ tự: to_physical_units -> align_axes -> lowpass -> resample -> baro_channels.
"""
from __future__ import annotations

import numpy as np

from har.config import DatasetConfig
from har.units import to_physical_units, align_axes
from har.filtering import apply_lowpass
from har.resampling import sync_imu_baro
from har.baro import baro_channels


def process_record(
    rec: dict,
    cfg: DatasetConfig,
    use_baro: bool = True,
    cutoff: float = 20.0,
    zero_phase: bool = False,
) -> np.ndarray:
    """
    Pipeline đầy đủ cho 1 bản ghi IMU (+ baro).

    Thứ tự:
      load -> to_physical_units -> align_axes -> lowpass (tần số gốc)
           -> resample_imu / resample_baro (về TARGET_FS)
           -> baro_channels -> stack

    Trả: (N, 6) nếu use_baro=False hoặc không có baro,
         (N, 8) nếu use_baro=True và có baro [acc_x,y,z, gyro_x,y,z, dp, dpdt].
    Tần số: TARGET_FS (50 Hz).
    """
    acc, gyro, baro = to_physical_units(rec, cfg)
    acc, gyro = align_axes(acc, gyro, cfg.axis_map)
    acc  = apply_lowpass(acc,  cfg.fs_imu, cutoff, zero_phase=zero_phase)
    gyro = apply_lowpass(gyro, cfg.fs_imu, cutoff, zero_phase=zero_phase)
    imu, p = sync_imu_baro(acc, gyro, baro if use_baro else None, cfg)
    if use_baro and p is not None:
        imu = np.hstack([imu, baro_channels(p)])
    return imu   # (N, 6) hoặc (N, 8) @ 50 Hz
