"""
har/config.py - Cấu hình dataset và tải YAML.

Chứa DatasetConfig, CONFIGS (giữ nguyên hằng số từ har_preprocess.py)
và hàm load_config() để đọc configs/default.yaml.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

G = 9.80665          # m/s²
TARGET_FS = 50.0     # Hz, tần số chung cho IMU + barometer


@dataclass
class DatasetConfig:
    """Thông số vật lý và cấu hình từng dataset."""
    name: str
    fs_imu: float
    fs_baro: float | None
    acc_to_ms2: float            # hệ số: giá trị thô -> m/s²
    gyro_to_rads: float          # hệ số: giá trị thô -> rad/s
    baro_to_hpa: float | None    # hệ số: giá trị thô -> hPa (None nếu đã là hPa)
    axis_map: np.ndarray = field(default_factory=lambda: np.eye(3))  # R: trục dataset -> trục chuẩn


# !! Các hệ số dưới đây lấy từ datasheet/paper theo trí nhớ - XÁC NHẬN lại với README
# của từng dataset trước khi chạy thật (sai hệ số = sai toàn bộ kết quả).
CONFIGS: dict[str, DatasetConfig] = {
    # LSM6DS3 +-8g: 0.244 mg/LSB; gyro +-2000 dps: 70 mdps/LSB; LPS25HB: 4096 LSB/hPa
    # FallAllD: file _B.dat đã ở hPa (cột 1) => baro_to_hpa = 1.0 (không cần đổi)
    "fallalld": DatasetConfig(
        "fallalld", 238.0, 10.0,
        0.244e-3 * G,          # acc_to_ms2: XÁC NHẬN hệ số với README FallAllD
        0.070 * np.pi / 180,   # gyro_to_rads: XÁC NHẬN hệ số với README FallAllD
        1.0,                   # baro_to_hpa: file _B.dat cột 0 đã là hPa
    ),
    # KFall: acc đã ở đơn vị g, gyro deg/s, 100 Hz
    "kfall": DatasetConfig("kfall", 100.0, None, G, np.pi / 180, None),
    # UCI HAR: total_acc ở g, gyro rad/s, 50 Hz
    "ucihar": DatasetConfig("ucihar", 50.0, None, G, 1.0, None),
    # SisFall: ADXL345 (32g/2^13 g/LSB), ITG3200 (4000/2^16 dps/LSB), 200 Hz
    "sisfall": DatasetConfig(
        "sisfall", 200.0, None,
        32 / 2**13 * G,              # acc_to_ms2: XÁC NHẬN hệ số với README SisFall
        4000 / 2**16 * np.pi / 180,  # gyro_to_rads: XÁC NHẬN hệ số với README SisFall
        None,
    ),
}


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Tải configs/default.yaml; trả dict. Dùng path gốc project làm cơ sở."""
    if config_path is None:
        # Tìm file config relative to src/har/config.py -> lên 3 cấp -> project root
        config_path = Path(__file__).resolve().parents[2] / "configs" / "default.yaml"
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Không tìm thấy config: {config_path}")
    with config_path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(cfg: dict[str, Any], key: str, dataset: str = "") -> Path:
    """Lấy đường dẫn từ config và resolve theo project root."""
    root = Path(__file__).resolve().parents[2]
    paths = cfg.get("paths", {})
    base = root / paths.get(key, key)
    return base / dataset if dataset else base
