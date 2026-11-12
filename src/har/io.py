"""
har/io.py - Loader dữ liệu cho từng dataset.

FallAllD (đã xác nhận cấu trúc thực tế):
  - Thư mục flat: data/raw/fallalld/
  - 4 file .dat per bản ghi: _A (Acc 3-cột int, 238Hz), _G (Gyro 3-cột int, 238Hz),
    _B (Baro+temp 2-cột float, 10Hz, cột 0 đã là hPa), _M (Mag 3-cột int, 238Hz)
  - Tên file: S{subject}_D{device}_A{activity}_T{trial}_{suffix}.dat
  - Regex: S(\\d+)_D(\\d+)_A(\\d+)_T(\\d+)_{suffix}

Các dataset khác (kfall, ucihar, sisfall): stub NotImplementedError - cần implement.
Thêm dataset mới: thêm hàm load_xxx() và đăng ký vào LOADERS dict ở cuối file.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

# Regex khớp tên file FallAllD thực tế: S01_D1_A013_T01_{suffix}.dat
FALLALLD_FILE_RE = re.compile(r"S(\d+)_D(\d+)_A(\d+)_T(\d+)_([A-Z])\.dat$", re.I)


# ----------------------------------------------------------------------------
# FallAllD loader
# ----------------------------------------------------------------------------

def build_index_fallalld(root: str | Path, waist_device: int | None = 1) -> pd.DataFrame:
    """
    Quét thư mục FallAllD flat, trả DataFrame:
    (path_stem, subject, device, activity, trial, is_fall).
    path_stem: đường dẫn không có _{suffix}.dat để ghép lại khi load.
    Chỉ giữ file _A.dat (anchor), các file còn lại suy ra từ stem.
    """
    rows = []
    root = Path(root)
    for p in sorted(root.rglob("*_A.dat")):
        m = FALLALLD_FILE_RE.match(p.name)
        if not m:
            continue
        s, d, a, t, _ = m.groups()
        s, d, a, t = int(s), int(d), int(a), int(t)
        if waist_device is not None and d != waist_device:
            continue
        stem = str(p.with_name(p.name[:-6]))  # bỏ "_A.dat" -> "S01_D1_A013_T01"
        rows.append(dict(
            path_stem=stem,
            subject=s,
            device=d,
            activity=a,
            trial=t,
            is_fall=(a >= 100),   # ADL: 1-44; Fall: 101-135 (đã xác nhận)
        ))
    return pd.DataFrame(rows).reset_index(drop=True)


def load_record_fallalld(path_stem: str | Path) -> dict:
    """
    Đọc 1 bản ghi FallAllD từ path_stem (không có _{suffix}.dat).
    Trả dict(acc (N,3) float, gyro (N,3) float, baro (M,) float).

    Cấu trúc file (đã xác nhận từ dữ liệu thực tế):
      _A.dat: 3 cột int, không có header -> acc thô (LSB), 238 Hz
      _G.dat: 3 cột int, không có header -> gyro thô (LSB), 238 Hz
      _B.dat: 2 cột float, không có header -> [hPa, °C], 10 Hz (đã là hPa, hệ số = 1.0)
      _M.dat: 3 cột int, không có header -> magnetometer (bỏ qua trong pipeline này)
    """
    stem = Path(path_stem)
    acc_path  = Path(str(stem) + "_A.dat")
    gyro_path = Path(str(stem) + "_G.dat")
    baro_path = Path(str(stem) + "_B.dat")

    for p in (acc_path, gyro_path, baro_path):
        if not p.exists():
            raise FileNotFoundError(f"Thiếu file FallAllD: {p}")

    acc  = np.loadtxt(acc_path,  delimiter=",", dtype=float)   # (N,3)
    gyro = np.loadtxt(gyro_path, delimiter=",", dtype=float)   # (N,3)
    baro_raw = np.loadtxt(baro_path, delimiter=",", dtype=float)  # (M,2)
    baro = baro_raw[:, 0]  # cột 0 = hPa; cột 1 = °C (bỏ qua nhiệt độ)

    return dict(acc=acc, gyro=gyro, baro=baro)


# ----------------------------------------------------------------------------
# KFall loader (stub - chưa implement)
# ----------------------------------------------------------------------------

def build_index_kfall(root: str | Path, waist_device: int | None = None) -> pd.DataFrame:
    """
    TODO: Implement loader cho KFall dataset.
    Cấu trúc thư mục và tên file: chưa xác nhận - cần xem README KFall.
    """
    raise NotImplementedError(
        "KFall loader chưa được implement.\n"
        "TODO: Xem cấu trúc thư mục data/raw/kfall và dán 5 dòng đầu file mẫu,\n"
        "      sau đó implement build_index_kfall() và load_record_kfall() trong io.py."
    )


def load_record_kfall(path: str | Path) -> dict:
    """TODO: Implement. Xem build_index_kfall."""
    raise NotImplementedError("KFall loader chưa được implement. Xem TODO trong build_index_kfall.")


# ----------------------------------------------------------------------------
# UCI HAR loader (stub - chưa implement)
# ----------------------------------------------------------------------------

def build_index_ucihar(root: str | Path, waist_device: int | None = None) -> pd.DataFrame:
    """
    TODO: Implement loader cho UCI HAR dataset.
    UCI HAR có cấu trúc đặc biệt (train/test split sẵn, file signals riêng).
    Cần xem README và cấu trúc thư mục data/raw/ucihar.
    """
    raise NotImplementedError(
        "UCI HAR loader chưa được implement.\n"
        "TODO: UCI HAR có thể đã tách train/test; cần quyết định có rebuild từ raw\n"
        "      hay dùng split sẵn. Xem data/raw/ucihar/README.txt."
    )


def load_record_ucihar(path: str | Path) -> dict:
    """TODO: Implement. Xem build_index_ucihar."""
    raise NotImplementedError("UCI HAR loader chưa được implement.")


# ----------------------------------------------------------------------------
# SisFall loader (stub - chưa implement)
# ----------------------------------------------------------------------------

def build_index_sisfall(root: str | Path, waist_device: int | None = None) -> pd.DataFrame:
    """
    TODO: Implement loader cho SisFall dataset.
    Cần xem cấu trúc thư mục data/raw/sisfall và tên file.
    """
    raise NotImplementedError(
        "SisFall loader chưa được implement.\n"
        "TODO: Xem cấu trúc data/raw/sisfall và dán 5 dòng đầu file mẫu."
    )


def load_record_sisfall(path: str | Path) -> dict:
    """TODO: Implement. Xem build_index_sisfall."""
    raise NotImplementedError("SisFall loader chưa được implement.")


# ----------------------------------------------------------------------------
# Registry - thêm dataset mới vào đây
# ----------------------------------------------------------------------------

LOADERS: dict[str, dict] = {
    "fallalld": {
        "build_index": build_index_fallalld,
        "load_record": load_record_fallalld,
        "path_col": "path_stem",   # tên cột chứa path trong index DataFrame
    },
    "kfall": {
        "build_index": build_index_kfall,
        "load_record": load_record_kfall,
        "path_col": "path",
    },
    "ucihar": {
        "build_index": build_index_ucihar,
        "load_record": load_record_ucihar,
        "path_col": "path",
    },
    "sisfall": {
        "build_index": build_index_sisfall,
        "load_record": load_record_sisfall,
        "path_col": "path",
    },
}


def get_loader(dataset: str) -> dict:
    """Lấy loader functions theo tên dataset. Raise ValueError nếu không tìm thấy."""
    if dataset not in LOADERS:
        raise ValueError(
            f"Dataset '{dataset}' không có trong registry. "
            f"Các dataset hỗ trợ: {list(LOADERS.keys())}"
        )
    return LOADERS[dataset]
