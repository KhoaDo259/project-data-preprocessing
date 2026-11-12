"""
har_preprocess.py - Pipeline tiền xử lý IMU + Barometer (Tuần 8, HK261-DAGD1-407)

Thứ tự khuyến nghị:
  load -> to_physical_units -> align_axes -> lowpass (tần số gốc)
       -> resample_imu / resample_baro (về TARGET_FS)
       -> baro_channels -> make_windows -> fit_scaler (CHỈ trên train fold) -> transform
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal
from sklearn.preprocessing import MinMaxScaler, StandardScaler

G = 9.80665          # m/s^2
TARGET_FS = 50.0     # Hz, tần số chung cho IMU + barometer


# ----------------------------------------------------------------------------
# 0. Cấu hình theo dataset (đổi dataset = đổi config, không sửa hàm)
# ----------------------------------------------------------------------------
@dataclass
class DatasetConfig:
    name: str
    fs_imu: float
    fs_baro: float | None
    acc_to_ms2: float            # hệ số: giá trị thô -> m/s^2
    gyro_to_rads: float          # hệ số: giá trị thô -> rad/s
    baro_to_hpa: float | None    # hệ số: giá trị thô -> hPa
    axis_map: np.ndarray = field(default_factory=lambda: np.eye(3))  # R: trục dataset -> trục chuẩn


# !! Các hệ số dưới đây lấy từ datasheet/paper theo trí nhớ - XÁC NHẬN lại với README
# của từng dataset trước khi chạy thật (sai hệ số = sai toàn bộ kết quả).
CONFIGS = {
    # LSM6DS3 +-8g: 0.244 mg/LSB; gyro +-2000 dps: 70 mdps/LSB; LPS25HB: 4096 LSB/hPa
    "fallalld": DatasetConfig("fallalld", 238.0, 10.0,
                              0.244e-3 * G, 0.070 * np.pi / 180, 1 / 4096),
    # KFall: acc đã ở đơn vị g, gyro deg/s, 100 Hz
    "kfall": DatasetConfig("kfall", 100.0, None, G, np.pi / 180, None),
    # UCI HAR: total_acc ở g, gyro rad/s, 50 Hz
    "ucihar": DatasetConfig("ucihar", 50.0, None, G, 1.0, None),
    # SisFall: ADXL345 (32g/2^13 g/LSB), ITG3200 (4000/2^16 dps/LSB), 200 Hz
    "sisfall": DatasetConfig("sisfall", 200.0, None, 32 / 2**13 * G,
                             4000 / 2**16 * np.pi / 180, None),
}


# ----------------------------------------------------------------------------
# 1. Đọc dữ liệu + EDA
# ----------------------------------------------------------------------------
# GIẢ ĐỊNH: mỗi bản ghi là 1 file CSV có các cột bên dưới và tên file dạng
# S{subject}_D{device}_A{activity}_T{trial}.csv. Sửa regex/cột cho khớp bản tải về.
FILE_RE = re.compile(r"S(\d+)_D(\d+)_A(\d+)_T(\d+)", re.I)
COLS = dict(acc=["Acc_x", "Acc_y", "Acc_z"],
            gyro=["Gyr_x", "Gyr_y", "Gyr_z"],
            baro=["Bar"])  # nếu baro nằm file/cột khác tần số -> viết lại loader này


def build_index(root: str, waist_device: int | None = None) -> pd.DataFrame:
    """Quét thư mục, trả DataFrame (path, subject, device, activity, trial, is_fall)."""
    rows = []
    for p in Path(root).rglob("*.csv"):
        m = FILE_RE.search(p.stem)
        if m:
            s, d, a, t = map(int, m.groups())
            rows.append(dict(path=str(p), subject=s, device=d, activity=a, trial=t,
                             is_fall=a >= 100))  # FallAllD: ADL 1-44, ngã 101-135
    df = pd.DataFrame(rows)
    if waist_device is not None:  # chỉ giữ thiết bị ở eo (xem Mục 4.3 báo cáo Tuần 7)
        df = df[df.device == waist_device]
    return df.reset_index(drop=True)


def load_record(path: str) -> dict:
    """Đọc 1 bản ghi -> dict(acc (N,3), gyro (N,3), baro (M,)) ở đơn vị thô."""
    df = pd.read_csv(path)
    baro = df[COLS["baro"][0]].dropna().to_numpy(float)  # M << N vì 10 Hz
    return dict(acc=df[COLS["acc"]].to_numpy(float),
                gyro=df[COLS["gyro"]].to_numpy(float), baro=baro)


def plot_label_distribution(df: pd.DataFrame, by: str = "activity"):
    fig, ax = plt.subplots(1, 2, figsize=(13, 4))
    df[by].value_counts().sort_index().plot.bar(ax=ax[0], title=f"Số bản ghi theo {by}")
    df.groupby("subject").size().plot.bar(ax=ax[1], title="Số bản ghi theo người")
    plt.tight_layout()
    return fig


def plot_fall_vs_adl(fall: dict, adl: dict, cfg: DatasetConfig):
    """So sánh tín hiệu thô (đã đổi đơn vị) của 1 té ngã và 1 ADL: acc, gyro, baro."""
    fig, axes = plt.subplots(3, 2, figsize=(13, 8), sharex="col")
    for j, (name, rec) in enumerate([("Té ngã", fall), ("ADL", adl)]):
        t_imu = np.arange(len(rec["acc"])) / cfg.fs_imu
        t_bar = np.arange(len(rec["baro"])) / cfg.fs_baro
        acc, gyro, baro = to_physical_units(rec, cfg)
        axes[0, j].plot(t_imu, acc);  axes[0, j].set_ylabel("Acc (m/s²)")
        axes[1, j].plot(t_imu, gyro); axes[1, j].set_ylabel("Gyro (rad/s)")
        axes[2, j].plot(t_bar, baro, ".-"); axes[2, j].set_ylabel("Áp suất (hPa)")
        axes[0, j].set_title(name); axes[2, j].set_xlabel("Thời gian (s)")
    plt.tight_layout()
    return fig


# ----------------------------------------------------------------------------
# 2. Đổi đơn vị + căn trục
# ----------------------------------------------------------------------------
def to_physical_units(rec: dict, cfg: DatasetConfig):
    acc = rec["acc"] * cfg.acc_to_ms2
    gyro = rec["gyro"] * cfg.gyro_to_rads
    baro = rec["baro"] * cfg.baro_to_hpa if cfg.baro_to_hpa else None
    return acc, gyro, baro


def align_axes(acc: np.ndarray, gyro: np.ndarray, R: np.ndarray):
    """Áp ma trận xoay/đổi dấu R (3x3) lên cả acc và gyro (cùng khung tham chiếu)."""
    return acc @ R.T, gyro @ R.T


def gravity_rotation(acc_static: np.ndarray, target=(0.0, 0.0, 1.0)) -> np.ndarray:
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


# ----------------------------------------------------------------------------
# 3. Lọc thông thấp Butterworth
# ----------------------------------------------------------------------------
def butter_lowpass_sos(cutoff: float, fs: float, order: int = 4):
    assert cutoff < fs / 2, "cutoff phải < Nyquist"
    return signal.butter(order, cutoff, btype="low", fs=fs, output="sos")


def apply_lowpass(x: np.ndarray, fs: float, cutoff: float = 20.0, order: int = 4,
                  zero_phase: bool = False) -> np.ndarray:
    """
    cutoff 20 Hz: giữ động học người (đi/chạy < 10 Hz, xung va chạm khi ngã đến ~15-20 Hz).
    zero_phase=False (sosfilt, causal) để khớp với bộ lọc chạy thời gian thực trên ESP32;
    True (sosfiltfilt) chỉ dùng khi phân tích offline.
    """
    sos = butter_lowpass_sos(cutoff, fs, order)
    f = signal.sosfiltfilt if zero_phase else signal.sosfilt
    return f(sos, x, axis=0)


# ----------------------------------------------------------------------------
# 4. Resampling về tần số chung
# ----------------------------------------------------------------------------
def resample_imu(x: np.ndarray, fs_in: float, fs_out: float = TARGET_FS) -> np.ndarray:
    """Polyphase resample (có anti-alias FIR sẵn). 238 -> 50 Hz = up 25 / down 119."""
    frac = Fraction(fs_out / fs_in).limit_denominator(1000)
    return signal.resample_poly(x, frac.numerator, frac.denominator, axis=0)


def resample_baro(p: np.ndarray, fs_in: float, n_out: int, fs_out: float = TARGET_FS):
    """Nội suy tuyến tính 10 -> 50 Hz theo timestamp (không dùng spline/FFT: tránh ringing)."""
    t_in = np.arange(len(p)) / fs_in
    t_out = np.arange(n_out) / fs_out
    return np.interp(t_out, t_in, p)  # ngoài khoảng -> giữ giá trị biên


def sync_imu_baro(acc, gyro, baro, cfg: DatasetConfig, fs_out: float = TARGET_FS):
    """Trả (imu (N,6), baro (N,)) cùng độ dài, cùng trục thời gian. Giả định cùng t=0."""
    imu = np.hstack([acc, gyro])
    imu_r = resample_imu(imu, cfg.fs_imu, fs_out)
    if baro is None:
        return imu_r, None
    return imu_r, resample_baro(baro, cfg.fs_baro, len(imu_r), fs_out)


# ----------------------------------------------------------------------------
# 5. Đặc trưng áp suất
# ----------------------------------------------------------------------------
def pressure_to_height(p_hpa: np.ndarray, p0: float | None = None) -> np.ndarray:
    """Độ cao tương đối (m) theo công thức khí áp chuẩn; p0 mặc định = trung vị 1 s đầu."""
    p0 = np.median(p_hpa[:int(TARGET_FS)]) if p0 is None else p0
    return 44330.0 * (1.0 - (p_hpa / p0) ** (1 / 5.255))


def baro_channels(p_hpa: np.ndarray, fs: float = TARGET_FS) -> np.ndarray:
    """
    Trả (N,2): [dp_hpa, dpdt_hpa_per_s].
      - làm mượt: median (khử spike) -> Savitzky-Golay
      - dpdt: đạo hàm Savitzky-Golay (đã tích hợp làm mượt, ổn hơn np.diff)
    Té ngã ~0.5-1 m => ~0.06-0.12 hPa trong <1 s, sát mức nhiễu của LPS25HB:
    nên kiểm tra bằng ablation (IMU vs IMU+baro) chứ đừng mặc định là có ích.
    """
    p = signal.medfilt(p_hpa, kernel_size=5)
    w = int(0.5 * fs) | 1                              # cửa sổ lẻ ~0.5 s
    p_s = signal.savgol_filter(p, w, polyorder=2)
    dpdt = signal.savgol_filter(p, w, polyorder=2, deriv=1, delta=1 / fs)
    return np.stack([p_s, dpdt], axis=1)


# ----------------------------------------------------------------------------
# 6. Cửa sổ + chuẩn hóa
# ----------------------------------------------------------------------------
def make_windows(x: np.ndarray, fs: float = TARGET_FS, win_s: float = 2.0,
                 overlap: float = 0.5, relative_cols: tuple[int, ...] = ()) -> np.ndarray:
    """
    Cắt (N,C) -> (n_win, win, C). relative_cols: cột trừ baseline đầu cửa sổ
    (dùng cho áp suất tuyệt đối để bỏ offset theo thời tiết/thiết bị).
    """
    win, step = int(win_s * fs), int(win_s * fs * (1 - overlap))
    idx = range(0, len(x) - win + 1, step)
    out = np.stack([x[i:i + win].copy() for i in idx]) if len(x) >= win else np.empty((0, win, x.shape[1]))
    for c in relative_cols:
        out[:, :, c] -= np.median(out[:, :int(0.2 * fs), c], axis=1, keepdims=True)
    return out


def fit_scaler(X_train: np.ndarray, method: str = "zscore"):
    """Fit theo từng kênh trên TẬP TRAIN của fold (tránh rò rỉ). X: (n, win, C)."""
    sc = StandardScaler() if method == "zscore" else MinMaxScaler()
    sc.fit(X_train.reshape(-1, X_train.shape[-1]))
    return sc


def transform_windows(X: np.ndarray, scaler) -> np.ndarray:
    n, w, c = X.shape
    return scaler.transform(X.reshape(-1, c)).reshape(n, w, c).astype(np.float32)


def save_scaler(scaler, path: str):
    joblib.dump(scaler, path)
    # mean/scale để nhúng vào firmware (z-score: (x-mean)/scale)
    if isinstance(scaler, StandardScaler):
        print("mean :", np.array2string(scaler.mean_, separator=", "))
        print("scale:", np.array2string(scaler.scale_, separator=", "))


# ----------------------------------------------------------------------------
# Pipeline 1 bản ghi (gọi hàng loạt cho mọi dataset qua cfg)
# ----------------------------------------------------------------------------
def process_record(rec: dict, cfg: DatasetConfig, use_baro: bool = True,
                   cutoff: float = 20.0, zero_phase: bool = False) -> np.ndarray:
    acc, gyro, baro = to_physical_units(rec, cfg)
    acc, gyro = align_axes(acc, gyro, cfg.axis_map)
    acc = apply_lowpass(acc, cfg.fs_imu, cutoff, zero_phase=zero_phase)
    gyro = apply_lowpass(gyro, cfg.fs_imu, cutoff, zero_phase=zero_phase)
    imu, p = sync_imu_baro(acc, gyro, baro if use_baro else None, cfg)
    if use_baro and p is not None:
        imu = np.hstack([imu, baro_channels(p)])
    return imu  # (N, 6) hoặc (N, 8) ở 50 Hz


if __name__ == "__main__":
    # smoke test với dữ liệu giả
    cfg = CONFIGS["fallalld"]
    n = int(10 * cfg.fs_imu)
    rec = dict(acc=np.random.randn(n, 3) * 100, gyro=np.random.randn(n, 3) * 100,
               baro=101325 * 4096 / 100 + np.random.randn(100) * 5)
    x = process_record(rec, cfg)
    W = make_windows(x, relative_cols=(6,))
    sc = fit_scaler(W)
    print(x.shape, W.shape, transform_windows(W, sc).mean().round(3))