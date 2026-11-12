"""
tests/test_smoke.py - Smoke tests với dữ liệu giả.

Kiểm tra:
  1. Shape sau resample 238->50 Hz
  2. Độ dài IMU và baro khớp nhau sau sync
  3. Scaler fit trên train có mean≈0, std≈1
  4. Không có người trùng giữa train/test trong cùng fold
"""
from __future__ import annotations

import numpy as np
import pytest

from har.config import CONFIGS, TARGET_FS
from har.filtering import apply_lowpass
from har.resampling import resample_imu, resample_baro, sync_imu_baro
from har.windowing import make_windows
from har.normalization import fit_scaler, transform_windows
from har.splits import make_folds


# -------------------------------------------------------------------------
# Fixtures
# -------------------------------------------------------------------------

@pytest.fixture
def fallalld_cfg():
    return CONFIGS["fallalld"]


@pytest.fixture
def fake_record(fallalld_cfg):
    """Bản ghi IMU + baro giả, 10 giây ở tần số FallAllD (238 Hz, 10 Hz)."""
    cfg = fallalld_cfg
    n_imu  = int(10 * cfg.fs_imu)   # 2380 mẫu
    n_baro = int(10 * cfg.fs_baro)  # 100 mẫu
    rng    = np.random.default_rng(0)
    return dict(
        acc  = rng.standard_normal((n_imu, 3)) * 100,   # thô (LSB)
        gyro = rng.standard_normal((n_imu, 3)) * 100,
        baro = 101325.0 + rng.standard_normal(n_baro),  # giả dữ liệu hPa thực tế
    )


# -------------------------------------------------------------------------
# Test 1: Shape sau resample
# -------------------------------------------------------------------------

def test_resample_imu_shape(fallalld_cfg):
    """238 -> 50 Hz: số mẫu ra phải xấp xỉ n_in * (50/238)."""
    cfg   = fallalld_cfg
    fs_in = cfg.fs_imu
    n_in  = int(5 * fs_in)   # 5 giây
    x     = np.random.randn(n_in, 6)
    x_out = resample_imu(x, fs_in, TARGET_FS)
    expected = round(n_in * TARGET_FS / fs_in)
    # cho phép lệch tối đa 2 mẫu do làm tròn phân số
    assert abs(len(x_out) - expected) <= 2, (
        f"resample_imu shape sai: {len(x_out)} vs expected~{expected}"
    )
    assert x_out.shape[1] == 6


def test_resample_baro_shape(fallalld_cfg):
    """10 -> 50 Hz: baro sau resample phải cùng độ dài với IMU resampled."""
    cfg    = fallalld_cfg
    n_imu  = int(5 * cfg.fs_imu)
    n_baro = int(5 * cfg.fs_baro)
    imu    = np.random.randn(n_imu, 6)
    baro   = np.random.randn(n_baro)
    imu_r  = resample_imu(imu, cfg.fs_imu, TARGET_FS)
    baro_r = resample_baro(baro, cfg.fs_baro, len(imu_r), TARGET_FS)
    assert len(imu_r) == len(baro_r), (
        f"IMU ({len(imu_r)}) và baro ({len(baro_r)}) khác độ dài sau resample"
    )


# -------------------------------------------------------------------------
# Test 2: IMU và baro cùng độ dài sau sync
# -------------------------------------------------------------------------

def test_sync_imu_baro_aligned(fake_record, fallalld_cfg):
    """sync_imu_baro phải trả IMU và baro cùng độ dài."""
    cfg  = fallalld_cfg
    rec  = fake_record
    # baro đã ở hPa (hệ số 1.0), nhân acc/gyro với hệ số để có đơn vị vật lý
    acc  = rec["acc"]  * cfg.acc_to_ms2
    gyro = rec["gyro"] * cfg.gyro_to_rads
    baro = rec["baro"] * cfg.baro_to_hpa   # baro_to_hpa = 1.0 cho fallalld
    imu_r, baro_r = sync_imu_baro(acc, gyro, baro, cfg, fs_out=TARGET_FS)
    assert len(imu_r) == len(baro_r), (
        f"Sau sync: IMU {len(imu_r)} mẫu, baro {len(baro_r)} mẫu - không khớp"
    )
    assert imu_r.shape[1] == 6


# -------------------------------------------------------------------------
# Test 3: Scaler trên train có mean≈0, std≈1
# -------------------------------------------------------------------------

def test_scaler_fit_zscore():
    """Sau fit_scaler zscore và transform_windows, train data phải có mean≈0, std≈1."""
    rng = np.random.default_rng(42)
    # Tạo windows giả với offset lớn để kiểm tra scaler thực sự chuẩn hóa
    X_train = rng.standard_normal((200, 100, 8)) * 5 + 10
    sc      = fit_scaler(X_train, method="zscore")
    X_norm  = transform_windows(X_train, sc)
    flat    = X_norm.reshape(-1, 8)
    np.testing.assert_allclose(flat.mean(axis=0), 0.0, atol=1e-5,
                                err_msg="mean sau z-score phải ≈ 0")
    np.testing.assert_allclose(flat.std(axis=0),  1.0, atol=1e-5,
                                err_msg="std sau z-score phải ≈ 1")


def test_scaler_no_leakage():
    """Scaler fit trên train không được dùng thống kê test -> test mean ≠ 0 nếu khác phân phối."""
    rng = np.random.default_rng(7)
    X_train = rng.standard_normal((200, 100, 6))
    X_test  = rng.standard_normal((50, 100, 6)) + 5.0   # offset khác hẳn
    sc      = fit_scaler(X_train, method="zscore")
    X_te_n  = transform_windows(X_test, sc)
    # test set mean sau transform KHÔNG gần 0 (vì offset khác)
    assert abs(X_te_n.reshape(-1, 6).mean()) > 1.0, (
        "Test set mean quá gần 0 - có thể scaler đã rò rỉ dữ liệu test"
    )


# -------------------------------------------------------------------------
# Test 4: Không có người trùng giữa train/test
# -------------------------------------------------------------------------

def test_no_subject_overlap():
    """Không có subject nào xuất hiện đồng thời trong train và test của cùng 1 fold."""
    rng      = np.random.default_rng(0)
    n        = 300
    # 10 người, mỗi người ~30 mẫu
    subjects = np.repeat(np.arange(10), 30)
    y        = rng.integers(0, 5, size=n)
    folds    = make_folds(y, subjects, n_splits=5, stratified=True, random_seed=42)

    for k, fold in enumerate(folds):
        tr_subj = set(subjects[fold["train"]])
        te_subj = set(subjects[fold["test"]])
        overlap = tr_subj & te_subj
        assert len(overlap) == 0, (
            f"Fold {k+1}: subject {overlap} xuất hiện trong cả train lẫn test!"
        )


# -------------------------------------------------------------------------
# Test 5: make_windows shape
# -------------------------------------------------------------------------

def test_make_windows_shape():
    """make_windows trả đúng shape (n_win, win_len, C)."""
    fs    = TARGET_FS
    win_s = 2.0
    ovl   = 0.5
    n     = int(10 * fs)   # 10 giây
    C     = 8
    x     = np.random.randn(n, C)
    W     = make_windows(x, fs=fs, win_s=win_s, overlap=ovl)
    win_len = int(win_s * fs)   # 100
    # số window mong đợi
    step = int(win_s * fs * (1 - ovl))
    n_expected = len(range(0, n - win_len + 1, step))
    assert W.shape == (n_expected, win_len, C), (
        f"make_windows shape sai: {W.shape} vs ({n_expected}, {win_len}, {C})"
    )
