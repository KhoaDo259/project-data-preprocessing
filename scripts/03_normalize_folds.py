"""
scripts/03_normalize_folds.py - Tạo subject-wise folds và normalize.

Chức năng:
  - Tải windows.npz từ 02_build_windows.py
  - Tạo GroupKFold / StratifiedGroupKFold (subject-wise)
  - Với mỗi fold: fit_scaler CHỈ trên train, transform train+test
  - Lưu scaler + in mean/scale cho firmware
  - Lưu folds.json

Chạy:
  python scripts/03_normalize_folds.py --dataset fallalld
  python scripts/03_normalize_folds.py --dataset fallalld --channels imu --n-splits 5
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np

from har.config import load_config, resolve_path, CONFIGS
from har.labels import CLASS_NAMES
from har.normalization import fit_scaler, transform_windows, save_scaler
from har.splits import make_folds, print_fold_stats, save_folds


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Tạo folds và normalize windows")
    p.add_argument("--dataset",   default="fallalld", choices=list(CONFIGS.keys()))
    p.add_argument("--config",    default=None, help="Đường dẫn configs/default.yaml")
    p.add_argument("--n-splits",  type=int, default=None,
                   help="Số fold (ghi đè config; mặc định trong config: 5)")
    p.add_argument("--channels",  default="imu_baro", choices=["imu", "imu_baro"],
                   help="Chọn kênh: 'imu'=0:6, 'imu_baro'=0:8 (mặc định: imu_baro)")
    p.add_argument("--stratified", action="store_true", default=True,
                   help="Dùng StratifiedGroupKFold (mặc định: True)")
    p.add_argument("--no-stratified", dest="stratified", action="store_false",
                   help="Dùng GroupKFold thay vì StratifiedGroupKFold")
    p.add_argument("--method",    default=None, choices=["zscore", "minmax"],
                   help="Phương pháp chuẩn hóa (ghi đè config)")
    return p.parse_args()


def main() -> None:
    args     = parse_args()
    cfg_dict = load_config(args.config)
    norm_cfg = cfg_dict.get("normalization", {})

    n_splits    = args.n_splits or norm_cfg.get("n_splits", 5)
    random_seed = norm_cfg.get("random_seed", 42)
    method      = args.method or norm_cfg.get("method", "zscore")

    # Đường dẫn
    proc_dir   = resolve_path(cfg_dict, "data_processed") / args.dataset
    scaler_dir = resolve_path(cfg_dict, "outputs_scalers")
    scaler_dir.mkdir(parents=True, exist_ok=True)

    npz_path = proc_dir / "windows.npz"
    if not npz_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy {npz_path}. Chạy 02_build_windows.py trước."
        )

    print(f"[norm] Tải: {npz_path}")
    data    = np.load(npz_path)
    X_full  = data["X"]          # (n, win, C) float32
    y       = data["y"]          # (n,)
    subject = data["subject"]    # (n,)

    # Chọn kênh
    n_ch = X_full.shape[-1]
    if args.channels == "imu":
        ch_slice = slice(0, 6)
        ch_label = "imu (0:6)"
    else:
        if n_ch < 8:
            print(f"[norm] WARN: windows chỉ có {n_ch} kênh, không có baro. Dùng tất cả kênh.")
            ch_slice = slice(0, n_ch)
        else:
            ch_slice = slice(0, 8)
        ch_label = f"imu_baro (0:{ch_slice.stop})"

    X = X_full[:, :, ch_slice]
    print(f"[norm] X shape sau chọn kênh ({ch_label}): {X.shape}")

    # Tạo folds
    print(f"\n[norm] Tạo {n_splits} folds (stratified={args.stratified}, seed={random_seed})...")
    folds = make_folds(y, subject, n_splits=n_splits,
                       stratified=args.stratified, random_seed=random_seed)

    # In thống kê
    print_fold_stats(folds, y, subject, class_names=CLASS_NAMES)

    # Normalize từng fold
    print(f"\n[norm] Phương pháp chuẩn hóa: {method}")
    for k, fold in enumerate(folds):
        tr = np.array(fold["train"])
        te = np.array(fold["test"])

        X_tr = X[tr]
        X_te = X[te]

        # Fit CHỈ trên train (tránh rò rỉ)
        sc = fit_scaler(X_tr, method=method)

        # Transform
        X_tr_norm = transform_windows(X_tr, sc)
        X_te_norm = transform_windows(X_te, sc)

        # Lưu scaler
        scaler_path = scaler_dir / f"{args.dataset}_fold{k+1}.joblib"
        save_scaler(sc, scaler_path, verbose=True)

        # Kiểm tra nhanh (mean≈0, std≈1 trên train sau z-score)
        if method == "zscore":
            flat = X_tr_norm.reshape(-1, X_tr_norm.shape[-1])
            print(f"  [fold {k+1}] train mean≈{flat.mean(axis=0).round(3)}, std≈{flat.std(axis=0).round(3)}")

    # Lưu folds.json
    folds_path = proc_dir / "folds.json"
    save_folds(folds, folds_path)

    print(f"\n[norm] Hoàn tất. Folds: {folds_path}")
    print(f"[norm] Scalers: {scaler_dir}/{args.dataset}_fold*.joblib")


if __name__ == "__main__":
    main()
