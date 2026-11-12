"""
scripts/02_build_windows.py - Xây dựng windows từ dữ liệu thô.

Chức năng:
  - Với mỗi bản ghi thiết bị eo: process_record -> make_windows
  - Fall mode: 'peak' (chỉ cửa sổ đỉnh |acc|) | 'all' (mọi cửa sổ)
  - ADL: lấy mọi cửa sổ
  - Lưu data/processed/{dataset}/windows.npz + meta.csv + params.json

Chạy:
  python scripts/02_build_windows.py --dataset fallalld
  python scripts/02_build_windows.py --dataset fallalld --fall-mode all --no-baro
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import pandas as pd
import scipy
import sklearn

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from har.config import load_config, resolve_path, CONFIGS, TARGET_FS
from har.io import get_loader
from har.labels import load_label_map, get_target_class, validate_label_map
from har.pipeline import process_record
from har.windowing import make_windows, find_peak_window


LABEL_MAP_FILES = {
    "fallalld": PROJECT_ROOT / "configs" / "labels_fallalld.yaml",
    # TODO: thêm label map cho kfall, ucihar, sisfall khi implement
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build windows từ dữ liệu IMU/baro thô")
    p.add_argument("--dataset",       default="fallalld", choices=list(CONFIGS.keys()))
    p.add_argument("--config",        default=None, help="Đường dẫn configs/default.yaml")
    p.add_argument("--fall-mode",     default=None, choices=["all", "peak"],
                   help="Chế độ lấy cửa sổ cho bản ghi ngã (ghi đè config)")
    p.add_argument("--no-baro",       action="store_true",
                   help="Không dùng barometer (X 6 kênh thay vì 8)")
    p.add_argument("--waist-device",  type=int, default=None,
                   help="Device ID ở eo (ghi đè config)")
    return p.parse_args()


def main() -> None:
    args   = parse_args()
    cfg_dict = load_config(args.config)
    cfg      = CONFIGS[args.dataset]

    dataset_cfg  = cfg_dict.get("datasets", {}).get(args.dataset, {})
    pipeline_cfg = cfg_dict.get("pipeline", {})
    wind_cfg     = cfg_dict.get("windowing", {})

    raw_subdir   = dataset_cfg.get("raw_subdir", args.dataset)
    raw_root     = resolve_path(cfg_dict, "data_raw") / raw_subdir
    out_dir      = resolve_path(cfg_dict, "data_processed") / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    waist_device = args.waist_device or dataset_cfg.get("waist_device")
    use_baro     = (not args.no_baro) and pipeline_cfg.get("baro_channels", True)
    fall_mode    = args.fall_mode or wind_cfg.get("fall_mode", "peak")
    win_s        = pipeline_cfg.get("win_s", 2.0)
    overlap      = pipeline_cfg.get("overlap", 0.5)
    cutoff       = pipeline_cfg.get("cutoff_hz", 20.0)
    zero_phase   = pipeline_cfg.get("zero_phase", False)
    rel_col      = pipeline_cfg.get("relative_baro_col", 6)
    relative_cols = (rel_col,) if use_baro else ()

    loader = get_loader(args.dataset)

    # Label map
    if args.dataset not in LABEL_MAP_FILES:
        raise FileNotFoundError(
            f"Chưa có label map cho dataset '{args.dataset}'. "
            "Không tự động gán nhãn; hãy thêm map trước khi build windows."
        )
    lmap_path = LABEL_MAP_FILES[args.dataset]
    if not lmap_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy label map: {lmap_path}. "
            "Hãy tạo và điền target_class trước khi build windows."
        )
    label_map = load_label_map(lmap_path)
    null_ids = validate_label_map(label_map)
    if null_ids:
        warnings.warn(
            f"Có {len(null_ids)} activity_id còn target_class=null: {null_ids}. "
            "Các bản ghi tương ứng sẽ bị bỏ qua; không tự gộp lớp.",
            stacklevel=2,
        )

    print(f"[build] Dataset: {args.dataset} | waist_device={waist_device} | use_baro={use_baro} | fall_mode={fall_mode}")

    # Build index
    df = loader["build_index"](raw_root, waist_device=waist_device)
    print(f"[build] Tổng bản ghi: {len(df)}")

    path_col = loader["path_col"]

    all_X, all_y, all_subj, all_rid = [], [], [], []

    for idx, row in df.iterrows():
        activity = row["activity"]
        subject  = row["subject"]
        is_fall  = row["is_fall"]
        path     = row[path_col]

        # Gán nhãn
        target_cls = get_target_class(activity, label_map, strict=False)
        if target_cls is None:
            continue

        # Xử lý bản ghi
        rec = loader["load_record"](path)
        data = process_record(
            rec, cfg, use_baro=use_baro, cutoff=cutoff, zero_phase=zero_phase
        )

        # Cắt cửa sổ
        windows = make_windows(data, fs=TARGET_FS, win_s=win_s, overlap=overlap,
                               relative_cols=relative_cols)
        if len(windows) == 0:
            continue

        # Fall mode
        if is_fall and fall_mode == "peak":
            peak_idx = find_peak_window(windows, acc_cols=(0, 1, 2))
            windows  = windows[[peak_idx]]

        n_win = len(windows)
        all_X.append(windows)
        all_y.extend([target_cls] * n_win)
        all_subj.extend([subject]  * n_win)
        all_rid.extend([idx]       * n_win)

        if (idx + 1) % 50 == 0:
            print(f"  [{idx+1}/{len(df)}] windows tích lũy: {sum(len(w) for w in all_X)}")

    if not all_X:
        print("[build] Không có cửa sổ nào được tạo. Kiểm tra label map và dữ liệu.")
        return

    X       = np.concatenate(all_X,  axis=0).astype(np.float32)
    y       = np.array(all_y,        dtype=np.int32)
    subject = np.array(all_subj,     dtype=np.int32)
    rec_id  = np.array(all_rid,      dtype=np.int32)

    print(f"\n[build] Kết quả: X={X.shape}, y={y.shape}, {len(np.unique(subject))} người")

    # Lưu windows.npz
    npz_path = out_dir / "windows.npz"
    np.savez_compressed(npz_path, X=X, y=y, subject=subject, record_id=rec_id)
    print(f"[build] Lưu: {npz_path}")

    # Lưu meta.csv
    meta = pd.DataFrame({"y": y, "subject": subject, "record_id": rec_id})
    meta_path = out_dir / "meta.csv"
    meta.to_csv(meta_path, index=False)
    print(f"[build] Lưu: {meta_path}")

    # Lưu params.json (để lặp lại thí nghiệm)
    params = {
        "dataset":       args.dataset,
        "use_baro":      use_baro,
        "fall_mode":     fall_mode,
        "waist_device":  waist_device,
        "target_fs":     TARGET_FS,
        "win_s":         win_s,
        "overlap":       overlap,
        "cutoff_hz":     cutoff,
        "filter_order":  4,
        "zero_phase":    zero_phase,
        "relative_cols": list(relative_cols),
        "n_channels":    X.shape[-1],
        "n_windows":     int(X.shape[0]),
        "n_subjects":    int(len(np.unique(subject))),
        "lib_versions": {
            "numpy":        np.__version__,
            "scipy":        scipy.__version__,
            "sklearn":      sklearn.__version__,
        },
        "random_seed":   cfg_dict.get("normalization", {}).get("random_seed", 42),
    }
    params_path = out_dir / "params.json"
    with open(params_path, "w", encoding="utf-8") as f:
        json.dump(params, f, indent=2, ensure_ascii=False)
    print(f"[build] Lưu: {params_path}")

    # Thống kê phân phối nhãn
    print("\n[build] Phân phối nhãn:")
    for c in sorted(np.unique(y)):
        print(f"  Lớp {c}: {(y==c).sum()} cửa sổ")


if __name__ == "__main__":
    main()
