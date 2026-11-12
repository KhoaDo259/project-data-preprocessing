"""
scripts/01_eda.py - Phân tích dữ liệu khám phá (EDA).

Chức năng:
  1. build_index -> lưu outputs/figures/label_dist.png
  2. Vẽ 1 bản ghi ngã vs 1 ADL -> outputs/figures/fall_vs_adl.png

Chạy:
  python scripts/01_eda.py --dataset fallalld
  python scripts/01_eda.py --dataset fallalld --fall-id 101 --adl-id 1 --no-show
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Thêm project root vào sys.path để import har.*
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import matplotlib

from har.config import load_config, resolve_path, CONFIGS
from har.io import get_loader

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="EDA: phân phối nhãn và so sánh té ngã vs ADL")
    p.add_argument("--dataset",   default="fallalld", choices=list(CONFIGS.keys()),
                   help="Tên dataset (mặc định: fallalld)")
    p.add_argument("--config",    default=None, help="Đường dẫn configs/default.yaml")
    p.add_argument("--fall-id",   type=int, default=None,
                   help="activity_id của bản ghi ngã cần vẽ (mặc định: lấy bản ghi ngã đầu tiên)")
    p.add_argument("--adl-id",    type=int, default=None,
                   help="activity_id của bản ghi ADL cần vẽ (mặc định: lấy bản ghi ADL đầu tiên)")
    p.add_argument("--no-show",   action="store_true",
                   help="Không hiện cửa sổ GUI, chỉ lưu file (dùng backend Agg)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.no_show:
        matplotlib.use("Agg")

    import matplotlib.pyplot as plt
    from har.eda import plot_label_distribution, plot_fall_vs_adl

    cfg_dict = load_config(args.config)
    cfg      = CONFIGS[args.dataset]
    loader   = get_loader(args.dataset)

    # Đường dẫn thư mục raw
    dataset_cfg = cfg_dict.get("datasets", {}).get(args.dataset, {})
    raw_subdir  = dataset_cfg.get("raw_subdir", args.dataset)
    raw_root    = resolve_path(cfg_dict, "data_raw") / raw_subdir
    fig_dir     = resolve_path(cfg_dict, "outputs_figures")
    fig_dir.mkdir(parents=True, exist_ok=True)

    waist_device = dataset_cfg.get("waist_device")

    print(f"[EDA] Dataset: {args.dataset} | raw: {raw_root}")

    # 1. Build index
    print("[EDA] Đang quét thư mục...")
    df = loader["build_index"](raw_root, waist_device=waist_device)
    print(f"[EDA] Tổng bản ghi: {len(df)} | Falls: {df['is_fall'].sum()} | ADLs: {(~df['is_fall']).sum()}")

    # 2. Phân phối nhãn
    fig1 = plot_label_distribution(df, by="activity")
    out1 = fig_dir / "label_dist.png"
    fig1.savefig(out1, dpi=150, bbox_inches="tight")
    print(f"[EDA] Lưu: {out1}")

    # 3. So sánh 1 bản ghi ngã vs 1 ADL
    falls = df[df["is_fall"]]
    adls  = df[~df["is_fall"]]

    if falls.empty or adls.empty:
        print("[EDA] Không đủ dữ liệu để vẽ fall vs ADL. Bỏ qua.")
        if not args.no_show:
            plt.show()
        return

    # Chọn bản ghi theo --fall-id / --adl-id hoặc lấy đầu tiên
    path_col = loader["path_col"]

    if args.fall_id is not None:
        row_fall = falls[falls["activity"] == args.fall_id]
        if row_fall.empty:
            print(f"[EDA] Không tìm thấy fall activity_id={args.fall_id}. Dùng bản ghi đầu tiên.")
            row_fall = falls.iloc[[0]]
    else:
        row_fall = falls.iloc[[0]]

    if args.adl_id is not None:
        row_adl = adls[adls["activity"] == args.adl_id]
        if row_adl.empty:
            print(f"[EDA] Không tìm thấy ADL activity_id={args.adl_id}. Dùng bản ghi đầu tiên.")
            row_adl = adls.iloc[[0]]
    else:
        row_adl = adls.iloc[[0]]

    print(f"[EDA] Vẽ fall: {row_fall.iloc[0][path_col]}")
    print(f"[EDA] Vẽ ADL:  {row_adl.iloc[0][path_col]}")

    rec_fall = loader["load_record"](row_fall.iloc[0][path_col])
    rec_adl  = loader["load_record"](row_adl.iloc[0][path_col])

    fig2 = plot_fall_vs_adl(rec_fall, rec_adl, cfg)
    out2 = fig_dir / "fall_vs_adl.png"
    fig2.savefig(out2, dpi=150, bbox_inches="tight")
    print(f"[EDA] Lưu: {out2}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
