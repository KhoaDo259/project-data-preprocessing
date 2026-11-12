# HAR Preprocessing Pipeline — TinyML ESP32-S3

Pipeline tiền xử lý dữ liệu IMU/barometer cho nhận dạng hoạt động người (HAR) với 5 lớp:
**Đứng (0) · Ngồi (1) · Đi bộ (2) · Chạy (3) · Té ngã (4)**

Dataset chính: **FallAllD** (IMU 238 Hz, barometer 10 Hz, thiết bị eo).
Dataset phụ: KFall, UCI HAR, SisFall (IMU only — stub, cần implement loader).

---

## Cài đặt

```bash
# 1. Tạo virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

# 2. Cài thư viện
pip install -r requirements.txt

# 3. Cài package ở chế độ editable (để import har.* hoạt động từ scripts/)
pip install -e .
```

---

## Thứ tự chạy

### Bước 0 — Chuẩn bị dữ liệu và nhãn

1. Đặt dữ liệu FallAllD dưới `data/raw/fallalld/`. Loader hỗ trợ thư mục
   phẳng hoặc thư mục con (ví dụ `data/raw/fallalld/FallAllD/`) với các file
   dạng `S01_D1_A013_T01_A.dat`, `_G.dat`, `_B.dat`, `_M.dat`.

2. Kiểm tra mapping đã xác nhận trong
   [`configs/labels_fallalld.yaml`](configs/labels_fallalld.yaml):
   ```yaml
   activity_map:
     13: {name: "Sitting", target_class: 1}
     101: {name: "Fall", target_class: 4}
     ...
   ```
   - Mapping tuần 8: Standing `14,41`; Sitting `13,42`; Walking
     `19–22,29–36`; Running `24,25,43,44`; Fall `101–135`.
   - `target_class: null` → bản ghi sẽ bị bỏ qua khi build windows.

### Bước 1 — EDA

```bash
# Vẽ phân phối nhãn + so sánh 1 bản ghi ngã vs ADL
python scripts/01_eda.py --dataset fallalld

# Không hiện GUI (CI / server)
python scripts/01_eda.py --dataset fallalld --no-show

# Chỉ định bản ghi cụ thể
python scripts/01_eda.py --dataset fallalld --fall-id 101 --adl-id 13
```

Kết quả: `outputs/figures/label_dist.png`, `outputs/figures/fall_vs_adl.png`

### Bước 2 — Build Windows

```bash
# FallAllD với baro (8 kênh), fall_mode=peak (mặc định)
python scripts/02_build_windows.py --dataset fallalld

# Không dùng baro (6 kênh IMU)
python scripts/02_build_windows.py --dataset fallalld --no-baro

# Lấy tất cả cửa sổ của bản ghi ngã (không chỉ đỉnh)
python scripts/02_build_windows.py --dataset fallalld --fall-mode all
```

Kết quả: `data/processed/fallalld/windows.npz` (X float32, y, subject, record_id),
`meta.csv`, `params.json`

Kết quả kiểm tra trên bộ dữ liệu hiện tại: `X=(18973, 100, 8)`, gồm 14 người.

### Bước 3 — Normalize & Folds

```bash
# 5 folds subject-wise, StandardScaler (z-score), 8 kênh imu+baro
python scripts/03_normalize_folds.py --dataset fallalld

# Chỉ 6 kênh IMU (dùng cùng windows.npz, không cần build lại)
python scripts/03_normalize_folds.py --dataset fallalld --channels imu

# Thay đổi số fold (chốt sau khi xem phân phối nhãn)
python scripts/03_normalize_folds.py --dataset fallalld --n-splits 10
```

Kết quả: `outputs/scalers/fallalld_fold{k}.joblib`, `data/processed/fallalld/folds.json`

### Chạy tests

```bash
pytest tests/ -v
```

---

## Thêm dataset mới

1. **Thêm config** vào [`src/har/config.py`](src/har/config.py) → dict `CONFIGS`:
   ```python
   "newdataset": DatasetConfig("newdataset", fs_imu=100.0, fs_baro=None,
                               acc_to_ms2=G, gyro_to_rads=np.pi/180, baro_to_hpa=None),
   ```

2. **Thêm loader** vào [`src/har/io.py`](src/har/io.py):
   ```python
   def build_index_newdataset(root, waist_device=None): ...
   def load_record_newdataset(path): ...
   ```

3. **Đăng ký** vào dict `LOADERS` ở cuối `io.py`:
   ```python
   "newdataset": {
       "build_index": build_index_newdataset,
       "load_record": load_record_newdataset,
       "path_col": "path",
   },
   ```

4. **Thêm cấu hình** vào `configs/default.yaml` → `datasets:`.

5. (Tùy chọn) **Tạo label map** `configs/labels_newdataset.yaml` và đăng ký trong `scripts/02_build_windows.py` → `LABEL_MAP_FILES`.

---

## Cấu trúc project

```
configs/
  default.yaml           # cấu hình đường dẫn, pipeline, fold
  labels_fallalld.yaml   # ánh xạ activity_id -> target_class
data/
  raw/fallalld/          # file .dat gốc (flat, không commit lên git)
  processed/fallalld/    # windows.npz, meta.csv, params.json, folds.json
outputs/
  figures/               # label_dist.png, fall_vs_adl.png
  scalers/               # fallalld_fold{k}.joblib
src/har/
  config.py    # DatasetConfig, CONFIGS, load_config
  io.py        # loader FallAllD (thực tế) + stub kfall/ucihar/sisfall
  units.py     # to_physical_units, align_axes, gravity_rotation
  filtering.py # butter_lowpass_sos, apply_lowpass
  resampling.py# resample_imu, resample_baro, sync_imu_baro
  baro.py      # pressure_to_height, baro_channels
  windowing.py # make_windows, find_peak_window
  normalization.py # fit_scaler, transform_windows, save_scaler
  labels.py    # load_label_map, get_target_class (strict null check)
  splits.py    # make_folds, print_fold_stats, save/load_folds
  eda.py       # plot_label_distribution, plot_fall_vs_adl
  pipeline.py  # process_record (pipeline đầy đủ 1 bản ghi)
scripts/
  01_eda.py
  02_build_windows.py
  03_normalize_folds.py
tests/
  test_smoke.py
```
