"""
har/labels.py - Ánh xạ activity_id -> target_class (5 lớp HAR).

Báo lỗi rõ ràng nếu còn target_class: null trong file YAML (Quy tắc 3).
Không tự gộp lớp.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import yaml


CLASS_NAMES = {0: "Standing", 1: "Sitting", 2: "Walking", 3: "Running", 4: "Fall"}


def load_label_map(yaml_path: str | Path) -> dict[int, Optional[int]]:
    """
    Tải labels_fallalld.yaml, trả {activity_id: target_class | None}.
    Các entry có target_class: null được giữ nguyên là None (bỏ qua khi build windows).
    """
    with open(yaml_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    activity_map = data.get("activity_map", {})
    return {int(aid): entry.get("target_class") for aid, entry in activity_map.items()}


def get_target_class(
    activity_id: int,
    label_map: dict[int, Optional[int]],
    strict: bool = True,
) -> Optional[int]:
    """
    Trả target_class cho activity_id.

    strict=True (mặc định): raise ValueError nếu target_class là null/None
    (phải điền YAML trước khi chạy build_windows).
    strict=False: trả None (bỏ qua bản ghi, dùng khi lọc sơ bộ).
    """
    if activity_id not in label_map:
        raise KeyError(
            f"activity_id={activity_id} không có trong label map. "
            f"Thêm vào configs/labels_fallalld.yaml."
        )
    cls = label_map[activity_id]
    if cls is None and strict:
        raise ValueError(
            f"activity_id={activity_id} có target_class: null trong YAML.\n"
            f"Hãy điền target_class (0-4) hoặc giữ null để bỏ qua bản ghi này.\n"
            f"Các lớp: {CLASS_NAMES}"
        )
    return cls


def validate_label_map(label_map: dict[int, Optional[int]]) -> list[int]:
    """
    Kiểm tra toàn bộ map, trả danh sách activity_id còn null.
    Dùng để in cảnh báo trước khi chạy build_windows.
    """
    nulls = [aid for aid, cls in label_map.items() if cls is None]
    return nulls
