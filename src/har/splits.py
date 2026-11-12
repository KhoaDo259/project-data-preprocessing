"""
har/splits.py - Tạo subject-wise cross-validation folds.

Dùng GroupKFold hoặc StratifiedGroupKFold để đảm bảo không có người trùng
giữa train/test trong cùng fold.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Iterator

import numpy as np
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold


def make_folds(
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int = 5,
    stratified: bool = True,
    random_seed: int = 42,
) -> list[dict]:
    """
    Tạo n_splits folds subject-wise.

    Trả list of dict: [{'train': [...indices...], 'test': [...indices...]}].
    stratified=True: dùng StratifiedGroupKFold (cân bằng lớp giữa các fold).
    stratified=False: dùng GroupKFold (đơn giản hơn, ít ràng buộc hơn).
    """
    if stratified:
        kf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
        splits = kf.split(np.zeros(len(y)), y, groups)
    else:
        kf = GroupKFold(n_splits=n_splits)
        splits = kf.split(np.zeros(len(y)), y, groups)

    folds = []
    for train_idx, test_idx in splits:
        folds.append({
            "train": train_idx.tolist(),
            "test":  test_idx.tolist(),
        })
    return folds


def print_fold_stats(
    folds: list[dict],
    y: np.ndarray,
    groups: np.ndarray,
    class_names: dict[int, str] | None = None,
) -> None:
    """In số người và số mẫu mỗi lớp trong train/test của từng fold. Cảnh báo fold thiếu lớp."""
    classes = sorted(np.unique(y))
    for k, fold in enumerate(folds):
        tr, te = np.array(fold["train"]), np.array(fold["test"])
        n_subjects_tr = len(np.unique(groups[tr]))
        n_subjects_te = len(np.unique(groups[te]))
        print(f"\nFold {k+1}/{len(folds)}:")
        print(f"  Train: {len(tr)} mẫu, {n_subjects_tr} người | Test: {len(te)} mẫu, {n_subjects_te} người")
        for c in classes:
            cname = class_names.get(c, str(c)) if class_names else str(c)
            n_tr = int((y[tr] == c).sum())
            n_te = int((y[te] == c).sum())
            print(f"    Lớp {c} ({cname}): train={n_tr}, test={n_te}")
            if n_te == 0:
                warnings.warn(f"Fold {k+1}: Lớp {c} ({cname}) không có mẫu trong test set!", stacklevel=2)
            if n_tr == 0:
                warnings.warn(f"Fold {k+1}: Lớp {c} ({cname}) không có mẫu trong train set!", stacklevel=2)


def save_folds(folds: list[dict], path: str | Path) -> None:
    """Lưu chỉ số fold vào JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(folds, f, indent=2)
    print(f"[folds] Lưu: {path}")


def load_folds(path: str | Path) -> list[dict]:
    """Tải folds.json, trả list of dict."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)
