"""Sinh bộ chia tập chuẩn chống rò rỉ dữ liệu (Zero-Leakage Splits) cho Oddball P300.

Tệp: scripts/data/generate_splits.py

Tạo tệp `{data_dir}/splits.json` từ `{data_dir}/metadata.csv` với seed cố định (42),
bao gồm 3 giao thức đánh giá chuẩn:
1. `within_subject`: StratifiedKFold(5) theo từng subject + tách 20% train thành val_idx.
2. `cross_subject_5fold`: StratifiedGroupKFold(5) theo subject_id + tách ~20% train_subjects làm val_idx.
3. `cross_subject_loso`: LeaveOneGroupOut() theo subject_id + tách val_subjects rời nhau.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    LeaveOneGroupOut,
    StratifiedGroupKFold,
    StratifiedKFold,
    StratifiedShuffleSplit,
)


def natural_sort_key(s: str) -> List[int | str]:
    """Sắp xếp tự nhiên theo thứ tự số (vd: sub-02 trước sub-10)."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", str(s))]


def compute_trial_uid_sha256(trial_uids: List[str]) -> str:
    """Tính hash SHA-256 cho danh sách trial_uid để khóa toàn vẹn dữ liệu."""
    payload = "\n".join(trial_uids).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_within_subject_splits(
    subjects: np.ndarray,
    y: np.ndarray,
    unique_subjects: List[str],
    seed: int,
) -> Dict[str, List[Dict[str, Any]]]:
    """Tạo StratifiedKFold(5) theo từng subject kèm tách 20% Train làm Val."""
    within_splits: Dict[str, List[Dict[str, Any]]] = {}

    for sub_id in unique_subjects:
        sub_indices = np.where(subjects == sub_id)[0]
        sub_y = y[sub_indices]

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        sub_folds: List[Dict[str, Any]] = []

        for fold_idx, (tr_val_pos, te_pos) in enumerate(skf.split(sub_indices, sub_y)):
            train_val_idx = sub_indices[tr_val_pos]
            test_idx = sub_indices[te_pos]
            train_val_y = y[train_val_idx]

            sss = StratifiedShuffleSplit(
                n_splits=1, test_size=0.2, random_state=seed + fold_idx
            )
            tr_pos, val_pos = next(sss.split(train_val_idx, train_val_y))
            train_idx = np.sort(train_val_idx[tr_pos])
            val_idx = np.sort(train_val_idx[val_pos])

            sub_folds.append(
                {
                    "fold": int(fold_idx),
                    "train_idx": train_idx.astype(int).tolist(),
                    "val_idx": val_idx.astype(int).tolist(),
                    "test_idx": np.sort(test_idx).astype(int).tolist(),
                }
            )

        within_splits[sub_id] = sub_folds

    return within_splits


def build_cross_subject_5fold_splits(
    subjects: np.ndarray,
    y: np.ndarray,
    seed: int,
) -> List[Dict[str, Any]]:
    """Tạo StratifiedGroupKFold(5) nhóm theo subject_id với val_idx tách theo subject."""
    unique_subjects = sorted(np.unique(subjects).tolist(), key=natural_sort_key)
    if len(unique_subjects) < 5:
        return []

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    dummy_x = np.zeros(len(y), dtype=np.int8)
    folds: List[Dict[str, Any]] = []

    for fold_idx, (tr_val_pos, te_pos) in enumerate(sgkf.split(dummy_x, y, groups=subjects)):
        test_subjects = sorted(np.unique(subjects[te_pos]).tolist(), key=natural_sort_key)
        train_val_subjects = sorted(
            np.unique(subjects[tr_val_pos]).tolist(), key=natural_sort_key
        )

        n_val = max(1, int(round(len(train_val_subjects) * 0.2)))
        rng = np.random.RandomState(seed + fold_idx)
        val_subjects = sorted(
            rng.choice(train_val_subjects, size=n_val, replace=False).tolist(),
            key=natural_sort_key,
        )
        val_set = set(val_subjects)
        train_subjects = [s for s in train_val_subjects if s not in val_set]

        train_idx = np.where(np.isin(subjects, train_subjects))[0]
        val_idx = np.where(np.isin(subjects, val_subjects))[0]
        test_idx = np.where(np.isin(subjects, test_subjects))[0]

        folds.append(
            {
                "fold": int(fold_idx),
                "train_subjects": train_subjects,
                "val_subjects": val_subjects,
                "test_subjects": test_subjects,
                "train_idx": train_idx.astype(int).tolist(),
                "val_idx": val_idx.astype(int).tolist(),
                "test_idx": test_idx.astype(int).tolist(),
            }
        )

    return folds


def build_cross_subject_loso_splits(
    subjects: np.ndarray,
    y: np.ndarray,
    unique_subjects: List[str],
    seed: int,
) -> List[Dict[str, Any]]:
    """Tạo Leave-One-Subject-Out (LOSO) với tập val_subjects tách rời 100%."""
    if len(unique_subjects) < 3:
        return []

    logo = LeaveOneGroupOut()
    dummy_x = np.zeros(len(y), dtype=np.int8)
    raw_folds = []

    for tr_val_pos, te_pos in logo.split(dummy_x, y, groups=subjects):
        test_subject = str(subjects[te_pos[0]])
        raw_folds.append((test_subject, tr_val_pos, te_pos))

    # Sắp xếp tự nhiên theo tên subject test để fold 0..K-1 ổn định
    raw_folds.sort(key=lambda item: natural_sort_key(item[0]))

    n_subjects = len(unique_subjects)
    folds: List[Dict[str, Any]] = []
    for fold_idx, (test_subject, _, _) in enumerate(raw_folds):
        rem_subjects = [s for s in unique_subjects if s != test_subject]
        n_val = 4 if n_subjects >= 30 else max(1, int(round(len(rem_subjects) * 0.15)))

        rng = np.random.RandomState(seed + fold_idx)
        val_subjects = sorted(
            rng.choice(rem_subjects, size=n_val, replace=False).tolist(),
            key=natural_sort_key,
        )
        val_set = set(val_subjects)
        train_subjects = [s for s in rem_subjects if s not in val_set]

        train_idx = np.where(np.isin(subjects, train_subjects))[0]
        val_idx = np.where(np.isin(subjects, val_subjects))[0]
        test_idx = np.where(subjects == test_subject)[0]

        folds.append(
            {
                "fold": int(fold_idx),
                "train_subjects": train_subjects,
                "val_subjects": val_subjects,
                "test_subjects": [test_subject],
                "train_idx": train_idx.astype(int).tolist(),
                "val_idx": val_idx.astype(int).tolist(),
                "test_idx": test_idx.astype(int).tolist(),
            }
        )

    return folds


def generate_splits(data_dir: Path, seed: int = 42) -> Path:
    """Đọc metadata.csv trong `data_dir` và xuất `splits.json`."""
    meta_path = data_dir / "metadata.csv"
    if not meta_path.exists():
        raise FileNotFoundError(f"Không tìm thấy metadata.csv tại: {meta_path.resolve()}")

    df = pd.read_csv(meta_path)
    required_cols = {"trial_uid", "subject_id", "label"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"metadata.csv thiếu các cột bắt buộc: {sorted(missing)}")

    trial_uids = df["trial_uid"].astype(str).tolist()
    subjects = df["subject_id"].astype(str).to_numpy()
    y = df["label"].astype(np.int64).to_numpy()
    unique_subjects = sorted(np.unique(subjects).tolist(), key=natural_sort_key)

    splits_payload = {
        "schema_version": "2.5.0",
        "seed": int(seed),
        "n_samples": int(len(df)),
        "n_subjects": int(len(unique_subjects)),
        "subjects": unique_subjects,
        "trial_uid_sha256": compute_trial_uid_sha256(trial_uids),
        "within_subject": build_within_subject_splits(subjects, y, unique_subjects, seed),
        "cross_subject_5fold": build_cross_subject_5fold_splits(subjects, y, seed),
        "cross_subject_loso": build_cross_subject_loso_splits(
            subjects, y, unique_subjects, seed
        ),
    }

    out_path = data_dir / "splits.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(splits_payload, f, indent=2)

    print(
        f"[generate_splits] Đã lưu {out_path} "
        f"(n_samples={len(df)}, n_subjects={len(unique_subjects)}, "
        f"5fold={len(splits_payload['cross_subject_5fold'])}, "
        f"loso={len(splits_payload['cross_subject_loso'])})"
    )
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sinh bộ chia tập chuẩn Zero-Leakage (splits.json) từ metadata.csv"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/processed"),
        help="Thư mục chứa metadata.csv và sẽ lưu splits.json (mặc định: data/processed)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed cố định cho StratifiedKFold / ShuffleSplit (mặc định: 42)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generate_splits(data_dir=args.data_dir, seed=args.seed)
