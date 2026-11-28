"""ML3 XGBoost training, checkpointing, metrics, and plots (CS5, LOSO & Within-Subject)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
from pathlib import Path
import re
import time
from typing import Any, Dict, List, Literal, Mapping, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_curve

try:
    from src.evaluation.visualization import OKABE_ITO, setup_publication_style
except Exception:  # Fallback identical to src.evaluation.visualization when MNE is absent on Colab
    OKABE_ITO: Dict[str, str] = {
        "blue": "#0072B2",
        "orange": "#D55E00",
        "green": "#009E73",
        "purple": "#CC79A7",
        "sky": "#56B4E9",
        "yellow": "#E69F00",
        "black": "#1F2937",
        "grey": "#6B7280",
        "red": "#B91C1C",
    }

    def setup_publication_style() -> None:
        plt.rcParams.update(
            {
                "font.family": "sans-serif",
                "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
                "font.size": 9.5,
                "axes.titlesize": 10.5,
                "axes.titleweight": "bold",
                "axes.labelsize": 9.5,
                "xtick.labelsize": 8.5,
                "ytick.labelsize": 8.5,
                "legend.fontsize": 8.2,
                "figure.titlesize": 12.0,
                "figure.titleweight": "bold",
                "axes.spines.top": False,
                "axes.spines.right": False,
                "axes.linewidth": 0.9,
                "grid.alpha": 0.28,
                "grid.linestyle": "--",
                "savefig.dpi": 300,
                "pdf.fonttype": 42,
                "ps.fonttype": 42,
            }
        )

from src.evaluation.metrics import (
    METRIC_NAMES,
    _sanitize_json_value,
    compute_binary_metrics,
    find_optimal_threshold,
)
from src.features.ml3_feature_store import (
    CS5_MAIN_EXPERIMENTS,
    FEATURE_SCENARIO_SPECS,
    FeatureScenarioSpec,
    LOSO_MAIN_EXPERIMENTS,
    ML3FeatureDataset,
    ScenarioMatrix,
    W8_WINDOW_NAMES,
    ensure_drive_path_ready,
    inspect_feature_store_metadata,
    load_ml3_feature_dataset,
    parse_ml3_feature_name,
)
from src.models.Ml3 import ML3XGBoostConfig, ML3XGBoostModel

ProtocolName = Literal["cs5", "loso", "within"]

SPLIT_FILE_BY_PROTOCOL: Dict[ProtocolName, Tuple[str, str]] = {
    "cs5": ("cross_5fold.json", "cross_subject_5fold"),
    "loso": ("loso.json", "cross_subject_loso"),
    "within": ("within_5fold.json", "within_subject"),
}

WITHIN_MAIN_EXPERIMENTS: Tuple[str, ...] = ("combined_w5",)

CANONICAL_EXPERIMENT_ORDER: Tuple[str, ...] = (
    "temporal",
    "wavelet_w5",
    "wavelet_w8",
    "combined_w5",
    "combined_w8",
    "wavelet_w5_gamma",
    "wavelet_w8_gamma",
    "combined_w5_gamma",
    "combined_w8_gamma",
)

PRIMARY_METRICS: Tuple[str, ...] = ("pr_auc", "roc_auc")
SUPPORTING_METRICS: Tuple[str, ...] = (
    "balanced_accuracy",
    "f1_target",
    "sensitivity",
    "specificity",
)
ALL_REPORTED_METRICS: Tuple[str, ...] = PRIMARY_METRICS + SUPPORTING_METRICS


def _natural_sort_key(value: str) -> list[int | str]:
    return [int(tok) if tok.isdigit() else tok.lower() for tok in re.split(r"(\d+)", str(value))]


def _canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(
        _sanitize_json_value(dict(payload)),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.stem}.tmp.json")
    tmp_path.write_text(
        json.dumps(_sanitize_json_value(dict(payload)), indent=2, allow_nan=False),
        encoding="utf-8",
    )
    tmp_path.replace(path)
    return path


def normalize_protocol_name(protocol: str) -> ProtocolName:
    cleaned = str(protocol).strip().lower()
    alias_map: Dict[str, ProtocolName] = {
        "cs5": "cs5",
        "cross_5fold": "cs5",
        "cross_subject_5fold": "cs5",
        "loso": "loso",
        "cross_subject_loso": "loso",
        "within": "within",
        "within_subject": "within",
        "within_5fold": "within",
        "ws": "within",
    }
    if cleaned not in alias_map:
        raise ValueError(
            f"Unsupported ML3 protocol {protocol!r}; expected 'cs5', 'loso', or 'within'"
        )
    return alias_map[cleaned]


def default_experiments_for_protocol(protocol: ProtocolName) -> Tuple[str, ...]:
    norm = normalize_protocol_name(protocol)
    if norm == "cs5":
        return CS5_MAIN_EXPERIMENTS
    if norm == "within":
        return WITHIN_MAIN_EXPERIMENTS
    return LOSO_MAIN_EXPERIMENTS


def validate_experiment_list(
    protocol: ProtocolName,
    experiments: Optional[Sequence[str]],
    *,
    allow_gamma_loso: bool = False,
) -> Tuple[str, ...]:
    norm = normalize_protocol_name(protocol)
    chosen = (
        tuple(dict.fromkeys(str(e).strip() for e in experiments))
        if experiments
        else default_experiments_for_protocol(norm)
    )
    if not chosen:
        raise ValueError("At least one ML3 experiment must be specified")

    unknown = [exp for exp in chosen if exp not in FEATURE_SCENARIO_SPECS]
    if unknown:
        raise ValueError(
            f"Unknown ML3 experiment(s): {unknown}. "
            f"Valid options: {list(CANONICAL_EXPERIMENT_ORDER)}"
        )

    if norm == "loso" and not allow_gamma_loso:
        gamma_exps = [exp for exp in chosen if FEATURE_SCENARIO_SPECS[exp].uses_gamma]
        if gamma_exps:
            raise ValueError(
                f"Gamma experiments {gamma_exps} do not run on LOSO by default. "
                "Run gamma verification on CS5 first (or pass --allow-gamma-loso explicitly)."
            )
    return chosen


def required_feature_groups_for_experiments(experiments: Sequence[str]) -> Tuple[str, ...]:
    groups: list[str] = []
    for exp in experiments:
        spec = FEATURE_SCENARIO_SPECS[exp]
        if spec.uses_temporal and "temporal" not in groups:
            groups.append("temporal")
        if spec.wavelet_windows is not None and "wavelet" not in groups:
            groups.append("wavelet")
    return tuple(groups)


@dataclass(frozen=True)
class SubjectFoldSplit:
    """Subject-level train/val/test partition for one CS5 or LOSO fold."""

    protocol: ProtocolName
    fold: int
    train_subjects: Tuple[str, ...]
    val_subjects: Tuple[str, ...]
    test_subjects: Tuple[str, ...]

    def __post_init__(self) -> None:
        if self.fold < 0:
            raise ValueError(f"fold index must be non-negative, got {self.fold}")
        if not self.train_subjects or not self.val_subjects or not self.test_subjects:
            raise ValueError(
                f"Fold {self.fold} must have non-empty train_subjects, val_subjects, and test_subjects"
            )
        s_train = set(self.train_subjects)
        s_val = set(self.val_subjects)
        s_test = set(self.test_subjects)
        if len(s_train) != len(self.train_subjects) or len(s_val) != len(self.val_subjects) or len(s_test) != len(self.test_subjects):
            raise ValueError(f"Fold {self.fold} contains duplicate subject IDs within a split")

        overlap_tv = sorted(s_train & s_val)
        overlap_tt = sorted(s_train & s_test)
        overlap_vt = sorted(s_val & s_test)
        if overlap_tv or overlap_tt or overlap_vt:
            raise ValueError(
                f"Subject leakage in {self.protocol} fold {self.fold}: "
                f"train&val={overlap_tv}, train&test={overlap_tt}, val&test={overlap_vt}"
            )

    @property
    def fold_key(self) -> str:
        return f"fold_{self.fold:02d}"

    def split_signature(self) -> Dict[str, Any]:
        return {
            "protocol": self.protocol,
            "fold": int(self.fold),
            "train_subjects": list(self.train_subjects),
            "val_subjects": list(self.val_subjects),
            "test_subjects": list(self.test_subjects),
        }


@dataclass(frozen=True)
class WithinFoldSplit:
    """Trial-level train/val/test partition for one (subject_id, fold) in Within-Subject 5-Fold."""

    protocol: ProtocolName
    subject_id: str
    fold: int
    train_local_idx: Tuple[int, ...]
    val_local_idx: Tuple[int, ...]
    test_local_idx: Tuple[int, ...]
    subject_n_trials: int

    def __post_init__(self) -> None:
        if self.protocol != "within":
            raise ValueError(f"WithinFoldSplit requires protocol='within', got {self.protocol!r}")
        if self.fold < 0:
            raise ValueError(f"fold index must be non-negative, got {self.fold}")
        if not self.subject_id:
            raise ValueError("subject_id must be non-empty")
        if not self.train_local_idx or not self.val_local_idx or not self.test_local_idx:
            raise ValueError(
                f"Within-subject split {self.subject_id} fold {self.fold} must have non-empty "
                "train, val, and test indices"
            )
        s_tr = set(self.train_local_idx)
        s_va = set(self.val_local_idx)
        s_te = set(self.test_local_idx)
        if (
            len(s_tr) != len(self.train_local_idx)
            or len(s_va) != len(self.val_local_idx)
            or len(s_te) != len(self.test_local_idx)
        ):
            raise ValueError(
                f"Duplicate trial indices in within-subject split {self.subject_id} fold {self.fold}"
            )
        if (s_tr & s_va) or (s_tr & s_te) or (s_va & s_te):
            raise ValueError(
                f"Trial leakage in within-subject split {self.subject_id} fold {self.fold}"
            )
        all_idx = s_tr | s_va | s_te
        if min(all_idx) < 0 or max(all_idx) >= self.subject_n_trials:
            raise ValueError(
                f"Local trial index out of bounds [0, {self.subject_n_trials}) "
                f"for {self.subject_id} fold {self.fold}"
            )

    @property
    def fold_key(self) -> str:
        return f"{self.subject_id}_fold_{self.fold:02d}"

    @property
    def train_subjects(self) -> Tuple[str, ...]:
        return (self.subject_id,)

    @property
    def val_subjects(self) -> Tuple[str, ...]:
        return (self.subject_id,)

    @property
    def test_subjects(self) -> Tuple[str, ...]:
        return (self.subject_id,)

    def split_signature(self) -> Dict[str, Any]:
        def _hash_indices(idxs: Tuple[int, ...]) -> str:
            return hashlib.sha256(",".join(str(i) for i in idxs).encode("utf-8")).hexdigest()[:16]

        return {
            "protocol": self.protocol,
            "subject_id": self.subject_id,
            "fold": int(self.fold),
            "subject_n_trials": int(self.subject_n_trials),
            "n_train": len(self.train_local_idx),
            "n_val": len(self.val_local_idx),
            "n_test": len(self.test_local_idx),
            "train_idx_sha": _hash_indices(self.train_local_idx),
            "val_idx_sha": _hash_indices(self.val_local_idx),
            "test_idx_sha": _hash_indices(self.test_local_idx),
        }


FoldSplit = SubjectFoldSplit | WithinFoldSplit


def _iter_fold_checkpoint_dirs(exp_dir: Path) -> List[Path]:
    """Return sorted per-fold checkpoint directories (handles ``fold_*`` and ``sub-*_fold_*``)."""
    if not exp_dir.is_dir():
        return []
    dirs = [
        p
        for p in exp_dir.iterdir()
        if p.is_dir() and (p.name.startswith("fold_") or "_fold_" in p.name)
    ]
    return sorted(dirs, key=lambda p: _natural_sort_key(p.name))


def load_ml3_splits(
    splits_dir: Path | str,
    protocol: ProtocolName | str,
) -> Tuple[Tuple[FoldSplit, ...], Dict[str, Any]]:
    """Load CS5, LOSO, or Within-Subject folds directly from ``splits/*.json``."""
    norm = normalize_protocol_name(protocol)
    filename, json_key = SPLIT_FILE_BY_PROTOCOL[norm]
    split_path = Path(splits_dir) / filename
    if not split_path.is_file():
        raise FileNotFoundError(f"Split file not found: {split_path}")

    raw = json.loads(split_path.read_text(encoding="utf-8"))

    if norm == "within":
        if json_key not in raw or not isinstance(raw[json_key], dict) or not raw[json_key]:
            raise ValueError(f"Split file {split_path} is missing non-empty dict '{json_key}'")
        within_folds: list[WithinFoldSplit] = []
        subject_order = raw.get("subjects") or sorted(raw[json_key].keys(), key=_natural_sort_key)
        for sub_id in subject_order:
            if sub_id not in raw[json_key]:
                raise ValueError(f"Subject {sub_id!r} missing from '{json_key}' in {split_path}")
            entries = raw[json_key][sub_id]
            if not isinstance(entries, list) or not entries:
                raise ValueError(f"Empty fold list for subject {sub_id!r} in {split_path}")

            first_union = sorted(
                set(int(i) for i in entries[0]["train_idx"])
                | set(int(i) for i in entries[0]["val_idx"])
                | set(int(i) for i in entries[0]["test_idx"])
            )
            sub_offset = first_union[0]
            sub_n_trials = len(first_union)
            if first_union != list(range(sub_offset, sub_offset + sub_n_trials)):
                raise ValueError(
                    f"Within-subject indices for {sub_id!r} in {split_path} are not contiguous"
                )

            seen_sub_folds: set[int] = set()
            for entry in entries:
                fold_idx = int(entry["fold"])
                if fold_idx in seen_sub_folds:
                    raise ValueError(
                        f"Duplicate fold index {fold_idx} for subject {sub_id!r} in {split_path}"
                    )
                seen_sub_folds.add(fold_idx)
                tr_local = tuple(int(i) - sub_offset for i in entry["train_idx"])
                va_local = tuple(int(i) - sub_offset for i in entry["val_idx"])
                te_local = tuple(int(i) - sub_offset for i in entry["test_idx"])
                within_folds.append(
                    WithinFoldSplit(
                        protocol="within",
                        subject_id=str(sub_id),
                        fold=fold_idx,
                        train_local_idx=tr_local,
                        val_local_idx=va_local,
                        test_local_idx=te_local,
                        subject_n_trials=sub_n_trials,
                    )
                )

        within_sorted = tuple(
            sorted(within_folds, key=lambda f: (_natural_sort_key(f.subject_id), f.fold))
        )
        meta = {
            "split_path": str(split_path),
            "schema_version": raw.get("schema_version"),
            "seed": raw.get("seed"),
            "n_subjects": raw.get("n_subjects", len(subject_order)),
            "subjects": list(subject_order),
            "trial_uid_sha256": raw.get("trial_uid_sha256"),
            "n_folds": len(within_sorted),
            "n_folds_per_subject": len(raw[json_key][subject_order[0]]),
        }
        return within_sorted, meta

    if json_key not in raw or not isinstance(raw[json_key], list) or not raw[json_key]:
        raise ValueError(f"Split file {split_path} is missing non-empty fold list '{json_key}'")

    folds: list[SubjectFoldSplit] = []
    seen_folds: set[int] = set()
    for entry in raw[json_key]:
        fold_idx = int(entry["fold"])
        if fold_idx in seen_folds:
            raise ValueError(f"Duplicate fold index {fold_idx} in {split_path}")
        seen_folds.add(fold_idx)
        folds.append(
            SubjectFoldSplit(
                protocol=norm,
                fold=fold_idx,
                train_subjects=tuple(str(s) for s in entry["train_subjects"]),
                val_subjects=tuple(str(s) for s in entry["val_subjects"]),
                test_subjects=tuple(str(s) for s in entry["test_subjects"]),
            )
        )

    folds_sorted = tuple(sorted(folds, key=lambda f: f.fold))
    meta = {
        "split_path": str(split_path),
        "schema_version": raw.get("schema_version"),
        "seed": raw.get("seed"),
        "n_subjects": raw.get("n_subjects"),
        "subjects": raw.get("subjects"),
        "trial_uid_sha256": raw.get("trial_uid_sha256"),
        "n_folds": len(folds_sorted),
    }
    return folds_sorted, meta


def resolve_subject_fold_indices(
    split: SubjectFoldSplit,
    subject_ids: np.ndarray,
    trial_uids: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map ``train_subjects / val_subjects / test_subjects`` to verified disjoint row indices."""
    subs = np.asarray(subject_ids, dtype=str)
    uids = np.asarray(trial_uids, dtype=str)
    if len(subs) != len(uids):
        raise ValueError("subject_ids and trial_uids must have equal length")

    available = set(np.unique(subs).tolist())
    required = set(split.train_subjects) | set(split.val_subjects) | set(split.test_subjects)
    missing = sorted(required - available)
    if missing:
        raise ValueError(
            f"Split {split.protocol} fold {split.fold} references subjects absent from features: {missing}"
        )

    train_idx = np.flatnonzero(np.isin(subs, split.train_subjects))
    val_idx = np.flatnonzero(np.isin(subs, split.val_subjects))
    test_idx = np.flatnonzero(np.isin(subs, split.test_subjects))

    if train_idx.size == 0 or val_idx.size == 0 or test_idx.size == 0:
        raise ValueError(f"Empty partition encountered in {split.protocol} fold {split.fold}")

    if (
        np.intersect1d(train_idx, val_idx).size
        or np.intersect1d(train_idx, test_idx).size
        or np.intersect1d(val_idx, test_idx).size
    ):
        raise ValueError(f"Trial index leakage detected in {split.protocol} fold {split.fold}")

    u_tr = set(uids[train_idx].tolist())
    u_va = set(uids[val_idx].tolist())
    u_te = set(uids[test_idx].tolist())
    if (u_tr & u_va) or (u_tr & u_te) or (u_va & u_te):
        raise ValueError(f"trial_uid leakage detected in {split.protocol} fold {split.fold}")

    return train_idx, val_idx, test_idx


def resolve_within_fold_indices(
    split: WithinFoldSplit,
    subject_ids: np.ndarray,
    trial_uids: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map a subject's within-subject local indices to verified disjoint global row indices."""
    subs = np.asarray(subject_ids, dtype=str)
    uids = np.asarray(trial_uids, dtype=str)
    if len(subs) != len(uids):
        raise ValueError("subject_ids and trial_uids must have equal length")

    sub_rows = np.flatnonzero(subs == split.subject_id)
    if sub_rows.size == 0:
        raise ValueError(
            f"Within-subject split {split.fold_key} references subject absent from features: "
            f"{split.subject_id!r}"
        )
    if sub_rows.size != split.subject_n_trials:
        raise ValueError(
            f"Trial count mismatch for {split.subject_id!r} in {split.fold_key}: "
            f"features have {sub_rows.size} trials, split expects {split.subject_n_trials}"
        )

    train_idx = sub_rows[np.asarray(split.train_local_idx, dtype=np.int64)]
    val_idx = sub_rows[np.asarray(split.val_local_idx, dtype=np.int64)]
    test_idx = sub_rows[np.asarray(split.test_local_idx, dtype=np.int64)]

    if (
        np.intersect1d(train_idx, val_idx).size
        or np.intersect1d(train_idx, test_idx).size
        or np.intersect1d(val_idx, test_idx).size
    ):
        raise ValueError(f"Trial index leakage detected in {split.fold_key}")

    u_tr = set(uids[train_idx].tolist())
    u_va = set(uids[val_idx].tolist())
    u_te = set(uids[test_idx].tolist())
    if (u_tr & u_va) or (u_tr & u_te) or (u_va & u_te):
        raise ValueError(f"trial_uid leakage detected in {split.fold_key}")

    return train_idx, val_idx, test_idx


def resolve_fold_indices(
    split: FoldSplit,
    subject_ids: np.ndarray,
    trial_uids: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Dispatch fold index resolution for CS5, LOSO, or Within-Subject splits."""
    if isinstance(split, WithinFoldSplit):
        return resolve_within_fold_indices(split, subject_ids, trial_uids)
    return resolve_subject_fold_indices(split, subject_ids, trial_uids)


def compute_fold_config_hash(
    *,
    protocol: ProtocolName,
    scenario: FeatureScenarioSpec,
    model_config: ML3XGBoostConfig,
    split: FoldSplit,
    threshold_metric: str = "balanced_accuracy",
) -> str:
    payload = {
        "schema_version": 1,
        "protocol": protocol,
        "feature_policy": scenario.policy_dict(),
        "model_parameters": model_config.semantic_dict(),
        "split": split.split_signature(),
        "threshold_metric": threshold_metric,
    }
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _format_metric_with_confusion(raw_metrics: Mapping[str, Any]) -> Dict[str, Any]:
    data = dict(raw_metrics)
    data["confusion_matrix"] = {
        "tp": int(data["tp"]),
        "tn": int(data["tn"]),
        "fp": int(data["fp"]),
        "fn": int(data["fn"]),
    }
    return data


def _compute_pooled_metrics_with_fixed_preds(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, Any]:
    """Compute trial-pooled metrics across folds where each fold used its own val threshold."""
    base = compute_binary_metrics(y_true, y_prob, threshold=0.5)
    y_arr = np.asarray(y_true, dtype=np.int64).ravel()
    pred_arr = np.asarray(y_pred, dtype=np.int64).ravel()

    pos_mask = y_arr == 1
    neg_mask = y_arr == 0
    pred_pos = pred_arr == 1
    pred_neg = pred_arr == 0

    tp = int(np.sum(pred_pos & pos_mask))
    fn = int(np.sum(pred_neg & pos_mask))
    fp = int(np.sum(pred_pos & neg_mask))
    tn = int(np.sum(pred_neg & neg_mask))
    n_pos = int(tp + fn)
    n_neg = int(tn + fp)

    sensitivity = float(tp / n_pos) if n_pos > 0 else float("nan")
    specificity = float(tn / n_neg) if n_neg > 0 else float("nan")
    if n_pos > 0 and n_neg > 0:
        bal_acc = float(0.5 * (sensitivity + specificity))
    elif n_pos > 0:
        bal_acc = sensitivity
    else:
        bal_acc = specificity
    f1_denom = 2 * tp + fp + fn
    f1_target = float((2 * tp) / f1_denom) if f1_denom > 0 else 0.0

    return {
        "pr_auc": base["pr_auc"],
        "roc_auc": base["roc_auc"],
        "balanced_accuracy": bal_acc,
        "f1_target": f1_target,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "n_samples": int(y_arr.size),
        "n_pos": n_pos,
        "n_neg": n_neg,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "confusion_matrix": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
    }


def summarize_scalar_values(values: Sequence[float | int | None]) -> Dict[str, Any]:
    finite = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not finite:
        return {
            "mean": float("nan"),
            "std": float("nan"),
            "median": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
            "n": 0,
        }
    arr = np.asarray(finite, dtype=np.float64)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1 if arr.size > 1 else 0)),
        "median": float(np.median(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "n": int(arr.size),
    }


def summarize_metric_records(records: Sequence[Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {
        metric: summarize_scalar_values([rec.get(metric) for rec in records])
        for metric in ALL_REPORTED_METRICS
    }


def _per_subject_metrics(
    subject_list: Sequence[str],
    subject_ids: np.ndarray,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
) -> Dict[str, Dict[str, Any]]:
    subs = np.asarray(subject_ids, dtype=str)
    result: Dict[str, Dict[str, Any]] = {}
    for sub in sorted(subject_list, key=_natural_sort_key):
        mask = subs == sub
        if not np.any(mask):
            continue
        m = _format_metric_with_confusion(
            compute_binary_metrics(y_true[mask], y_prob[mask], threshold=threshold)
        )
        m["subject_id"] = sub
        result[sub] = m
    return result


def _validate_and_load_fold_checkpoint(
    fold_dir: Path,
    *,
    expected_hash: str,
    resume: bool,
    overwrite: bool,
) -> Optional[Tuple[Dict[str, Any], Dict[str, np.ndarray]]]:
    if not fold_dir.exists():
        return None

    manifest_path = fold_dir / "manifest.json"
    metrics_path = fold_dir / "metrics.json"
    preds_path = fold_dir / "predictions.npz"
    model_path = fold_dir / "model.json"

    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        saved_hash = str(manifest.get("config_hash", ""))
        if saved_hash != expected_hash:
            if overwrite and not resume:
                return None
            raise ValueError(
                f"Checkpoint không tương thích tại '{fold_dir}': config_hash không khớp "
                f"(saved={saved_hash[:12]}..., expected={expected_hash[:12]}...). "
                "Model parameters, feature policy hoặc split đã thay đổi."
            )

        if overwrite and not resume:
            return None

        if not resume:
            raise ValueError(
                f"Checkpoint đã tồn tại tại '{fold_dir}'. Dùng --resume để tái sử dụng "
                "hoặc --overwrite để ghi đè."
            )

        missing = [
            p.name
            for p in (model_path, metrics_path, preds_path)
            if not p.is_file()
        ]
        if missing:
            raise ValueError(
                f"Checkpoint tại '{fold_dir}' bị thiếu file {missing}; "
                "hãy xóa thư mục fold này hoặc chạy lại với --overwrite."
            )

        fold_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        with np.load(preds_path, allow_pickle=False) as payload:
            preds = {k: np.asarray(payload[k]) for k in payload.files}
        return fold_metrics, preds

    # Incomplete fold directory without manifest.json (e.g., interrupted Colab session):
    for partial in fold_dir.glob("*"):
        if partial.is_file():
            partial.unlink()
    return None


def train_or_load_fold(
    *,
    scenario_matrix: ScenarioMatrix,
    split: FoldSplit,
    model_config: ML3XGBoostConfig,
    fold_dir: Path,
    threshold_metric: str = "balanced_accuracy",
    resume: bool = False,
    overwrite: bool = False,
) -> Tuple[Dict[str, Any], Dict[str, np.ndarray], str]:
    """Train a single CS5, LOSO, or Within-Subject fold or reuse its verified checkpoint."""
    expected_hash = compute_fold_config_hash(
        protocol=split.protocol,
        scenario=scenario_matrix.scenario,
        model_config=model_config,
        split=split,
        threshold_metric=threshold_metric,
    )

    cached = _validate_and_load_fold_checkpoint(
        fold_dir,
        expected_hash=expected_hash,
        resume=resume,
        overwrite=overwrite,
    )
    if cached is not None:
        fold_metrics, preds = cached
        return fold_metrics, preds, "reused"

    train_idx, val_idx, test_idx = resolve_fold_indices(
        split,
        scenario_matrix.subject_ids,
        scenario_matrix.trial_uids,
    )

    X_train = scenario_matrix.values[train_idx]
    y_train = scenario_matrix.targets[train_idx]
    X_val = scenario_matrix.values[val_idx]
    y_val = scenario_matrix.targets[val_idx]
    X_test = scenario_matrix.values[test_idx]
    y_test = scenario_matrix.targets[test_idx]

    model = ML3XGBoostModel(config=model_config)
    t_start = time.perf_counter()
    model.fit_fold(X_train, y_train, X_val, y_val)
    train_duration_sec = float(time.perf_counter() - t_start)

    train_prob = model.predict_proba_positive(X_train)
    val_prob = model.predict_proba_positive(X_val)
    test_prob = model.predict_proba_positive(X_test)

    # Threshold is strictly optimized on validation only; never on test.
    threshold = find_optimal_threshold(y_val, val_prob, metric=threshold_metric)

    val_pred = (val_prob >= threshold).astype(np.int64)
    test_pred = (test_prob >= threshold).astype(np.int64)

    val_subs = scenario_matrix.subject_ids[val_idx]
    test_subs = scenario_matrix.subject_ids[test_idx]

    train_pooled = _format_metric_with_confusion(
        compute_binary_metrics(y_train, train_prob, threshold=threshold)
    )
    val_pooled = _format_metric_with_confusion(
        compute_binary_metrics(y_val, val_prob, threshold=threshold)
    )
    val_per_subject = _per_subject_metrics(
        split.val_subjects, val_subs, y_val, val_prob, threshold=threshold
    )
    val_subject_macro = summarize_metric_records(list(val_per_subject.values()))

    test_pooled = _format_metric_with_confusion(
        compute_binary_metrics(y_test, test_prob, threshold=threshold)
    )
    test_per_subject = _per_subject_metrics(
        split.test_subjects, test_subs, y_test, test_prob, threshold=threshold
    )
    test_subject_macro = summarize_metric_records(list(test_per_subject.values()))

    assert model.best_iteration_ is not None
    assert model.scale_pos_weight_ is not None

    split_subject_id: Optional[str] = (
        split.subject_id if isinstance(split, WithinFoldSplit) else None
    )

    fold_metrics: Dict[str, Any] = {
        "schema_version": 1,
        "protocol": split.protocol,
        "experiment": scenario_matrix.scenario.name,
        "fold": int(split.fold),
        "fold_key": split.fold_key,
        **({"subject_id": split_subject_id} if split_subject_id is not None else {}),
        "config_hash": expected_hash,
        "n_features": int(scenario_matrix.n_features),
        "train_subjects": list(split.train_subjects),
        "val_subjects": list(split.val_subjects),
        "test_subjects": list(split.test_subjects),
        "n_train": int(len(y_train)),
        "n_train_frequent": int(np.sum(y_train == 0)),
        "n_train_target": int(np.sum(y_train == 1)),
        "n_val": int(len(y_val)),
        "n_test": int(len(y_test)),
        "scale_pos_weight": float(model.scale_pos_weight_),
        "best_iteration": int(model.best_iteration_),
        "n_trees_used": int(model.best_iteration_ + 1),
        "best_val_aucpr": model.best_val_score_,
        "threshold_metric": threshold_metric,
        "threshold": float(threshold),
        "train_duration_sec": train_duration_sec,
        "train_pooled": train_pooled,
        "validation_pooled": val_pooled,
        "validation_per_subject": val_per_subject,
        "validation_subject_macro": val_subject_macro,
        "test_pooled": test_pooled,
        "test_per_subject": test_per_subject,
        "test_subject_macro": test_subject_macro,
    }

    predictions_payload: Dict[str, np.ndarray] = {
        "subject_id": np.asarray(test_subs, dtype=str),
        "fold": np.full(len(test_idx), int(split.fold), dtype=np.int64),
        "trial_idx": np.asarray(test_idx, dtype=np.int64),
        "trial_uid": np.asarray(scenario_matrix.trial_uids[test_idx], dtype=str),
        "y_true": np.asarray(y_test, dtype=np.int64),
        "y_prob": np.asarray(test_prob, dtype=np.float64),
        "y_pred": np.asarray(test_pred, dtype=np.int64),
        "threshold": np.full(len(test_idx), float(threshold), dtype=np.float64),
        "val_subject_id": np.asarray(val_subs, dtype=str),
        "val_trial_idx": np.asarray(val_idx, dtype=np.int64),
        "val_trial_uid": np.asarray(scenario_matrix.trial_uids[val_idx], dtype=str),
        "val_y_true": np.asarray(y_val, dtype=np.int64),
        "val_y_prob": np.asarray(val_prob, dtype=np.float64),
        "val_y_pred": np.asarray(val_pred, dtype=np.int64),
    }

    fold_dir.mkdir(parents=True, exist_ok=True)
    model.save_model(fold_dir / "model.json")

    tmp_preds = fold_dir / "predictions.tmp.npz"
    np.savez_compressed(tmp_preds, **predictions_payload)
    tmp_preds.replace(fold_dir / "predictions.npz")

    _atomic_write_json(fold_dir / "metrics.json", fold_metrics)
    if model.learning_curve_ is not None:
        _atomic_write_json(
            fold_dir / "learning_curve.json",
            {
                "fold": int(split.fold),
                "fold_key": split.fold_key,
                **({"subject_id": split_subject_id} if split_subject_id is not None else {}),
                "best_iteration": int(model.best_iteration_),
                **model.learning_curve_,
            },
        )

    manifest_payload: Dict[str, Any] = {
        "schema_version": 1,
        "protocol": split.protocol,
        "experiment": scenario_matrix.scenario.name,
        "fold": int(split.fold),
        "fold_key": split.fold_key,
        **({"subject_id": split_subject_id} if split_subject_id is not None else {}),
        "config_hash": expected_hash,
        "feature_policy": scenario_matrix.scenario.policy_dict(),
        "model_config": model_config.report_dict(),
        "split": split.split_signature(),
        "threshold_metric": threshold_metric,
        "n_features": int(scenario_matrix.n_features),
        "best_iteration": int(model.best_iteration_),
        "train_duration_sec": train_duration_sec,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "files": ["model.json", "metrics.json", "predictions.npz"],
    }
    _atomic_write_json(fold_dir / "manifest.json", manifest_payload)

    del model
    gc.collect()
    return fold_metrics, predictions_payload, "trained"


def aggregate_experiment_results(
    *,
    protocol: ProtocolName,
    experiment: str,
    exp_dir: Path,
    model_config: ML3XGBoostConfig,
    fold_metrics_list: Sequence[Mapping[str, Any]],
    fold_preds_list: Sequence[Mapping[str, np.ndarray]],
) -> Dict[str, Any]:
    """Aggregate both trial-level pooled metrics and subject-macro metrics across folds."""
    if not fold_metrics_list:
        raise ValueError(f"No fold results available to aggregate for {experiment}")

    ordered_pairs = sorted(
        zip(fold_metrics_list, fold_preds_list),
        key=lambda pair: (
            _natural_sort_key(str(pair[0].get("subject_id") or "")),
            int(pair[0]["fold"]),
        ),
    )
    folds_sorted = [dict(p[0]) for p in ordered_pairs]
    preds_sorted = [p[1] for p in ordered_pairs]

    # 1. Pooled out-of-fold test predictions & validation predictions
    all_test_y = np.concatenate([p["y_true"] for p in preds_sorted], axis=0)
    all_test_prob = np.concatenate([p["y_prob"] for p in preds_sorted], axis=0)
    all_test_pred = np.concatenate([p["y_pred"] for p in preds_sorted], axis=0)
    overall_test_pooled = _compute_pooled_metrics_with_fixed_preds(
        all_test_y, all_test_prob, all_test_pred
    )

    all_val_y = np.concatenate([p["val_y_true"] for p in preds_sorted], axis=0)
    all_val_prob = np.concatenate([p["val_y_prob"] for p in preds_sorted], axis=0)
    all_val_pred = np.concatenate([p["val_y_pred"] for p in preds_sorted], axis=0)
    overall_val_pooled = _compute_pooled_metrics_with_fixed_preds(
        all_val_y, all_val_prob, all_val_pred
    )

    # Fold-level pooled summary (Mean ± SD & median across folds)
    test_fold_pooled_summary = summarize_metric_records(
        [f["test_pooled"] for f in folds_sorted]
    )
    val_fold_pooled_summary = summarize_metric_records(
        [f["validation_pooled"] for f in folds_sorted]
    )
    train_records = [f["train_pooled"] for f in folds_sorted if "train_pooled" in f]
    train_fold_pooled_summary = (
        summarize_metric_records(train_records) if train_records else None
    )

    # 2. Subject-macro aggregation (equal weight per subject; primary cross-subject metric)
    subject_buckets: Dict[str, list[Dict[str, Any]]] = {}
    for f_rec in folds_sorted:
        fold_idx = int(f_rec["fold"])
        for sub_id, sub_m in f_rec["test_per_subject"].items():
            entry = dict(sub_m)
            entry["fold"] = fold_idx
            subject_buckets.setdefault(sub_id, []).append(entry)

    per_subject_test: Dict[str, Dict[str, Any]] = {}
    for sub_id in sorted(subject_buckets, key=_natural_sort_key):
        entries = subject_buckets[sub_id]
        if len(entries) == 1:
            per_subject_test[sub_id] = entries[0]
        else:
            agg_sub = summarize_metric_records(entries)
            per_subject_test[sub_id] = {
                "subject_id": sub_id,
                "fold": [e["fold"] for e in entries],
                "n_folds": len(entries),
                **{m: agg_sub[m]["mean"] for m in ALL_REPORTED_METRICS},
                **{f"{m}_std": agg_sub[m]["std"] for m in ALL_REPORTED_METRICS},
                "n_samples": int(sum(int(e.get("n_samples", 0)) for e in entries)),
                "n_pos": int(sum(int(e.get("n_pos", 0)) for e in entries)),
                "n_neg": int(sum(int(e.get("n_neg", 0)) for e in entries)),
                "tp": int(sum(int(e.get("tp", 0)) for e in entries)),
                "tn": int(sum(int(e.get("tn", 0)) for e in entries)),
                "fp": int(sum(int(e.get("fp", 0)) for e in entries)),
                "fn": int(sum(int(e.get("fn", 0)) for e in entries)),
            }

    test_subject_macro_summary = summarize_metric_records(list(per_subject_test.values()))
    fold_subject_macro_means = [
        {m: f["test_subject_macro"][m]["mean"] for m in ALL_REPORTED_METRICS}
        for f in folds_sorted
    ]
    test_fold_subject_macro_summary = summarize_metric_records(fold_subject_macro_means)

    val_fold_subject_macro_means = [
        {m: f["validation_subject_macro"][m]["mean"] for m in ALL_REPORTED_METRICS}
        for f in folds_sorted
    ]
    val_fold_subject_macro_summary = summarize_metric_records(val_fold_subject_macro_means)

    best_iter_stats = summarize_scalar_values([f["best_iteration"] for f in folds_sorted])
    duration_values = [float(f["train_duration_sec"]) for f in folds_sorted]
    duration_stats = summarize_scalar_values(duration_values)
    duration_stats["total_sec"] = float(np.sum(duration_values))

    n_features = int(folds_sorted[0]["n_features"])
    scenario_spec = FEATURE_SCENARIO_SPECS[experiment]

    summary_payload: Dict[str, Any] = {
        "schema_version": 1,
        "protocol": protocol,
        "experiment": experiment,
        "feature_policy": scenario_spec.policy_dict(),
        "model_config": model_config.report_dict(),
        "n_features": n_features,
        "n_folds": len(folds_sorted),
        "n_test_subjects": len(per_subject_test),
        "best_iteration": best_iter_stats,
        "train_duration_sec": duration_stats,
        "test": {
            "subject_macro": test_subject_macro_summary,
            "fold_subject_macro": test_fold_subject_macro_summary,
            "fold_pooled": test_fold_pooled_summary,
            "overall_trial_pooled": overall_test_pooled,
        },
        "validation": {
            "fold_subject_macro": val_fold_subject_macro_summary,
            "fold_pooled": val_fold_pooled_summary,
            "overall_trial_pooled": overall_val_pooled,
        },
        "train": (
            {"fold_pooled": train_fold_pooled_summary}
            if train_fold_pooled_summary is not None
            else {}
        ),
        "per_subject": per_subject_test,
        "per_fold": folds_sorted,
    }
    _atomic_write_json(exp_dir / "experiment_summary.json", summary_payload)

    # Export convenient CSV tables for per-fold and per-subject inspection
    fold_rows = []
    for f in folds_sorted:
        tp_m = f["test_pooled"]
        tsm = f["test_subject_macro"]
        vp_m = f["validation_pooled"]
        fold_rows.append(
            {
                "protocol": protocol,
                "experiment": experiment,
                **({"subject_id": f["subject_id"]} if "subject_id" in f else {}),
                "fold": int(f["fold"]),
                "test_subjects": ",".join(f["test_subjects"]),
                "n_features": int(f["n_features"]),
                "scale_pos_weight": float(f["scale_pos_weight"]),
                "best_iteration": int(f["best_iteration"]),
                "threshold": float(f["threshold"]),
                "train_duration_sec": float(f["train_duration_sec"]),
                "val_pr_auc": vp_m["pr_auc"],
                "val_roc_auc": vp_m["roc_auc"],
                "val_balanced_accuracy": vp_m["balanced_accuracy"],
                "test_pooled_pr_auc": tp_m["pr_auc"],
                "test_pooled_roc_auc": tp_m["roc_auc"],
                "test_pooled_balanced_accuracy": tp_m["balanced_accuracy"],
                "test_pooled_f1_target": tp_m["f1_target"],
                "test_pooled_sensitivity": tp_m["sensitivity"],
                "test_pooled_specificity": tp_m["specificity"],
                "test_subject_macro_pr_auc": tsm["pr_auc"]["mean"],
                "test_subject_macro_roc_auc": tsm["roc_auc"]["mean"],
                "test_subject_macro_balanced_accuracy": tsm["balanced_accuracy"]["mean"],
                "test_subject_macro_f1_target": tsm["f1_target"]["mean"],
                "tp": int(tp_m["tp"]),
                "tn": int(tp_m["tn"]),
                "fp": int(tp_m["fp"]),
                "fn": int(tp_m["fn"]),
            }
        )
    pd.DataFrame(fold_rows).to_csv(exp_dir / "per_fold_metrics.csv", index=False)

    subject_rows = []
    for sub_id in sorted(per_subject_test, key=_natural_sort_key):
        s_rec = per_subject_test[sub_id]
        subject_rows.append(
            {
                "protocol": protocol,
                "experiment": experiment,
                "subject_id": sub_id,
                "fold": s_rec.get("fold"),
                **({"n_folds": s_rec["n_folds"]} if "n_folds" in s_rec else {}),
                "pr_auc": s_rec.get("pr_auc"),
                "roc_auc": s_rec.get("roc_auc"),
                "balanced_accuracy": s_rec.get("balanced_accuracy"),
                "f1_target": s_rec.get("f1_target"),
                "sensitivity": s_rec.get("sensitivity"),
                "specificity": s_rec.get("specificity"),
                **(
                    {
                        "pr_auc_std": s_rec.get("pr_auc_std"),
                        "roc_auc_std": s_rec.get("roc_auc_std"),
                        "balanced_accuracy_std": s_rec.get("balanced_accuracy_std"),
                        "f1_target_std": s_rec.get("f1_target_std"),
                    }
                    if "pr_auc_std" in s_rec
                    else {}
                ),
                "n_samples": s_rec.get("n_samples"),
                "n_pos": s_rec.get("n_pos"),
                "n_neg": s_rec.get("n_neg"),
                "tp": s_rec.get("tp"),
                "tn": s_rec.get("tn"),
                "fp": s_rec.get("fp"),
                "fn": s_rec.get("fn"),
            }
        )
    pd.DataFrame(subject_rows).to_csv(exp_dir / "per_subject_metrics.csv", index=False)

    oof_tmp = exp_dir / "oof_predictions.tmp.npz"
    np.savez_compressed(
        oof_tmp,
        subject_id=np.concatenate([p["subject_id"] for p in preds_sorted], axis=0),
        fold=np.concatenate([p["fold"] for p in preds_sorted], axis=0),
        trial_idx=np.concatenate([p["trial_idx"] for p in preds_sorted], axis=0),
        trial_uid=np.concatenate([p["trial_uid"] for p in preds_sorted], axis=0),
        y_true=all_test_y,
        y_prob=all_test_prob,
        y_pred=all_test_pred,
        threshold=np.concatenate([p["threshold"] for p in preds_sorted], axis=0),
    )
    oof_tmp.replace(exp_dir / "oof_predictions.npz")

    return summary_payload


def _discover_completed_experiment_summaries(protocol_dir: Path) -> Dict[str, Dict[str, Any]]:
    summaries: Dict[str, Dict[str, Any]] = {}
    if not protocol_dir.is_dir():
        return summaries

    ordered_names = list(CANONICAL_EXPERIMENT_ORDER) + sorted(
        p.name
        for p in protocol_dir.iterdir()
        if p.is_dir() and p.name not in CANONICAL_EXPERIMENT_ORDER
    )
    for name in ordered_names:
        summary_file = protocol_dir / name / "experiment_summary.json"
        if summary_file.is_file():
            summaries[name] = json.loads(summary_file.read_text(encoding="utf-8"))
    return summaries


def _compute_paired_subject_deltas(
    summaries: Mapping[str, Mapping[str, Any]],
    exp_a: str,
    exp_b: str,
) -> Optional[Dict[str, Any]]:
    """Compute paired ``exp_a - exp_b`` differences across shared test subjects and folds."""
    if exp_a not in summaries or exp_b not in summaries:
        return None
    subs_a = summaries[exp_a].get("per_subject", {})
    subs_b = summaries[exp_b].get("per_subject", {})
    shared_subs = sorted(set(subs_a) & set(subs_b), key=_natural_sort_key)
    if not shared_subs:
        return None

    subject_deltas: Dict[str, Dict[str, Any]] = {}
    per_sub_vectors: Dict[str, List[float]] = {}
    for metric in ALL_REPORTED_METRICS:
        diffs: list[float] = []
        for sub in shared_subs:
            va = subs_a[sub].get(metric)
            vb = subs_b[sub].get(metric)
            if va is not None and vb is not None and math.isfinite(float(va)) and math.isfinite(float(vb)):
                diffs.append(float(va) - float(vb))
        subject_deltas[metric] = summarize_scalar_values(diffs)
        per_sub_vectors[metric] = diffs

    folds_a = {
        (str(f.get("subject_id") or ""), int(f["fold"])): f
        for f in summaries[exp_a].get("per_fold", [])
    }
    folds_b = {
        (str(f.get("subject_id") or ""), int(f["fold"])): f
        for f in summaries[exp_b].get("per_fold", [])
    }
    shared_folds = sorted(
        set(folds_a) & set(folds_b),
        key=lambda k: (_natural_sort_key(k[0]), k[1]),
    )
    fold_deltas: Dict[str, Dict[str, Any]] = {}
    per_fold_vectors: Dict[str, List[float]] = {}
    for metric in ALL_REPORTED_METRICS:
        f_diffs: list[float] = []
        for k in shared_folds:
            va = folds_a[k]["test_pooled"].get(metric)
            vb = folds_b[k]["test_pooled"].get(metric)
            if va is not None and vb is not None and math.isfinite(float(va)) and math.isfinite(float(vb)):
                f_diffs.append(float(va) - float(vb))
        fold_deltas[metric] = summarize_scalar_values(f_diffs)
        per_fold_vectors[metric] = f_diffs

    return {
        "comparison": f"{exp_a} - {exp_b}",
        "lhs": exp_a,
        "rhs": exp_b,
        "n_shared_subjects": len(shared_subs),
        "n_shared_folds": len(shared_folds),
        "subject_paired_delta": subject_deltas,
        "fold_pooled_paired_delta": fold_deltas,
        "_subject_vectors": per_sub_vectors,
        "_fold_vectors": per_fold_vectors,
    }


def write_protocol_comparison_and_plots(
    protocol: ProtocolName,
    protocol_dir: Path,
    *,
    features_dir: Optional[Path | str] = None,
) -> Dict[str, Any]:
    """Build protocol comparison tables and generate CS5/LOSO plots without auto-picking a winner."""
    summaries = _discover_completed_experiment_summaries(protocol_dir)
    if not summaries:
        return {}

    comparison_rows = []
    for exp_name, summary in summaries.items():
        sm = summary["test"]["subject_macro"]
        fp = summary["test"]["fold_pooled"]
        op = summary["test"]["overall_trial_pooled"]
        bi = summary["best_iteration"]
        td = summary["train_duration_sec"]
        comparison_rows.append(
            {
                "protocol": protocol,
                "experiment": exp_name,
                "n_features": summary["n_features"],
                "n_folds": summary["n_folds"],
                "n_subjects": summary["n_test_subjects"],
                "subject_macro_pr_auc_mean": sm["pr_auc"]["mean"],
                "subject_macro_pr_auc_std": sm["pr_auc"]["std"],
                "subject_macro_pr_auc_median": sm["pr_auc"]["median"],
                "subject_macro_roc_auc_mean": sm["roc_auc"]["mean"],
                "subject_macro_roc_auc_std": sm["roc_auc"]["std"],
                "subject_macro_roc_auc_median": sm["roc_auc"]["median"],
                "subject_macro_bal_acc_mean": sm["balanced_accuracy"]["mean"],
                "subject_macro_bal_acc_std": sm["balanced_accuracy"]["std"],
                "subject_macro_f1_target_mean": sm["f1_target"]["mean"],
                "subject_macro_f1_target_std": sm["f1_target"]["std"],
                "fold_pooled_pr_auc_mean": fp["pr_auc"]["mean"],
                "fold_pooled_pr_auc_std": fp["pr_auc"]["std"],
                "fold_pooled_roc_auc_mean": fp["roc_auc"]["mean"],
                "fold_pooled_roc_auc_std": fp["roc_auc"]["std"],
                "overall_pooled_pr_auc": op["pr_auc"],
                "overall_pooled_roc_auc": op["roc_auc"],
                "overall_pooled_bal_acc": op["balanced_accuracy"],
                "overall_pooled_f1_target": op["f1_target"],
                "best_iteration_mean": bi["mean"],
                "best_iteration_median": bi["median"],
                "train_time_mean_sec": td["mean"],
                "train_time_std_sec": td["std"],
                "train_time_total_sec": td["total_sec"],
            }
        )
    pd.DataFrame(comparison_rows).to_csv(protocol_dir / "comparison_table.csv", index=False)

    candidate_pairs = [
        ("wavelet_w8", "wavelet_w5"),
        ("combined_w8", "combined_w5"),
        ("combined_w5", "temporal"),
        ("combined_w8", "temporal"),
        ("combined_w5", "wavelet_w5"),
        ("combined_w8", "wavelet_w8"),
        ("wavelet_w5", "temporal"),
        ("wavelet_w8", "temporal"),
        ("combined_w5_gamma", "combined_w5"),
        ("combined_w8_gamma", "combined_w8"),
    ]
    paired_deltas: Dict[str, Any] = {}
    paired_with_vectors: Dict[str, Any] = {}
    for lhs, rhs in candidate_pairs:
        delta = _compute_paired_subject_deltas(summaries, lhs, rhs)
        if delta is not None:
            key = f"{lhs}_minus_{rhs}"
            paired_with_vectors[key] = delta
            clean_delta = {k: v for k, v in delta.items() if not k.startswith("_")}
            paired_deltas[key] = clean_delta

    generated_plots: List[str] = []
    if protocol == "cs5":
        generated_plots.extend(_generate_cs5_plots(protocol_dir, summaries, paired_with_vectors))
    elif protocol == "within":
        generated_plots.extend(_generate_within_plots(protocol_dir, summaries, paired_with_vectors))
    else:
        generated_plots.extend(_generate_loso_plots(protocol_dir, summaries, paired_with_vectors))
    generated_plots.extend(
        generate_ml3_diagnostic_plots(
            protocol=protocol,
            protocol_dir=protocol_dir,
            summaries=summaries,
            features_dir=features_dir,
        )
    )

    comparison_payload = {
        "schema_version": 1,
        "protocol": protocol,
        "note": (
            "No automatic winner rule is applied; inspect subject-macro metrics, "
            "trial-pooled metrics, paired deltas, feature counts, and training times."
        ),
        "experiments": list(summaries.keys()),
        "rows": comparison_rows,
        "paired_deltas": paired_deltas,
        "plots": generated_plots,
    }
    _atomic_write_json(protocol_dir / "comparison_summary.json", comparison_payload)
    return comparison_payload


# ── Publication Visualization Suite (setup_publication_style + OKABE_ITO) ────

_EXP_COLORS: Dict[str, str] = {
    "temporal": OKABE_ITO["blue"],
    "wavelet_w5": OKABE_ITO["yellow"],
    "wavelet_w8": OKABE_ITO["orange"],
    "combined_w5": OKABE_ITO["green"],
    "combined_w8": OKABE_ITO["purple"],
    "combined_w5_gamma": OKABE_ITO["sky"],
    "combined_w8_gamma": OKABE_ITO["red"],
    "wavelet_w5_gamma": OKABE_ITO["sky"],
    "wavelet_w8_gamma": OKABE_ITO["red"],
}

_EXP_MARKERS: Dict[str, str] = {
    "temporal": "o",
    "wavelet_w5": "s",
    "wavelet_w8": "^",
    "combined_w5": "D",
    "combined_w8": "P",
    "combined_w5_gamma": "v",
    "combined_w8_gamma": "X",
}

_EXP_LABELS: Dict[str, str] = {
    "temporal": "Temporal (T)",
    "wavelet_w5": "Wavelet W5",
    "wavelet_w8": "Wavelet W8",
    "combined_w5": "Combined W5 (C5)",
    "combined_w8": "Combined W8 (C8)",
    "combined_w5_gamma": "Combined W5+Gamma",
    "combined_w8_gamma": "Combined W8+Gamma",
    "wavelet_w5_gamma": "Wavelet W5+Gamma",
    "wavelet_w8_gamma": "Wavelet W8+Gamma",
}

_WINDOW_DISPLAY: Dict[str, str] = {
    "baseline_m200_0": "[-200, 0]",
    "post_0_200": "[0, 200]",
    "post_100_300": "[100, 300]",
    "post_200_400": "[200, 400]",
    "post_300_500": "[300, 500]",
    "post_400_600": "[400, 600]",
    "post_500_700": "[500, 700]",
    "post_600_800": "[600, 800]",
}


def _exp_color(name: str) -> str:
    return _EXP_COLORS.get(name, OKABE_ITO["grey"])


def _exp_marker(name: str) -> str:
    return _EXP_MARKERS.get(name, "o")


def _exp_label(name: str) -> str:
    return _EXP_LABELS.get(name, name)


def _save_pub_png(fig: plt.Figure, out_path: Path) -> str:
    target = ensure_drive_path_ready(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target.with_name(f"{target.stem}.tmp.png")
    fig.savefig(tmp_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    tmp_path.replace(target)
    return target.name


def _generate_cs5_plots(
    protocol_dir: Path,
    summaries: Mapping[str, Mapping[str, Any]],
    paired_deltas: Mapping[str, Mapping[str, Any]],
) -> List[str]:
    setup_publication_style()
    exp_names = list(summaries.keys())
    x = np.arange(len(exp_names))
    colors = [_exp_color(e) for e in exp_names]
    labels = [_exp_label(e) for e in exp_names]
    rng = np.random.default_rng(42)
    saved_files: List[str] = []

    # 1. cs5_metric_comparison.png
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 9.6), constrained_layout=True)
    fig.suptitle("ML3 Cross-Subject 5-Fold — Trial-Pooled Metrics, Dimensionality & Train Time")

    for ax, metric, title in (
        (axes[0, 0], "pr_auc", "(A) Test Fold-Pooled PR-AUC (Mean ± SD)"),
        (axes[0, 1], "roc_auc", "(B) Test Fold-Pooled ROC-AUC (Mean ± SD)"),
    ):
        means = [summaries[e]["test"]["fold_pooled"][metric]["mean"] for e in exp_names]
        stds = [summaries[e]["test"]["fold_pooled"][metric]["std"] for e in exp_names]
        bars = ax.bar(
            x,
            means,
            width=0.58,
            yerr=stds,
            capsize=4,
            color=colors,
            alpha=0.84,
            edgecolor=OKABE_ITO["black"],
            linewidth=0.8,
            error_kw=dict(ecolor=OKABE_ITO["black"], lw=1.1),
        )
        for idx, e in enumerate(exp_names):
            fold_vals = [f["test_pooled"][metric] for f in summaries[e]["per_fold"]]
            jitter = rng.uniform(-0.10, 0.10, size=len(fold_vals))
            ax.scatter(
                np.full(len(fold_vals), idx) + jitter,
                fold_vals,
                color=OKABE_ITO["black"],
                s=22,
                zorder=4,
                alpha=0.82,
            )
            if math.isfinite(means[idx]):
                sd_v = stds[idx] if math.isfinite(stds[idx]) else 0.0
                ax.text(
                    bars[idx].get_x() + bars[idx].get_width() / 2.0,
                    min(1.03, means[idx] + sd_v + 0.012),
                    f"{means[idx]:.3f}±{sd_v:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=8.2,
                    color=OKABE_ITO["black"],
                )
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=18, ha="right")
        ax.set_title(title)
        ax.set_ylim(0.0, 1.09)
        ax.grid(axis="y")

    ax_sup = axes[1, 0]
    width = 0.18
    sup_metrics = [
        ("balanced_accuracy", "Bal Acc", OKABE_ITO["blue"]),
        ("f1_target", "F1 Target", OKABE_ITO["green"]),
        ("sensitivity", "Sensitivity", OKABE_ITO["orange"]),
        ("specificity", "Specificity", OKABE_ITO["sky"]),
    ]
    for m_i, (m_key, m_label, m_col) in enumerate(sup_metrics):
        m_means = [summaries[e]["test"]["fold_pooled"][m_key]["mean"] for e in exp_names]
        m_stds = [summaries[e]["test"]["fold_pooled"][m_key]["std"] for e in exp_names]
        offset = (m_i - 1.5) * width
        ax_sup.bar(
            x + offset,
            m_means,
            width=width,
            yerr=m_stds,
            capsize=2.5,
            color=m_col,
            edgecolor=OKABE_ITO["black"],
            linewidth=0.5,
            label=m_label,
            alpha=0.88,
        )
    ax_sup.set_xticks(x)
    ax_sup.set_xticklabels(labels, rotation=18, ha="right")
    ax_sup.set_ylim(0.0, 1.06)
    ax_sup.set_title("(C) Supporting Metrics (Fold-Pooled Mean ± SD)")
    ax_sup.legend(ncol=2, frameon=False)
    ax_sup.grid(axis="y")

    ax_cost = axes[1, 1]
    feat_counts = [summaries[e]["n_features"] for e in exp_names]
    time_means = [summaries[e]["train_duration_sec"]["mean"] for e in exp_names]
    time_stds = [summaries[e]["train_duration_sec"]["std"] for e in exp_names]

    b_feat = ax_cost.bar(
        x - 0.18,
        feat_counts,
        width=0.34,
        color=OKABE_ITO["blue"],
        alpha=0.85,
        edgecolor=OKABE_ITO["black"],
        linewidth=0.6,
        label="Features",
    )
    ax_cost.set_ylabel("Feature Count", color=OKABE_ITO["blue"])
    ax_cost.tick_params(axis="y", labelcolor=OKABE_ITO["blue"])
    for idx, val in enumerate(feat_counts):
        ax_cost.text(
            b_feat[idx].get_x() + b_feat[idx].get_width() / 2.0,
            val * 1.01,
            f"{val:,}",
            ha="center",
            va="bottom",
            fontsize=7.8,
            color=OKABE_ITO["black"],
        )

    ax_time = ax_cost.twinx()
    ax_time.spines["top"].set_visible(False)
    ax_time.bar(
        x + 0.18,
        time_means,
        width=0.34,
        yerr=time_stds,
        capsize=3.5,
        color=OKABE_ITO["orange"],
        alpha=0.85,
        edgecolor=OKABE_ITO["black"],
        linewidth=0.6,
        label="Train Time (s)",
    )
    ax_time.set_ylabel("Fold Train Time (s, Mean ± SD)", color=OKABE_ITO["orange"])
    ax_time.tick_params(axis="y", labelcolor=OKABE_ITO["orange"])
    ax_cost.set_xticks(x)
    ax_cost.set_xticklabels(labels, rotation=18, ha="right")
    ax_cost.set_title("(D) Feature Count & Per-Fold Training Time")
    ax_cost.grid(axis="y")
    saved_files.append(_save_pub_png(fig, protocol_dir / "cs5_metric_comparison.png"))

    # 2. cs5_subject_macro_comparison.png
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 9.6), constrained_layout=True)
    fig.suptitle("ML3 Cross-Subject 5-Fold — Subject-Macro Generalization (N = 39 Subjects, Equal Weight)")
    for ax, (metric, title) in zip(
        axes.ravel(),
        (
            ("pr_auc", "(A) Subject-Macro PR-AUC"),
            ("roc_auc", "(B) Subject-Macro ROC-AUC"),
            ("balanced_accuracy", "(C) Subject-Macro Balanced Accuracy"),
            ("f1_target", "(D) Subject-Macro F1-Target"),
        ),
    ):
        means = [summaries[e]["test"]["subject_macro"][metric]["mean"] for e in exp_names]
        stds = [summaries[e]["test"]["subject_macro"][metric]["std"] for e in exp_names]
        medians = [summaries[e]["test"]["subject_macro"][metric]["median"] for e in exp_names]
        ax.bar(
            x,
            means,
            width=0.56,
            yerr=stds,
            capsize=4,
            color=colors,
            alpha=0.76,
            edgecolor=OKABE_ITO["black"],
            linewidth=0.8,
            error_kw=dict(ecolor=OKABE_ITO["black"], lw=1.1),
        )
        ax.scatter(
            x,
            medians,
            marker="D",
            color=OKABE_ITO["yellow"],
            edgecolor=OKABE_ITO["black"],
            s=46,
            zorder=5,
            label="Median across subjects",
        )
        for idx, e in enumerate(exp_names):
            sub_vals = [
                float(s[metric])
                for s in summaries[e]["per_subject"].values()
                if s.get(metric) is not None and math.isfinite(float(s[metric]))
            ]
            jitter = rng.uniform(-0.16, 0.16, size=len(sub_vals))
            ax.scatter(
                np.full(len(sub_vals), idx) + jitter,
                sub_vals,
                color=OKABE_ITO["black"],
                s=14,
                alpha=0.45,
                zorder=4,
            )
            if math.isfinite(means[idx]):
                sd_v = stds[idx] if math.isfinite(stds[idx]) else 0.0
                ax.text(
                    idx,
                    min(1.04, means[idx] + sd_v + 0.014),
                    f"{means[idx]:.3f}±{sd_v:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=8.2,
                    color=OKABE_ITO["black"],
                )
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=18, ha="right")
        ax.set_ylim(0.0, 1.10)
        ax.set_title(title)
        ax.legend(loc="lower right", frameon=False)
        ax.grid(axis="y")
    saved_files.append(_save_pub_png(fig, protocol_dir / "cs5_subject_macro_comparison.png"))

    # 3. cs5_fold_profiles.png (Discrete Fold 0..4 ticks + distinct Okabe-Ito markers)
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 9.2), constrained_layout=True)
    fig.suptitle("ML3 Cross-Subject 5-Fold — Consistency Across Held-Out Subject Partitions")
    all_fold_ids = sorted(
        {int(r["fold"]) for e in exp_names for r in summaries[e]["per_fold"]}
    )
    for ax, field_kind, title, ylabel in (
        (axes[0, 0], "pr_auc", "(A) Test PR-AUC by Partition", "PR-AUC"),
        (axes[0, 1], "roc_auc", "(B) Test ROC-AUC by Partition", "ROC-AUC"),
        (axes[1, 0], "balanced_accuracy", "(C) Test Balanced Accuracy by Partition", "Balanced Accuracy"),
        (axes[1, 1], "best_iteration", "(D) Early-Stopping Best Iteration", "Trees Selected (Val AUCPR)"),
    ):
        for e in exp_names:
            f_recs = sorted(summaries[e]["per_fold"], key=lambda r: int(r["fold"]))
            f_ids = [int(r["fold"]) for r in f_recs]
            if field_kind == "best_iteration":
                y_vals = [int(r["best_iteration"]) + 1 for r in f_recs]
            else:
                y_vals = [float(r["test_pooled"][field_kind]) for r in f_recs]
            ax.plot(
                f_ids,
                y_vals,
                marker=_exp_marker(e),
                markersize=6.2,
                linewidth=1.9,
                color=_exp_color(e),
                label=_exp_label(e),
            )
        if field_kind == "best_iteration":
            max_trees = int(summaries[exp_names[0]].get("model_config", {}).get("n_estimators", 2000))
            ax.axhline(
                max_trees,
                color=OKABE_ITO["red"],
                ls=":",
                lw=1.0,
                alpha=0.75,
                label=f"Max trees ({max_trees})",
            )
        ax.set_xticks(all_fold_ids)
        ax.set_xticklabels([f"Fold {k}" for k in all_fold_ids])
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True)
        ax.legend(frameon=False, ncol=2 if len(exp_names) > 4 else 1)
    saved_files.append(_save_pub_png(fig, protocol_dir / "cs5_fold_profiles.png"))

    # 4. cs5_window_paired_deltas.png
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 6.2), constrained_layout=True)
    fig.suptitle("ML3 Cross-Subject 5-Fold — Paired Per-Subject Differences (N = 39 Subjects)")
    delta_keys = [
        k
        for k in (
            "wavelet_w8_minus_wavelet_w5",
            "combined_w8_minus_combined_w5",
            "combined_w5_minus_temporal",
            "combined_w8_minus_temporal",
            "combined_w5_minus_wavelet_w5",
            "combined_w8_minus_wavelet_w8",
            "combined_w5_gamma_minus_combined_w5",
            "combined_w8_gamma_minus_combined_w8",
        )
        if k in paired_deltas
    ]
    for ax, metric, title in (
        (axes[0], "pr_auc", "(A) Paired Δ PR-AUC (LHS − RHS)"),
        (axes[1], "roc_auc", "(B) Paired Δ ROC-AUC (LHS − RHS)"),
    ):
        ax.axhline(0.0, color=OKABE_ITO["red"], linestyle="--", linewidth=1.15)
        if delta_keys:
            dx = np.arange(len(delta_keys))
            d_labels = [paired_deltas[k]["comparison"] for k in delta_keys]
            d_means = [paired_deltas[k]["subject_paired_delta"][metric]["mean"] for k in delta_keys]
            d_stds = [paired_deltas[k]["subject_paired_delta"][metric]["std"] for k in delta_keys]
            bar_cols = [
                OKABE_ITO["green"] if (math.isfinite(m) and m >= 0.0) else OKABE_ITO["orange"]
                for m in d_means
            ]
            ax.bar(
                dx,
                d_means,
                width=0.56,
                yerr=d_stds,
                capsize=4,
                color=bar_cols,
                alpha=0.78,
                edgecolor=OKABE_ITO["black"],
                linewidth=0.8,
            )
            for idx, k in enumerate(delta_keys):
                vec = paired_deltas[k]["_subject_vectors"][metric]
                jitter = rng.uniform(-0.15, 0.15, size=len(vec))
                ax.scatter(
                    np.full(len(vec), idx) + jitter,
                    vec,
                    color=OKABE_ITO["black"],
                    s=14,
                    alpha=0.45,
                    zorder=4,
                )
                if math.isfinite(d_means[idx]):
                    ax.text(
                        idx,
                        d_means[idx],
                        f"{d_means[idx]:+.4f}",
                        ha="center",
                        va="bottom" if d_means[idx] >= 0 else "top",
                        fontsize=8.2,
                        fontweight="bold",
                        color=OKABE_ITO["black"],
                    )
            ax.set_xticks(dx)
            ax.set_xticklabels(d_labels, rotation=22, ha="right")
        else:
            ax.text(
                0.5,
                0.5,
                "Run ≥2 comparable scenarios to view paired deltas",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
        ax.set_ylabel(f"Δ {metric.upper()}")
        ax.set_title(title)
        ax.grid(axis="y")
    saved_files.append(_save_pub_png(fig, protocol_dir / "cs5_window_paired_deltas.png"))
    return saved_files


def _generate_loso_plots(
    protocol_dir: Path,
    summaries: Mapping[str, Mapping[str, Any]],
    paired_deltas: Mapping[str, Mapping[str, Any]],
) -> List[str]:
    setup_publication_style()
    exp_names = list(summaries.keys())
    x = np.arange(len(exp_names))
    colors = [_exp_color(e) for e in exp_names]
    labels = [_exp_label(e) for e in exp_names]
    rng = np.random.default_rng(42)
    saved_files: List[str] = []

    # 1. loso_metric_distribution.png
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 9.6), constrained_layout=True)
    fig.suptitle("ML3 LOSO (N = 39 Held-Out Subjects) — Distributions, Dimensionality & Train Time")
    for ax, metric, title in (
        (axes[0, 0], "pr_auc", "(A) LOSO Per-Subject PR-AUC"),
        (axes[0, 1], "roc_auc", "(B) LOSO Per-Subject ROC-AUC"),
        (axes[1, 0], "balanced_accuracy", "(C) LOSO Per-Subject Balanced Accuracy"),
    ):
        series = []
        for e in exp_names:
            vals = [
                float(s[metric])
                for s in summaries[e]["per_subject"].values()
                if s.get(metric) is not None and math.isfinite(float(s[metric]))
            ]
            series.append(vals if vals else [float("nan")])
        bp = ax.boxplot(
            series,
            positions=x,
            widths=0.46,
            patch_artist=True,
            showfliers=False,
            medianprops=dict(color=OKABE_ITO["black"], lw=1.4),
        )
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c)
            patch.set_edgecolor(OKABE_ITO["black"])
            patch.set_alpha(0.60)
        for idx, e in enumerate(exp_names):
            vals = series[idx]
            mean_v = summaries[e]["test"]["subject_macro"][metric]["mean"]
            std_v = summaries[e]["test"]["subject_macro"][metric]["std"]
            jitter = rng.uniform(-0.14, 0.14, size=len(vals))
            ax.scatter(
                np.full(len(vals), idx) + jitter,
                vals,
                color=OKABE_ITO["black"],
                s=16,
                alpha=0.50,
                zorder=4,
            )
            ax.errorbar(
                [idx],
                [mean_v],
                yerr=[std_v],
                fmt="D",
                color=OKABE_ITO["yellow"],
                markeredgecolor=OKABE_ITO["black"],
                ecolor=OKABE_ITO["black"],
                capsize=4,
                zorder=5,
            )
            if math.isfinite(mean_v):
                ax.text(
                    idx,
                    min(1.04, max(vals) + 0.02),
                    f"{mean_v:.3f}±{std_v:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=8.2,
                    color=OKABE_ITO["black"],
                )
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15, ha="right")
        ax.set_ylim(0.0, 1.09)
        ax.set_title(title)
        ax.grid(axis="y")

    ax_cost = axes[1, 1]
    feat_counts = [summaries[e]["n_features"] for e in exp_names]
    time_means = [summaries[e]["train_duration_sec"]["mean"] for e in exp_names]
    time_stds = [summaries[e]["train_duration_sec"]["std"] for e in exp_names]
    b_feat = ax_cost.bar(
        x - 0.18,
        feat_counts,
        width=0.34,
        color=OKABE_ITO["blue"],
        alpha=0.85,
        edgecolor=OKABE_ITO["black"],
        linewidth=0.6,
    )
    ax_cost.set_ylabel("Feature Count", color=OKABE_ITO["blue"])
    ax_cost.tick_params(axis="y", labelcolor=OKABE_ITO["blue"])
    for idx, val in enumerate(feat_counts):
        ax_cost.text(
            b_feat[idx].get_x() + b_feat[idx].get_width() / 2.0,
            val * 1.01,
            f"{val:,}",
            ha="center",
            va="bottom",
            fontsize=7.8,
        )
    ax_time = ax_cost.twinx()
    ax_time.spines["top"].set_visible(False)
    ax_time.bar(
        x + 0.18,
        time_means,
        width=0.34,
        yerr=time_stds,
        capsize=3.5,
        color=OKABE_ITO["orange"],
        alpha=0.85,
        edgecolor=OKABE_ITO["black"],
        linewidth=0.6,
    )
    ax_time.set_ylabel("Fold Train Time (s, Mean ± SD)", color=OKABE_ITO["orange"])
    ax_time.tick_params(axis="y", labelcolor=OKABE_ITO["orange"])
    ax_cost.set_xticks(x)
    ax_cost.set_xticklabels(labels, rotation=15, ha="right")
    ax_cost.set_title("(D) Feature Count & Per-Subject Fold Train Time")
    ax_cost.grid(axis="y")
    saved_files.append(_save_pub_png(fig, protocol_dir / "loso_metric_distribution.png"))

    # 2. loso_per_subject_metrics.png
    all_subjects = sorted(
        {sub for e in exp_names for sub in summaries[e]["per_subject"].keys()},
        key=_natural_sort_key,
    )
    sx = np.arange(len(all_subjects))
    fig, axes = plt.subplots(2, 1, figsize=(15.5, 8.6), constrained_layout=True, sharex=True)
    fig.suptitle("ML3 LOSO — Per-Subject Generalization Profile Across 39 Subjects")
    for ax, metric, title in (
        (axes[0], "pr_auc", "(A) Held-Out Subject PR-AUC"),
        (axes[1], "roc_auc", "(B) Held-Out Subject ROC-AUC"),
    ):
        for e in exp_names:
            sub_map = summaries[e]["per_subject"]
            y_vals = [
                float(sub_map[s][metric]) if s in sub_map and sub_map[s].get(metric) is not None else float("nan")
                for s in all_subjects
            ]
            ax.plot(
                sx,
                y_vals,
                marker=_exp_marker(e),
                markersize=4.6,
                linewidth=1.5,
                color=_exp_color(e),
                label=_exp_label(e),
                alpha=0.92,
            )
        ax.set_ylabel(metric.upper())
        ax.set_title(title)
        ax.grid(True)
        ax.legend(loc="lower right", ncol=len(exp_names), frameon=False)
    axes[1].set_xticks(sx)
    axes[1].set_xticklabels(all_subjects, rotation=55, ha="right", fontsize=8.0)
    saved_files.append(_save_pub_png(fig, protocol_dir / "loso_per_subject_metrics.png"))

    # 3. loso_c5_vs_c8_deltas.png
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 6.2), constrained_layout=True)
    fig.suptitle("ML3 LOSO — Paired Per-Subject Differences (C5 vs C8 & Combined vs Temporal)")
    loso_delta_keys = [
        k
        for k in (
            "combined_w8_minus_combined_w5",
            "combined_w5_minus_temporal",
            "combined_w8_minus_temporal",
        )
        if k in paired_deltas
    ]
    for ax, metric, title in (
        (axes[0], "pr_auc", "(A) LOSO Paired Δ PR-AUC"),
        (axes[1], "roc_auc", "(B) LOSO Paired Δ ROC-AUC"),
    ):
        ax.axhline(0.0, color=OKABE_ITO["red"], linestyle="--", linewidth=1.15)
        if loso_delta_keys:
            dx = np.arange(len(loso_delta_keys))
            d_labels = [paired_deltas[k]["comparison"] for k in loso_delta_keys]
            d_means = [paired_deltas[k]["subject_paired_delta"][metric]["mean"] for k in loso_delta_keys]
            d_stds = [paired_deltas[k]["subject_paired_delta"][metric]["std"] for k in loso_delta_keys]
            bar_cols = [
                OKABE_ITO["green"] if (math.isfinite(m) and m >= 0.0) else OKABE_ITO["orange"]
                for m in d_means
            ]
            ax.bar(
                dx,
                d_means,
                width=0.52,
                yerr=d_stds,
                capsize=4,
                color=bar_cols,
                alpha=0.78,
                edgecolor=OKABE_ITO["black"],
                linewidth=0.8,
            )
            for idx, k in enumerate(loso_delta_keys):
                vec = paired_deltas[k]["_subject_vectors"][metric]
                jitter = rng.uniform(-0.14, 0.14, size=len(vec))
                ax.scatter(
                    np.full(len(vec), idx) + jitter,
                    vec,
                    color=OKABE_ITO["black"],
                    s=15,
                    alpha=0.48,
                    zorder=4,
                )
                if math.isfinite(d_means[idx]):
                    ax.text(
                        idx,
                        d_means[idx],
                        f"{d_means[idx]:+.4f}±{d_stds[idx]:.4f}",
                        ha="center",
                        va="bottom" if d_means[idx] >= 0 else "top",
                        fontsize=8.2,
                        fontweight="bold",
                    )
            ax.set_xticks(dx)
            ax.set_xticklabels(d_labels, rotation=18, ha="right")
        else:
            ax.text(
                0.5,
                0.5,
                "Run ≥2 LOSO experiments (temporal, combined_w5, combined_w8) for paired deltas",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
        ax.set_ylabel(f"Δ {metric.upper()}")
        ax.set_title(title)
        ax.grid(axis="y")
    saved_files.append(_save_pub_png(fig, protocol_dir / "loso_c5_vs_c8_deltas.png"))
    return saved_files


def _generate_within_plots(
    protocol_dir: Path,
    summaries: Mapping[str, Mapping[str, Any]],
    paired_deltas: Mapping[str, Mapping[str, Any]],
) -> List[str]:
    del paired_deltas  # reserved for multi-experiment within-subject comparisons
    setup_publication_style()
    exp_names = list(summaries.keys())
    x = np.arange(len(exp_names))
    colors = [_exp_color(e) for e in exp_names]
    labels = [_exp_label(e) for e in exp_names]
    rng = np.random.default_rng(42)
    saved_files: List[str] = []

    n_subs = max(int(summaries[e]["n_test_subjects"]) for e in exp_names)

    # 1. within_metric_distribution.png
    fig, axes = plt.subplots(2, 2, figsize=(13.6, 9.2), constrained_layout=True)
    fig.suptitle(
        f"ML3 Within-Subject 5-Fold (N = {n_subs} Subjects) — Metric Distributions & Train Time"
    )
    for ax, metric, title in (
        (axes[0, 0], "pr_auc", "(A) Within-Subject Per-Subject PR-AUC (5-Fold Mean)"),
        (axes[0, 1], "roc_auc", "(B) Within-Subject Per-Subject ROC-AUC (5-Fold Mean)"),
        (axes[1, 0], "balanced_accuracy", "(C) Within-Subject Balanced Accuracy (5-Fold Mean)"),
    ):
        series = []
        for e in exp_names:
            vals = [
                float(s[metric])
                for s in summaries[e]["per_subject"].values()
                if s.get(metric) is not None and math.isfinite(float(s[metric]))
            ]
            series.append(vals if vals else [float("nan")])
        bp = ax.boxplot(
            series,
            positions=x,
            widths=0.42,
            patch_artist=True,
            showfliers=False,
            medianprops=dict(color=OKABE_ITO["black"], lw=1.4),
        )
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c)
            patch.set_edgecolor(OKABE_ITO["black"])
            patch.set_alpha(0.60)
        for idx, e in enumerate(exp_names):
            vals = series[idx]
            mean_v = summaries[e]["test"]["subject_macro"][metric]["mean"]
            std_v = summaries[e]["test"]["subject_macro"][metric]["std"]
            jitter = rng.uniform(-0.12, 0.12, size=len(vals))
            ax.scatter(
                np.full(len(vals), idx) + jitter,
                vals,
                color=OKABE_ITO["black"],
                s=18,
                alpha=0.55,
                zorder=4,
            )
            ax.errorbar(
                [idx],
                [mean_v],
                yerr=[std_v if math.isfinite(std_v) else 0.0],
                fmt="D",
                color=OKABE_ITO["yellow"],
                markeredgecolor=OKABE_ITO["black"],
                ecolor=OKABE_ITO["black"],
                capsize=4,
                zorder=5,
            )
            if math.isfinite(mean_v):
                sd_v = std_v if math.isfinite(std_v) else 0.0
                ax.text(
                    idx,
                    min(1.04, max(vals) + 0.02),
                    f"{mean_v:.3f}±{sd_v:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=8.4,
                    color=OKABE_ITO["black"],
                )
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=12, ha="right")
        ax.set_ylim(0.0, 1.09)
        ax.set_title(title)
        ax.grid(axis="y")

    ax_cost = axes[1, 1]
    feat_counts = [summaries[e]["n_features"] for e in exp_names]
    time_means = [summaries[e]["train_duration_sec"]["mean"] for e in exp_names]
    time_stds = [summaries[e]["train_duration_sec"]["std"] for e in exp_names]
    b_feat = ax_cost.bar(
        x - 0.18,
        feat_counts,
        width=0.34,
        color=OKABE_ITO["blue"],
        alpha=0.85,
        edgecolor=OKABE_ITO["black"],
        linewidth=0.6,
    )
    ax_cost.set_ylabel("Feature Count", color=OKABE_ITO["blue"])
    ax_cost.tick_params(axis="y", labelcolor=OKABE_ITO["blue"])
    for idx, val in enumerate(feat_counts):
        ax_cost.text(
            b_feat[idx].get_x() + b_feat[idx].get_width() / 2.0,
            val * 1.01,
            f"{val:,}",
            ha="center",
            va="bottom",
            fontsize=8.0,
        )
    ax_time = ax_cost.twinx()
    ax_time.spines["top"].set_visible(False)
    ax_time.bar(
        x + 0.18,
        time_means,
        width=0.34,
        yerr=time_stds,
        capsize=3.5,
        color=OKABE_ITO["orange"],
        alpha=0.85,
        edgecolor=OKABE_ITO["black"],
        linewidth=0.6,
    )
    ax_time.set_ylabel("Per-Fold Train Time (s, Mean ± SD)", color=OKABE_ITO["orange"])
    ax_time.tick_params(axis="y", labelcolor=OKABE_ITO["orange"])
    ax_cost.set_xticks(x)
    ax_cost.set_xticklabels(labels, rotation=12, ha="right")
    ax_cost.set_title("(D) Feature Count & Single-Fold Train Time")
    ax_cost.grid(axis="y")
    saved_files.append(_save_pub_png(fig, protocol_dir / "within_metric_distribution.png"))

    # 2. within_per_subject_metrics.png
    all_subjects = sorted(
        {sub for e in exp_names for sub in summaries[e]["per_subject"].keys()},
        key=_natural_sort_key,
    )
    sx = np.arange(len(all_subjects))
    fig, axes = plt.subplots(2, 1, figsize=(15.5, 8.6), constrained_layout=True, sharex=True)
    fig.suptitle(
        f"ML3 Within-Subject 5-Fold — Per-Subject Profile (Mean ± SD Across 5 Folds, N = {len(all_subjects)})"
    )
    for ax, metric, title in (
        (axes[0], "pr_auc", "(A) Within-Subject Per-Subject PR-AUC (5-Fold Mean ± SD)"),
        (axes[1], "roc_auc", "(B) Within-Subject Per-Subject ROC-AUC (5-Fold Mean ± SD)"),
    ):
        for e in exp_names:
            sub_map = summaries[e]["per_subject"]
            y_vals = [
                float(sub_map[s][metric]) if s in sub_map and sub_map[s].get(metric) is not None else float("nan")
                for s in all_subjects
            ]
            y_errs = [
                float(sub_map[s].get(f"{metric}_std", 0.0))
                if s in sub_map and sub_map[s].get(f"{metric}_std") is not None and math.isfinite(float(sub_map[s].get(f"{metric}_std", 0.0)))
                else 0.0
                for s in all_subjects
            ]
            col = _exp_color(e)
            macro_mean = summaries[e]["test"]["subject_macro"][metric]["mean"]
            ax.errorbar(
                sx,
                y_vals,
                yerr=y_errs,
                marker=_exp_marker(e),
                markersize=4.8,
                linewidth=1.5,
                capsize=2.5,
                color=col,
                label=f"{_exp_label(e)} (Cohort Mean={macro_mean:.3f})",
                alpha=0.92,
            )
            if math.isfinite(macro_mean):
                ax.axhline(macro_mean, color=col, ls="--", lw=1.0, alpha=0.65)
        ax.set_ylabel(metric.upper())
        ax.set_ylim(0.0, 1.04)
        ax.set_title(title)
        ax.grid(True)
        ax.legend(loc="lower right", ncol=max(1, len(exp_names)), frameon=False)
    axes[1].set_xticks(sx)
    axes[1].set_xticklabels(all_subjects, rotation=55, ha="right", fontsize=8.0)
    saved_files.append(_save_pub_png(fig, protocol_dir / "within_per_subject_metrics.png"))
    return saved_files


def backfill_learning_curves_from_checkpoints(
    *,
    protocol_dir: Path | str,
    features_dir: Path | str,
    splits_dir: Path | str,
    protocol: ProtocolName | str,
    device: str = "cuda",
    n_curve_points: int = 20,
    verbose: bool = True,
) -> int:
    """Reconstruct ``learning_curve.json`` & ``train_pooled`` from existing ``model.json`` checkpoints.

    This allows inspecting training vs validation overfitting curves on already
    completed runs (such as CS5) in a few seconds without retraining any fold.
    """
    p_dir = Path(protocol_dir)
    norm_proto = normalize_protocol_name(protocol)
    summaries = _discover_completed_experiment_summaries(p_dir)
    if not summaries:
        return 0

    missing_exps: list[str] = []
    for exp_name in summaries:
        exp_dir = p_dir / exp_name
        for fold_dir in _iter_fold_checkpoint_dirs(exp_dir):
            if not (fold_dir / "model.json").is_file():
                continue
            lc_ok = (fold_dir / "learning_curve.json").is_file()
            m_path = fold_dir / "metrics.json"
            m_data = json.loads(m_path.read_text(encoding="utf-8")) if m_path.is_file() else {}
            if not lc_ok or "train_pooled" not in m_data:
                if exp_name not in missing_exps:
                    missing_exps.append(exp_name)

    if not missing_exps:
        return 0

    all_folds, _ = load_ml3_splits(splits_dir, norm_proto)
    fold_by_key = {f.fold_key: f for f in all_folds}
    req_groups = required_feature_groups_for_experiments(missing_exps)
    needs_gamma = any(FEATURE_SCENARIO_SPECS[e].uses_gamma for e in missing_exps)
    dataset = load_ml3_feature_dataset(
        features_dir,
        required_groups=req_groups,
        include_gamma_support=needs_gamma,
    )

    updated_folds = 0
    cfg = ML3XGBoostConfig(device=device)
    import xgboost as xgb

    for exp_name in missing_exps:
        t_exp = time.perf_counter()
        scenario_matrix = dataset.build_scenario(exp_name)
        exp_dir = p_dir / exp_name
        fold_metrics_list = []
        fold_preds_list = []

        # Build ONE DMatrix per experiment (shared across all folds & tree checkpoints!)
        # Reusing a single immutable DMatrix enables XGBoost's internal prediction cache
        # and avoids rebuilding ~700MB DMatrix buffers per fold/round.
        full_dm = xgb.DMatrix(scenario_matrix.values)
        exp_updated = 0

        for fold_dir in _iter_fold_checkpoint_dirs(exp_dir):
            m_path = fold_dir / "metrics.json"
            p_path = fold_dir / "predictions.npz"
            mod_path = fold_dir / "model.json"
            if not (m_path.is_file() and p_path.is_file() and mod_path.is_file()):
                continue

            f_metrics = json.loads(m_path.read_text(encoding="utf-8"))
            lc_path = fold_dir / "learning_curve.json"
            need_train = "train_pooled" not in f_metrics
            need_lc = not lc_path.is_file()

            if need_train or need_lc:
                fold_idx = int(f_metrics["fold"])
                f_key = str(f_metrics.get("fold_key") or fold_dir.name)
                split = fold_by_key[f_key]
                train_idx, val_idx, _ = resolve_fold_indices(
                    split, scenario_matrix.subject_ids, scenario_matrix.trial_uids
                )
                y_tr = scenario_matrix.targets[train_idx]
                y_va = scenario_matrix.targets[val_idx]

                best_iter = int(f_metrics["best_iteration"])
                model = ML3XGBoostModel.load_model(mod_path, config=cfg, best_iteration=best_iter)
                assert model.estimator_ is not None
                booster = model.estimator_.get_booster()
                booster.set_param({"device": device})

                if need_train:
                    p_all_best = np.asarray(
                        booster.predict(full_dm, iteration_range=(0, best_iter + 1)),
                        dtype=np.float64,
                    )
                    f_metrics["train_pooled"] = _format_metric_with_confusion(
                        compute_binary_metrics(
                            y_tr,
                            p_all_best[train_idx],
                            threshold=float(f_metrics["threshold"]),
                        )
                    )
                    _atomic_write_json(m_path, f_metrics)

                if need_lc:
                    total_rounds = int(booster.num_boosted_rounds())
                    grid = np.unique(
                        np.concatenate(
                            [
                                np.linspace(
                                    1,
                                    max(1, total_rounds),
                                    num=min(n_curve_points, max(1, total_rounds)),
                                    dtype=int,
                                ),
                                np.array([min(total_rounds, best_iter + 1), total_rounds], dtype=int),
                            ]
                        )
                    )
                    tr_curve: list[float] = []
                    va_curve: list[float] = []
                    for ntree in grid:
                        p_all = booster.predict(full_dm, iteration_range=(0, int(ntree)))
                        tr_curve.append(float(average_precision_score(y_tr, p_all[train_idx])))
                        va_curve.append(float(average_precision_score(y_va, p_all[val_idx])))
                    _atomic_write_json(
                        lc_path,
                        {
                            "fold": fold_idx,
                            "best_iteration": best_iter,
                            "rounds": [int(v) for v in grid.tolist()],
                            "train_aucpr": tr_curve,
                            "val_aucpr": va_curve,
                        },
                    )

                updated_folds += 1
                exp_updated += 1
                del booster, model

            fold_metrics_list.append(f_metrics)
            with np.load(p_path, allow_pickle=False) as payload:
                fold_preds_list.append({k: np.asarray(payload[k]) for k in payload.files})

        if verbose:
            print(
                f"[Backfill][{norm_proto.upper()}][{exp_name}] "
                f"updated {exp_updated} fold(s) in {time.perf_counter() - t_exp:.1f}s"
            )

        if fold_metrics_list:
            aggregate_experiment_results(
                protocol=norm_proto,
                experiment=exp_name,
                exp_dir=exp_dir,
                model_config=cfg,
                fold_metrics_list=fold_metrics_list,
                fold_preds_list=fold_preds_list,
            )
        del full_dm, scenario_matrix
        gc.collect()

    return updated_folds


def generate_ml3_diagnostic_plots(
    *,
    protocol: ProtocolName | str,
    protocol_dir: Path | str,
    summaries: Optional[Mapping[str, Mapping[str, Any]]] = None,
    features_dir: Optional[Path | str] = None,
) -> List[str]:
    """Generate Overfitting, ROC/PR, Feature Gain Breakdown, and Calibration plots."""
    setup_publication_style()
    norm_proto = normalize_protocol_name(protocol)
    p_dir = Path(protocol_dir)
    resolved_summaries = (
        dict(summaries) if summaries is not None else _discover_completed_experiment_summaries(p_dir)
    )
    if not resolved_summaries:
        return []

    exp_names = list(resolved_summaries.keys())
    saved: List[str] = []

    # ── Diagnostic 1: Overfitting & Learning Curves ({proto}_overfitting_diagnostics.png) ──
    curves_by_exp: Dict[str, List[Dict[str, Any]]] = {}
    for e in exp_names:
        f_curves = []
        for fold_dir in _iter_fold_checkpoint_dirs(p_dir / e):
            lc_file = fold_dir / "learning_curve.json"
            if lc_file.is_file():
                f_curves.append(json.loads(lc_file.read_text(encoding="utf-8")))
        if f_curves:
            curves_by_exp[e] = f_curves

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.2), constrained_layout=True)
    fig.suptitle(f"ML3 {norm_proto.upper()} — Training Overfitting & Generalization Diagnostics")

    ax_lc = axes[0]
    if curves_by_exp:
        for e, lc_list in curves_by_exp.items():
            col = _exp_color(e)
            max_r = max(max(c["rounds"]) for c in lc_list)
            common_r = np.linspace(1, max_r, num=min(120, max_r))
            tr_interp = np.vstack([np.interp(common_r, c["rounds"], c["train_aucpr"]) for c in lc_list])
            va_interp = np.vstack([np.interp(common_r, c["rounds"], c["val_aucpr"]) for c in lc_list])
            tr_mean = tr_interp.mean(axis=0)
            va_mean = va_interp.mean(axis=0)
            va_std = va_interp.std(axis=0, ddof=1 if len(lc_list) > 1 else 0)

            ax_lc.plot(common_r, tr_mean, ls="--", lw=1.35, color=col, alpha=0.75)
            ax_lc.plot(common_r, va_mean, ls="-", lw=2.0, color=col, label=f"{_exp_label(e)} (Val — / Train --)")
            ax_lc.fill_between(common_r, va_mean - va_std, va_mean + va_std, color=col, alpha=0.14)
            mean_best = float(np.mean([int(c["best_iteration"]) + 1 for c in lc_list]))
            ax_lc.axvline(mean_best, color=col, ls=":", lw=1.0, alpha=0.75)
        ax_lc.set_xlabel("Boosting Trees")
        ax_lc.set_ylabel("PR-AUC")
        ax_lc.set_ylim(0.25, 1.03)
        ax_lc.legend(fontsize=7.4, frameon=False, loc="lower right")
    else:
        ax_lc.text(
            0.5,
            0.5,
            "Run backfill_learning_curves_from_checkpoints()\nto reconstruct Train/Val curves from model.json",
            ha="center",
            va="center",
            transform=ax_lc.transAxes,
        )
    ax_lc.set_title("(A) Train (--) vs Val (—) Boosting PR-AUC")
    ax_lc.grid(True)

    # Panel (B) & (C): Train vs Validation vs Test Split Comparison
    x = np.arange(len(exp_names))
    w = 0.25
    for ax, metric, title in (
        (axes[1], "pr_auc", "(B) Train vs Val vs Test PR-AUC Gap"),
        (axes[2], "balanced_accuracy", "(C) Train vs Val vs Test Balanced Accuracy Gap"),
    ):
        tr_means = []
        for e in exp_names:
            tr_block = resolved_summaries[e].get("train", {}).get("fold_pooled", {})
            tr_means.append(tr_block.get(metric, {}).get("mean", float("nan")))
        va_means = [resolved_summaries[e]["validation"]["fold_pooled"][metric]["mean"] for e in exp_names]
        te_means = [resolved_summaries[e]["test"]["fold_pooled"][metric]["mean"] for e in exp_names]
        te_stds = [resolved_summaries[e]["test"]["fold_pooled"][metric]["std"] for e in exp_names]

        if any(math.isfinite(v) for v in tr_means):
            ax.bar(x - w, tr_means, width=w, color=OKABE_ITO["sky"], edgecolor=OKABE_ITO["black"], lw=0.6, label="Train")
        ax.bar(x, va_means, width=w, color=OKABE_ITO["yellow"], edgecolor=OKABE_ITO["black"], lw=0.6, label="Validation")
        ax.bar(
            x + w,
            te_means,
            width=w,
            yerr=te_stds,
            capsize=3,
            color=OKABE_ITO["green"],
            edgecolor=OKABE_ITO["black"],
            lw=0.6,
            label="Test",
        )
        for idx in range(len(exp_names)):
            if math.isfinite(va_means[idx]) and math.isfinite(te_means[idx]):
                gap = va_means[idx] - te_means[idx]
                top_y = max(
                    [v for v in (tr_means[idx], va_means[idx], te_means[idx]) if math.isfinite(v)]
                )
                ax.text(
                    idx,
                    min(1.04, top_y + 0.025),
                    f"Val−Test:\n{gap:+.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=7.3,
                    color=OKABE_ITO["black"],
                )
        ax.set_xticks(x)
        ax.set_xticklabels([_exp_label(e) for e in exp_names], rotation=18, ha="right")
        ax.set_ylim(0.0, 1.12)
        ax.set_title(title)
        ax.legend(frameon=False, loc="lower right")
        ax.grid(axis="y")

    saved.append(_save_pub_png(fig, p_dir / f"{norm_proto}_overfitting_diagnostics.png"))

    # ── Diagnostic 2: Out-of-Fold ROC & PR Curves ({proto}_roc_pr_curves.png) ──
    oof_available = [e for e in exp_names if (p_dir / e / "oof_predictions.npz").is_file()]
    if oof_available:
        fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.4), constrained_layout=True)
        fig.suptitle(f"ML3 {norm_proto.upper()} — Out-of-Fold Precision–Recall & ROC Curves")
        ax_pr, ax_roc = axes
        prevalence: Optional[float] = None

        for e in oof_available:
            with np.load(p_dir / e / "oof_predictions.npz", allow_pickle=False) as oof:
                y_true = oof["y_true"].astype(np.int64)
                y_prob = oof["y_prob"].astype(np.float64)
            if len(np.unique(y_true)) < 2:
                continue
            if prevalence is None:
                prevalence = float(np.mean(y_true == 1))
            prec, rec, _ = precision_recall_curve(y_true, y_prob)
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            ap = resolved_summaries[e]["test"]["subject_macro"]["pr_auc"]["mean"]
            auc_v = resolved_summaries[e]["test"]["subject_macro"]["roc_auc"]["mean"]
            ax_pr.plot(rec, prec, lw=2.0, color=_exp_color(e), label=f"{_exp_label(e)} (Macro AP={ap:.3f})")
            ax_roc.plot(fpr, tpr, lw=2.0, color=_exp_color(e), label=f"{_exp_label(e)} (Macro AUC={auc_v:.3f})")

        if prevalence is not None:
            ax_pr.axhline(
                prevalence,
                color=OKABE_ITO["grey"],
                ls="--",
                lw=1.0,
                label=f"Target prevalence ({prevalence:.3f})",
            )
        ax_pr.set_xlabel("Recall (Sensitivity)")
        ax_pr.set_ylabel("Precision")
        ax_pr.set_xlim(0.0, 1.0)
        ax_pr.set_ylim(0.0, 1.02)
        ax_pr.set_title("(A) Out-of-Fold Precision–Recall Curve")
        ax_pr.legend(loc="lower left", frameon=False)
        ax_pr.grid(True)

        ax_roc.plot([0, 1], [0, 1], color=OKABE_ITO["grey"], ls="--", lw=1.0, label="Chance (0.500)")
        ax_roc.set_xlabel("False Positive Rate (1 − Specificity)")
        ax_roc.set_ylabel("True Positive Rate (Sensitivity)")
        ax_roc.set_xlim(0.0, 1.0)
        ax_roc.set_ylim(0.0, 1.02)
        ax_roc.set_title("(B) Out-of-Fold ROC Curve")
        ax_roc.legend(loc="lower right", frameon=False)
        ax_roc.grid(True)
        saved.append(_save_pub_png(fig, p_dir / f"{norm_proto}_roc_pr_curves.png"))

        # ── Diagnostic 3: Confusion Matrix & Probability Separation ({proto}_confusion_and_prob_dist.png) ──
        ref_exps = [e for e in ("temporal", "combined_w5", "combined_w8") if e in oof_available]
        if not ref_exps:
            ref_exps = oof_available[:2]
        n_cm = min(2, len(ref_exps))
        chosen_cm = [ref_exps[0], ref_exps[-1]] if len(ref_exps) >= 2 else [ref_exps[0]]

        fig, axes = plt.subplots(1, n_cm + 1, figsize=(5.2 * (n_cm + 1), 4.6), constrained_layout=True)
        if n_cm + 1 == 2:
            axes_list = [axes[0], axes[1]]
        else:
            axes_list = list(axes)
        fig.suptitle(f"ML3 {norm_proto.upper()} — Normalized Confusion Matrices & Score Separation")

        for idx_c, e in enumerate(chosen_cm):
            ax_c = axes_list[idx_c]
            op = resolved_summaries[e]["test"]["overall_trial_pooled"]
            tn, fp, fn, tp = int(op["tn"]), int(op["fp"]), int(op["fn"]), int(op["tp"])
            cm_raw = np.array([[tn, fp], [fn, tp]], dtype=np.float64)
            row_sums = np.maximum(cm_raw.sum(axis=1, keepdims=True), 1.0)
            cm_norm = cm_raw / row_sums * 100.0
            im = ax_c.imshow(cm_norm, cmap="Blues", vmin=0.0, vmax=100.0)
            for r_i in range(2):
                for c_i in range(2):
                    txt_col = "white" if cm_norm[r_i, c_i] > 55.0 else OKABE_ITO["black"]
                    ax_c.text(
                        c_i,
                        r_i,
                        f"{cm_norm[r_i, c_i]:.1f}%\n({int(cm_raw[r_i, c_i]):,})",
                        ha="center",
                        va="center",
                        fontsize=9.0,
                        fontweight="bold",
                        color=txt_col,
                    )
            ax_c.set_xticks([0, 1])
            ax_c.set_yticks([0, 1])
            ax_c.set_xticklabels(["Pred Frequent", "Pred Target"])
            ax_c.set_yticklabels(["True Frequent", "True Target"])
            ax_c.set_title(f"({chr(65 + idx_c)}) {_exp_label(e)}")
            ax_c.spines["top"].set_visible(True)
            ax_c.spines["right"].set_visible(True)

        ax_dist = axes_list[-1]
        best_ref = chosen_cm[-1]
        with np.load(p_dir / best_ref / "oof_predictions.npz", allow_pickle=False) as oof:
            yt = oof["y_true"].astype(np.int64)
            yp = oof["y_prob"].astype(np.float64)
            thrs = oof["threshold"].astype(np.float64)
        bins = np.linspace(0.0, 1.0, 42)
        ax_dist.hist(
            yp[yt == 0],
            bins=bins,
            density=True,
            color=OKABE_ITO["blue"],
            alpha=0.55,
            label="True Frequent (0)",
        )
        ax_dist.hist(
            yp[yt == 1],
            bins=bins,
            density=True,
            color=OKABE_ITO["orange"],
            alpha=0.58,
            label="True Target (1)",
        )
        mean_thr = float(np.mean(thrs))
        ax_dist.axvline(
            mean_thr,
            color=OKABE_ITO["red"],
            ls="--",
            lw=1.5,
            label=f"Mean Val Threshold ({mean_thr:.3f})",
        )
        ax_dist.set_xlabel("Predicted Probability P(Target)")
        ax_dist.set_ylabel("Density")
        ax_dist.set_title(f"({chr(65 + len(chosen_cm))}) Score Separation — {_exp_label(best_ref)}")
        ax_dist.legend(frameon=False)
        ax_dist.grid(axis="y")
        saved.append(_save_pub_png(fig, p_dir / f"{norm_proto}_confusion_and_prob_dist.png"))

    # ── Diagnostic 4: XGBoost Gain Importance Breakdown ({proto}_feature_importance_breakdown.png) ──
    imp_target = next(
        (e for e in ("combined_w8", "combined_w5", "temporal", "wavelet_w8", "wavelet_w5") if e in exp_names),
        None,
    )
    if imp_target is not None:
        feat_names = _reconstruct_scenario_feature_names(p_dir, imp_target, features_dir=features_dir)
        model_files = [
            fd / "model.json"
            for fd in _iter_fold_checkpoint_dirs(p_dir / imp_target)
            if (fd / "model.json").is_file()
        ]
        if feat_names and model_files:
            n_feat = len(feat_names)
            fold_gains = []
            for mf in model_files:
                m_obj = ML3XGBoostModel.load_model(mf, config=ML3XGBoostConfig(device="cpu"))
                g = m_obj.gain_importance_vector(n_feat)
                total_g = float(g.sum())
                if total_g > 0.0:
                    fold_gains.append(g / total_g * 100.0)
                del m_obj
            if fold_gains:
                gain_mat = np.vstack(fold_gains)  # (n_folds, n_feat) in %
                parsed_cols = [parse_ml3_feature_name(n) for n in feat_names]

                fig, axes = plt.subplots(1, 3, figsize=(16.8, 5.4), constrained_layout=True)
                fig.suptitle(
                    f"ML3 {norm_proto.upper()} — Neurophysiological Feature Gain Breakdown ({_exp_label(imp_target)})"
                )

                # Panel A: Gain % by Time Window (stacked Temporal vs Wavelet)
                ax_win = axes[0]
                win_order = [w for w in W8_WINDOW_NAMES if any(p.window == w for p in parsed_cols)]
                temp_win_pct = []
                wav_win_pct = []
                for w_name in win_order:
                    t_idx = [i for i, p in enumerate(parsed_cols) if p.window == w_name and p.group == "temporal"]
                    w_idx = [i for i, p in enumerate(parsed_cols) if p.window == w_name and p.group == "wavelet"]
                    temp_win_pct.append(float(gain_mat[:, t_idx].sum(axis=1).mean()) if t_idx else 0.0)
                    wav_win_pct.append(float(gain_mat[:, w_idx].sum(axis=1).mean()) if w_idx else 0.0)

                wx = np.arange(len(win_order))
                ax_win.bar(wx, temp_win_pct, width=0.58, color=OKABE_ITO["blue"], edgecolor=OKABE_ITO["black"], lw=0.6, label="Temporal Gain %")
                ax_win.bar(
                    wx,
                    wav_win_pct,
                    width=0.58,
                    bottom=temp_win_pct,
                    color=OKABE_ITO["green"],
                    edgecolor=OKABE_ITO["black"],
                    lw=0.6,
                    label="Wavelet Gain %",
                )
                ax_win.set_xticks(wx)
                ax_win.set_xticklabels([_WINDOW_DISPLAY.get(w, w) for w in win_order], rotation=28, ha="right")
                ax_win.set_ylabel("Share of Total XGBoost Gain (%)")
                ax_win.set_title("(A) Gain Distribution Across Epoch Windows (ms)")
                ax_win.legend(frameon=False)
                ax_win.grid(axis="y")

                # Panel B: Top 15 EEG Channels by Gain %
                ax_ch = axes[1]
                channels = sorted({p.channel for p in parsed_cols})
                ch_means = []
                ch_stds = []
                for ch in channels:
                    c_idx = [i for i, p in enumerate(parsed_cols) if p.channel == ch]
                    ch_fold_sums = gain_mat[:, c_idx].sum(axis=1)
                    ch_means.append(float(ch_fold_sums.mean()))
                    ch_stds.append(float(ch_fold_sums.std(ddof=1 if len(fold_gains) > 1 else 0)))
                order_ch = np.argsort(ch_means)[::-1][:15][::-1]
                cy = np.arange(len(order_ch))
                ax_ch.barh(
                    cy,
                    [ch_means[i] for i in order_ch],
                    xerr=[ch_stds[i] for i in order_ch],
                    capsize=3,
                    color=OKABE_ITO["purple"],
                    edgecolor=OKABE_ITO["black"],
                    lw=0.6,
                    alpha=0.85,
                )
                ax_ch.set_yticks(cy)
                ax_ch.set_yticklabels([channels[i] for i in order_ch])
                ax_ch.set_xlabel("Share of Total Gain (%, Mean ± SD)")
                ax_ch.set_title("(B) Top 15 Electrodes by Gain")
                ax_ch.grid(axis="x")

                # Panel C: Gain % by Feature Statistic / Band
                ax_ft = axes[2]
                feat_types = list(dict.fromkeys(f"{p.group}:{p.feature}" for p in parsed_cols))
                ft_means = []
                ft_cols = []
                ft_labels = []
                for ft in feat_types:
                    grp, fn = ft.split(":", 1)
                    f_idx = [i for i, p in enumerate(parsed_cols) if p.group == grp and p.feature == fn]
                    ft_means.append(float(gain_mat[:, f_idx].sum(axis=1).mean()))
                    ft_cols.append(OKABE_ITO["blue"] if grp == "temporal" else OKABE_ITO["green"])
                    ft_labels.append(fn.replace("_", " "))
                order_ft = np.argsort(ft_means)
                fy = np.arange(len(order_ft))
                ax_ft.barh(
                    fy,
                    [ft_means[i] for i in order_ft],
                    color=[ft_cols[i] for i in order_ft],
                    edgecolor=OKABE_ITO["black"],
                    lw=0.6,
                    alpha=0.86,
                )
                ax_ft.set_yticks(fy)
                ax_ft.set_yticklabels([ft_labels[i] for i in order_ft])
                ax_ft.set_xlabel("Share of Total Gain (%)")
                ax_ft.set_title("(C) Gain by Temporal Stat & Wavelet Band")
                ax_ft.grid(axis="x")

                saved.append(_save_pub_png(fig, p_dir / f"{norm_proto}_feature_importance_breakdown.png"))

    return saved


def _reconstruct_scenario_feature_names(
    protocol_dir: Path,
    experiment: str,
    *,
    features_dir: Optional[Path | str] = None,
) -> Optional[Tuple[str, ...]]:
    """Reconstruct exact scenario feature_names from feature .npz files or channel metadata."""
    channels: Optional[List[str]] = None

    candidate_fdirs: List[Path] = []
    if features_dir is not None:
        candidate_fdirs.append(Path(features_dir))
    candidate_fdirs.extend(
        [
            Path("/content/features"),
            protocol_dir.parents[1] / "features",
        ]
    )
    for cand in candidate_fdirs:
        for grp in ("temporal", "wavelet"):
            g_dir = cand / grp
            if g_dir.is_dir():
                npz_files = sorted(p for p in g_dir.glob("*.npz") if not p.name.endswith(".part.npz"))
                if npz_files:
                    with np.load(npz_files[0], allow_pickle=False) as payload:
                        if "feature_names" in payload.files:
                            raw_names = [str(x) for x in payload["feature_names"].tolist()]
                            channels = list(dict.fromkeys(parse_ml3_feature_name(n).channel for n in raw_names))
                            break
        if channels:
            break

    if not channels:
        channel_json = Path(__file__).resolve().parents[2] / "data" / "processed" / "channel_names.json"
        if channel_json.is_file():
            channels = [str(c) for c in json.loads(channel_json.read_text(encoding="utf-8"))]

    if not channels:
        summary_file = protocol_dir / experiment / "experiment_summary.json"
        if not summary_file.is_file():
            return None
        summary = json.loads(summary_file.read_text(encoding="utf-8"))
        n_feat = int(summary["n_features"])
        spec = FEATURE_SCENARIO_SPECS.get(experiment)
        if spec is None:
            return None
        per_ch = (56 if spec.uses_temporal else 0) + (
            len(spec.wavelet_windows) * (6 if spec.uses_gamma else 5) if spec.wavelet_windows else 0
        )
        if per_ch <= 0 or n_feat % per_ch != 0:
            return None
        n_ch = n_feat // per_ch
        channels = [f"Ch{i + 1:03d}" for i in range(n_ch)]

    spec = FEATURE_SCENARIO_SPECS.get(experiment)
    if spec is None or not channels:
        return None

    from src.features.ml3_features import TEMPORAL_STATISTICS
    from src.features.ml3_feature_store import H4_BAND_FEATURES, GAMMA_ENERGY_FEATURE, H4_ENTROPY_FEATURE

    names: list[str] = []
    if spec.uses_temporal:
        for w in W8_WINDOW_NAMES:
            for ch in channels:
                for st in TEMPORAL_STATISTICS:
                    names.append(f"temporal|channel={ch}|window={w}|feature={st}")
    if spec.wavelet_windows is not None:
        w_feats = list(H4_BAND_FEATURES) + ([GAMMA_ENERGY_FEATURE] if spec.uses_gamma else []) + [H4_ENTROPY_FEATURE]
        for w in spec.wavelet_windows:
            for ch in channels:
                for wf in w_feats:
                    names.append(f"wavelet|channel={ch}|window={w}|feature={wf}")
    return tuple(names)



# ── High-Level Planning & Execution Orchestrator ─────────────────────────────

def _filter_ml3_splits(
    all_folds: Tuple[FoldSplit, ...],
    protocol: ProtocolName,
    *,
    folds_filter: Optional[Sequence[int]] = None,
    subjects_filter: Optional[Sequence[str]] = None,
) -> Tuple[FoldSplit, ...]:
    selected = all_folds
    if subjects_filter is not None:
        wanted_subs = {str(s).strip() for s in subjects_filter if str(s).strip()}
        if protocol == "within":
            available_subs = {f.subject_id for f in all_folds if isinstance(f, WithinFoldSplit)}
            missing_subs = sorted(wanted_subs - available_subs, key=_natural_sort_key)
            if missing_subs:
                raise ValueError(
                    f"Requested subjects not present in {protocol}: {missing_subs}"
                )
            selected = tuple(
                f for f in selected if isinstance(f, WithinFoldSplit) and f.subject_id in wanted_subs
            )
        else:
            selected = tuple(
                f for f in selected if any(s in wanted_subs for s in f.test_subjects)
            )
            if not selected:
                raise ValueError(
                    f"No {protocol} folds matched requested subjects: {sorted(wanted_subs)}"
                )

    if folds_filter is not None:
        wanted = set(int(f) for f in folds_filter)
        available = {f.fold for f in selected}
        missing = sorted(wanted - available)
        if missing:
            raise ValueError(f"Requested fold indices not present in {protocol}: {missing}")
        selected = tuple(f for f in selected if f.fold in wanted)

    return selected


def build_ml3_execution_plan(
    *,
    features_dir: Path | str,
    splits_dir: Path | str,
    protocol: ProtocolName | str,
    experiments: Optional[Sequence[str]] = None,
    output_dir: Path | str,
    model_config: ML3XGBoostConfig = ML3XGBoostConfig(),
    folds_filter: Optional[Sequence[int]] = None,
    subjects_filter: Optional[Sequence[str]] = None,
    threshold_metric: str = "balanced_accuracy",
    allow_gamma_loso: bool = False,
    mount_drive: bool = False,
    drive_mount_point: Path | str = "/content/drive",
) -> Dict[str, Any]:
    """Validate inputs and build a dry-run execution summary (used by ``--plan-only``)."""
    norm_protocol = normalize_protocol_name(protocol)
    chosen_experiments = validate_experiment_list(
        norm_protocol, experiments, allow_gamma_loso=allow_gamma_loso
    )
    req_groups = required_feature_groups_for_experiments(chosen_experiments)
    all_folds, split_meta = load_ml3_splits(splits_dir, norm_protocol)
    selected_folds = _filter_ml3_splits(
        all_folds,
        norm_protocol,
        folds_filter=folds_filter,
        subjects_filter=subjects_filter,
    )

    store_info = inspect_feature_store_metadata(
        features_dir,
        required_groups=req_groups,
        mount_drive=mount_drive,
        mount_point=drive_mount_point,
    )
    out_root = ensure_drive_path_ready(
        output_dir, mount_drive=mount_drive, mount_point=drive_mount_point
    )

    split_subjects = set()
    for f in selected_folds:
        split_subjects.update(f.train_subjects)
        split_subjects.update(f.val_subjects)
        split_subjects.update(f.test_subjects)
    missing_subs = sorted(split_subjects - set(store_info["subjects"]))
    if missing_subs:
        raise ValueError(
            f"Feature store at {store_info['features_dir']} is missing subjects required by splits: {missing_subs}"
        )

    n_ch = int(store_info["n_channels"])
    scenario_details = []
    for exp in chosen_experiments:
        spec = FEATURE_SCENARIO_SPECS[exp]
        n_feat = 0
        if spec.uses_temporal:
            n_feat += 8 * n_ch * 7
        if spec.wavelet_windows is not None:
            n_feat += len(spec.wavelet_windows) * n_ch * (6 if spec.uses_gamma else 5)
        scenario_details.append(
            {
                "experiment": exp,
                "expected_n_features": n_feat,
                "output_dir": str(out_root / norm_protocol / exp),
                "policy": spec.policy_dict(),
            }
        )

    return {
        "protocol": norm_protocol,
        "features_dir": store_info["features_dir"],
        "output_dir": str(out_root / norm_protocol),
        "required_feature_groups": list(req_groups),
        "n_subjects_available": store_info["n_subjects"],
        "n_channels": n_ch,
        "split_metadata": split_meta,
        "n_folds_selected": len(selected_folds),
        "fold_indices": (
            [f.fold_key for f in selected_folds]
            if norm_protocol == "within"
            else [f.fold for f in selected_folds]
        ),
        "threshold_metric": threshold_metric,
        "model_config": model_config.report_dict(),
        "experiments": scenario_details,
    }


def run_ml3_experiments(
    *,
    features_dir: Path | str,
    splits_dir: Path | str,
    protocol: ProtocolName | str,
    experiments: Optional[Sequence[str]] = None,
    output_dir: Path | str,
    model_config: ML3XGBoostConfig = ML3XGBoostConfig(),
    folds_filter: Optional[Sequence[int]] = None,
    subjects_filter: Optional[Sequence[str]] = None,
    threshold_metric: str = "balanced_accuracy",
    resume: bool = False,
    overwrite: bool = False,
    allow_gamma_loso: bool = False,
    mount_drive: bool = False,
    drive_mount_point: Path | str = "/content/drive",
    verbose: bool = True,
) -> Dict[str, Any]:
    """Execute requested ML3 feature scenarios across CS5, LOSO, or Within-Subject folds."""
    norm_protocol = normalize_protocol_name(protocol)
    chosen_experiments = validate_experiment_list(
        norm_protocol, experiments, allow_gamma_loso=allow_gamma_loso
    )
    req_groups = required_feature_groups_for_experiments(chosen_experiments)
    needs_gamma = any(FEATURE_SCENARIO_SPECS[e].uses_gamma for e in chosen_experiments)

    all_folds, split_meta = load_ml3_splits(splits_dir, norm_protocol)
    selected_folds = _filter_ml3_splits(
        all_folds,
        norm_protocol,
        folds_filter=folds_filter,
        subjects_filter=subjects_filter,
    )

    needed_subjects = sorted(
        {
            s
            for f in selected_folds
            for s in (*f.train_subjects, *f.val_subjects, *f.test_subjects)
        },
        key=_natural_sort_key,
    )

    out_root = ensure_drive_path_ready(
        output_dir, mount_drive=mount_drive, mount_point=drive_mount_point
    )
    protocol_dir = out_root / norm_protocol
    protocol_dir.mkdir(parents=True, exist_ok=True)

    dataset: ML3FeatureDataset = load_ml3_feature_dataset(
        features_dir,
        subjects=needed_subjects,
        required_groups=req_groups,
        include_gamma_support=needs_gamma,
        mount_drive=mount_drive,
        mount_point=drive_mount_point,
    )

    experiment_summaries: Dict[str, Any] = {}
    fold_statuses: Dict[str, Dict[str, int]] = {}

    for exp_name in chosen_experiments:
        scenario_matrix = dataset.build_scenario(exp_name)
        exp_dir = protocol_dir / exp_name
        exp_dir.mkdir(parents=True, exist_ok=True)

        if verbose:
            print(
                f"[ML3][{norm_protocol.upper()}][{exp_name}] "
                f"samples={scenario_matrix.n_samples:,}, features={scenario_matrix.n_features:,}, "
                f"folds={len(selected_folds)}, device={model_config.device}"
            )

        fold_metrics_list: list[Dict[str, Any]] = []
        fold_preds_list: list[Dict[str, np.ndarray]] = []
        trained_count = 0
        reused_count = 0

        for split in selected_folds:
            fold_dir = exp_dir / split.fold_key
            f_metrics, f_preds, status = train_or_load_fold(
                scenario_matrix=scenario_matrix,
                split=split,
                model_config=model_config,
                fold_dir=fold_dir,
                threshold_metric=threshold_metric,
                resume=resume,
                overwrite=overwrite,
            )
            fold_metrics_list.append(f_metrics)
            fold_preds_list.append(f_preds)
            if status == "reused":
                reused_count += 1
            else:
                trained_count += 1

            if verbose:
                tp = f_metrics["test_pooled"]
                print(
                    f"  {split.fold_key} [{status}] "
                    f"best_iter={f_metrics['best_iteration']} "
                    f"thr={f_metrics['threshold']:.3f} "
                    f"PR-AUC={tp['pr_auc']:.4f} "
                    f"ROC-AUC={tp['roc_auc']:.4f} "
                    f"BalAcc={tp['balanced_accuracy']:.4f} "
                    f"F1={tp['f1_target']:.4f} "
                    f"time={f_metrics['train_duration_sec']:.1f}s"
                )

        summary = aggregate_experiment_results(
            protocol=norm_protocol,
            experiment=exp_name,
            exp_dir=exp_dir,
            model_config=model_config,
            fold_metrics_list=fold_metrics_list,
            fold_preds_list=fold_preds_list,
        )
        experiment_summaries[exp_name] = summary
        fold_statuses[exp_name] = {"trained": trained_count, "reused": reused_count}

        del scenario_matrix
        gc.collect()

    comparison = write_protocol_comparison_and_plots(
        norm_protocol,
        protocol_dir,
        features_dir=features_dir,
    )

    return {
        "protocol": norm_protocol,
        "output_dir": str(protocol_dir),
        "split_metadata": split_meta,
        "fold_statuses": fold_statuses,
        "experiments": experiment_summaries,
        "comparison": comparison,
    }
