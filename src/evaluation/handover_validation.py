"""Reusable validation for the group handover schema (Proposal section 3.5)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

PER_FOLD_COLUMNS = (
    "subject_id",
    "fold",
    "bal_acc",
    "roc_auc",
    "pr_auc",
    "f1_target",
    "sensitivity",
    "specificity",
)
METRIC_COLUMNS = PER_FOLD_COLUMNS[2:]
PREDICTION_KEYS = (
    "subject_id",
    "fold",
    "trial_idx",
    "trial_uid",
    "y_true",
    "y_prob",
    "y_pred",
)


@dataclass(frozen=True)
class ValidationIssue:
    path: Path
    message: str


@dataclass(frozen=True)
class HandoverValidationReport:
    files_checked: int
    issues: tuple[ValidationIssue, ...]

    @property
    def is_valid(self) -> bool:
        return not self.issues

    def raise_for_errors(self) -> None:
        if self.issues:
            details = "\n".join(f"- {issue.path}: {issue.message}" for issue in self.issues)
            raise ValueError(f"Handover validation failed:\n{details}")


def validate_per_fold_csv(path: Path) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    try:
        frame = pd.read_csv(path)
    except Exception as exc:
        return [ValidationIssue(path, f"cannot read CSV: {exc}")]

    if tuple(frame.columns) != PER_FOLD_COLUMNS:
        issues.append(ValidationIssue(path, f"expected columns {PER_FOLD_COLUMNS}"))
        return issues
    if frame.empty:
        issues.append(ValidationIssue(path, "file contains no folds"))
        return issues
    if frame[["subject_id", "fold"]].isna().any().any():
        issues.append(ValidationIssue(path, "subject_id or fold contains missing values"))
    if frame.duplicated(["subject_id", "fold"]).any():
        issues.append(ValidationIssue(path, "duplicate (subject_id, fold) rows"))

    for column in METRIC_COLUMNS:
        try:
            values = frame[column].to_numpy(dtype=np.float64)
        except (TypeError, ValueError):
            issues.append(ValidationIssue(path, f"{column} is not numeric"))
            continue
        if not np.isfinite(values).all():
            issues.append(ValidationIssue(path, f"{column} contains NaN or infinite values"))
        elif not ((values >= 0.0) & (values <= 1.0)).all():
            issues.append(ValidationIssue(path, f"{column} contains values outside [0, 1]"))
    return issues


def validate_predictions_npz(path: Path) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    try:
        with np.load(path, allow_pickle=False) as payload:
            missing = sorted(set(PREDICTION_KEYS) - set(payload.files))
            if missing:
                return [ValidationIssue(path, f"missing arrays: {missing}")]
            arrays = {key: np.asarray(payload[key]) for key in PREDICTION_KEYS}
    except Exception as exc:
        return [ValidationIssue(path, f"cannot read NPZ: {exc}")]

    lengths = {key: len(value) for key, value in arrays.items()}
    if len(set(lengths.values())) != 1:
        issues.append(ValidationIssue(path, f"arrays have different lengths: {lengths}"))
        return issues
    if not lengths["y_true"]:
        issues.append(ValidationIssue(path, "prediction arrays are empty"))
        return issues
    if len(np.unique(arrays["trial_uid"].astype(str))) != lengths["trial_uid"]:
        issues.append(ValidationIssue(path, "trial_uid values are not unique"))
    for key in ("y_true", "y_pred"):
        if not set(np.unique(arrays[key]).tolist()).issubset({0, 1}):
            issues.append(ValidationIssue(path, f"{key} contains values outside {{0, 1}}"))
    try:
        probabilities = arrays["y_prob"].astype(np.float64)
        if not np.isfinite(probabilities).all():
            issues.append(ValidationIssue(path, "y_prob contains NaN or infinite values"))
        elif not ((probabilities >= 0.0) & (probabilities <= 1.0)).all():
            issues.append(ValidationIssue(path, "y_prob contains values outside [0, 1]"))
    except (TypeError, ValueError):
        issues.append(ValidationIssue(path, "y_prob is not numeric"))
    return issues


def _is_visible_relative_path(path: Path, root: Path) -> bool:
    return not any(part.startswith(".") for part in path.relative_to(root).parts)


def discover_handover_artifacts(directory: Path) -> tuple[list[Path], list[Path]]:
    """Find handover CSV and NPZ artifacts in ``directory`` and its branch subfolders."""

    if not directory.exists():
        return [], []
    if directory.is_file():
        csv_paths = [directory] if directory.name.endswith("_per_fold.csv") else []
        npz_paths = [directory] if directory.name.endswith("_preds.npz") else []
        return csv_paths, npz_paths

    csv_paths = sorted(
        path
        for path in directory.rglob("*_per_fold.csv")
        if path.is_file() and _is_visible_relative_path(path, directory)
    )
    npz_paths = sorted(
        path
        for path in directory.rglob("*_preds.npz")
        if path.is_file() and _is_visible_relative_path(path, directory)
    )
    return csv_paths, npz_paths


def validate_handover_directory(
    directory: Path,
    *,
    require_prediction_pair: bool = False,
) -> HandoverValidationReport:
    """Validate every handover artifact present across ``directory`` and branch subfolders."""

    csv_paths, npz_paths = discover_handover_artifacts(directory)
    issues: list[ValidationIssue] = []
    if not csv_paths and not npz_paths:
        issues.append(ValidationIssue(directory, "no handover artifacts found"))

    seen_names: dict[str, Path] = {}
    for path in [*csv_paths, *npz_paths]:
        previous = seen_names.get(path.name)
        if previous is not None and previous != path:
            issues.append(
                ValidationIssue(path, f"duplicate artifact filename also present at {previous}")
            )
        else:
            seen_names[path.name] = path

    for path in csv_paths:
        issues.extend(validate_per_fold_csv(path))
    for path in npz_paths:
        issues.extend(validate_predictions_npz(path))

    if require_prediction_pair:
        csv_stems = {path.with_name(path.name.removesuffix("_per_fold.csv")) for path in csv_paths}
        npz_stems = {path.with_name(path.name.removesuffix("_preds.npz")) for path in npz_paths}
        for stem in sorted(csv_stems - npz_stems):
            issues.append(
                ValidationIssue(stem.with_name(f"{stem.name}_preds.npz"), "missing prediction pair")
            )
        for stem in sorted(npz_stems - csv_stems):
            issues.append(
                ValidationIssue(stem.with_name(f"{stem.name}_per_fold.csv"), "missing per-fold pair")
            )

    return HandoverValidationReport(len(csv_paths) + len(npz_paths), tuple(issues))
