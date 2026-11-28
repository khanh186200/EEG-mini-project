"""Input preparation contracts for paired RQ2 statistics.

The statistical backend is deliberately not executed by this scaffold. This
module establishes the critical invariant that Within folds are averaged to
one row per subject before models are paired.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import pandas as pd


@dataclass(frozen=True)
class RQ2Config:
    metrics: tuple[str, ...] = ("roc_auc", "bal_acc")
    protocols: tuple[str, ...] = ("within", "loso")
    alpha: float = 0.05

    def __post_init__(self) -> None:
        if not 0.0 < self.alpha < 1.0:
            raise ValueError("alpha must lie strictly between 0 and 1")


def aggregate_subject_scores(frame: pd.DataFrame, metric: str) -> pd.Series:
    """Return exactly one score per subject, averaging Within folds if present."""

    required = {"subject_id", metric}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    if frame[["subject_id", metric]].isna().any().any():
        raise ValueError("subject scores contain missing values")
    return frame.groupby("subject_id", sort=True)[metric].mean()


def build_paired_subject_table(
    model_frames: Mapping[str, pd.DataFrame],
    metric: str,
) -> pd.DataFrame:
    """Inner-align model scores so every retained row is a valid paired subject."""

    if len(model_frames) < 2:
        raise ValueError("paired statistics require at least two models")
    columns = {
        model_name: aggregate_subject_scores(frame, metric)
        for model_name, frame in model_frames.items()
    }
    paired = pd.concat(columns, axis=1, join="inner").sort_index()
    paired.index.name = "subject_id"
    if paired.empty:
        raise ValueError("models have no subjects in common")
    if paired.isna().any().any():
        raise ValueError("paired table unexpectedly contains missing values")
    return paired
