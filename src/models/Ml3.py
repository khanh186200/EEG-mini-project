"""XGBoost model configuration and estimator wrapper for ML3 P300 experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Protocol, Sequence, Tuple, runtime_checkable

import numpy as np

try:
    import xgboost as xgb
except ImportError:  # pragma: no cover - surfaced with a clear runtime message
    xgb = None


# Optional small post-feature-comparison search space (validation-only; off during comparison)
POST_COMPARISON_TUNING_GRID: Dict[str, Tuple[int | float, ...]] = {
    "max_depth": (2, 3, 4),
    "min_child_weight": (5.0, 10.0, 20.0),
    "colsample_bytree": (0.10, 0.15, 0.25),
    "learning_rate": (0.03, 0.05),
    "reg_lambda": (2.0, 5.0, 10.0),
}


@dataclass(frozen=True)
class TreeModelSpec:
    """Serializable model request."""

    name: str
    parameters: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("model name must not be empty")
        if "gamma" in self.parameters:
            raise ValueError(
                "Use 'min_split_loss' instead of 'gamma' for XGBoost hyperparameters "
                "to avoid confusion with the EEG gamma band."
            )


@runtime_checkable
class BinaryProbabilisticEstimator(Protocol):
    """Minimum estimator surface needed by ML3 evaluation."""

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        **kwargs: Any,
    ) -> "BinaryProbabilisticEstimator": ...

    def predict_proba(self, X: np.ndarray) -> np.ndarray: ...


@runtime_checkable
class TreeEstimatorFactory(Protocol):
    """Factory protocol for tree estimators."""

    def create(self, spec: TreeModelSpec, *, random_seed: int) -> BinaryProbabilisticEstimator: ...


@dataclass(frozen=True)
class ML3XGBoostConfig:
    """Fixed XGBoost hyperparameter policy shared across all ML3 feature scenarios.

    Note: XGBoost's split regularization parameter is explicitly named
    ``min_split_loss`` (never ``gamma``) in configs and reports so it cannot be
    confused with the 30-40 Hz EEG gamma band.
    """

    objective: str = "binary:logistic"
    tree_method: str = "hist"
    device: str = "cuda"
    n_estimators: int = 2000
    learning_rate: float = 0.05
    max_depth: int = 3
    min_child_weight: float = 10.0
    subsample: float = 0.8
    colsample_bytree: float = 0.15
    reg_alpha: float = 0.1
    reg_lambda: float = 5.0
    min_split_loss: float = 0.0
    max_bin: int = 256
    eval_metric: str = "aucpr"
    early_stopping_rounds: int = 100
    random_state: int = 42
    n_jobs: int = -1
    learning_rate_schedule: Optional[Tuple[float, ...]] = None

    def __post_init__(self) -> None:
        if self.device not in {"cuda", "cpu"} and not self.device.startswith("cuda:"):
            raise ValueError(f"Unsupported XGBoost device: {self.device!r}")
        if self.n_estimators <= 0:
            raise ValueError("n_estimators must be positive")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if self.max_depth <= 0:
            raise ValueError("max_depth must be positive")
        if self.min_child_weight < 0.0:
            raise ValueError("min_child_weight must be non-negative")
        if not (0.0 < self.subsample <= 1.0):
            raise ValueError("subsample must lie in (0, 1]")
        if not (0.0 < self.colsample_bytree <= 1.0):
            raise ValueError("colsample_bytree must lie in (0, 1]")
        if self.reg_alpha < 0.0 or self.reg_lambda < 0.0 or self.min_split_loss < 0.0:
            raise ValueError("regularization parameters must be non-negative")
        if self.max_bin < 2:
            raise ValueError("max_bin must be >= 2")
        if self.early_stopping_rounds <= 0:
            raise ValueError("early_stopping_rounds must be positive")
        if self.learning_rate_schedule is not None:
            if len(self.learning_rate_schedule) == 0 or any(
                lr <= 0.0 for lr in self.learning_rate_schedule
            ):
                raise ValueError("learning_rate_schedule must contain positive learning rates")

    def semantic_dict(self) -> Dict[str, Any]:
        """Hyperparameters that define mathematical model behavior for checkpoint hashing."""
        return {
            "model_family": "xgboost",
            "objective": self.objective,
            "tree_method": self.tree_method,
            "n_estimators": int(self.n_estimators),
            "learning_rate": float(self.learning_rate),
            "max_depth": int(self.max_depth),
            "min_child_weight": float(self.min_child_weight),
            "subsample": float(self.subsample),
            "colsample_bytree": float(self.colsample_bytree),
            "reg_alpha": float(self.reg_alpha),
            "reg_lambda": float(self.reg_lambda),
            "min_split_loss": float(self.min_split_loss),
            "max_bin": int(self.max_bin),
            "eval_metric": self.eval_metric,
            "early_stopping_rounds": int(self.early_stopping_rounds),
            "random_state": int(self.random_state),
            "learning_rate_schedule": (
                [float(v) for v in self.learning_rate_schedule]
                if self.learning_rate_schedule is not None
                else None
            ),
        }

    def report_dict(self) -> Dict[str, Any]:
        """Full configuration dictionary for manifests and reports (uses min_split_loss)."""
        payload = self.semantic_dict()
        payload["device"] = self.device
        payload["n_jobs"] = int(self.n_jobs)
        return payload


def compute_scale_pos_weight(y_train: np.ndarray) -> float:
    """Compute ``N_{Frequent,train} / N_{Target,train}`` strictly from train labels."""
    labels = np.asarray(y_train, dtype=np.int64).ravel()
    if labels.size == 0:
        raise ValueError("y_train must not be empty")
    unique = set(np.unique(labels).tolist())
    if not unique.issubset({0, 1}):
        raise ValueError(f"y_train must contain binary labels in {{0, 1}}, got {sorted(unique)}")
    n_frequent = int(np.sum(labels == 0))
    n_target = int(np.sum(labels == 1))
    if n_frequent == 0 or n_target == 0:
        raise ValueError(
            f"Both Frequent (0) and Target (1) classes must exist in y_train "
            f"(got n_frequent={n_frequent}, n_target={n_target})"
        )
    return float(n_frequent / n_target)


class ML3XGBoostModel:
    """Validation-stopped XGBoost binary classifier with strict zero-test-leakage."""

    def __init__(self, config: ML3XGBoostConfig = ML3XGBoostConfig()) -> None:
        if xgb is None:
            raise ImportError(
                "xgboost>=2.0.0 is required for ML3 training; install requirements.txt"
            )
        self.config = config
        self.scale_pos_weight_: Optional[float] = None
        self.best_iteration_: Optional[int] = None
        self.best_val_score_: Optional[float] = None
        self.learning_curve_: Optional[Dict[str, list[float | int]]] = None
        self.estimator_: Optional[Any] = None

    def _build_callbacks(self) -> Optional[list[Any]]:
        if self.config.learning_rate_schedule is None:
            return None
        schedule = list(self.config.learning_rate_schedule)

        def _lr_fn(boosting_round: int) -> float:
            idx = min(boosting_round, len(schedule) - 1)
            return float(schedule[idx])

        return [xgb.callback.LearningRateScheduler(_lr_fn)]

    def fit_fold(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> "ML3XGBoostModel":
        """Fit on ``(X_train, y_train)`` with early stopping strictly on ``(X_val, y_val)``."""
        X_tr = np.asarray(X_train, dtype=np.float32)
        y_tr = np.asarray(y_train, dtype=np.int64).ravel()
        X_va = np.asarray(X_val, dtype=np.float32)
        y_va = np.asarray(y_val, dtype=np.int64).ravel()

        if X_tr.ndim != 2 or X_va.ndim != 2:
            raise ValueError("X_train and X_val must be 2D feature matrices")
        if X_tr.shape[1] != X_va.shape[1]:
            raise ValueError("X_train and X_val feature counts do not match")
        if len(y_tr) != X_tr.shape[0] or len(y_va) != X_va.shape[0]:
            raise ValueError("Label array length does not match feature matrix rows")
        if len(np.unique(y_va)) < 2:
            raise ValueError("Validation split must contain both classes for PR-AUC early stopping")

        spw = compute_scale_pos_weight(y_tr)
        callbacks = self._build_callbacks()

        clf_kwargs: Dict[str, Any] = {
            "objective": self.config.objective,
            "tree_method": self.config.tree_method,
            "device": self.config.device,
            "n_estimators": self.config.n_estimators,
            "learning_rate": self.config.learning_rate,
            "max_depth": self.config.max_depth,
            "min_child_weight": self.config.min_child_weight,
            "subsample": self.config.subsample,
            "colsample_bytree": self.config.colsample_bytree,
            "reg_alpha": self.config.reg_alpha,
            "reg_lambda": self.config.reg_lambda,
            "min_split_loss": self.config.min_split_loss,
            "max_bin": self.config.max_bin,
            "eval_metric": self.config.eval_metric,
            "early_stopping_rounds": self.config.early_stopping_rounds,
            "scale_pos_weight": spw,
            "random_state": self.config.random_state,
            "n_jobs": self.config.n_jobs,
        }
        if callbacks is not None:
            clf_kwargs["callbacks"] = callbacks

        # XGBoost monitors the LAST entry in eval_set (validation_1 = X_va) for
        # early stopping while logging per-round train/val aucpr curves.
        clf = xgb.XGBClassifier(**clf_kwargs)
        clf.fit(
            X_tr,
            y_tr,
            eval_set=[(X_tr, y_tr), (X_va, y_va)],
            verbose=False,
        )

        best_iter = getattr(clf, "best_iteration", None)
        if best_iter is None:
            best_iter = clf.get_booster().num_boosted_rounds() - 1
        best_score = getattr(clf, "best_score", None)

        evals_res = clf.evals_result() or {}
        metric_key = self.config.eval_metric
        tr_hist = [float(v) for v in evals_res.get("validation_0", {}).get(metric_key, [])]
        va_hist = [float(v) for v in evals_res.get("validation_1", {}).get(metric_key, [])]

        self.estimator_ = clf
        self.scale_pos_weight_ = spw
        self.best_iteration_ = int(best_iter)
        self.best_val_score_ = float(best_score) if best_score is not None else None
        self.learning_curve_ = {
            "rounds": list(range(1, len(va_hist) + 1)),
            "train_aucpr": tr_hist,
            "val_aucpr": va_hist,
        }
        return self

    def gain_importance_vector(self, n_features: int) -> np.ndarray:
        """Extract total/mean gain importance aligned to column indices ``0 .. n_features-1``."""
        if self.estimator_ is None:
            raise RuntimeError("ML3XGBoostModel has not been fitted or loaded")
        raw_scores = self.estimator_.get_booster().get_score(importance_type="gain")
        vec = np.zeros(int(n_features), dtype=np.float64)
        for key, val in raw_scores.items():
            if key.startswith("f") and key[1:].isdigit():
                idx = int(key[1:])
                if 0 <= idx < n_features:
                    vec[idx] = float(val)
        return vec

    def predict_proba_positive(self, X: np.ndarray) -> np.ndarray:
        """Predict P(Target=1) using only trees up to ``best_iteration + 1``."""
        if self.estimator_ is None:
            raise RuntimeError("ML3XGBoostModel has not been fitted or loaded")
        X_arr = np.asarray(X, dtype=np.float32)
        if self.best_iteration_ is not None:
            proba = self.estimator_.predict_proba(
                X_arr,
                iteration_range=(0, int(self.best_iteration_) + 1),
            )
        else:
            proba = self.estimator_.predict_proba(X_arr)
        return np.asarray(proba[:, 1], dtype=np.float64)

    def save_model(self, path: Path | str) -> Path:
        """Save booster/classifier atomically to ``model.json``."""
        if self.estimator_ is None:
            raise RuntimeError("Cannot save an unfitted ML3XGBoostModel")
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = target.with_name(f"{target.stem}.tmp{target.suffix or '.json'}")
        self.estimator_.save_model(str(tmp_path))
        tmp_path.replace(target)
        return target

    @classmethod
    def load_model(
        cls,
        path: Path | str,
        *,
        config: ML3XGBoostConfig = ML3XGBoostConfig(),
        best_iteration: Optional[int] = None,
    ) -> "ML3XGBoostModel":
        """Reload a saved ``model.json`` checkpoint."""
        if xgb is None:
            raise ImportError(
                "xgboost>=2.0.0 is required for ML3 training; install requirements.txt"
            )
        instance = cls(config=config)
        clf = xgb.XGBClassifier()
        clf.load_model(str(path))
        instance.estimator_ = clf
        loaded_best = best_iteration
        if loaded_best is None:
            loaded_best = getattr(clf, "best_iteration", None)
        instance.best_iteration_ = int(loaded_best) if loaded_best is not None else None
        return instance
