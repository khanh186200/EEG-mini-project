"""Huấn luyện DL1 (MS-CNN-BiGRU) theo thứ tự Within, LOSO và CS5."""

import argparse
import hashlib
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import torch

REPO_DIR = Path(__file__).resolve().parents[2]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from src.evaluation.metrics import (
    METRIC_NAMES,
    PREDICTION_COLUMNS,
    aggregate_metrics,
    save_metrics_json,
    save_predictions_csv,
)
from src.training.dataset_dl1 import get_protocol_folds, load_eeg_dataset

DL1_SOURCES = (
    "src/models/ms_cnn_bigru.py",
    "src/training/dataset_dl1.py",
    "src/training/trainer_dl1.py",
)
SEED = 42
PATIENCE = 15


@dataclass(frozen=True)
class Experiment:
    protocol: str
    variant: str
    label: str
    name: str
    fold_count: int
    reference: str

    def run_name(self, mode):
        prefix = "pilot" if mode == "pilot" else "dl1"
        return f"{prefix}_{self.name}"


METHODS = {
    "within": (
        Experiment("within_subject", "clean", "within", "ws_ms_cnn_bigru", 195, "within_5fold.json"),
    ),
    "loso": (
        Experiment("cross_subject_loso", "clean", "loso", "loso_ms_cnn_bigru", 39, "loso.json"),
    ),
    "cs5": (
        Experiment("cross_subject_5fold", "clean", "cs5_clean", "cs5_ms_cnn_bigru_clean", 5, "cross_5fold.json"),
        Experiment("cross_subject_5fold", "no_ica", "cs5_no_ica", "cs5_ms_cnn_bigru_no_ica", 5, "cross_5fold.json"),
    ),
}


@dataclass
class FoldResult:
    metrics: dict
    predictions: pd.DataFrame


def _experiment_config(experiment, bundle, mode):
    return dict(
        exp_name=experiment.run_name(mode), model="ms_cnn_bigru", model_kwargs={},
        protocol=experiment.protocol, data_dir=str(bundle.data_dir), data_variant=bundle.data_variant,
        epochs=1 if mode == "pilot" else 80, patience=PATIENCE, seed=SEED, lr=1e-3,
        weight_decay=1e-2, batch_size=32 if experiment.protocol == "within_subject" else 64,
        early_stop_metric="val_roc_auc", threshold_metric="balanced_accuracy",
    )


def _load_folds(experiment, bundle, mode):
    reference = REPO_DIR / "splits" / experiment.reference
    protocol = experiment.protocol
    # CS5 đã được kiểm tra trên clean và ánh xạ sang index của từng biến thể trong dataset loader.
    if protocol != "cross_subject_5fold":
        locked = json.loads(reference.read_text(encoding="utf-8"))
        if locked["trial_uid_sha256"] != bundle.trial_uid_sha256 or locked[protocol] != bundle.splits[protocol]:
            raise ValueError(f"{experiment.label}: dữ liệu hoặc fold khác chuẩn của nhóm")
    folds = get_protocol_folds(bundle.splits, protocol)
    if len(folds) != experiment.fold_count:
        raise ValueError(f"{experiment.label}: cần {experiment.fold_count} fold, tìm được {len(folds)}")
    if mode == "full":
        return folds
    if protocol == "cross_subject_5fold":
        return [next(fold for fold in folds if fold["fold"] == 0)]
    return folds[:1]


def _run_id(bundle, folds, config):
    code_hash = hashlib.sha256()
    for source in DL1_SOURCES:
        code_hash.update((REPO_DIR / source).read_bytes())
    if config["protocol"] == "cross_subject_5fold":
        code_hash.update(json.dumps(folds, sort_keys=True).encode("utf-8"))
    return f'{bundle.trial_uid_sha256}:{code_hash.hexdigest()}:{config["epochs"]}:seed{SEED}'


def _read_predictions(path):
    return pd.read_csv(path, dtype={"trial_uid": str, "subject_id": str}, float_precision="round_trip")


def _validate_fold(metrics, predictions, bundle, fold):
    metadata = ("fold_key", "protocol", "fold", "subject_id", "train_subjects", "val_subjects", "test_subjects")
    if any(metrics.get(key) != fold[key] for key in metadata if key in fold):
        raise ValueError("Metadata fold khác split đã khóa")
    if not set(METRIC_NAMES).issubset(metrics) or not set(PREDICTION_COLUMNS).issubset(predictions):
        raise ValueError("Thiếu cột metrics hoặc dự đoán")
    test_idx = fold["test_idx"]
    ids = bundle.metadata["trial_uid"].iloc[test_idx].astype(str)
    indexed = predictions.set_index("trial_uid", verify_integrity=True)
    if len(indexed) != len(ids) or not ids.isin(indexed.index).all():
        raise ValueError("Danh sách trial test khác split đã khóa")
    aligned = indexed.loc[ids]
    if (not np.array_equal(aligned["y_true"], bundle.y[test_idx])
            or not np.array_equal(aligned["subject_id"], bundle.subject_ids[test_idx])
            or not aligned["protocol"].eq(fold["protocol"]).all()
            or not aligned["fold"].eq(fold["fold"]).all()
            or not aligned["y_prob"].between(0, 1).all()
            or not aligned["threshold"].between(0, 1).all()
            or not aligned["y_pred"].isin([0, 1]).all()):
        raise ValueError("Dự đoán hoặc nhãn test không hợp lệ")
    return FoldResult(metrics, aligned.reset_index())


def _read_experiment(directory, config, bundle, folds, run_id):
    name = config["exp_name"]
    metrics_path = directory / f"metrics_{name}.json"
    predictions_path = directory / f"predictions_{name}.csv"
    try:
        report = json.loads(metrics_path.read_text(encoding="utf-8"))
        saved_config = report["config"]
        if (report["trial_uid_sha256"] != bundle.trial_uid_sha256
                or any(saved_config.get(key) != value for key, value in config.items() if key != "data_dir")
                or saved_config.get("run_id", run_id) != run_id):
            raise ValueError("Cấu hình, dữ liệu hoặc mã huấn luyện đã thay đổi")
        stored_metrics = {row["fold_key"]: row for row in report["per_fold"]}
        predictions = _read_predictions(predictions_path)
        group_columns = ["subject_id", "fold"] if config["protocol"] == "within_subject" else "fold"
        stored_predictions = {group: rows for group, rows in predictions.groupby(group_columns, sort=False)}
    except FileNotFoundError:
        return {}
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"  Báo cáo trên Drive cần kiểm tra lại theo từng fold: {exc}", flush=True)
        return {}

    recovered = {}
    for fold in folds:
        key = fold["fold_key"]
        group = (fold["subject_id"], fold["fold"]) if config["protocol"] == "within_subject" else fold["fold"]
        try:
            recovered[key] = _validate_fold(stored_metrics[key], stored_predictions[group], bundle, fold)
        except (ValueError, KeyError, TypeError) as exc:
            print(f"  Kết quả tổng hợp thiếu hoặc sai cho {key}: {exc}", flush=True)
    return recovered


def _read_fold(cache, legacy, bundle, fold, run_id):
    key = fold["fold_key"]
    try:
        record = json.loads((cache / f"{key}_metrics.json").read_text(encoding="utf-8"))
        if record.pop("run_id", None) == run_id:
            predictions = _read_predictions(cache / f"{key}_pred.csv")
            return _validate_fold(record, predictions, bundle, fold)
    except FileNotFoundError:
        pass
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"  Cache {key} cần khôi phục: {exc}", flush=True)

    try:
        record = json.loads((legacy / f"{key}.json").read_text(encoding="utf-8"))
        if record.get("run_id") == run_id:
            metrics = dict(record["metrics"], best_epoch=record["best_epoch"], threshold=record["threshold"])
            return _validate_fold(metrics, pd.DataFrame(record["predictions"]), bundle, fold)
    except FileNotFoundError:
        pass
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"  Kết quả cũ {key} cần khôi phục: {exc}", flush=True)
    return None


def _write_fold(cache, key, result, run_id):
    cache.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=cache) as folder:
        stage = Path(folder)
        predictions = save_predictions_csv(result.predictions, stage / f"{key}_pred.csv")
        metrics = stage / f"{key}_metrics.json"
        metrics.write_text(json.dumps({**result.metrics, "run_id": run_id}), encoding="utf-8")
        shutil.move(str(predictions), str(cache / predictions.name))
        shutil.move(str(metrics), str(cache / metrics.name))


def _handover(results_dir, label, bundle, metrics, predictions):
    ids = bundle.metadata["trial_uid"].astype(str)
    indexed = predictions.set_index("trial_uid", verify_integrity=True)
    if not ids.is_unique or len(indexed) != len(ids) or not ids.isin(indexed.index).all():
        raise ValueError(f"{label}: dự đoán thiếu hoặc trùng trial")
    aligned = indexed.loc[ids]
    directory = results_dir / "metrics" / "dl1"
    handover = results_dir / "handover_standard"
    handover.mkdir(parents=True, exist_ok=True)
    prefix = f"ms_cnn_bigru_{label}"
    scores = pd.DataFrame(metrics).rename(columns={"balanced_accuracy": "bal_acc"})
    columns = ["subject_id", "fold", "bal_acc", "roc_auc", "pr_auc", "f1_target", "sensitivity", "specificity"]
    with TemporaryDirectory(dir=handover) as folder:
        stage = Path(folder)
        fold_file = stage / f"{prefix}_per_fold.csv"
        pred_file = stage / f"{prefix}_preds.npz"
        scores[columns].to_csv(fold_file, index=False)
        np.savez_compressed(
            pred_file, subject_id=aligned["subject_id"].to_numpy(dtype=str),
            fold=aligned["fold"].to_numpy(dtype=np.int64), trial_idx=np.arange(len(ids), dtype=np.int64),
            trial_uid=ids.to_numpy(dtype=str), y_true=aligned["y_true"].to_numpy(dtype=np.int64),
            y_prob=aligned["y_prob"].to_numpy(dtype=np.float64), y_pred=aligned["y_pred"].to_numpy(dtype=np.int64),
        )
        csv_file = save_predictions_csv(aligned.reset_index(), stage / f"handover_predictions_dl1_{label}.csv")
        shutil.move(str(csv_file), str(directory / csv_file.name))
        for path in (fold_file, pred_file):
            shutil.move(str(path), str(handover / path.name))


def _save_experiment(results_dir, experiment, config, bundle, results, mode, run_id):
    directory = results_dir / "metrics" / "dl1"
    directory.mkdir(parents=True, exist_ok=True)
    name = config["exp_name"]
    metrics = [result.metrics for result in results]
    predictions = pd.concat([result.predictions for result in results], ignore_index=True)
    agg = aggregate_metrics(metrics, experiment.protocol)
    if mode == "full":
        _handover(results_dir, experiment.label, bundle, metrics, predictions)
    with TemporaryDirectory(dir=directory) as folder:
        stage = Path(folder)
        csv_file = save_predictions_csv(predictions, stage / f"predictions_{name}.csv")
        json_file = save_metrics_json(
            stage / f"metrics_{name}.json", {**config, "run_id": run_id}, bundle.trial_uid_sha256,
            agg["summary"], metrics, agg["per_subject"],
        )
        for path in (csv_file, json_file):
            shutil.move(str(path), str(directory / path.name))
    print(f"  Kết quả: {directory / json_file.name}", flush=True)


def _restore_handover(results_dir, experiment, bundle, folds, results):
    label = experiment.label
    handover = results_dir / "handover_standard"
    prefix = f"ms_cnn_bigru_{label}"
    paths = (
        results_dir / "metrics" / "dl1" / f"handover_predictions_dl1_{label}.csv",
        handover / f"{prefix}_per_fold.csv",
        handover / f"{prefix}_preds.npz",
    )
    if all(path.is_file() and path.stat().st_size for path in paths):
        return
    ordered = [results[fold["fold_key"]] for fold in folds]
    metrics = [result.metrics for result in ordered]
    predictions = pd.concat([result.predictions for result in ordered], ignore_index=True)
    _handover(results_dir, label, bundle, metrics, predictions)


def _run_experiment(experiment, bundle, mode, results_dir):
    folds = _load_folds(experiment, bundle, mode)
    config = _experiment_config(experiment, bundle, mode)
    run_id = _run_id(bundle, folds, config)
    root = results_dir / "checkpoints" / "dl1" / config["exp_name"]
    cache = root / "cache"
    legacy = results_dir / "dl1" / mode / experiment.label
    directory = results_dir / "metrics" / "dl1"
    print(f"  Kiểm tra kết quả đã lưu: {experiment.label} ({len(folds)} fold)", flush=True)
    results = _read_experiment(directory, config, bundle, folds, run_id)
    if len(results) == len(folds):
        print(f"  {experiment.label}: dùng lại đủ {len(folds)} fold đã lưu.", flush=True)
        if mode == "full":
            _restore_handover(results_dir, experiment, bundle, folds, results)
        return

    for fold in folds:
        key = fold["fold_key"]
        if key not in results:
            cached = _read_fold(cache, legacy, bundle, fold, run_id)
            if cached is not None:
                results[key] = cached

    pending = [fold for fold in folds if fold["fold_key"] not in results]
    print(f"  {experiment.label}: dùng lại {len(results)}/{len(folds)} fold, cần train {len(pending)} fold.", flush=True)
    if pending:
        from src.training.trainer_dl1 import train_single_fold

        checkpoint_dir = root if experiment.protocol != "within_subject" else None
        for number, fold in enumerate(pending, start=1):
            key = fold["fold_key"]
            print(f"  [{number}/{len(pending)}] Train {experiment.label} {key}", flush=True)
            trained = train_single_fold(
                bundle, fold, epochs=config["epochs"], patience=PATIENCE, seed=SEED,
                checkpoint_dir=checkpoint_dir,
            )
            metrics = dict(trained["fold_metrics"], best_epoch=trained["best_epoch"], threshold=trained["best_threshold"])
            result = _validate_fold(metrics, trained["predictions_df"], bundle, fold)
            _write_fold(cache, key, result, run_id)
            results[key] = result
            print(f"    ROC-AUC={metrics['roc_auc']:.4f}, BA={metrics['balanced_accuracy']:.4f}", flush=True)

    _save_experiment(results_dir, experiment, config, bundle, [results[f["fold_key"]] for f in folds], mode, run_id)


def _save_subject_summary(results_dir, data_dir, trial_uid_sha256):
    directory = results_dir / "metrics" / "dl1"
    tables = []
    for method in ("within", "loso"):
        name = METHODS[method][0].run_name("full")
        report_path = directory / f"metrics_{name}.json"
        if not report_path.is_file():
            return
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
            if report["trial_uid_sha256"] != trial_uid_sha256:
                return
            table = pd.DataFrame.from_dict(report["per_subject"], orient="index")[list(METRIC_NAMES)]
        except (OSError, ValueError, KeyError, TypeError) as exc:
            print(f"Chưa tổng hợp theo subject được từ {report_path.name}: {exc}", flush=True)
            return
        prefix = "ws" if method == "within" else "loso"
        tables.append(table.add_prefix(f"{prefix}_"))
    summary = pd.concat(tables, axis=1).rename_axis("subject_id")
    qc = data_dir / "preprocessing_qc_summary.csv"
    if qc.is_file():
        summary = summary.join(pd.read_csv(qc).set_index("subject_id")[["n_clean_epochs_kept", "n_frequent", "n_target", "drop_rate_pct"]])
    audit = REPO_DIR / "reports" / "audit_per_subject.csv"
    if audit.is_file():
        summary = summary.join(pd.read_csv(audit).set_index("subject_id")[["hit_rate_pct"]])
    summary.to_csv(directory / "handover_per_subject_summary_dl1.csv")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True, help="Thư mục EEG đã xử lý")
    parser.add_argument("--results-dir", type=Path, required=True, help="Thư mục lưu kết quả")
    parser.add_argument("--method", choices=("all", *METHODS), default="all",
                        help="Mặc định chạy lần lượt Within, LOSO và CS5")
    parser.add_argument("--mode", choices=("full", "pilot", "cs5-full", "cs5-pilot"), default="full")
    args = parser.parse_args(argv)
    method, mode = args.method, args.mode
    if mode.startswith("cs5-"):
        if method not in ("all", "cs5"):
            parser.error("Chế độ cs5-full/cs5-pilot chỉ dùng với phương thức CS5.")
        method, mode = "cs5", mode.removeprefix("cs5-")
    methods = list(METHODS) if method == "all" else [method]
    device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"DL1 MS-CNN-BiGRU | {mode.upper()} | {device}", flush=True)
    print(f"Kết quả: {args.results_dir}", flush=True)
    clean_hash = None
    for step, name in enumerate(methods, start=1):
        print(f"\n[{step}/{len(methods)}] {name.upper()}", flush=True)
        for experiment in METHODS[name]:
            cs5_path = REPO_DIR / "splits" / experiment.reference if name == "cs5" else None
            bundle = load_eeg_dataset(args.data_dir, data_variant=experiment.variant, cs5_path=cs5_path)
            _run_experiment(experiment, bundle, mode, args.results_dir)
            if name in ("within", "loso"):
                clean_hash = bundle.trial_uid_sha256
            del bundle
    if mode == "full" and clean_hash is not None:
        _save_subject_summary(args.results_dir, args.data_dir, clean_hash)
    print("\nHoàn tất DL1.", flush=True)


if __name__ == "__main__":
    main()
