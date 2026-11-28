"""CLI chạy thực nghiệm huấn luyện & đánh giá DL2 chuẩn Zero Leakage.

Ví dụ:
    py -3.14 scripts/training/train_dl2.py --data-dir data/processed --model p300_atcnet \
        --protocol cross_subject_5fold --epochs 2 --exp-name smoke_atcnet
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

# Đảm bảo import được `src.*` khi chạy trực tiếp từ thư mục con `scripts/training`.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.evaluation.metrics import (
    METRIC_NAMES,
    PREDICTION_COLUMNS,
    aggregate_metrics,
    save_metrics_json,
    save_predictions_csv,
)
from src.training.dataset_dl2 import (
    VALID_DATA_VARIANTS,
    VALID_PROTOCOLS,
    get_protocol_folds,
    load_eeg_dataset,
)
from src.training.trainer_dl2 import train_single_fold


def _default_exp_name(args: argparse.Namespace) -> str:
    shared_tag = "shared" if args.shared_branch else "indep"
    return (
        f"p300_atcnet_{args.readout}_w{args.n_windows}_{shared_tag}_"
        f"{args.protocol}_{args.data_variant}_s{args.seed}"
    )


def _resolve_model_kwargs(args: argparse.Namespace) -> Dict[str, Any]:
    dropout_val = 0.3 if args.dropout is None else float(args.dropout)
    spatial_norm_val = 2.0 if args.spatial_max_norm is None else float(args.spatial_max_norm)
    return {
        "readout": args.readout,
        "n_windows": int(args.n_windows),
        "shared_branch": bool(args.shared_branch),
        "tcn_filters": int(args.tcn_filters),
        "temp_kernel_1": int(args.temp_kernel_1),
        "spatial_max_norm": spatial_norm_val,
        "cv_dropout": dropout_val,
        "at_dropout": dropout_val,
        "tcn_dropout": dropout_val,
    }


def _try_load_cached_fold(
    ckpt_dir: Path,
    cache_dir: Path,
    fold_key: str,
) -> Optional[tuple[Dict[str, Any], pd.DataFrame]]:
    ckpt_path = ckpt_dir / f"{fold_key}.pt"
    pred_path = cache_dir / f"{fold_key}_pred.csv"
    metrics_path = cache_dir / f"{fold_key}_metrics.json"
    if not (ckpt_path.exists() and pred_path.exists() and metrics_path.exists()):
        return None
    try:
        df_pred = pd.read_csv(pred_path)
        if list(df_pred.columns) != PREDICTION_COLUMNS or df_pred.isna().any().any():
            return None
        with open(metrics_path, "r", encoding="utf-8") as f:
            fold_metrics = json.load(f)
        return fold_metrics, df_pred
    except Exception:
        return None


def _save_cached_fold(
    cache_dir: Path,
    fold_key: str,
    fold_metrics: Dict[str, Any],
    predictions_df: pd.DataFrame,
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    predictions_df[PREDICTION_COLUMNS].to_csv(cache_dir / f"{fold_key}_pred.csv", index=False)
    with open(cache_dir / f"{fold_key}_metrics.json", "w", encoding="utf-8") as f:
        json.dump(fold_metrics, f, indent=2)


def run_experiment(args: argparse.Namespace) -> Dict[str, Any]:
    """Chạy toàn bộ vòng lặp thực nghiệm theo tham số CLI và xuất file kết quả chuẩn."""
    exp_name = args.exp_name or _default_exp_name(args)
    output_dir = Path(args.output_dir)
    metrics_dir = output_dir / "metrics" / "dl2"
    ckpt_dir = output_dir / "checkpoints" / "dl2" / exp_name
    cache_dir = ckpt_dir / "cache"

    bundle = load_eeg_dataset(data_dir=args.data_dir, data_variant=args.data_variant)
    fold_specs = get_protocol_folds(
        bundle.splits,
        protocol=args.protocol,
        subjects=args.subjects,
        folds=args.folds,
    )
    if not fold_specs:
        raise ValueError(
            f"Không có fold nào khớp bộ lọc: protocol={args.protocol}, "
            f"subjects={args.subjects}, folds={args.folds}."
        )

    model_kwargs = _resolve_model_kwargs(args)
    trial_uids = bundle.metadata["trial_uid"].astype(str).tolist()
    subject_ids = bundle.subject_ids.astype(str).tolist()

    per_fold_results: List[Dict[str, Any]] = []
    prediction_frames: List[pd.DataFrame] = []

    print(
        f"[train_dl2] exp={exp_name} | model={args.model} | variant={args.data_variant} | "
        f"protocol={args.protocol} | n_folds={len(fold_specs)} | epochs={args.epochs}"
    )

    for idx, spec in enumerate(fold_specs):
        fold_key = spec["fold_key"]
        if args.resume:
            cached = _try_load_cached_fold(ckpt_dir, cache_dir, fold_key)
            if cached is not None:
                cached_metrics, cached_pred_df = cached
                per_fold_results.append(cached_metrics)
                prediction_frames.append(cached_pred_df)
                print(f"  [{idx + 1}/{len(fold_specs)}] {fold_key}: RESUME (cached)")
                continue

        fold_meta = {
            "fold": spec["fold"],
            "subject_id": spec["subject_id"],
            "protocol": spec["protocol"],
            "fold_key": fold_key,
        }
        for k in ("train_subjects", "val_subjects", "test_subjects"):
            if k in spec:
                fold_meta[k] = spec[k]

        res = train_single_fold(
            X=bundle.X,
            y=bundle.y,
            train_idx=spec["train_idx"],
            val_idx=spec["val_idx"],
            test_idx=spec["test_idx"],
            model_name=args.model,
            model_kwargs=model_kwargs,
            protocol=args.protocol,
            fold_key=fold_key,
            fold_meta=fold_meta,
            trial_uids=trial_uids,
            subject_ids=subject_ids,
            epochs=args.epochs,
            patience=args.patience,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            early_stop_metric=args.early_stop_metric,
            threshold_metric=args.threshold_metric,
            seed=args.seed,
            device=args.device,
            use_amp=args.use_amp,
            checkpoint_dir=ckpt_dir,
            save_checkpoint=args.save_checkpoints,
        )

        fm = res["fold_metrics"]
        pred_df = res["predictions_df"]
        per_fold_results.append(fm)
        prediction_frames.append(pred_df)

        if args.save_checkpoints:
            _save_cached_fold(cache_dir, fold_key, fm, pred_df)

        print(
            f"  [{idx + 1}/{len(fold_specs)}] {fold_key}: "
            f"bAcc={fm['balanced_accuracy']:.4f} | ROC={fm['roc_auc']:.4f} | "
            f"PR={fm['pr_auc']:.4f} | thr={fm['threshold']:.3f} | "
            f"ep={fm['best_epoch']} | {fm['train_time_sec']:.2f}s"
        )

    all_preds_df = pd.concat(prediction_frames, ignore_index=True)
    pred_csv_path = save_predictions_csv(
        all_preds_df, metrics_dir / f"predictions_{exp_name}.csv"
    )

    agg = aggregate_metrics(per_fold_results, protocol=args.protocol)
    config_payload = {
        "exp_name": exp_name,
        "data_dir": str(args.data_dir),
        "data_variant": args.data_variant,
        "model": args.model,
        "model_kwargs": model_kwargs,
        "protocol": args.protocol,
        "subjects": list(args.subjects) if args.subjects else None,
        "folds": list(args.folds) if args.folds is not None else None,
        "epochs": int(args.epochs),
        "patience": int(args.patience),
        "batch_size": int(args.batch_size),
        "effective_batch_size": (
            min(int(args.batch_size), 32)
            if args.protocol == "within_subject"
            else int(args.batch_size)
        ),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "seed": int(args.seed),
        "early_stop_metric": args.early_stop_metric,
        "threshold_metric": args.threshold_metric,
        "use_amp": bool(args.use_amp),
        "save_checkpoints": bool(args.save_checkpoints),
    }

    metrics_json_path = save_metrics_json(
        out_path=metrics_dir / f"metrics_{exp_name}.json",
        config=config_payload,
        trial_uid_sha256=bundle.trial_uid_sha256,
        summary=agg["summary"],
        per_fold=per_fold_results,
        per_subject=agg["per_subject"],
    )

    summary_str = " | ".join(
        f"{m}={agg['summary'][m]['mean']:.4f}+/-{agg['summary'][m]['std']:.4f}"
        for m in METRIC_NAMES
    )
    print(f"[train_dl2] Summary ({agg['aggregation_unit']}): {summary_str}")
    print(f"[train_dl2] Saved: {pred_csv_path} & {metrics_json_path}")

    return {
        "exp_name": exp_name,
        "predictions_path": pred_csv_path,
        "metrics_path": metrics_json_path,
        "checkpoint_dir": ckpt_dir,
        "summary": agg["summary"],
        "per_subject": agg["per_subject"],
        "per_fold": per_fold_results,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Huấn luyện và đánh giá các mô hình DL nhánh 2 (P300-ATCNet)."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/processed"),
        help="Thư mục dữ liệu đã tiền xử lý (mặc định: data/processed).",
    )
    parser.add_argument(
        "--data-variant",
        type=str,
        choices=list(VALID_DATA_VARIANTS),
        default="clean",
        help="Biến thể dữ liệu: clean hoặc no_ica (mặc định: clean).",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["p300_atcnet"],
        default="p300_atcnet",
        help="Kiến trúc mô hình: p300_atcnet.",
    )
    parser.add_argument(
        "--protocol",
        type=str,
        choices=list(VALID_PROTOCOLS),
        default="cross_subject_5fold",
        help="Giao thức chia tập: within_subject | cross_subject_5fold | cross_subject_loso.",
    )
    parser.add_argument(
        "--subjects",
        nargs="+",
        default=None,
        help="Tùy chọn lọc danh sách subject (vd: sub-03 sub-05 sub-07).",
    )
    parser.add_argument(
        "--folds",
        nargs="+",
        type=int,
        default=None,
        help="Tùy chọn lọc chỉ số fold (vd: 0 1).",
    )

    # Tham số cấu hình P300ATCNet
    parser.add_argument(
        "--readout",
        type=str,
        choices=["avg_pool", "attn_pool", "last_step"],
        default="avg_pool",
        help="Chế độ readout cho P300ATCNet (mặc định: avg_pool).",
    )
    parser.add_argument(
        "--n-windows",
        type=int,
        default=2,
        help="Số cửa sổ trượt cho P300ATCNet (mặc định: 2).",
    )
    parser.add_argument(
        "--shared-branch",
        action="store_true",
        help="Dùng chung 1 nhánh AT+TC giữa các cửa sổ trượt của P300ATCNet.",
    )
    parser.add_argument(
        "--tcn-filters",
        type=int,
        default=32,
        help="Số kênh đầu ra TCN cho P300ATCNet (mặc định: 32).",
    )
    parser.add_argument(
        "--temp-kernel-1",
        type=int,
        default=125,
        help="Độ dài temporal kernel đầu tiên của P300ATCNet (mặc định: 125).",
    )
    parser.add_argument(
        "--spatial-max-norm",
        type=float,
        default=None,
        help="Ngưỡng L2 max-norm lớp spatial_conv (mặc định: 2.0).",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=None,
        help="Tỷ lệ dropout (mặc định: 0.3).",
    )

    # Hyperparameters huấn luyện
    parser.add_argument("--epochs", type=int, default=80, help="Số epoch tối đa (mặc định: 80).")
    parser.add_argument(
        "--patience",
        type=int,
        default=15,
        help="Số epoch kiên nhẫn cho Early Stopping (mặc định: 15).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size (mặc định: 64; tự giới hạn <= 32 cho within_subject).",
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate AdamW (mặc định: 1e-3).")
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-2,
        help="Weight decay AdamW (mặc định: 1e-2).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed cố định (mặc định: 42).")
    parser.add_argument(
        "--early-stop-metric",
        type=str,
        choices=["val_roc_auc", "val_pr_auc"],
        default="val_roc_auc",
        help="Chỉ số theo dõi trên Validation để Early Stopping (mặc định: val_roc_auc).",
    )
    parser.add_argument(
        "--threshold-metric",
        type=str,
        choices=["balanced_accuracy", "f1_target"],
        default="balanced_accuracy",
        help="Chỉ số tối ưu ngưỡng trên Validation (mặc định: balanced_accuracy).",
    )
    parser.add_argument(
        "--use-amp",
        action="store_true",
        help="Bật mixed precision AMP (mặc định tắt, giữ FP32 ổn định).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Thiết bị chạy (vd: cuda, cpu); mặc định tự chọn CUDA nếu sẵn có.",
    )

    # Quản lý thực nghiệm & output
    parser.add_argument(
        "--exp-name",
        type=str,
        default=None,
        help="Tên định danh thực nghiệm; nếu bỏ trống sẽ tự sinh theo cấu hình.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Bỏ qua các fold đã có sẵn checkpoint .pt và cache dự đoán.",
    )
    parser.add_argument(
        "--save-checkpoints",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Có lưu checkpoint .pt theo từng fold hay không (mặc định: True).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
        help="Thư mục gốc lưu metrics và checkpoints (mặc định: results).",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    run_experiment(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
