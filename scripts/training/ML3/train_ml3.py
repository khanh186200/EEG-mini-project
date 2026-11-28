"""CLI entrypoint for ML3 P300 XGBoost experiments (CS5, LOSO & Within-Subject) with `/content` SSD staging.

Colab / VS Code Colab Extension fast workflow (stages features from Drive -> `/content/features`,
saves checkpoints & plots back to Google Drive)::

    python scripts/training/ML3/train_ml3.py \
      --features-dir /content/drive/MyDrive/EEG-mini-project/features \
      --stage-to-local --local-features-dir /content/features \
      --splits-dir splits \
      --protocol cs5 \
      --experiments temporal wavelet_w5 wavelet_w8 combined_w5 combined_w8 \
      --output-dir /content/drive/MyDrive/EEG-mini-project/results/ml3 \
      --device cuda \
      --resume

LOSO evaluation::

    python scripts/training/ML3/train_ml3.py \
      --features-dir /content/drive/MyDrive/EEG-mini-project/features \
      --stage-to-local --local-features-dir /content/features \
      --splits-dir splits \
      --protocol loso \
      --experiments temporal combined_w5 combined_w8 \
      --output-dir /content/drive/MyDrive/EEG-mini-project/results/ml3 \
      --device cuda \
      --resume

Within-Subject 5-Fold evaluation (default: `combined_w5`)::

    python scripts/training/ML3/train_ml3.py \
      --features-dir /content/drive/MyDrive/EEG-mini-project/features \
      --stage-to-local --local-features-dir /content/features \
      --splits-dir splits \
      --protocol within \
      --experiments combined_w5 \
      --output-dir /content/drive/MyDrive/EEG-mini-project/results/ml3 \
      --device cuda \
      --resume
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Dict, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.ml3_feature_store import connect_google_drive, stage_features_to_local
from src.models.Ml3 import ML3XGBoostConfig
from src.training.ml3_training import (
    CANONICAL_EXPERIMENT_ORDER,
    build_ml3_execution_plan,
    normalize_protocol_name,
    required_feature_groups_for_experiments,
    run_ml3_experiments,
    validate_experiment_list,
)


def setup_colab_drive_paths(
    drive_project_dir: Path | str = "/content/drive/MyDrive/EEG-mini-project",
    *,
    stage_to_content: bool = True,
    local_features_dir: Path | str = "/content/features",
    mount_point: Path | str = "/content/drive",
    force_remount: bool = False,
) -> Dict[str, Path]:
    """Verify Google Drive mount and optionally stage ``features/`` into ``/content/features``."""
    connect_google_drive(mount_point=mount_point, force_remount=force_remount)
    project_root = Path(drive_project_dir)
    drive_features_dir = project_root / "features"
    temporal_dir = drive_features_dir / "temporal"
    wavelet_dir = drive_features_dir / "wavelet"
    output_dir = project_root / "results" / "ml3"

    missing = [str(p) for p in (temporal_dir, wavelet_dir) if not p.is_dir()]
    if missing:
        raise FileNotFoundError(
            f"Đã kết nối Google Drive nhưng không tìm thấy thư mục feature bắt buộc: {missing}"
        )

    active_features_dir = (
        stage_features_to_local(
            drive_features_dir,
            local_features_dir,
            required_groups=("temporal", "wavelet"),
            mount_drive=False,
            mount_point=mount_point,
        )
        if stage_to_content
        else drive_features_dir
    )

    return {
        "project_dir": project_root,
        "drive_features_dir": drive_features_dir,
        "features_dir": active_features_dir,
        "temporal_dir": active_features_dir / "temporal",
        "wavelet_dir": active_features_dir / "wavelet",
        "output_dir": output_dir,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train and evaluate ML3 XGBoost feature scenarios on CS5 and LOSO."
    )
    parser.add_argument(
        "--features-dir",
        type=Path,
        required=True,
        help="Root directory containing features/temporal and features/wavelet (Drive or /content/features).",
    )
    parser.add_argument(
        "--stage-to-local",
        action="store_true",
        help="Copy required features from --features-dir (e.g. Google Drive) to --local-features-dir (/content/features) before loading.",
    )
    parser.add_argument(
        "--local-features-dir",
        type=Path,
        default=Path("/content/features"),
        help="Local Colab SSD directory for staged features when --stage-to-local is set (default: /content/features).",
    )
    parser.add_argument(
        "--splits-dir",
        type=Path,
        default=Path("splits"),
        help="Directory containing cross_5fold.json, loso.json, and within_5fold.json (default: splits).",
    )
    parser.add_argument(
        "--protocol",
        choices=("cs5", "loso", "within"),
        required=True,
        help="Evaluation protocol: 'cs5' (Cross-Subject 5-Fold), 'loso', or 'within' (Within-Subject 5-Fold).",
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        choices=CANONICAL_EXPERIMENT_ORDER,
        default=None,
        help=(
            "Feature scenarios to run. Defaults to all 5 main scenarios for cs5, "
            "(temporal, combined_w5, combined_w8) for loso, and (combined_w5,) for within."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Root output directory for ML3 checkpoints, metrics, and plots (e.g. results/ml3 on Drive).",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="XGBoost execution device ('cuda' or 'cpu'; default: cuda).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse completed fold checkpoints when config_hash matches.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing fold checkpoints when not using --resume.",
    )
    parser.add_argument(
        "--folds",
        nargs="+",
        type=int,
        default=None,
        help="Optional subset of fold indices to execute.",
    )
    parser.add_argument(
        "--subjects",
        nargs="+",
        type=str,
        default=None,
        help="Optional subset of subject IDs to execute (useful for within-subject runs).",
    )
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Validate splits and feature directories and print execution plan without training.",
    )
    parser.add_argument(
        "--mount-drive",
        action="store_true",
        help="Explicitly attempt Google Drive mount before accessing feature/output paths.",
    )
    parser.add_argument(
        "--drive-mount-point",
        type=Path,
        default=Path("/content/drive"),
        help="Colab Google Drive mount point (default: /content/drive).",
    )
    parser.add_argument(
        "--allow-gamma-loso",
        action="store_true",
        help="Explicitly permit running *_gamma experiments under LOSO (disabled by default).",
    )
    parser.add_argument(
        "--threshold-metric",
        choices=("balanced_accuracy", "f1_target"),
        default="balanced_accuracy",
        help="Validation metric used to select decision threshold (default: balanced_accuracy).",
    )

    xgb_group = parser.add_argument_group("XGBoost Hyperparameters (Fixed Policy Defaults)")
    xgb_group.add_argument("--n-estimators", type=int, default=2000)
    xgb_group.add_argument("--learning-rate", type=float, default=0.05)
    xgb_group.add_argument("--max-depth", type=int, default=3)
    xgb_group.add_argument("--min-child-weight", type=float, default=10.0)
    xgb_group.add_argument("--subsample", type=float, default=0.8)
    xgb_group.add_argument("--colsample-bytree", type=float, default=0.15)
    xgb_group.add_argument("--reg-alpha", type=float, default=0.1)
    xgb_group.add_argument("--reg-lambda", type=float, default=5.0)
    xgb_group.add_argument(
        "--min-split-loss",
        type=float,
        default=0.0,
        help="XGBoost split loss regularization (named min_split_loss to avoid EEG gamma confusion).",
    )
    xgb_group.add_argument("--max-bin", type=int, default=256)
    xgb_group.add_argument("--early-stopping-rounds", type=int, default=100)
    xgb_group.add_argument("--seed", type=int, default=42)
    xgb_group.add_argument("--n-jobs", type=int, default=-1)
    return parser


def _build_model_config(args: argparse.Namespace) -> ML3XGBoostConfig:
    return ML3XGBoostConfig(
        device=args.device,
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        max_depth=args.max_depth,
        min_child_weight=args.min_child_weight,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        reg_alpha=args.reg_alpha,
        reg_lambda=args.reg_lambda,
        min_split_loss=args.min_split_loss,
        max_bin=args.max_bin,
        early_stopping_rounds=args.early_stopping_rounds,
        random_state=args.seed,
        n_jobs=args.n_jobs,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.resume and args.overwrite:
        raise ValueError("Cannot combine --resume and --overwrite simultaneously")

    model_config = _build_model_config(args)
    norm_protocol = normalize_protocol_name(args.protocol)
    chosen_experiments = validate_experiment_list(
        norm_protocol, args.experiments, allow_gamma_loso=args.allow_gamma_loso
    )
    req_groups = required_feature_groups_for_experiments(chosen_experiments)

    effective_features_dir = args.features_dir
    if args.stage_to_local and not args.plan_only:
        effective_features_dir = stage_features_to_local(
            args.features_dir,
            args.local_features_dir,
            required_groups=req_groups,
            mount_drive=args.mount_drive,
            mount_point=args.drive_mount_point,
        )

    if args.plan_only:
        plan = build_ml3_execution_plan(
            features_dir=effective_features_dir,
            splits_dir=args.splits_dir,
            protocol=norm_protocol,
            experiments=chosen_experiments,
            output_dir=args.output_dir,
            model_config=model_config,
            folds_filter=args.folds,
            subjects_filter=args.subjects,
            threshold_metric=args.threshold_metric,
            allow_gamma_loso=args.allow_gamma_loso,
            mount_drive=args.mount_drive,
            drive_mount_point=args.drive_mount_point,
        )
        if args.stage_to_local:
            plan["stage_to_local"] = {
                "source_features_dir": str(args.features_dir),
                "local_features_dir": str(args.local_features_dir),
            }
        print(json.dumps(plan, indent=2))
        print("Plan only: no model was trained.")
        return 0

    result = run_ml3_experiments(
        features_dir=effective_features_dir,
        splits_dir=args.splits_dir,
        protocol=norm_protocol,
        experiments=chosen_experiments,
        output_dir=args.output_dir,
        model_config=model_config,
        folds_filter=args.folds,
        subjects_filter=args.subjects,
        threshold_metric=args.threshold_metric,
        resume=args.resume,
        overwrite=args.overwrite,
        allow_gamma_loso=args.allow_gamma_loso,
        mount_drive=args.mount_drive,
        drive_mount_point=args.drive_mount_point,
        verbose=True,
    )

    comp = result.get("comparison", {})
    plots = comp.get("plots", [])
    print(f"\nCompleted ML3 {result['protocol'].upper()} experiments -> {result['output_dir']}")
    for exp_name, status_counts in result["fold_statuses"].items():
        sm = result["experiments"][exp_name]["test"]["subject_macro"]
        fp = result["experiments"][exp_name]["test"]["fold_pooled"]
        print(
            f"  - {exp_name}: trained={status_counts['trained']}, reused={status_counts['reused']} | "
            f"Subject-Macro PR-AUC={sm['pr_auc']['mean']:.4f}±{sm['pr_auc']['std']:.4f}, "
            f"ROC-AUC={sm['roc_auc']['mean']:.4f}±{sm['roc_auc']['std']:.4f} | "
            f"Fold-Pooled PR-AUC={fp['pr_auc']['mean']:.4f}±{fp['pr_auc']['std']:.4f}"
        )
    if plots:
        print(f"Generated plots: {', '.join(plots)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
