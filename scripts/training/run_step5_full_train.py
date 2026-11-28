"""Điều phối Full Training Bước 5 (39 subjects), kiểm tra toàn vẹn, xuất file bàn giao và sinh báo cáo.

Chạy tuần tự theo thứ tự ưu tiên với `--epochs 80 --patience 15 --lr 1e-3 --weight-decay 1e-2 --seed 42 --resume`:
  Nhóm 1 (BẮT BUỘC):
    1. dl2_ws_p300_atcnet        (within_subject, 39 subjects = 195 folds, bs=32, clean)
    2. dl2_loso_p300_atcnet      (cross_subject_loso, 39 folds, bs=64, clean)
    3. dl2_cs5_p300_atcnet_clean (cross_subject_5fold, 5 folds, bs=64, clean)
    4. dl2_cs5_p300_atcnet_no_ica(cross_subject_5fold, 5 folds, bs=64, no_ica)

Sau khi train xong:
  - Kiểm tra toàn vẹn 20,886 dòng dự đoán khớp 100% `data/processed/metadata.csv`, không NaN/Inf, threshold in [0.05, 0.95].
  - Xuất `handover_predictions_dl2_within.csv`, `handover_predictions_dl2_loso.csv`, `handover_per_subject_summary_dl2.csv` tại `results/metrics/dl2/`.
  - Sinh báo cáo `reports/STEP5_FULL_RESULTS.md` gồm 4 bảng chuẩn.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.training.train_dl2 import build_arg_parser as build_train_arg_parser
from scripts.training.train_dl2 import run_experiment
from src.evaluation.metrics import METRIC_NAMES, PREDICTION_COLUMNS
from src.training.dataset_dl2 import natural_sort_key


@dataclass(frozen=True)
class Step5ExpSpec:
    """Đặc tả một thực nghiệm huấn luyện trong Bước 5."""

    exp_name: str
    group: int
    display_name: str
    cli_args: Tuple[str, ...]


ATCNET_MODEL_ARGS: Tuple[str, ...] = (
    "--model", "p300_atcnet",
    "--readout", "attn_pool",
    "--n-windows", "2",
    "--tcn-filters", "32",
    "--temp-kernel-1", "125",
    "--spatial-max-norm", "2.0",
    "--dropout", "0.3",
)

GROUP1_SPECS: Tuple[Step5ExpSpec, ...] = (
    Step5ExpSpec(
        exp_name="dl2_ws_p300_atcnet",
        group=1,
        display_name="P300-ATCNet (attn_pool) — Within-Subject",
        cli_args=(
            "--protocol", "within_subject",
            "--batch-size", "32",
            "--data-variant", "clean",
            *ATCNET_MODEL_ARGS,
        ),
    ),
    Step5ExpSpec(
        exp_name="dl2_loso_p300_atcnet",
        group=1,
        display_name="P300-ATCNet (attn_pool) — Cross-Subject LOSO",
        cli_args=(
            "--protocol", "cross_subject_loso",
            "--batch-size", "64",
            "--data-variant", "clean",
            *ATCNET_MODEL_ARGS,
        ),
    ),
    Step5ExpSpec(
        exp_name="dl2_cs5_p300_atcnet_clean",
        group=1,
        display_name="P300-ATCNet (attn_pool) — CS 5-Fold (clean)",
        cli_args=(
            "--protocol", "cross_subject_5fold",
            "--batch-size", "64",
            "--data-variant", "clean",
            *ATCNET_MODEL_ARGS,
        ),
    ),
    Step5ExpSpec(
        exp_name="dl2_cs5_p300_atcnet_no_ica",
        group=1,
        display_name="P300-ATCNet (attn_pool) — CS 5-Fold (no_ica)",
        cli_args=(
            "--protocol", "cross_subject_5fold",
            "--batch-size", "64",
            "--data-variant", "no_ica",
            *ATCNET_MODEL_ARGS,
        ),
    ),
)


def _cleanup_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _run_spec(
    spec: Step5ExpSpec,
    data_dir: Path,
    output_dir: Path,
    common_args: Sequence[str],
) -> Dict[str, Any]:
    parser = build_train_arg_parser()
    argv = [
        "--exp-name", spec.exp_name,
        "--data-dir", str(data_dir),
        "--output-dir", str(output_dir),
        *common_args,
        *spec.cli_args,
    ]
    args = parser.parse_args(argv)
    t0 = time.perf_counter()
    res = run_experiment(args)
    wall = time.perf_counter() - t0
    _cleanup_memory()
    print(f"[step5] Finished {spec.exp_name} in {wall:.2f}s ({wall / 60.0:.2f}m)")
    return res


def _load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _validate_predictions_integrity(
    pred_path: Path,
    metadata_df: pd.DataFrame,
    expected_protocol: str,
) -> pd.DataFrame:
    """Kiểm tra toàn vẹn 20,886 dòng dự đoán khớp metadata.csv và threshold in [0.05, 0.95]."""
    if not pred_path.exists():
        raise FileNotFoundError(f"Thiếu file predictions bắt buộc: {pred_path.resolve()}")

    df = pd.read_csv(pred_path)
    expected_n = len(metadata_df)
    assert list(df.columns) == PREDICTION_COLUMNS, (
        f"{pred_path.name}: cột không khớp {list(df.columns)} != {PREDICTION_COLUMNS}"
    )
    assert len(df) == expected_n, (
        f"{pred_path.name}: kỳ vọng {expected_n} dòng, thực tế có {len(df)} dòng."
    )
    assert not df.isna().any().any(), f"{pred_path.name}: chứa giá trị NaN."
    assert np.isfinite(df["y_prob"].to_numpy(dtype=np.float64)).all(), (
        f"{pred_path.name}: y_prob chứa Inf."
    )
    assert df["trial_uid"].is_unique, f"{pred_path.name}: có trial_uid trùng lặp."

    meta_uids = metadata_df["trial_uid"].astype(str).tolist()
    pred_uids = df["trial_uid"].astype(str).tolist()
    assert set(pred_uids) == set(meta_uids), (
        f"{pred_path.name}: tập trial_uid không khớp 100% với metadata.csv."
    )
    assert ((df["threshold"] >= 0.05 - 1e-9) & (df["threshold"] <= 0.95 + 1e-9)).all(), (
        f"{pred_path.name}: phát hiện threshold ngoài [0.05, 0.95] "
        f"(min={df['threshold'].min()}, max={df['threshold'].max()})."
    )
    assert set(df["protocol"].astype(str).unique()) == {expected_protocol}, (
        f"{pred_path.name}: protocol không khớp {expected_protocol}."
    )

    # Sắp xếp đúng thứ tự dòng của metadata.csv và đối chiếu y_true
    aligned = df.set_index("trial_uid").loc[meta_uids].reset_index()[PREDICTION_COLUMNS]
    assert np.array_equal(
        aligned["y_true"].to_numpy(dtype=np.int64),
        metadata_df["label"].to_numpy(dtype=np.int64),
    ), f"{pred_path.name}: y_true không khớp metadata['label']."
    assert np.array_equal(
        aligned["subject_id"].astype(str).to_numpy(),
        metadata_df["subject_id"].astype(str).to_numpy(),
    ), f"{pred_path.name}: subject_id không khớp metadata['subject_id']."

    return aligned


def export_handover_files(
    data_dir: Path,
    metrics_dir: Path,
    audit_csv_path: Path,
) -> Dict[str, Path]:
    """Kiểm tra toàn vẹn và xuất 3 file bàn giao chuẩn hóa tại `results/metrics/dl2/`."""
    meta_df = pd.read_csv(data_dir / "metadata.csv")
    assert len(meta_df) == 20886, f"metadata.csv có {len(meta_df)} dòng, kỳ vọng 20886."

    # 1. Validate tất cả prediction CSV của Bước 5
    ws_atc_aligned = _validate_predictions_integrity(
        metrics_dir / "predictions_dl2_ws_p300_atcnet.csv",
        meta_df,
        "within_subject",
    )
    loso_atc_aligned = _validate_predictions_integrity(
        metrics_dir / "predictions_dl2_loso_p300_atcnet.csv",
        meta_df,
        "cross_subject_loso",
    )

    # 2. Xuất 2 file predictions bàn giao cho Khánh & TV4
    handover_ws_path = metrics_dir / "handover_predictions_dl2_within.csv"
    handover_loso_path = metrics_dir / "handover_predictions_dl2_loso.csv"
    ws_atc_aligned.to_csv(handover_ws_path, index=False)
    loso_atc_aligned.to_csv(handover_loso_path, index=False)

    # 3. Xuất bảng tổng hợp per-subject (39 dòng) sẵn sàng chạy Wilcoxon / Paired t-test
    qc_df = pd.read_csv(data_dir / "preprocessing_qc_summary.csv")
    qc_map = {str(r["subject_id"]): r.to_dict() for _, r in qc_df.iterrows()}

    audit_map: Dict[str, Dict[str, Any]] = {}
    if audit_csv_path.exists():
        audit_df = pd.read_csv(audit_csv_path)
        audit_map = {str(r["subject_id"]): r.to_dict() for _, r in audit_df.iterrows()}

    ws_atc_json = _load_json(metrics_dir / "metrics_dl2_ws_p300_atcnet.json")
    loso_atc_json = _load_json(metrics_dir / "metrics_dl2_loso_p300_atcnet.json")

    subjects = sorted(qc_map.keys(), key=natural_sort_key)
    assert len(subjects) == 39, f"Kỳ vọng 39 subjects, nhận {len(subjects)}."

    rows: List[Dict[str, Any]] = []
    for sub_id in subjects:
        qc = qc_map[sub_id]
        aud = audit_map.get(sub_id, {})
        row: Dict[str, Any] = {
            "subject_id": sub_id,
            "n_clean_epochs_kept": int(qc["n_clean_epochs_kept"]),
            "n_frequent": int(qc["n_frequent"]),
            "n_target": int(qc["n_target"]),
            "drop_rate_pct": float(qc["drop_rate_pct"]),
            "hit_rate_pct": float(aud["hit_rate_pct"]) if "hit_rate_pct" in aud else float("nan"),
        }
        for prefix, jdata in (("ws", ws_atc_json), ("loso", loso_atc_json)):
            sub_entry = jdata["per_subject"][sub_id]
            for m in METRIC_NAMES:
                row[f"{prefix}_{m}"] = float(sub_entry[m])
        rows.append(row)

    summary_df = pd.DataFrame(rows)
    assert len(summary_df) == 39
    handover_summary_path = metrics_dir / "handover_per_subject_summary_dl2.csv"
    summary_df.to_csv(handover_summary_path, index=False)

    return {
        "within": handover_ws_path,
        "loso": handover_loso_path,
        "summary": handover_summary_path,
    }


def _compute_run_timing_stats(
    metrics_data: Dict[str, Any],
) -> Dict[str, float]:
    per_fold = metrics_data["per_fold"]
    cfg = metrics_data["config"]
    max_epochs = int(cfg["epochs"])
    patience = int(cfg["patience"])
    total_sec = float(sum(float(f["train_time_sec"]) for f in per_fold))
    total_epochs = int(
        sum(min(int(f["best_epoch"]) + patience, max_epochs) for f in per_fold)
    )
    n_folds = len(per_fold)
    return {
        "n_folds": float(n_folds),
        "total_epochs": float(total_epochs),
        "total_sec": total_sec,
        "sec_per_fold": total_sec / max(n_folds, 1),
        "sec_per_epoch": total_sec / max(total_epochs, 1),
    }


def _fmt_mean_sd(mean_val: float, std_val: float, decimals: int = 4) -> str:
    if not math.isfinite(mean_val):
        return "NaN"
    return f"{mean_val:.{decimals}f} ± {std_val:.{decimals}f}"


def _subset_summary(
    per_subject: Dict[str, Dict[str, Any]],
    subject_ids: Sequence[str],
) -> Dict[str, Tuple[float, float]]:
    out: Dict[str, Tuple[float, float]] = {}
    for m in METRIC_NAMES:
        vals = np.array([float(per_subject[s][m]) for s in subject_ids], dtype=np.float64)
        out[m] = (
            float(np.mean(vals)),
            float(np.std(vals, ddof=1 if len(vals) > 1 else 0)),
        )
    return out


def generate_step5_report(
    data_dir: Path,
    metrics_dir: Path,
    audit_csv_path: Path,
    report_path: Path,
) -> Path:
    """Sinh báo cáo `reports/STEP5_FULL_RESULTS.md` gồm đúng 4 bảng phân tích."""
    loaded: Dict[str, Dict[str, Any]] = {
        "ws_atc": _load_json(metrics_dir / "metrics_dl2_ws_p300_atcnet.json"),
        "loso_atc": _load_json(metrics_dir / "metrics_dl2_loso_p300_atcnet.json"),
        "cs5_clean": _load_json(metrics_dir / "metrics_dl2_cs5_p300_atcnet_clean.json"),
        "cs5_no_ica": _load_json(metrics_dir / "metrics_dl2_cs5_p300_atcnet_no_ica.json"),
    }

    qc_df = pd.read_csv(data_dir / "preprocessing_qc_summary.csv")
    qc_map = {str(r["subject_id"]): r.to_dict() for _, r in qc_df.iterrows()}
    audit_df = pd.read_csv(audit_csv_path) if audit_csv_path.exists() else pd.DataFrame()
    audit_map = (
        {str(r["subject_id"]): r.to_dict() for _, r in audit_df.iterrows()}
        if not audit_df.empty
        else {}
    )
    all_39_subs = sorted(qc_map.keys(), key=natural_sort_key)

    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    lines: List[str] = [
        "# BƯỚC 5 — BÁO CÁO KẾT QUẢ FULL TRAINING (39 SUBJECTS)",
        "",
        f"- **Tập dữ liệu**: `{data_dir.as_posix()}` (39 subjects sạch, 20,886 epochs)",
        f"- **Phần cứng thực thi**: `{gpu_name}`",
        "- **Mô hình chủ lực DL2**: `P300ATCNet(readout='attn_pool', n_windows=2, shared_branch=False, tcn_filters=32, temp_kernel_1=125, spatial_max_norm=2.0, dropout=0.3)`",
        "- **Hai bản vá kỹ thuật áp dụng trước Full Run**:",
        "  1. Kẹp ngưỡng quyết định `find_optimal_threshold()` gọn trong `[0.05, 0.95]` (ưu tiên gần `0.5` khi hòa).",
        "  2. Tối ưu GPU-Resident (`ChannelStandardScaler` `float64` trên GPU + mini-batch GPU `torch.randperm` + `spatial_conv` cuBLAS batched GEMM trong `trainer_dl2.py`).",
        "",
    ]

    # ------------------------------------------------------------------
    # BẢNG 1 — Tốc độ sau tối ưu GPU-resident
    # ------------------------------------------------------------------
    lines.extend([
        "### Bảng 1 — Tốc độ huấn luyện sau tối ưu GPU-Resident trên RTX 5060 Ti",
        "",
        "| Thực nghiệm | Số folds | Tổng epochs | Giây/epoch | Giây/fold | Tổng thời gian |",
        "| :--- | ---: | ---: | ---: | ---: | ---: |",
    ])


    speed_rows: List[Tuple[str, str]] = [
        ("`dl2_ws_p300_atcnet` (Within, 39 subs)", "ws_atc"),
        ("`dl2_loso_p300_atcnet` (LOSO, 39 subs)", "loso_atc"),
        ("`dl2_cs5_p300_atcnet_clean` (CS5 clean)", "cs5_clean"),
        ("`dl2_cs5_p300_atcnet_no_ica` (CS5 no_ica)", "cs5_no_ica"),
    ]

    total_step5_sec = 0.0
    for label, key in speed_rows:
        st5 = _compute_run_timing_stats(loaded[key])
        total_step5_sec += st5["total_sec"]
        lines.append(
            f"| {label} | {int(st5['n_folds'])} | {int(st5['total_epochs']):,} | "
            f"{st5['sec_per_epoch']:.3f}s | {st5['sec_per_fold']:.2f}s | "
            f"{st5['total_sec']:.1f}s ({st5['total_sec'] / 60.0:.2f}m) |"
        )

    lines.append("")
    lines.append(
        f"> Tổng thời gian huấn luyện thực tế của các thực nghiệm Bước 5: **`{total_step5_sec:.1f}s` (`{total_step5_sec / 60.0:.2f} phút`)**."
    )
    lines.append("")

    # ------------------------------------------------------------------
    # BẢNG 2 — Hiệu năng Full 39 Subjects & Kiểm định ghép cặp
    # ------------------------------------------------------------------
    lines.extend([
        "### Bảng 2 — Hiệu năng Full 39 Subjects & So sánh đối chứng (`Mean ± SD`)",
        "",
        "| Giao thức | Mô hình / Biến thể | Folds / Unit | ROC-AUC | PR-AUC | Balanced Acc | F1-Target | Sensitivity | Specificity |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ])

    perf_rows: List[Tuple[str, str, str, Dict[str, Any]]] = [
        ("Within-Subject", "P300-ATCNet (`attn_pool`)", "195 / 39 subs", loaded["ws_atc"]),
        ("Cross-Subject LOSO", "P300-ATCNet (`attn_pool`)", "39 / 39 subs", loaded["loso_atc"]),
    ]
    perf_rows.extend([
        ("Cross-Subject 5-Fold", "P300-ATCNet (`clean`)", "5 / fold", loaded["cs5_clean"]),
        ("Cross-Subject 5-Fold", "P300-ATCNet (`no_ica`)", "5 / fold", loaded["cs5_no_ica"]),
    ])

    ordered_6 = ("roc_auc", "pr_auc", "balanced_accuracy", "f1_target", "sensitivity", "specificity")
    for proto_label, model_label, unit_label, jdata in perf_rows:
        s = jdata["summary"]
        vals_str = " | ".join(
            _fmt_mean_sd(float(s[m]["mean"]), float(s[m]["std"])) for m in ordered_6
        )
        lines.append(f"| {proto_label} | {model_label} | {unit_label} | {vals_str} |")

    lines.append("")

    # ------------------------------------------------------------------
    # BẢNG 3 — Sensitivity Analysis (39 vs 37 vs 31 subjects)
    # ------------------------------------------------------------------
    subs_drop_gt30 = {s for s in all_39_subs if float(qc_map[s]["drop_rate_pct"]) > 30.0}
    subs_hit_lt90 = {
        s for s in all_39_subs
        if s in audit_map and float(audit_map[s]["hit_rate_pct"]) < 90.0
    }
    subs_37 = [s for s in all_39_subs if s not in subs_drop_gt30]
    subs_31 = [s for s in all_39_subs if s not in subs_hit_lt90]

    lines.extend([
        "### Bảng 3 — Sensitivity Analysis (39 subjects vs. 37 subjects loại `drop > 30%` vs. 31 subjects loại `Hit Rate < 90%`)",
        "",
        f"- **37 subjects**: loại `{', '.join(sorted(subs_drop_gt30, key=natural_sort_key))}` (`drop_rate_pct > 30%`).",
        f"- **31 subjects**: loại `{', '.join(sorted(subs_hit_lt90, key=natural_sort_key))}` (`Hit Rate < 90%`).",
        "",
        "| Giao thức & Mô hình | Cohort | N | ROC-AUC | PR-AUC | Balanced Acc | F1-Target | Sensitivity | Specificity |",
        "| :--- | :--- | ---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ])

    sens_targets: List[Tuple[str, Dict[str, Any]]] = [
        ("Within-Subject (`P300-ATCNet`)", loaded["ws_atc"]),
        ("Cross-Subject LOSO (`P300-ATCNet`)", loaded["loso_atc"]),
    ]

    for name, jdata in sens_targets:
        per_sub = jdata["per_subject"]
        sub39 = _subset_summary(per_sub, all_39_subs)
        sub37 = _subset_summary(per_sub, subs_37)
        sub31 = _subset_summary(per_sub, subs_31)
        for cohort_label, n_sub, stats_dict in (
            ("Full (chuẩn)", len(all_39_subs), sub39),
            ("Loại drop > 30%", len(subs_37), sub37),
            ("Loại Hit < 90%", len(subs_31), sub31),
        ):
            vals_str = " | ".join(
                _fmt_mean_sd(stats_dict[m][0], stats_dict[m][1]) for m in ordered_6
            )
            lines.append(f"| {name} | {cohort_label} | {n_sub} | {vals_str} |")
        d37 = " | ".join(f"{sub37[m][0] - sub39[m][0]:+.4f}" for m in ordered_6)
        d31 = " | ".join(f"{sub31[m][0] - sub39[m][0]:+.4f}" for m in ordered_6)
        lines.append(f"| {name} | **Δ (37 − 39)** | — | {d37} |")
        lines.append(f"| {name} | **Δ (31 − 39)** | — | {d31} |")
    lines.append("")

    # ------------------------------------------------------------------
    # BẢNG 4 — Top 5 subjects cao nhất & thấp nhất (Within & LOSO)
    # ------------------------------------------------------------------
    lines.extend([
        "### Bảng 4 — Top 5 Subjects Hiệu năng Cao nhất và Thấp nhất (`P300-ATCNet attn_pool`)",
        "",
        "| Giao thức | Nhóm xếp hạng | Subject | ROC-AUC | PR-AUC | Balanced Acc | F1-Target | Clean Epochs (Target) | Drop Rate (%) | Hit Rate (%) |",
        "| :--- | :--- | :--- | ---: | ---: | ---: | ---: | :---: | ---: | ---: |",
    ])

    for proto_label, jdata in (
        ("Within-Subject", loaded["ws_atc"]),
        ("Cross-Subject LOSO", loaded["loso_atc"]),
    ):
        per_sub = jdata["per_subject"]
        ranked = sorted(
            all_39_subs,
            key=lambda s: (float(per_sub[s]["roc_auc"]), float(per_sub[s]["pr_auc"])),
            reverse=True,
        )
        top5 = ranked[:5]
        bot5 = list(reversed(ranked[-5:]))
        for group_label, group_subs in (("Top 5 Cao nhất", top5), ("Bottom 5 Thấp nhất", bot5)):
            for idx_rank, s in enumerate(group_subs, start=1):
                rec = per_sub[s]
                qc = qc_map[s]
                hit_val = (
                    f"{float(audit_map[s]['hit_rate_pct']):.2f}%"
                    if s in audit_map
                    else "—"
                )
                lines.append(
                    f"| {proto_label} | {group_label} #{idx_rank} | `{s}` | "
                    f"{float(rec['roc_auc']):.4f} | {float(rec['pr_auc']):.4f} | "
                    f"{float(rec['balanced_accuracy']):.4f} | {float(rec['f1_target']):.4f} | "
                    f"{int(qc['n_clean_epochs_kept'])} ({int(qc['n_target'])}) | "
                    f"{float(qc['drop_rate_pct']):.2f}% | {hit_val} |"
                )
    lines.append("")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def run_step5_pipeline(
    data_dir: Path,
    output_dir: Path,
    report_path: Path,
    audit_csv_path: Path,
    *,
    report_only: bool = False,
) -> Path:
    """Chạy toàn bộ pipeline Bước 5, kiểm tra toàn vẹn, xuất bàn giao và sinh báo cáo."""
    metrics_dir = output_dir / "metrics" / "dl2"
    common_args = [
        "--epochs", "80",
        "--patience", "15",
        "--lr", "1e-3",
        "--weight-decay", "1e-2",
        "--seed", "42",
        "--resume",
    ]

    if not report_only:
        # Nhóm 1 — Bắt buộc
        for idx, spec in enumerate(GROUP1_SPECS, start=1):
            print(f"\n========== [Nhóm 1 | {idx}/{len(GROUP1_SPECS)}] {spec.exp_name} ==========")
            _run_spec(spec, data_dir, output_dir, common_args)


    handover_paths = export_handover_files(
        data_dir=data_dir,
        metrics_dir=metrics_dir,
        audit_csv_path=audit_csv_path,
    )
    print(
        "[step5] Exported handover files: "
        + ", ".join(f"{k}={v}" for k, v in handover_paths.items())
    )

    saved_report = generate_step5_report(
        data_dir=data_dir,
        metrics_dir=metrics_dir,
        audit_csv_path=audit_csv_path,
        report_path=report_path,
    )
    print(f"[step5] Generated report: {saved_report}")
    return saved_report


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Chạy Full Training Bước 5 (39 subjects), kiểm tra toàn vẹn & xuất báo cáo."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--report-path", type=Path, default=Path("reports/STEP5_FULL_RESULTS.md"))
    parser.add_argument(
        "--audit-csv", type=Path, default=Path("reports/audit_per_subject.csv")
    )
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args(argv)

    run_step5_pipeline(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        report_path=args.report_path,
        audit_csv_path=args.audit_csv,
        report_only=args.report_only,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
