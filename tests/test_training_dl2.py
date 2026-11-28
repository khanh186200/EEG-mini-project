"""Bộ kiểm thử Zero Leakage, Threshold/Metrics và End-to-End Smoke Test cho DL2."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from captum.attr import IntegratedGradients

from scripts.training.train_dl2 import main as train_dl2_main
from src.evaluation.metrics import (
    METRIC_NAMES,
    PREDICTION_COLUMNS,
    aggregate_metrics,
    compute_binary_metrics,
    find_optimal_threshold,
)
from src.training.dataset_dl2 import (
    ChannelStandardScaler,
    EEGTrialDataset,
    load_eeg_dataset,
)
from src.training.trainer_dl2 import load_checkpoint_for_eval, train_single_fold

def _resolve_data_dir() -> Path:
    env_dir = os.environ.get("EEG_DATA_DIR")
    if env_dir:
        return Path(env_dir)
    return Path("data/processed")


PROCESSED_DATA_DIR = _resolve_data_dir()


def test_zero_leakage_scaler_and_volts_precision(tmp_path: Path) -> None:
    """Kiểm tra ChannelStandardScaler chống sập đơn vị 1e-5 V và chống rò rỉ test."""
    rng = np.random.default_rng(42)
    n_channels = 127
    n_times = 251

    # Dữ liệu train ở thang Volts (~1e-5 V) với độ lệch và phương sai khác nhau theo kênh
    ch_means_tr = rng.uniform(-2e-5, 2e-5, size=(1, n_channels, 1))
    ch_stds_tr = rng.uniform(0.5e-5, 2.0e-5, size=(1, n_channels, 1))
    X_train = (ch_means_tr + ch_stds_tr * rng.standard_normal((40, n_channels, n_times))).astype(
        np.float32
    )

    # Dữ liệu test có phân bố dịch mạnh (+3 SD, độ lệch chuẩn gấp 2.5 lần)
    X_test = (
        (ch_means_tr + 3.0 * ch_stds_tr)
        + (2.5 * ch_stds_tr) * rng.standard_normal((16, n_channels, n_times))
    ).astype(np.float32)

    scaler = ChannelStandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    assert X_train_scaled.dtype == np.float32
    assert X_test_scaled.dtype == np.float32
    assert X_train_scaled.shape == (40, n_channels, n_times)
    assert X_test_scaled.shape == (16, n_channels, n_times)

    # Kiểm chứng toàn bộ 127 kênh sau chuẩn hóa train có mean ≈ 0 và std ≈ 1 (không bị eps=1e-5 bóp méo)
    tr_mean = X_train_scaled.mean(axis=(0, 2))
    tr_std = X_train_scaled.std(axis=(0, 2))
    assert np.max(np.abs(tr_mean)) < 1e-5, f"Mean train lệch quá mức: {np.max(np.abs(tr_mean))}"
    assert np.max(np.abs(tr_std - 1.0)) < 1e-3, (
        f"Std train lệch khỏi 1.0 (nghi lỗi đơn vị Volts): max diff={np.max(np.abs(tr_std - 1.0))}"
    )

    # Kiểm chứng X_test giữ nguyên độ lệch phân bố (tuyệt đối không refit trên test)
    te_mean = X_test_scaled.mean(axis=(0, 2))
    te_std = X_test_scaled.std(axis=(0, 2))
    assert np.all(te_mean > 2.0), f"X_test bị refit hoặc mất độ lệch mean: min={te_mean.min()}"
    assert np.all(te_std > 1.8), f"X_test bị refit hoặc mất độ lệch std: min={te_std.min()}"

    # Kiểm chứng state_dict() / from_state_dict() qua torch.save(weights_only=True)
    state_path = tmp_path / "scaler.pt"
    torch.save({"scaler_state_dict": scaler.state_dict()}, state_path)
    loaded_ckpt = torch.load(state_path, map_location="cpu", weights_only=True)
    scaler_reloaded = ChannelStandardScaler.from_state_dict(loaded_ckpt["scaler_state_dict"])

    X_test_reloaded = scaler_reloaded.transform(X_test)
    assert np.array_equal(X_test_scaled, X_test_reloaded)

    # Kiểm chứng EEGTrialDataset trả về shape và dtype đúng chuẩn
    y_dummy = rng.integers(0, 2, size=40, dtype=np.int64)
    ds = EEGTrialDataset(X_train_scaled, y_dummy)
    x0, y0 = ds[0]
    assert x0.shape == (127, 251) and x0.dtype == torch.float32
    assert y0.shape == (1,) and y0.dtype == torch.float32


def test_threshold_tuning_and_metrics() -> None:
    """Kiểm tra find_optimal_threshold, compute_binary_metrics và aggregate_metrics."""
    rng = np.random.default_rng(123)
    n_neg, n_pos = 260, 40  # Tỷ lệ lệch lớp 6.5:1
    y_true = np.concatenate([np.zeros(n_neg, dtype=np.int64), np.ones(n_pos, dtype=np.int64)])
    p_neg = rng.beta(2.0, 8.0, size=n_neg)  # tập trung quanh ~0.20
    p_pos = rng.beta(4.5, 5.5, size=n_pos)  # tập trung quanh ~0.45 (lệch dưới 0.5)
    y_prob = np.concatenate([p_neg, p_pos])

    opt_thr_ba = find_optimal_threshold(y_true, y_prob, metric="balanced_accuracy")
    opt_thr_f1 = find_optimal_threshold(y_true, y_prob, metric="f1_target")
    assert 0.05 <= opt_thr_ba <= 0.95
    assert 0.05 <= opt_thr_f1 <= 0.95

    # Kiểm chứng kẹp [0.05, 0.95] khi xác suất lệch cực trị & phá hòa ưu tiên gần 0.5
    y_ext = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int64)
    p_ext_low = np.array([1e-5, 2e-5, 3e-5, 4e-5, 0.02, 0.025, 0.03, 0.04], dtype=np.float64)
    thr_ext_low = find_optimal_threshold(y_ext, p_ext_low, metric="balanced_accuracy")
    assert 0.05 <= thr_ext_low <= 0.95

    p_ext_sep = np.array([1e-4, 2e-4, 3e-4, 4e-4, 0.9991, 0.9992, 0.9995, 0.9999], dtype=np.float64)
    thr_tie_ba = find_optimal_threshold(y_ext, p_ext_sep, metric="balanced_accuracy")
    thr_tie_f1 = find_optimal_threshold(y_ext, p_ext_sep, metric="f1_target")
    assert np.isclose(thr_tie_ba, 0.5), f"Kỳ vọng phá hòa chọn 0.5, nhận {thr_tie_ba}"
    assert np.isclose(thr_tie_f1, 0.5), f"Kỳ vọng phá hòa chọn 0.5, nhận {thr_tie_f1}"

    m_opt = compute_binary_metrics(y_true, y_prob, threshold=opt_thr_ba)
    m_def = compute_binary_metrics(y_true, y_prob, threshold=0.5)

    for key in METRIC_NAMES:
        assert key in m_opt and np.isfinite(m_opt[key])
    assert m_opt["balanced_accuracy"] >= m_def["balanced_accuracy"]
    assert m_opt["n_samples"] == 300
    assert m_opt["n_pos"] == 40
    assert m_opt["n_neg"] == 260
    assert m_opt["tp"] + m_opt["fn"] == 40
    assert m_opt["tn"] + m_opt["fp"] == 260

    # Edge case chỉ có 1 class -> fallback 0.5 an toàn
    assert find_optimal_threshold(np.zeros(20, dtype=np.int64), np.linspace(0.1, 0.9, 20)) == 0.5
    assert find_optimal_threshold(np.ones(20, dtype=np.int64), np.linspace(0.1, 0.9, 20)) == 0.5
    single_cls_metrics = compute_binary_metrics(
        np.zeros(10, dtype=np.int64), np.linspace(0.1, 0.9, 10), threshold=0.5
    )
    assert np.isnan(single_cls_metrics["roc_auc"])
    assert np.isnan(single_cls_metrics["pr_auc"])
    assert single_cls_metrics["n_neg"] == 10 and single_cls_metrics["n_pos"] == 0

    # Kiểm tra aggregate_metrics cho within_subject (trung bình theo subject trước)
    within_folds = [
        {"protocol": "within_subject", "subject_id": "sub-03", "fold": 0, **{m: 0.70 for m in METRIC_NAMES}},
        {"protocol": "within_subject", "subject_id": "sub-03", "fold": 1, **{m: 0.80 for m in METRIC_NAMES}},
        {"protocol": "within_subject", "subject_id": "sub-05", "fold": 0, **{m: 0.90 for m in METRIC_NAMES}},
        {"protocol": "within_subject", "subject_id": "sub-05", "fold": 1, **{m: 0.90 for m in METRIC_NAMES}},
    ]
    agg_within = aggregate_metrics(within_folds, protocol="within_subject")
    assert agg_within["aggregation_unit"] == "subject"
    assert agg_within["n_subjects"] == 2
    assert np.isclose(agg_within["per_subject"]["sub-03"]["balanced_accuracy"], 0.75)
    assert np.isclose(agg_within["per_subject"]["sub-05"]["balanced_accuracy"], 0.90)
    # Mean trên 2 subjects = (0.75 + 0.90) / 2 = 0.825 (khác với trung bình thô theo fold nếu số fold lệch)
    assert np.isclose(agg_within["summary"]["balanced_accuracy"]["mean"], 0.825)


def test_dataset_loader_sha256_and_no_ica(tmp_path: Path) -> None:
    """Kiểm tra load_eeg_dataset với clean (khóa SHA-256) và no_ica (đồng bộ 5-fold)."""
    if not (PROCESSED_DATA_DIR / "X_clean.npy").exists():
        pytest.skip("Không có data/processed để kiểm tra loader.")

    clean_bundle = load_eeg_dataset(PROCESSED_DATA_DIR, data_variant="clean")
    assert clean_bundle.X.shape[1:] == (127, 251)
    assert clean_bundle.X.dtype == np.float32

    no_ica_bundle = load_eeg_dataset(PROCESSED_DATA_DIR, data_variant="no_ica")
    assert no_ica_bundle.X.shape[1:] == (127, 251)
    assert no_ica_bundle.X.shape[0] >= clean_bundle.X.shape[0]
    assert len(no_ica_bundle.splits["cross_subject_5fold"]) == len(
        clean_bundle.splits["cross_subject_5fold"]
    )
    for f_clean, f_no_ica in zip(
        clean_bundle.splits["cross_subject_5fold"],
        no_ica_bundle.splits["cross_subject_5fold"],
    ):
        assert f_clean["train_subjects"] == f_no_ica["train_subjects"]
        assert f_clean["val_subjects"] == f_no_ica["val_subjects"]
        assert f_clean["test_subjects"] == f_no_ica["test_subjects"]
        tr, va, te = set(f_no_ica["train_idx"]), set(f_no_ica["val_idx"]), set(f_no_ica["test_idx"])
        assert tr.isdisjoint(va) and tr.isdisjoint(te) and va.isdisjoint(te)

    # Giả lập sai SHA-256 -> bắt buộc raise ValueError (chỉ ghi lát cắt 8 trials đầu)
    bad_dir = tmp_path / "bad_sha"
    bad_dir.mkdir()
    np.save(bad_dir / "X_clean.npy", clean_bundle.X[:8])
    np.save(bad_dir / "y.npy", clean_bundle.y[:8])
    np.save(bad_dir / "subject_ids.npy", clean_bundle.subject_ids[:8])
    clean_bundle.metadata.iloc[:8].to_csv(bad_dir / "metadata.csv", index=False)
    with open(PROCESSED_DATA_DIR / "splits.json", "r", encoding="utf-8") as f:
        tampered_splits = json.load(f)
    tampered_splits["trial_uid_sha256"] = "0" * 64
    with open(bad_dir / "splits.json", "w", encoding="utf-8") as f:
        json.dump(tampered_splits, f)

    with pytest.raises(ValueError, match="SHA-256"):
        load_eeg_dataset(bad_dir, data_variant="clean")

    # Kiểm tra train_single_fold chặn chồng chéo train/val/test
    with pytest.raises(ValueError, match="chồng chéo"):
        train_single_fold(
            X=clean_bundle.X,
            y=clean_bundle.y,
            train_idx=[0, 1, 2, 3],
            val_idx=[3, 4, 5],
            test_idx=[6, 7, 8],
            model_name="p300_atcnet",
            epochs=1,
            save_checkpoint=False,
        )


def _assert_smoke_outputs_and_captum(
    output_dir: Path,
    exp_name: str,
    ckpt_filename: str,
    device: str,
    bundle_X: np.ndarray,
) -> None:
    pred_path = output_dir / "metrics" / "dl2" / f"predictions_{exp_name}.csv"
    metrics_path = output_dir / "metrics" / "dl2" / f"metrics_{exp_name}.json"
    ckpt_path = output_dir / "checkpoints" / "dl2" / exp_name / ckpt_filename

    assert pred_path.exists(), f"Thiếu predictions CSV: {pred_path}"
    assert metrics_path.exists(), f"Thiếu metrics JSON: {metrics_path}"
    assert ckpt_path.exists(), f"Thiếu checkpoint .pt: {ckpt_path}"

    df_pred = pd.read_csv(pred_path)
    assert list(df_pred.columns) == PREDICTION_COLUMNS
    assert len(df_pred) > 0
    assert not df_pred.isna().any().any()
    assert ((df_pred["y_prob"] >= 0.0) & (df_pred["y_prob"] <= 1.0)).all()
    assert set(df_pred["y_pred"].unique()).issubset({0, 1})

    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics_payload = json.load(f)
    for top_key in ("config", "trial_uid_sha256", "summary", "per_subject", "per_fold"):
        assert top_key in metrics_payload
    for m in METRIC_NAMES:
        assert m in metrics_payload["summary"]
        assert np.isfinite(metrics_payload["summary"][m]["mean"])

    # Reload checkpoint + scaler_state_dict và kiểm chứng forward() + Captum IG trên 2 mẫu test
    model, scaler, ckpt = load_checkpoint_for_eval(ckpt_path, device=device)
    for req_key in (
        "model_name",
        "model_kwargs",
        "model_state_dict",
        "scaler_state_dict",
        "best_threshold",
        "best_epoch",
        "best_val_metric",
        "fold_meta",
        "seed",
    ):
        assert req_key in ckpt

    x_sample_scaled = scaler.transform(bundle_X[:2])
    x_tensor = torch.from_numpy(x_sample_scaled).to(device)
    with torch.inference_mode():
        logits = model(x_tensor)
    assert logits.shape == (2, 1)
    assert torch.isfinite(logits).all()

    ig = IntegratedGradients(model)
    baselines = torch.zeros_like(x_tensor)
    attributions = ig.attribute(x_tensor, baselines=baselines, target=0, n_steps=20)
    assert attributions.shape == (2, 127, 251)
    assert torch.isfinite(attributions).all()
    assert attributions.abs().sum().item() > 0.0


def test_end_to_end_smoke_on_processed_data(tmp_path: Path) -> None:
    """Smoke test 2 epochs trên data/processed cho P300ATCNet + Captum IG."""
    if not (PROCESSED_DATA_DIR / "X_clean.npy").exists():
        pytest.skip("Không tìm thấy data/processed để chạy smoke test.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    bundle = load_eeg_dataset(PROCESSED_DATA_DIR, data_variant="clean")

    # 1. P300ATCNet (avg_pool) trên 1 fold cross_subject_5fold
    t1 = time.perf_counter()
    rc2 = train_dl2_main(
        [
            "--data-dir",
            str(PROCESSED_DATA_DIR),
            "--data-variant",
            "clean",
            "--model",
            "p300_atcnet",
            "--readout",
            "avg_pool",
            "--protocol",
            "cross_subject_5fold",
            "--folds",
            "0",
            "--epochs",
            "2",
            "--batch-size",
            "64",
            "--device",
            device,
            "--output-dir",
            str(tmp_path),
            "--exp-name",
            "smoke_atcnet",
        ]
    )
    atcnet_sec = time.perf_counter() - t1
    assert rc2 == 0
    _assert_smoke_outputs_and_captum(
        output_dir=tmp_path,
        exp_name="smoke_atcnet",
        ckpt_filename="fold0.pt",
        device=device,
        bundle_X=bundle.X,
    )

    # 2. Kiểm tra --resume bỏ qua fold đã cache thành công
    t2 = time.perf_counter()
    rc3 = train_dl2_main(
        [
            "--data-dir",
            str(PROCESSED_DATA_DIR),
            "--data-variant",
            "clean",
            "--model",
            "p300_atcnet",
            "--readout",
            "avg_pool",
            "--protocol",
            "cross_subject_5fold",
            "--folds",
            "0",
            "--epochs",
            "2",
            "--device",
            device,
            "--resume",
            "--output-dir",
            str(tmp_path),
            "--exp-name",
            "smoke_atcnet",
        ]
    )
    resume_sec = time.perf_counter() - t2
    assert rc3 == 0

    print(
        f"\n[Smoke Timing | device={device.upper()}] "
        f"P300ATCNet cross-5fold (fold0, 2 ep): {atcnet_sec:.3f}s | "
        f"Resume cached fold: {resume_sec:.3f}s"
    )
