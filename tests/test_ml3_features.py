from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.features.ml3_features import (
    DEFAULT_TIME_WINDOWS,
    FeatureExtractionConfig,
    SubjectEpochs,
    TemporalFeatureExtractor,
    WaveletConfig,
    WaveletFeatureExtractor,
    extract_subject_features,
)
from src.features.ml3_feature_store import (
    extract_feature_groups,
    load_processed_feature_source,
)


class _MeanOnlyExtractor:
    feature_group = "placeholder"

    def transform(self, epochs: np.ndarray, *, sampling_rate_hz: float):
        assert sampling_rate_hz == 250.0
        return epochs.mean(axis=2), [f"placeholder_ch_{i}" for i in range(epochs.shape[1])]

    def configuration(self):
        return {"feature_group": self.feature_group}


def test_generic_feature_shell_batches_without_selecting_real_features() -> None:
    epochs = np.arange(5 * 2 * 4, dtype=np.float32).reshape(5, 2, 4)
    data = SubjectEpochs(
        subject_id="sub-01",
        epochs=epochs,
        epoch_ids=np.arange(5),
        targets=np.array([0, 1, 0, 0, 1]),
    )
    table = extract_subject_features(
        data,
        _MeanOnlyExtractor(),
        FeatureExtractionConfig(batch_size=2),
    )
    assert table.values.shape == (5, 2)
    assert table.values.dtype == np.float32
    assert table.feature_names == ("placeholder_ch_0", "placeholder_ch_1")
    assert table.subject_ids.tolist() == ["sub-01"] * 5


def _feature_index(names, *, group: str, window: str, feature: str) -> int:
    expected = f"{group}|channel=Cz|window={window}|feature={feature}"
    return tuple(names).index(expected)


def test_temporal_features_preserve_polarity_and_expected_count() -> None:
    times = np.linspace(-0.2, 0.8, 251)
    epochs = np.stack(
        (
            np.full((1, 251), 2e-6),
            np.full((1, 251), -3e-6),
        )
    ).astype(np.float32)
    extractor = TemporalFeatureExtractor(times, ["Cz"])
    values, names = extractor.transform(epochs, sampling_rate_hz=250.0)

    assert values.shape == (2, len(DEFAULT_TIME_WINDOWS) * 7)
    mean_idx = _feature_index(
        names, group="temporal", window="post_0_200", feature="mean_amplitude"
    )
    auc_idx = _feature_index(
        names, group="temporal", window="post_0_200", feature="absolute_auc"
    )
    polarity_idx = _feature_index(
        names, group="temporal", window="post_0_200", feature="polarity_index"
    )
    slope_idx = _feature_index(
        names, group="temporal", window="post_0_200", feature="regression_slope"
    )
    assert values[0, mean_idx] == pytest.approx(2e-6)
    assert values[1, mean_idx] == pytest.approx(-3e-6)
    assert values[:, auc_idx].tolist() == pytest.approx([4e-7, 6e-7])
    assert values[:, polarity_idx].tolist() == pytest.approx([1.0, -1.0])
    assert values[:, slope_idx].tolist() == pytest.approx([0.0, 0.0], abs=1e-18)


def test_temporal_results_do_not_depend_on_outer_batch_size() -> None:
    rng = np.random.default_rng(42)
    times = np.linspace(-0.2, 0.8, 251)
    data = SubjectEpochs(
        "sub-01",
        rng.normal(size=(5, 2, 251)).astype(np.float32),
        np.asarray([f"trial-{i}" for i in range(5)]),
        np.array([0, 1, 0, 0, 1]),
    )
    extractor = TemporalFeatureExtractor(times, ["Cz", "Pz"])
    one = extract_subject_features(data, extractor, FeatureExtractionConfig(batch_size=1))
    four = extract_subject_features(data, extractor, FeatureExtractionConfig(batch_size=4))
    np.testing.assert_allclose(one.values, four.values, rtol=0.0, atol=0.0)
    assert one.feature_names == four.feature_names


def test_wavelet_features_detect_alpha_sine_when_pywavelets_available() -> None:
    pytest.importorskip("pywt")
    times = np.linspace(-0.2, 0.8, 251)
    signal = np.sin(2.0 * np.pi * 10.0 * times)[None, None, :]
    extractor = WaveletFeatureExtractor(
        times,
        ["Cz"],
        config=WaveletConfig(signal_batch_size=1),
    )
    values, names = extractor.transform(signal, sampling_rate_hz=250.0)
    alpha_idx = _feature_index(
        names, group="wavelet", window="post_300_500", feature="alpha_energy"
    )
    energy_indices = [
        _feature_index(
            names,
            group="wavelet",
            window="post_300_500",
            feature=f"{band}_energy",
        )
        for band in ("delta", "theta", "alpha", "beta", "gamma")
    ]
    entropy_idx = _feature_index(
        names, group="wavelet", window="post_300_500", feature="wavelet_entropy"
    )
    assert values.shape == (1, len(DEFAULT_TIME_WINDOWS) * 6)
    assert values[0, alpha_idx] == pytest.approx(max(values[0, energy_indices]))
    assert 0.0 <= values[0, entropy_idx] <= 1.0


def test_resumable_subject_store_with_temporal_features(tmp_path) -> None:
    processed = tmp_path / "processed"
    processed.mkdir()
    times = np.linspace(-0.2, 0.8, 251)
    rng = np.random.default_rng(7)
    np.save(processed / "X_clean.npy", rng.normal(size=(4, 1, 251)).astype(np.float32))
    np.save(processed / "y.npy", np.array([0, 1, 0, 1], dtype=np.int64))
    np.save(processed / "subject_ids.npy", np.array(["sub-01", "sub-01", "sub-02", "sub-02"]))
    np.save(processed / "times.npy", times)
    (processed / "channel_names.json").write_text(json.dumps(["Cz"]), encoding="utf-8")
    pd.DataFrame(
        {
            "trial_uid": [f"trial-{i}" for i in range(4)],
            "subject_id": ["sub-01", "sub-01", "sub-02", "sub-02"],
            "label": [0, 1, 0, 1],
        }
    ).to_csv(processed / "metadata.csv", index=False)

    source = load_processed_feature_source(processed)
    extractor = TemporalFeatureExtractor(source.times_s, source.channel_names)
    output = tmp_path / "features"
    first = extract_feature_groups(
        source,
        output,
        [extractor],
        FeatureExtractionConfig(batch_size=1),
    )
    second = extract_feature_groups(
        source,
        output,
        [extractor],
        FeatureExtractionConfig(batch_size=2),
    )
    assert {record["status"] for record in first["files"]} == {"written"}
    assert {record["status"] for record in second["files"]} == {"reused"}
    assert (output / "feature_manifest.json").is_file()
    with np.load(output / "temporal/sub-01.npz", allow_pickle=False) as saved:
        assert saved["features"].shape == (2, 56)
        assert saved["trial_uid"].astype(str).tolist() == ["trial-0", "trial-1"]
