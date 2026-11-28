"""ML3 temporal and continuous-wavelet feature extraction.

The implementation is independent from every deep-learning data loader. It
operates on NumPy arrays/memmaps, extracts one subject in bounded batches, and
keeps identifiers beside every feature row for Within/LOSO use.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Protocol, Sequence, runtime_checkable

import numpy as np
from scipy.integrate import trapezoid

try:
    import pywt
except ImportError:  # pragma: no cover - covered by the clear runtime error below
    pywt = None


TEMPORAL_STATISTICS = (
    "mean_amplitude",
    "std",
    "min",
    "max",
    "absolute_auc",
    "polarity_index",
    "regression_slope",
)

FREQUENCY_BANDS_HZ = (
    ("delta", 1.0, 4.0),
    ("theta", 4.0, 8.0),
    ("alpha", 8.0, 13.0),
    ("beta", 13.0, 30.0),
    ("gamma", 30.0, 40.0),
)


@dataclass(frozen=True)
class TimeWindow:
    name: str
    start_s: float
    stop_s: float

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("window name must not be empty")
        if self.stop_s <= self.start_s:
            raise ValueError("window stop_s must be greater than start_s")


DEFAULT_TIME_WINDOWS = (
    TimeWindow("baseline_m200_0", -0.2, 0.0),
    TimeWindow("post_0_200", 0.0, 0.2),
    TimeWindow("post_100_300", 0.1, 0.3),
    TimeWindow("post_200_400", 0.2, 0.4),
    TimeWindow("post_300_500", 0.3, 0.5),
    TimeWindow("post_400_600", 0.4, 0.6),
    TimeWindow("post_500_700", 0.5, 0.7),
    TimeWindow("post_600_800", 0.6, 0.8),
)


@dataclass(frozen=True)
class FeatureExtractionConfig:
    """Shared runtime settings; feature definitions live in their extractors."""

    sampling_rate_hz: float = 250.0
    batch_size: int = 16
    output_dtype: str = "float32"

    def __post_init__(self) -> None:
        if self.sampling_rate_hz <= 0:
            raise ValueError("sampling_rate_hz must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if not np.issubdtype(np.dtype(self.output_dtype), np.floating):
            raise ValueError("output_dtype must be a floating dtype")


@dataclass(frozen=True)
class WaveletConfig:
    """Reproducible CWT settings for the agreed five frequency bands."""

    wavelet: str = "cmor1.5-1.0"
    frequency_step_hz: float = 1.0
    method: str = "fft"
    signal_batch_size: int = 32

    def __post_init__(self) -> None:
        if self.frequency_step_hz <= 0:
            raise ValueError("frequency_step_hz must be positive")
        if self.method not in {"conv", "fft", "auto"}:
            raise ValueError("CWT method must be conv, fft, or auto")
        if self.signal_batch_size <= 0:
            raise ValueError("signal_batch_size must be positive")


@dataclass(frozen=True)
class SubjectEpochs:
    """One subject's epochs and identifiers, potentially backed by a memmap."""

    subject_id: str
    epochs: np.ndarray
    epoch_ids: np.ndarray
    targets: np.ndarray

    def __post_init__(self) -> None:
        if not self.subject_id:
            raise ValueError("subject_id must not be empty")
        if self.epochs.ndim != 3:
            raise ValueError("epochs must have shape (n_epochs, n_channels, n_times)")
        n_epochs = self.epochs.shape[0]
        if len(self.epoch_ids) != n_epochs or len(self.targets) != n_epochs:
            raise ValueError("epoch_ids and targets must align one-to-one with epochs")
        if not set(np.unique(self.targets).tolist()).issubset({0, 1}):
            raise ValueError("targets must be binary values in {0, 1}")


@dataclass(frozen=True)
class FeatureTable:
    """Feature matrix with the identifiers needed by Within and LOSO splits."""

    values: np.ndarray
    feature_names: tuple[str, ...]
    subject_ids: np.ndarray
    epoch_ids: np.ndarray
    targets: np.ndarray

    def __post_init__(self) -> None:
        if self.values.ndim != 2:
            raise ValueError("values must have shape (n_epochs, n_features)")
        n_epochs, n_features = self.values.shape
        if n_features != len(self.feature_names):
            raise ValueError("feature_names must align one-to-one with feature columns")
        if any(len(arr) != n_epochs for arr in (self.subject_ids, self.epoch_ids, self.targets)):
            raise ValueError("identifiers and targets must align one-to-one with feature rows")
        if len(set(self.feature_names)) != len(self.feature_names):
            raise ValueError("feature_names must be unique")
        if not np.isfinite(self.values).all():
            raise ValueError("feature matrix contains NaN or infinite values")


@runtime_checkable
class FeatureExtractor(Protocol):
    """Interface shared by temporal and wavelet extractors."""

    feature_group: str

    def transform(
        self,
        epochs: np.ndarray,
        *,
        sampling_rate_hz: float,
    ) -> tuple[np.ndarray, Sequence[str]]:
        """Return ``(features, names)`` for one epoch batch."""

    def configuration(self) -> dict[str, object]:
        """Return JSON-serializable settings for the feature manifest."""


def _window_mask(times_s: np.ndarray, window: TimeWindow) -> np.ndarray:
    tolerance = max(np.finfo(np.float64).eps * 64, 1e-12)
    return (times_s >= window.start_s - tolerance) & (times_s <= window.stop_s + tolerance)


def _validate_axes(
    times_s: np.ndarray,
    channel_names: Sequence[str],
    windows: Sequence[TimeWindow],
) -> tuple[np.ndarray, tuple[str, ...], tuple[TimeWindow, ...]]:
    times = np.asarray(times_s, dtype=np.float64)
    channels = tuple(str(name) for name in channel_names)
    resolved_windows = tuple(windows)
    if times.ndim != 1 or len(times) < 2 or not np.all(np.diff(times) > 0):
        raise ValueError("times_s must be a strictly increasing one-dimensional array")
    if not channels or len(set(channels)) != len(channels):
        raise ValueError("channel_names must be non-empty and unique")
    if not resolved_windows or len({window.name for window in resolved_windows}) != len(resolved_windows):
        raise ValueError("windows must be non-empty and have unique names")
    tolerance = np.finfo(np.float64).eps * 64
    for window in resolved_windows:
        if window.start_s < times[0] - tolerance or window.stop_s > times[-1] + tolerance:
            raise ValueError(f"window {window.name} lies outside the epoch time axis")
        if np.count_nonzero(_window_mask(times, window)) < 2:
            raise ValueError(f"window {window.name} contains fewer than two samples")
    return times, channels, resolved_windows


def _sampling_rate_from_times(times_s: np.ndarray) -> float:
    return float(1.0 / np.median(np.diff(times_s)))


def _validate_epoch_batch(
    epochs: np.ndarray,
    times_s: np.ndarray,
    channel_names: Sequence[str],
    sampling_rate_hz: float,
) -> np.ndarray:
    values = np.asarray(epochs)
    if values.ndim != 3:
        raise ValueError("epochs must have shape (n_epochs, n_channels, n_times)")
    if values.shape[1:] != (len(channel_names), len(times_s)):
        raise ValueError("epoch channel/time dimensions do not match extractor axes")
    if not np.isfinite(values).all():
        raise ValueError("input epochs contain NaN or infinite values")
    inferred_rate = _sampling_rate_from_times(times_s)
    if not np.isclose(inferred_rate, sampling_rate_hz, rtol=1e-4, atol=1e-4):
        raise ValueError(
            f"sampling rate mismatch: times imply {inferred_rate:.6f} Hz, "
            f"received {sampling_rate_hz:.6f} Hz"
        )
    return values.astype(np.float64, copy=False)


def _feature_name(group: str, channel: str, window: str, feature: str) -> str:
    return f"{group}|channel={channel}|window={window}|feature={feature}"


class TemporalFeatureExtractor:
    """Seven agreed time-domain statistics for every channel/window."""

    feature_group = "temporal"

    def __init__(
        self,
        times_s: np.ndarray,
        channel_names: Sequence[str],
        windows: Sequence[TimeWindow] = DEFAULT_TIME_WINDOWS,
    ) -> None:
        self.times_s, self.channel_names, self.windows = _validate_axes(
            times_s, channel_names, windows
        )
        self._feature_names = tuple(
            _feature_name(self.feature_group, channel, window.name, statistic)
            for window in self.windows
            for channel in self.channel_names
            for statistic in TEMPORAL_STATISTICS
        )

    def configuration(self) -> dict[str, object]:
        return {
            "feature_group": self.feature_group,
            "statistics": list(TEMPORAL_STATISTICS),
            "windows": [asdict(window) for window in self.windows],
            "auc": "trapezoidal positive/negative area over seconds",
            "polarity_index": "(positive_auc-negative_auc)/(positive_auc+negative_auc)",
            "slope": "least-squares amplitude per second",
        }

    def transform(
        self,
        epochs: np.ndarray,
        *,
        sampling_rate_hz: float,
    ) -> tuple[np.ndarray, Sequence[str]]:
        data = _validate_epoch_batch(
            epochs, self.times_s, self.channel_names, sampling_rate_hz
        )
        window_features: list[np.ndarray] = []
        for window in self.windows:
            mask = _window_mask(self.times_s, window)
            window_times = self.times_s[mask]
            signal = data[..., mask]
            mean = signal.mean(axis=-1)
            std = signal.std(axis=-1, ddof=0)
            minimum = signal.min(axis=-1)
            maximum = signal.max(axis=-1)
            positive_auc = trapezoid(np.maximum(signal, 0.0), x=window_times, axis=-1)
            negative_auc = trapezoid(np.maximum(-signal, 0.0), x=window_times, axis=-1)
            absolute_auc = positive_auc + negative_auc
            polarity = np.divide(
                positive_auc - negative_auc,
                absolute_auc,
                out=np.zeros_like(absolute_auc),
                where=absolute_auc > 0.0,
            )
            centered_time = window_times - window_times.mean()
            slope = np.sum(signal * centered_time, axis=-1) / np.sum(centered_time**2)
            window_features.append(
                np.stack(
                    (mean, std, minimum, maximum, absolute_auc, polarity, slope),
                    axis=-1,
                ).reshape(data.shape[0], -1)
            )

        return np.concatenate(window_features, axis=1), self._feature_names


class WaveletFeatureExtractor:
    """CWT band-energy and normalized entropy for every channel/window."""

    feature_group = "wavelet"

    def __init__(
        self,
        times_s: np.ndarray,
        channel_names: Sequence[str],
        windows: Sequence[TimeWindow] = DEFAULT_TIME_WINDOWS,
        config: WaveletConfig = WaveletConfig(),
    ) -> None:
        if pywt is None:
            raise ImportError(
                "PyWavelets is required for wavelet features; install requirements.txt"
            )
        self.times_s, self.channel_names, self.windows = _validate_axes(
            times_s, channel_names, windows
        )
        self.config = config
        self.frequencies_hz, self._band_indices = self._build_frequency_grid(config)
        feature_labels = tuple(f"{name}_energy" for name, _, _ in FREQUENCY_BANDS_HZ) + (
            "wavelet_entropy",
        )
        self._feature_names = tuple(
            _feature_name(self.feature_group, channel, window.name, feature)
            for window in self.windows
            for channel in self.channel_names
            for feature in feature_labels
        )

    @staticmethod
    def _build_frequency_grid(
        config: WaveletConfig,
    ) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
        low = FREQUENCY_BANDS_HZ[0][1]
        high = FREQUENCY_BANDS_HZ[-1][2]
        count = int(round((high - low) / config.frequency_step_hz)) + 1
        frequencies = low + np.arange(count, dtype=np.float64) * config.frequency_step_hz
        frequencies = frequencies[frequencies <= high + 1e-12]
        band_indices: list[np.ndarray] = []
        for index, (_, band_low, band_high) in enumerate(FREQUENCY_BANDS_HZ):
            if index == len(FREQUENCY_BANDS_HZ) - 1:
                mask = (frequencies >= band_low) & (frequencies <= band_high)
            else:
                mask = (frequencies >= band_low) & (frequencies < band_high)
            indices = np.flatnonzero(mask)
            if not len(indices):
                raise ValueError(
                    f"frequency_step_hz={config.frequency_step_hz} leaves a band empty"
                )
            band_indices.append(indices)
        return frequencies, tuple(band_indices)

    def configuration(self) -> dict[str, object]:
        return {
            "feature_group": self.feature_group,
            "bands_hz": [
                {"name": name, "low_hz": low, "high_hz": high}
                for name, low, high in FREQUENCY_BANDS_HZ
            ],
            "band_boundary_rule": "[low, high), final gamma includes 40 Hz",
            "windows": [asdict(window) for window in self.windows],
            "wavelet": self.config.wavelet,
            "frequency_step_hz": self.config.frequency_step_hz,
            "frequencies_hz": self.frequencies_hz.tolist(),
            "method": self.config.method,
            "energy": "mean squared CWT magnitude over band frequencies and window samples",
            "entropy": "Shannon entropy of five band energies normalized by log(5)",
        }

    def transform(
        self,
        epochs: np.ndarray,
        *,
        sampling_rate_hz: float,
    ) -> tuple[np.ndarray, Sequence[str]]:
        data = _validate_epoch_batch(
            epochs, self.times_s, self.channel_names, sampling_rate_hz
        )
        flat_signals = data.reshape(-1, data.shape[-1])
        n_signals = flat_signals.shape[0]
        result = np.empty((n_signals, len(self.windows), 6), dtype=np.float64)
        scales = pywt.frequency2scale(
            self.config.wavelet,
            self.frequencies_hz / sampling_rate_hz,
        )

        for start in range(0, n_signals, self.config.signal_batch_size):
            stop = min(start + self.config.signal_batch_size, n_signals)
            coefficients, returned_frequencies = pywt.cwt(
                flat_signals[start:stop],
                scales,
                self.config.wavelet,
                sampling_period=1.0 / sampling_rate_hz,
                method=self.config.method,
                axis=-1,
            )
            if not np.allclose(
                returned_frequencies,
                self.frequencies_hz,
                rtol=1e-5,
                atol=1e-5,
            ):
                raise RuntimeError("PyWavelets returned frequencies different from requested grid")
            power = np.abs(coefficients) ** 2
            for window_index, window in enumerate(self.windows):
                mask = _window_mask(self.times_s, window)
                energies = np.stack(
                    [
                        power[indices, :, :][:, :, mask].mean(axis=(0, 2))
                        for indices in self._band_indices
                    ],
                    axis=-1,
                )
                total = energies.sum(axis=-1, keepdims=True)
                probabilities = np.divide(
                    energies,
                    total,
                    out=np.zeros_like(energies),
                    where=total > 0.0,
                )
                log_probabilities = np.zeros_like(probabilities)
                np.log(
                    probabilities,
                    out=log_probabilities,
                    where=probabilities > 0.0,
                )
                entropy = -(probabilities * log_probabilities).sum(axis=-1) / np.log(5.0)
                result[start:stop, window_index, :5] = energies
                result[start:stop, window_index, 5] = entropy

        result = result.reshape(data.shape[0], data.shape[1], len(self.windows), 6)
        result = result.transpose(0, 2, 1, 3).reshape(data.shape[0], -1)
        return result, self._feature_names


def extract_subject_features(
    data: SubjectEpochs,
    extractor: FeatureExtractor,
    config: FeatureExtractionConfig = FeatureExtractionConfig(),
) -> FeatureTable:
    """Apply one extractor in bounded batches and preserve subject identifiers."""

    chunks: list[np.ndarray] = []
    expected_names: tuple[str, ...] | None = None

    for start in range(0, data.epochs.shape[0], config.batch_size):
        stop = min(start + config.batch_size, data.epochs.shape[0])
        values, names = extractor.transform(
            np.asarray(data.epochs[start:stop]),
            sampling_rate_hz=config.sampling_rate_hz,
        )
        values = np.asarray(values, dtype=config.output_dtype)
        names_tuple = tuple(str(name) for name in names)
        if values.ndim != 2 or values.shape[0] != stop - start:
            raise ValueError("extractor returned an invalid feature matrix shape")
        if values.shape[1] != len(names_tuple):
            raise ValueError("extractor returned a different number of names and columns")
        if expected_names is None:
            expected_names = names_tuple
        elif names_tuple != expected_names:
            raise ValueError("extractor changed feature names or column order between batches")
        if not np.isfinite(values).all():
            raise ValueError("extractor returned NaN or infinite values")
        chunks.append(values)

    if not chunks:
        raise ValueError("cannot extract features from an empty subject")

    matrix = np.concatenate(chunks, axis=0)
    return FeatureTable(
        values=matrix,
        feature_names=expected_names or (),
        subject_ids=np.repeat(np.asarray([data.subject_id], dtype=str), matrix.shape[0]),
        epoch_ids=np.asarray(data.epoch_ids),
        targets=np.asarray(data.targets),
    )
