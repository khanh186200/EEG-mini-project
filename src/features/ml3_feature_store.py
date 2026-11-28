"""Processed-data adapter and resumable per-subject ML3 feature storage."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
import re
import shutil
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.features.ml3_features import (
    DEFAULT_TIME_WINDOWS,
    FeatureExtractionConfig,
    FeatureExtractor,
    FeatureTable,
    SubjectEpochs,
    TEMPORAL_STATISTICS,
    extract_subject_features,
)


REQUIRED_PROCESSED_FILES = (
    "X_clean.npy",
    "y.npy",
    "subject_ids.npy",
    "times.npy",
    "channel_names.json",
    "metadata.csv",
)


@dataclass(frozen=True)
class ProcessedFeatureSource:
    input_dir: Path
    epochs: np.ndarray
    targets: np.ndarray
    subject_ids: np.ndarray
    trial_uids: np.ndarray
    times_s: np.ndarray
    channel_names: tuple[str, ...]

    @property
    def subjects(self) -> tuple[str, ...]:
        return tuple(sorted(np.unique(self.subject_ids.astype(str)).tolist()))

    def subject_data(self, subject_id: str) -> SubjectEpochs:
        indices = np.flatnonzero(self.subject_ids.astype(str) == subject_id)
        if not len(indices):
            raise KeyError(f"unknown subject: {subject_id}")
        # The preprocessing pipeline writes subjects contiguously. Preserve a
        # memmap view in that common case; fall back to indexed loading safely.
        if np.array_equal(indices, np.arange(indices[0], indices[-1] + 1)):
            selector: slice | np.ndarray = slice(int(indices[0]), int(indices[-1]) + 1)
        else:
            selector = indices
        return SubjectEpochs(
            subject_id=subject_id,
            epochs=self.epochs[selector],
            epoch_ids=self.trial_uids[selector],
            targets=self.targets[selector],
        )


def load_processed_feature_source(input_dir: Path) -> ProcessedFeatureSource:
    """Open the repository's processed contract without loading EEG into RAM."""

    root = Path(input_dir)
    missing = [name for name in REQUIRED_PROCESSED_FILES if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"processed directory is missing: {missing}")

    epochs = np.load(root / "X_clean.npy", mmap_mode="r")
    targets = np.load(root / "y.npy", mmap_mode="r")
    subject_ids = np.load(root / "subject_ids.npy", mmap_mode="r")
    times_s = np.load(root / "times.npy")
    channel_names = tuple(
        str(name)
        for name in json.loads((root / "channel_names.json").read_text(encoding="utf-8"))
    )
    metadata = pd.read_csv(root / "metadata.csv", usecols=["trial_uid", "subject_id", "label"])
    trial_uids = metadata["trial_uid"].astype(str).to_numpy()

    n_epochs = epochs.shape[0]
    if epochs.ndim != 3:
        raise ValueError("X_clean.npy must have shape (n_epochs, n_channels, n_times)")
    if epochs.shape[1:] != (len(channel_names), len(times_s)):
        raise ValueError("X_clean shape does not match channel_names.json and times.npy")
    if any(len(array) != n_epochs for array in (targets, subject_ids, trial_uids, metadata)):
        raise ValueError("processed arrays and metadata do not contain the same number of epochs")
    if not metadata["trial_uid"].is_unique:
        raise ValueError("metadata trial_uid values must be unique")
    if not np.array_equal(metadata["label"].to_numpy(dtype=np.int64), targets):
        raise ValueError("metadata labels do not match y.npy")
    if not np.array_equal(metadata["subject_id"].astype(str).to_numpy(), subject_ids.astype(str)):
        raise ValueError("metadata subject_id values do not match subject_ids.npy")
    if not set(np.unique(targets).tolist()).issubset({0, 1}):
        raise ValueError("y.npy must contain binary values in {0, 1}")

    return ProcessedFeatureSource(
        input_dir=root,
        epochs=epochs,
        targets=targets,
        subject_ids=subject_ids,
        trial_uids=trial_uids,
        times_s=np.asarray(times_s, dtype=np.float64),
        channel_names=channel_names,
    )


def _stable_json(payload: Mapping[str, object]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _configuration_payload(
    extractor: FeatureExtractor,
    runtime: FeatureExtractionConfig,
    source: ProcessedFeatureSource,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "extractor": extractor.configuration(),
        "runtime": asdict(runtime),
        "source": {
            "n_channels": len(source.channel_names),
            "n_times": len(source.times_s),
            "time_start_s": float(source.times_s[0]),
            "time_stop_s": float(source.times_s[-1]),
            "channel_names": list(source.channel_names),
        },
    }


def _configuration_hash(payload: Mapping[str, object]) -> str:
    # Batch size is operational only: changing it must not invalidate already
    # extracted values, which are explicitly tested for batch invariance.
    semantic_payload = dict(payload)
    runtime = dict(semantic_payload.get("runtime", {}))
    runtime.pop("batch_size", None)
    semantic_payload["runtime"] = runtime
    return hashlib.sha256(_stable_json(semantic_payload).encode("utf-8")).hexdigest()


def _validate_existing_output(
    path: Path,
    *,
    subject: SubjectEpochs,
    expected_hash: str,
) -> dict[str, object]:
    try:
        with np.load(path, allow_pickle=False) as saved:
            required = {
                "features",
                "feature_names",
                "subject_id",
                "trial_uid",
                "target",
                "config_sha256",
                "config_json",
            }
            missing = required - set(saved.files)
            if missing:
                raise ValueError(f"missing arrays: {sorted(missing)}")
            if str(saved["subject_id"].item()) != subject.subject_id:
                raise ValueError("subject_id does not match")
            if str(saved["config_sha256"].item()) != expected_hash:
                raise ValueError("extractor configuration changed")
            if not np.array_equal(saved["trial_uid"].astype(str), subject.epoch_ids.astype(str)):
                raise ValueError("trial_uid order changed")
            if not np.array_equal(saved["target"].astype(np.int64), subject.targets.astype(np.int64)):
                raise ValueError("targets changed")
            shape = tuple(int(value) for value in saved["features"].shape)
            if shape[0] != len(subject.targets):
                raise ValueError("feature row count does not match subject epochs")
            if shape[1] != len(saved["feature_names"]):
                raise ValueError("feature names do not match feature columns")
    except Exception as exc:
        raise ValueError(
            f"existing output {path} is incompatible ({exc}); use --overwrite"
        ) from exc
    return {"path": str(path), "shape": list(shape), "status": "reused"}


def save_subject_feature_table(
    path: Path,
    *,
    table,
    feature_group: str,
    configuration: Mapping[str, object],
    configuration_hash: str,
) -> None:
    """Write one self-describing feature file atomically."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary,
        features=table.values,
        feature_names=np.asarray(table.feature_names, dtype=str),
        feature_group=np.asarray(feature_group),
        subject_id=np.asarray(str(table.subject_ids[0])),
        trial_uid=np.asarray(table.epoch_ids, dtype=str),
        target=np.asarray(table.targets, dtype=np.int64),
        config_sha256=np.asarray(configuration_hash),
        config_json=np.asarray(_stable_json(configuration)),
    )
    temporary.replace(path)


def extract_feature_groups(
    source: ProcessedFeatureSource,
    output_dir: Path,
    extractors: Sequence[FeatureExtractor],
    runtime: FeatureExtractionConfig,
    *,
    subjects: Iterable[str] | None = None,
    overwrite: bool = False,
) -> dict[str, object]:
    """Extract requested groups per subject with resume-safe output files."""

    extractor_by_group = {extractor.feature_group: extractor for extractor in extractors}
    if not extractor_by_group:
        raise ValueError("at least one extractor is required")
    if len(extractor_by_group) != len(extractors):
        raise ValueError("feature_group names must be unique")

    selected_subjects = tuple(subjects) if subjects is not None else source.subjects
    unknown = sorted(set(selected_subjects) - set(source.subjects))
    if unknown:
        raise ValueError(f"unknown subjects requested: {unknown}")

    root = Path(output_dir)
    records: list[dict[str, object]] = []
    group_configurations = {
        group: _configuration_payload(extractor, runtime, source)
        for group, extractor in extractor_by_group.items()
    }
    group_hashes = {
        group: _configuration_hash(configuration)
        for group, configuration in group_configurations.items()
    }

    for subject_id in selected_subjects:
        subject = source.subject_data(subject_id)
        for group, extractor in extractor_by_group.items():
            output_path = root / group / f"{subject_id}.npz"
            if output_path.exists() and not overwrite:
                record = _validate_existing_output(
                    output_path,
                    subject=subject,
                    expected_hash=group_hashes[group],
                )
            else:
                table = extract_subject_features(subject, extractor, runtime)
                save_subject_feature_table(
                    output_path,
                    table=table,
                    feature_group=group,
                    configuration=group_configurations[group],
                    configuration_hash=group_hashes[group],
                )
                record = {
                    "path": str(output_path),
                    "shape": list(table.values.shape),
                    "status": "written",
                }
            records.append({"subject_id": subject_id, "feature_group": group, **record})

    manifest: dict[str, object] = {
        "schema_version": 1,
        "input_dir": str(source.input_dir),
        "output_dir": str(root),
        "subjects": list(selected_subjects),
        "feature_groups": list(extractor_by_group),
        "configurations": group_configurations,
        "configuration_sha256": group_hashes,
        "files": records,
    }
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "feature_manifest.json"
    temporary_manifest = root / "feature_manifest.tmp.json"
    temporary_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    temporary_manifest.replace(manifest_path)
    return manifest


# ── ML3 Training Feature Loader, Scenario Builder & Google Drive Connector ──

W8_WINDOW_NAMES: Tuple[str, ...] = tuple(window.name for window in DEFAULT_TIME_WINDOWS)
W5_WINDOW_NAMES: Tuple[str, ...] = (
    "baseline_m200_0",
    "post_0_200",
    "post_200_400",
    "post_400_600",
    "post_600_800",
)

H4_BAND_FEATURES: Tuple[str, ...] = (
    "delta_energy",
    "theta_energy",
    "alpha_energy",
    "beta_energy",
)
GAMMA_ENERGY_FEATURE = "gamma_energy"
RAW_WAVELET_ENTROPY_FEATURE = "wavelet_entropy"
H4_ENTROPY_FEATURE = "h4_entropy"

CS5_MAIN_EXPERIMENTS: Tuple[str, ...] = (
    "temporal",
    "wavelet_w5",
    "wavelet_w8",
    "combined_w5",
    "combined_w8",
)
LOSO_MAIN_EXPERIMENTS: Tuple[str, ...] = (
    "temporal",
    "combined_w5",
    "combined_w8",
)


@dataclass(frozen=True)
class FeatureScenarioSpec:
    """Policy definition for one ML3 feature experiment."""

    name: str
    uses_temporal: bool
    wavelet_windows: Optional[Tuple[str, ...]]
    uses_gamma: bool = False

    def policy_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "uses_temporal": self.uses_temporal,
            "temporal_windows": list(W8_WINDOW_NAMES) if self.uses_temporal else [],
            "temporal_statistics": list(TEMPORAL_STATISTICS) if self.uses_temporal else [],
            "wavelet_windows": list(self.wavelet_windows) if self.wavelet_windows else [],
            "wavelet_bands": (
                list(H4_BAND_FEATURES) + ([GAMMA_ENERGY_FEATURE] if self.uses_gamma else [])
                if self.wavelet_windows
                else []
            ),
            "wavelet_entropy": H4_ENTROPY_FEATURE if self.wavelet_windows else None,
            "uses_gamma": self.uses_gamma,
            "dtype": "float32",
        }


FEATURE_SCENARIO_SPECS: Dict[str, FeatureScenarioSpec] = {
    "temporal": FeatureScenarioSpec(
        name="temporal",
        uses_temporal=True,
        wavelet_windows=None,
        uses_gamma=False,
    ),
    "wavelet_w5": FeatureScenarioSpec(
        name="wavelet_w5",
        uses_temporal=False,
        wavelet_windows=W5_WINDOW_NAMES,
        uses_gamma=False,
    ),
    "wavelet_w8": FeatureScenarioSpec(
        name="wavelet_w8",
        uses_temporal=False,
        wavelet_windows=W8_WINDOW_NAMES,
        uses_gamma=False,
    ),
    "combined_w5": FeatureScenarioSpec(
        name="combined_w5",
        uses_temporal=True,
        wavelet_windows=W5_WINDOW_NAMES,
        uses_gamma=False,
    ),
    "combined_w8": FeatureScenarioSpec(
        name="combined_w8",
        uses_temporal=True,
        wavelet_windows=W8_WINDOW_NAMES,
        uses_gamma=False,
    ),
    "wavelet_w5_gamma": FeatureScenarioSpec(
        name="wavelet_w5_gamma",
        uses_temporal=False,
        wavelet_windows=W5_WINDOW_NAMES,
        uses_gamma=True,
    ),
    "wavelet_w8_gamma": FeatureScenarioSpec(
        name="wavelet_w8_gamma",
        uses_temporal=False,
        wavelet_windows=W8_WINDOW_NAMES,
        uses_gamma=True,
    ),
    "combined_w5_gamma": FeatureScenarioSpec(
        name="combined_w5_gamma",
        uses_temporal=True,
        wavelet_windows=W5_WINDOW_NAMES,
        uses_gamma=True,
    ),
    "combined_w8_gamma": FeatureScenarioSpec(
        name="combined_w8_gamma",
        uses_temporal=True,
        wavelet_windows=W8_WINDOW_NAMES,
        uses_gamma=True,
    ),
}


@dataclass(frozen=True)
class ParsedFeatureName:
    """Structured representation of ``group|channel=...|window=...|feature=...``."""

    raw: str
    group: str
    channel: str
    window: str
    feature: str


def _natural_subject_key(value: str) -> list[int | str]:
    return [int(tok) if tok.isdigit() else tok.lower() for tok in re.split(r"(\d+)", str(value))]


def is_google_drive_path(
    path: Path | str,
    mount_point: Path | str = "/content/drive",
) -> bool:
    """Return True when ``path`` lives under the Colab Google Drive mount root."""
    norm_path = str(path).replace("\\", "/").rstrip("/")
    norm_mount = str(mount_point).replace("\\", "/").rstrip("/")
    return norm_path == norm_mount or norm_path.startswith(f"{norm_mount}/")


def connect_google_drive(
    mount_point: Path | str = "/content/drive",
    *,
    force_remount: bool = False,
) -> Path:
    """Mount Google Drive in Colab if not yet mounted and return the mount path.

    When called inside an interactive Colab notebook cell, ``google.colab.drive``
    is invoked directly. When called from a shell subprocess (``!python ...``)
    after Drive has already been mounted, the existing mount is reused without
    re-authentication.
    """
    mount_root = Path(mount_point)
    drive_ready = (mount_root / "MyDrive").exists() or (mount_root / "My Drive").exists()
    if drive_ready and not force_remount:
        return mount_root

    try:
        from google.colab import drive as colab_drive  # type: ignore[import-not-found]
    except ImportError as exc:
        if drive_ready:
            return mount_root
        raise RuntimeError(
            f"Google Drive chưa được mount tại '{mount_root}' và môi trường hiện tại "
            "không có `google.colab`. Nếu chạy trên Google Colab, hãy chạy trước:\n"
            "  from google.colab import drive\n"
            f"  drive.mount('{mount_root.as_posix()}')"
        ) from exc

    try:
        colab_drive.mount(mount_root.as_posix(), force_remount=force_remount)
    except Exception as exc:
        raise RuntimeError(
            f"Không thể tự động mount Google Drive tại '{mount_root}'.\n"
            "- Nếu dùng VS Code Google Colab Extension: nhấn Ctrl+Shift+P -> "
            "'Colab: Mount Google Drive to Server...'.\n"
            "- Nếu dùng Colab trên trình duyệt: chạy cell:\n"
            "  from google.colab import drive\n"
            f"  drive.mount('{mount_root.as_posix()}')"
        ) from exc

    if not ((mount_root / "MyDrive").exists() or (mount_root / "My Drive").exists()):
        raise RuntimeError(
            f"Đã gọi mount Google Drive nhưng không tìm thấy '{mount_root / 'MyDrive'}'."
        )
    return mount_root


def ensure_drive_path_ready(
    path: Path | str,
    *,
    mount_drive: bool = False,
    mount_point: Path | str = "/content/drive",
) -> Path:
    """Ensure Google Drive is mounted when ``path`` points into Colab Drive."""
    resolved = Path(path)
    if mount_drive or is_google_drive_path(resolved, mount_point=mount_point):
        connect_google_drive(mount_point=mount_point)
    return resolved


def stage_features_to_local(
    source_features_dir: Path | str,
    local_features_dir: Path | str = "/content/features",
    *,
    required_groups: Sequence[str] = ("temporal", "wavelet"),
    subjects: Optional[Sequence[str]] = None,
    overwrite: bool = False,
    mount_drive: bool = False,
    mount_point: Path | str = "/content/drive",
    verbose: bool = True,
) -> Path:
    """Copy per-subject ``.npz`` features from Google Drive to local ``/content/features`` SSD.

    Files already present in ``local_features_dir`` with matching byte size are
    reused without copying again, keeping repeated runs in the same Colab kernel
    instantaneous while avoiding slow Google Drive FUSE random reads.
    """
    src_root = ensure_drive_path_ready(
        source_features_dir, mount_drive=mount_drive, mount_point=mount_point
    )
    dst_root = Path(local_features_dir)
    if src_root.exists() and dst_root.exists() and src_root.resolve() == dst_root.resolve():
        return dst_root

    groups = tuple(dict.fromkeys(required_groups))
    available_subjects = discover_feature_subjects(
        src_root,
        required_groups=groups,
        mount_drive=False,
        mount_point=mount_point,
    )
    if subjects is not None:
        missing = sorted(set(subjects) - set(available_subjects))
        if missing:
            raise ValueError(f"Requested subjects missing in {src_root}: {missing}")
        selected_subjects = tuple(sorted(dict.fromkeys(subjects), key=_natural_subject_key))
    else:
        selected_subjects = available_subjects

    dst_root.mkdir(parents=True, exist_ok=True)
    src_manifest = src_root / "feature_manifest.json"
    if src_manifest.is_file():
        dst_manifest = dst_root / "feature_manifest.json"
        if overwrite or not dst_manifest.is_file() or dst_manifest.stat().st_size != src_manifest.stat().st_size:
            tmp_manifest = dst_root / "feature_manifest.part.json"
            shutil.copy2(src_manifest, tmp_manifest)
            tmp_manifest.replace(dst_manifest)

    copied_files = 0
    reused_files = 0
    copied_bytes = 0

    for group in groups:
        dst_group_dir = dst_root / group
        dst_group_dir.mkdir(parents=True, exist_ok=True)
        for sub_id in selected_subjects:
            src_file = src_root / group / f"{sub_id}.npz"
            dst_file = dst_group_dir / f"{sub_id}.npz"
            src_size = src_file.stat().st_size
            if not overwrite and dst_file.is_file() and dst_file.stat().st_size == src_size:
                reused_files += 1
                continue
            tmp_file = dst_group_dir / f"{sub_id}.part.npz"
            shutil.copy2(src_file, tmp_file)
            tmp_file.replace(dst_file)
            copied_files += 1
            copied_bytes += src_size

    if verbose:
        print(
            f"[Stage Features] {src_root} -> {dst_root} | "
            f"copied={copied_files}, reused={reused_files}, "
            f"transferred={copied_bytes / (1024 * 1024):.1f} MB"
        )
    return dst_root


def parse_ml3_feature_name(name: str) -> ParsedFeatureName:
    """Parse ``{group}|channel={ch}|window={win}|feature={feat}``."""
    parts = str(name).split("|")
    if len(parts) != 4:
        raise ValueError(f"Invalid ML3 feature name format: {name!r}")
    group = parts[0].strip()
    kv: Dict[str, str] = {}
    for token in parts[1:]:
        if "=" not in token:
            raise ValueError(f"Invalid key=value segment in feature name: {name!r}")
        key, val = token.split("=", 1)
        kv[key.strip()] = val.strip()
    if set(kv) != {"channel", "window", "feature"} or not group:
        raise ValueError(f"Malformed ML3 feature metadata in: {name!r}")
    if not all(kv.values()):
        raise ValueError(f"Empty metadata value in feature name: {name!r}")
    return ParsedFeatureName(
        raw=str(name),
        group=group,
        channel=kv["channel"],
        window=kv["window"],
        feature=kv["feature"],
    )


def select_window_columns(
    values: np.ndarray,
    feature_names: Sequence[str],
    allowed_windows: Sequence[str],
) -> Tuple[np.ndarray, Tuple[str, ...]]:
    """Select columns whose parsed ``window`` belongs to ``allowed_windows``."""
    matrix = np.asarray(values)
    if matrix.ndim != 2 or matrix.shape[1] != len(feature_names):
        raise ValueError("values shape does not align with feature_names")

    window_set = set(allowed_windows)
    if not window_set:
        raise ValueError("allowed_windows must not be empty")

    keep_indices: list[int] = []
    kept_names: list[str] = []
    seen_windows: set[str] = set()
    for idx, fname in enumerate(feature_names):
        parsed = parse_ml3_feature_name(fname)
        if parsed.window in window_set:
            keep_indices.append(idx)
            kept_names.append(parsed.raw)
            seen_windows.add(parsed.window)

    missing_windows = sorted(window_set - seen_windows)
    if missing_windows:
        raise ValueError(f"Requested windows missing from feature_names: {missing_windows}")

    selected = np.ascontiguousarray(matrix[:, keep_indices], dtype=np.float32)
    return selected, tuple(kept_names)


def compute_h4_wavelet_features(
    raw_values: np.ndarray,
    raw_feature_names: Sequence[str],
    *,
    windows: Sequence[str] = W8_WINDOW_NAMES,
    include_gamma: bool = False,
) -> Tuple[np.ndarray, Tuple[str, ...]]:
    """Build wavelet features using four-band Shannon entropy ``H4`` in float32.

    ``H4 = -sum_{b in {delta,theta,alpha,beta}} p_b log(p_b) / log(4)``.
    The pre-extracted 5-band ``wavelet_entropy`` column is never read. When
    ``include_gamma=False`` (default), ``gamma_energy`` is also completely
    ignored; when ``include_gamma=True``, ``gamma_energy`` is appended while
    ``H4`` remains strictly computed from the four non-gamma bands.
    """
    matrix = np.asarray(raw_values)
    if matrix.ndim != 2 or matrix.shape[1] != len(raw_feature_names):
        raise ValueError("raw_values shape does not align with raw_feature_names")

    target_windows = set(windows)
    if not target_windows:
        raise ValueError("windows must not be empty")

    pair_order: list[Tuple[str, str]] = []
    pair_columns: Dict[Tuple[str, str], Dict[str, int]] = {}
    found_windows: set[str] = set()

    for col_idx, fname in enumerate(raw_feature_names):
        parsed = parse_ml3_feature_name(fname)
        if parsed.group != "wavelet":
            raise ValueError(f"Expected wavelet feature group, got {parsed.group!r} in {fname!r}")
        if parsed.window not in target_windows:
            continue
        found_windows.add(parsed.window)
        key = (parsed.window, parsed.channel)
        if key not in pair_columns:
            pair_order.append(key)
            pair_columns[key] = {}
        if parsed.feature in pair_columns[key]:
            raise ValueError(f"Duplicate wavelet feature column for {key}: {parsed.feature}")
        pair_columns[key][parsed.feature] = col_idx

    missing_windows = sorted(target_windows - found_windows)
    if missing_windows:
        raise ValueError(f"Wavelet feature file is missing windows: {missing_windows}")
    if not pair_order:
        raise ValueError("No wavelet (window, channel) columns found")

    required_feats = list(H4_BAND_FEATURES) + ([GAMMA_ENERGY_FEATURE] if include_gamma else [])
    for key in pair_order:
        missing_feats = [f for f in required_feats if f not in pair_columns[key]]
        if missing_feats:
            raise ValueError(f"Missing wavelet features {missing_feats} for window/channel {key}")

    band_idx = {
        band: np.asarray([pair_columns[key][band] for key in pair_order], dtype=np.int64)
        for band in H4_BAND_FEATURES
    }
    energies_4 = np.stack(
        [matrix[:, band_idx[band]].astype(np.float64, copy=False) for band in H4_BAND_FEATURES],
        axis=-1,
    )
    if not np.isfinite(energies_4).all():
        raise ValueError("Wavelet four-band energies contain NaN or infinite values")
    if (energies_4 < -1e-5).any():
        raise ValueError("Wavelet four-band energies must be non-negative")
    energies_4 = np.maximum(energies_4, 0.0)

    total_4 = energies_4.sum(axis=-1, keepdims=True)
    probs_4 = np.divide(
        energies_4,
        total_4,
        out=np.zeros_like(energies_4),
        where=total_4 > 0.0,
    )
    log_probs_4 = np.zeros_like(probs_4)
    np.log(probs_4, out=log_probs_4, where=probs_4 > 0.0)
    h4 = -(probs_4 * log_probs_4).sum(axis=-1) / np.log(4.0)

    if include_gamma:
        gamma_idx = np.asarray(
            [pair_columns[key][GAMMA_ENERGY_FEATURE] for key in pair_order],
            dtype=np.int64,
        )
        gamma_vals = matrix[:, gamma_idx].astype(np.float64, copy=False)
        if not np.isfinite(gamma_vals).all():
            raise ValueError("Wavelet gamma_energy contains NaN or infinite values")
        stacked = np.concatenate([energies_4, gamma_vals[..., None], h4[..., None]], axis=-1)
        out_feat_labels = (*H4_BAND_FEATURES, GAMMA_ENERGY_FEATURE, H4_ENTROPY_FEATURE)
    else:
        stacked = np.concatenate([energies_4, h4[..., None]], axis=-1)
        out_feat_labels = (*H4_BAND_FEATURES, H4_ENTROPY_FEATURE)

    out_values = np.ascontiguousarray(
        stacked.reshape(matrix.shape[0], len(pair_order) * len(out_feat_labels)),
        dtype=np.float32,
    )
    out_names = tuple(
        f"wavelet|channel={channel}|window={window}|feature={feat}"
        for window, channel in pair_order
        for feat in out_feat_labels
    )
    return out_values, out_names


def _read_subject_npz(path: Path, *, expected_group: str) -> Dict[str, Any]:
    with np.load(path, allow_pickle=False) as payload:
        required = {"features", "feature_names", "subject_id", "trial_uid", "target"}
        missing = sorted(required - set(payload.files))
        if missing:
            raise ValueError(f"{path} is missing required arrays: {missing}")

        features = np.asarray(payload["features"], dtype=np.float32)
        feature_names = tuple(str(x) for x in payload["feature_names"].tolist())
        subject_raw = payload["subject_id"]
        if subject_raw.ndim == 0:
            file_subject = str(subject_raw.item())
        else:
            unique_subs = set(subject_raw.astype(str).tolist())
            if len(unique_subs) != 1:
                raise ValueError(f"{path} contains multiple subject_ids: {sorted(unique_subs)}")
            file_subject = next(iter(unique_subs))

        if file_subject != path.stem:
            raise ValueError(
                f"{path} subject_id {file_subject!r} does not match filename stem {path.stem!r}"
            )
        if "feature_group" in payload.files:
            fg = str(payload["feature_group"].item())
            if fg != expected_group:
                raise ValueError(
                    f"{path} has feature_group={fg!r}, expected {expected_group!r}"
                )

        trial_uids = payload["trial_uid"].astype(str)
        targets = payload["target"].astype(np.int64)
        config_sha = (
            str(payload["config_sha256"].item()) if "config_sha256" in payload.files else None
        )

    if features.ndim != 2:
        raise ValueError(f"{path} features must be 2D, got shape {features.shape}")
    n_rows, n_cols = features.shape
    if n_rows == 0:
        raise ValueError(f"{path} contains zero trials")
    if n_cols != len(feature_names):
        raise ValueError(f"{path} column count ({n_cols}) != len(feature_names) ({len(feature_names)})")
    if len(set(feature_names)) != len(feature_names):
        raise ValueError(f"{path} contains duplicate feature_names")
    if len(trial_uids) != n_rows or len(targets) != n_rows:
        raise ValueError(f"{path} trial_uid/target length mismatch with features rows")
    if len(set(trial_uids.tolist())) != n_rows:
        raise ValueError(f"{path} contains duplicate trial_uid entries")
    if not set(np.unique(targets).tolist()).issubset({0, 1}):
        raise ValueError(f"{path} targets must be binary in {{0, 1}}")

    return {
        "subject_id": file_subject,
        "features": features,
        "feature_names": feature_names,
        "trial_uids": trial_uids,
        "targets": targets,
        "config_sha256": config_sha,
    }


def _join_temporal_and_wavelet_subject(
    temporal_rec: Mapping[str, Any],
    wavelet_rec: Mapping[str, Any],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    """Align one subject's temporal and wavelet tables strictly by ``trial_uid``."""
    sub_t = str(temporal_rec["subject_id"])
    sub_w = str(wavelet_rec["subject_id"])
    if sub_t != sub_w:
        raise ValueError(f"Subject mismatch between temporal ({sub_t}) and wavelet ({sub_w})")

    uids_t = np.asarray(temporal_rec["trial_uids"], dtype=str)
    uids_w = np.asarray(wavelet_rec["trial_uids"], dtype=str)
    set_t = set(uids_t.tolist())
    set_w = set(uids_w.tolist())
    if set_t != set_w:
        only_t = sorted(set_t - set_w)[:5]
        only_w = sorted(set_w - set_t)[:5]
        raise ValueError(
            f"trial_uid mismatch for {sub_t}: only_in_temporal={only_t}, only_in_wavelet={only_w}"
        )

    wav_index = {uid: idx for idx, uid in enumerate(uids_w.tolist())}
    order = np.asarray([wav_index[uid] for uid in uids_t.tolist()], dtype=np.int64)

    targets_t = np.asarray(temporal_rec["targets"], dtype=np.int64)
    targets_w = np.asarray(wavelet_rec["targets"], dtype=np.int64)[order]
    if not np.array_equal(targets_t, targets_w):
        raise ValueError(f"Target labels do not match after trial_uid join for {sub_t}")

    temp_vals = np.ascontiguousarray(temporal_rec["features"], dtype=np.float32)
    wav_vals = np.ascontiguousarray(wavelet_rec["features"][order], dtype=np.float32)
    return temp_vals, wav_vals, uids_t, targets_t, sub_t


@dataclass(frozen=True)
class ScenarioMatrix:
    """Materialized float32 matrix and metadata for one ML3 experiment."""

    scenario: FeatureScenarioSpec
    values: np.ndarray
    feature_names: Tuple[str, ...]
    subject_ids: np.ndarray
    trial_uids: np.ndarray
    targets: np.ndarray

    @property
    def n_samples(self) -> int:
        return int(self.values.shape[0])

    @property
    def n_features(self) -> int:
        return int(self.values.shape[1])


@dataclass
class ML3FeatureDataset:
    """Aligned per-trial ML3 features ready for CS5/LOSO scenario construction."""

    features_dir: Path
    subjects: Tuple[str, ...]
    subject_ids: np.ndarray
    trial_uids: np.ndarray
    targets: np.ndarray
    temporal_values: Optional[np.ndarray]
    temporal_feature_names: Tuple[str, ...]
    wavelet_w8_values: Optional[np.ndarray]
    wavelet_w8_feature_names: Tuple[str, ...]
    wavelet_w8_gamma_values: Optional[np.ndarray] = None
    wavelet_w8_gamma_feature_names: Tuple[str, ...] = ()
    manifest_metadata: Optional[Dict[str, Any]] = None
    group_hashes: Dict[str, Optional[str]] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.group_hashes is None:
            self.group_hashes = {}

    @property
    def n_samples(self) -> int:
        return int(len(self.targets))

    @property
    def n_channels(self) -> int:
        names = self.temporal_feature_names or self.wavelet_w8_feature_names
        return len({parse_ml3_feature_name(name).channel for name in names})

    @property
    def trial_uid_sha256(self) -> str:
        joined = "\n".join(self.trial_uids.astype(str).tolist())
        return hashlib.sha256(joined.encode("utf-8")).hexdigest()

    def scenario_feature_count(self, scenario_name: str) -> int:
        if scenario_name not in FEATURE_SCENARIO_SPECS:
            raise ValueError(
                f"Unknown ML3 experiment {scenario_name!r}; "
                f"available: {sorted(FEATURE_SCENARIO_SPECS)}"
            )
        spec = FEATURE_SCENARIO_SPECS[scenario_name]
        n_ch = self.n_channels
        total = 0
        if spec.uses_temporal:
            total += len(W8_WINDOW_NAMES) * n_ch * len(TEMPORAL_STATISTICS)
        if spec.wavelet_windows is not None:
            per_win_ch = 6 if spec.uses_gamma else 5
            total += len(spec.wavelet_windows) * n_ch * per_win_ch
        return total

    def build_scenario(self, scenario_name: str) -> ScenarioMatrix:
        """Construct the requested float32 feature scenario without recomputing CWT."""
        if scenario_name not in FEATURE_SCENARIO_SPECS:
            raise ValueError(
                f"Unknown ML3 experiment {scenario_name!r}; "
                f"available: {sorted(FEATURE_SCENARIO_SPECS)}"
            )
        spec = FEATURE_SCENARIO_SPECS[scenario_name]
        blocks: list[np.ndarray] = []
        names: list[str] = []

        if spec.uses_temporal:
            if self.temporal_values is None:
                raise ValueError(f"Experiment {scenario_name!r} requires temporal features")
            blocks.append(self.temporal_values)
            names.extend(self.temporal_feature_names)

        if spec.wavelet_windows is not None:
            if spec.uses_gamma:
                if self.wavelet_w8_gamma_values is None:
                    raise ValueError(
                        f"Experiment {scenario_name!r} requires wavelet gamma features; "
                        "load dataset with include_gamma_support=True"
                    )
                base_vals = self.wavelet_w8_gamma_values
                base_names = self.wavelet_w8_gamma_feature_names
            else:
                if self.wavelet_w8_values is None:
                    raise ValueError(f"Experiment {scenario_name!r} requires wavelet features")
                base_vals = self.wavelet_w8_values
                base_names = self.wavelet_w8_feature_names

            if tuple(spec.wavelet_windows) == W8_WINDOW_NAMES:
                wav_vals, wav_names = base_vals, base_names
            else:
                wav_vals, wav_names = select_window_columns(
                    base_vals, base_names, spec.wavelet_windows
                )
            blocks.append(wav_vals)
            names.extend(wav_names)

        if len(blocks) == 1:
            matrix = np.ascontiguousarray(blocks[0], dtype=np.float32)
        else:
            matrix = np.ascontiguousarray(np.concatenate(blocks, axis=1), dtype=np.float32)

        if matrix.dtype != np.float32:
            matrix = matrix.astype(np.float32)
        if not np.isfinite(matrix).all():
            raise ValueError(f"Scenario {scenario_name!r} contains NaN or infinite values")

        return ScenarioMatrix(
            scenario=spec,
            values=matrix,
            feature_names=tuple(names),
            subject_ids=self.subject_ids,
            trial_uids=self.trial_uids,
            targets=self.targets,
        )


def discover_feature_subjects(
    features_dir: Path | str,
    *,
    required_groups: Sequence[str] = ("temporal", "wavelet"),
    mount_drive: bool = False,
    mount_point: Path | str = "/content/drive",
) -> Tuple[str, ...]:
    """Inspect ``features/temporal`` and ``features/wavelet`` without loading matrices."""
    root = ensure_drive_path_ready(
        features_dir, mount_drive=mount_drive, mount_point=mount_point
    )
    if not root.is_dir():
        raise FileNotFoundError(f"Features directory not found: {root}")

    groups = tuple(dict.fromkeys(required_groups))
    if not groups:
        raise ValueError("required_groups must not be empty")

    group_subjects: Dict[str, set[str]] = {}
    for group in groups:
        gdir = root / group
        if not gdir.is_dir():
            raise FileNotFoundError(f"Missing feature group directory: {gdir}")
        subs = {p.stem for p in gdir.glob("*.npz") if not p.name.endswith(".tmp.npz")}
        if not subs:
            raise FileNotFoundError(f"No per-subject .npz files found in {gdir}")
        group_subjects[group] = subs

    first_group = groups[0]
    reference = group_subjects[first_group]
    for group in groups[1:]:
        if group_subjects[group] != reference:
            only_ref = sorted(reference - group_subjects[group])
            only_other = sorted(group_subjects[group] - reference)
            raise ValueError(
                f"Subject file mismatch between '{first_group}' and '{group}': "
                f"only_in_{first_group}={only_ref}, only_in_{group}={only_other}"
            )

    return tuple(sorted(reference, key=_natural_subject_key))


def inspect_feature_store_metadata(
    features_dir: Path | str,
    *,
    required_groups: Sequence[str] = ("temporal", "wavelet"),
    mount_drive: bool = False,
    mount_point: Path | str = "/content/drive",
) -> Dict[str, Any]:
    """Lightweight inspection of feature store subjects and channel count for ``--plan-only``."""
    root = ensure_drive_path_ready(
        features_dir, mount_drive=mount_drive, mount_point=mount_point
    )
    subjects = discover_feature_subjects(
        root,
        required_groups=required_groups,
        mount_drive=False,
        mount_point=mount_point,
    )
    sample_sub = subjects[0]
    groups = tuple(dict.fromkeys(required_groups))
    channel_sets: Dict[str, set[str]] = {}
    raw_counts: Dict[str, int] = {}
    for group in groups:
        rec = _read_subject_npz(root / group / f"{sample_sub}.npz", expected_group=group)
        parsed = [parse_ml3_feature_name(name) for name in rec["feature_names"]]
        channel_sets[group] = {p.channel for p in parsed}
        raw_counts[group] = len(rec["feature_names"])

    ref_channels = channel_sets[groups[0]]
    for group in groups[1:]:
        if channel_sets[group] != ref_channels:
            raise ValueError(f"Channel set mismatch between '{groups[0]}' and '{group}'")

    manifest_path = root / "feature_manifest.json"
    manifest_present = manifest_path.is_file()
    return {
        "features_dir": str(root),
        "subjects": list(subjects),
        "n_subjects": len(subjects),
        "n_channels": len(ref_channels),
        "raw_feature_counts": raw_counts,
        "manifest_present": manifest_present,
    }


def load_ml3_feature_dataset(
    features_dir: Path | str,
    *,
    subjects: Optional[Sequence[str]] = None,
    required_groups: Sequence[str] = ("temporal", "wavelet"),
    include_gamma_support: bool = False,
    mount_drive: bool = False,
    mount_point: Path | str = "/content/drive",
) -> ML3FeatureDataset:
    """Load per-subject temporal/wavelet ``.npz`` files, join on ``trial_uid``, and compute ``H4``."""
    root = ensure_drive_path_ready(
        features_dir, mount_drive=mount_drive, mount_point=mount_point
    )
    groups = tuple(dict.fromkeys(required_groups))
    available_subjects = discover_feature_subjects(
        root,
        required_groups=groups,
        mount_drive=False,
        mount_point=mount_point,
    )
    if subjects is not None:
        missing_subs = sorted(set(subjects) - set(available_subjects))
        if missing_subs:
            raise ValueError(f"Requested subjects not found in {root}: {missing_subs}")
        selected_subjects = tuple(sorted(dict.fromkeys(subjects), key=_natural_subject_key))
    else:
        selected_subjects = available_subjects

    manifest_path = root / "feature_manifest.json"
    manifest_data: Optional[Dict[str, Any]] = None
    if manifest_path.is_file():
        manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))

    need_temporal = "temporal" in groups
    need_wavelet = "wavelet" in groups
    if not (need_temporal or need_wavelet):
        raise ValueError(f"Unsupported required_groups: {groups}")

    temp_chunks: list[np.ndarray] = []
    wav_w8_chunks: list[np.ndarray] = []
    wav_w8_gamma_chunks: list[np.ndarray] = []
    sub_id_chunks: list[np.ndarray] = []
    uid_chunks: list[np.ndarray] = []
    target_chunks: list[np.ndarray] = []

    expected_temp_names: Optional[Tuple[str, ...]] = None
    expected_raw_wav_names: Optional[Tuple[str, ...]] = None
    resolved_w8_names: Tuple[str, ...] = ()
    resolved_w8_gamma_names: Tuple[str, ...] = ()
    group_hashes: Dict[str, Optional[str]] = {}

    for subject_id in selected_subjects:
        temp_rec = (
            _read_subject_npz(root / "temporal" / f"{subject_id}.npz", expected_group="temporal")
            if need_temporal
            else None
        )
        wav_rec = (
            _read_subject_npz(root / "wavelet" / f"{subject_id}.npz", expected_group="wavelet")
            if need_wavelet
            else None
        )

        if temp_rec is not None:
            if expected_temp_names is None:
                expected_temp_names = temp_rec["feature_names"]
                # Validate temporal schema (8 windows, 7 statistics)
                parsed_temp = [parse_ml3_feature_name(n) for n in expected_temp_names]
                if any(p.group != "temporal" for p in parsed_temp):
                    raise ValueError("Temporal feature_names contain non-temporal group prefix")
                temp_windows = {p.window for p in parsed_temp}
                if temp_windows != set(W8_WINDOW_NAMES):
                    raise ValueError(
                        f"Temporal windows {sorted(temp_windows)} do not match W8 {list(W8_WINDOW_NAMES)}"
                    )
                group_hashes["temporal"] = temp_rec["config_sha256"]
            elif temp_rec["feature_names"] != expected_temp_names:
                raise ValueError(
                    f"Inconsistent temporal feature_names in subject {subject_id}"
                )
            elif group_hashes.get("temporal") != temp_rec["config_sha256"]:
                raise ValueError(
                    f"Inconsistent temporal config_sha256 in subject {subject_id}"
                )

        if wav_rec is not None:
            if expected_raw_wav_names is None:
                expected_raw_wav_names = wav_rec["feature_names"]
                group_hashes["wavelet"] = wav_rec["config_sha256"]
            elif wav_rec["feature_names"] != expected_raw_wav_names:
                raise ValueError(
                    f"Inconsistent wavelet feature_names in subject {subject_id}"
                )
            elif group_hashes.get("wavelet") != wav_rec["config_sha256"]:
                raise ValueError(
                    f"Inconsistent wavelet config_sha256 in subject {subject_id}"
                )

        if temp_rec is not None and wav_rec is not None:
            t_vals, raw_w_vals, uids, targets, resolved_sub = _join_temporal_and_wavelet_subject(
                temp_rec, wav_rec
            )
            if not np.isfinite(t_vals).all():
                raise ValueError(f"Temporal features for {subject_id} contain NaN/Inf")
            temp_chunks.append(t_vals)
        elif temp_rec is not None:
            t_vals = np.ascontiguousarray(temp_rec["features"], dtype=np.float32)
            if not np.isfinite(t_vals).all():
                raise ValueError(f"Temporal features for {subject_id} contain NaN/Inf")
            temp_chunks.append(t_vals)
            raw_w_vals = None
            uids = temp_rec["trial_uids"]
            targets = temp_rec["targets"]
            resolved_sub = temp_rec["subject_id"]
        else:
            assert wav_rec is not None
            raw_w_vals = np.ascontiguousarray(wav_rec["features"], dtype=np.float32)
            uids = wav_rec["trial_uids"]
            targets = wav_rec["targets"]
            resolved_sub = wav_rec["subject_id"]

        if raw_w_vals is not None:
            assert expected_raw_wav_names is not None
            w8_vals, w8_names = compute_h4_wavelet_features(
                raw_w_vals,
                expected_raw_wav_names,
                windows=W8_WINDOW_NAMES,
                include_gamma=False,
            )
            wav_w8_chunks.append(w8_vals)
            resolved_w8_names = w8_names

            if include_gamma_support:
                w8_g_vals, w8_g_names = compute_h4_wavelet_features(
                    raw_w_vals,
                    expected_raw_wav_names,
                    windows=W8_WINDOW_NAMES,
                    include_gamma=True,
                )
                wav_w8_gamma_chunks.append(w8_g_vals)
                resolved_w8_gamma_names = w8_g_names

        n_rows = len(targets)
        sub_id_chunks.append(np.repeat(np.asarray([resolved_sub], dtype=str), n_rows))
        uid_chunks.append(np.asarray(uids, dtype=str))
        target_chunks.append(np.asarray(targets, dtype=np.int64))

    all_trial_uids = np.concatenate(uid_chunks, axis=0)
    if len(set(all_trial_uids.tolist())) != len(all_trial_uids):
        raise ValueError("Duplicate trial_uid detected across subjects")

    all_subject_ids = np.concatenate(sub_id_chunks, axis=0)
    all_targets = np.concatenate(target_chunks, axis=0)
    temporal_matrix = (
        np.ascontiguousarray(np.concatenate(temp_chunks, axis=0), dtype=np.float32)
        if temp_chunks
        else None
    )
    wavelet_w8_matrix = (
        np.ascontiguousarray(np.concatenate(wav_w8_chunks, axis=0), dtype=np.float32)
        if wav_w8_chunks
        else None
    )
    wavelet_w8_gamma_matrix = (
        np.ascontiguousarray(np.concatenate(wav_w8_gamma_chunks, axis=0), dtype=np.float32)
        if wav_w8_gamma_chunks
        else None
    )

    return ML3FeatureDataset(
        features_dir=root,
        subjects=selected_subjects,
        subject_ids=all_subject_ids,
        trial_uids=all_trial_uids,
        targets=all_targets,
        temporal_values=temporal_matrix,
        temporal_feature_names=expected_temp_names or (),
        wavelet_w8_values=wavelet_w8_matrix,
        wavelet_w8_feature_names=resolved_w8_names,
        wavelet_w8_gamma_values=wavelet_w8_gamma_matrix,
        wavelet_w8_gamma_feature_names=resolved_w8_gamma_names,
        manifest_metadata=manifest_data,
        group_hashes=group_hashes,
    )
