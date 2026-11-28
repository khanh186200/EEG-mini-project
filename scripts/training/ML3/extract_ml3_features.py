"""Extract resumable per-subject ML3 features from the processed data contract.

Colab example (after copying processed data to ``/content``)::

    python scripts/training/extract_ml3_features.py \
      --input-dir /content/processed \
      --output-dir /content/drive/MyDrive/EEG-mini-project/features
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features import (
    FeatureExtractionConfig,
    TemporalFeatureExtractor,
    WaveletConfig,
    WaveletFeatureExtractor,
    extract_feature_groups,
    load_processed_feature_source,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract temporal and wavelet ML3 features by subject."
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--groups",
        nargs="+",
        choices=("temporal", "wavelet"),
        default=("temporal", "wavelet"),
    )
    parser.add_argument(
        "--subjects",
        nargs="+",
        default=None,
        help="Optional subset such as sub-01 sub-02; defaults to all subjects.",
    )
    parser.add_argument("--sampling-rate", type=float, default=250.0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--wavelet", default="cmor1.5-1.0")
    parser.add_argument("--frequency-step-hz", type=float, default=1.0)
    parser.add_argument("--cwt-method", choices=("conv", "fft", "auto"), default="fft")
    parser.add_argument("--wavelet-signal-batch-size", type=int, default=32)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Validate inputs and print the extraction plan without computing features.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    source = load_processed_feature_source(args.input_dir)
    selected_subjects = tuple(args.subjects) if args.subjects else source.subjects
    unknown = sorted(set(selected_subjects) - set(source.subjects))
    if unknown:
        raise ValueError(f"unknown subjects requested: {unknown}")

    plan = {
        "input_dir": str(args.input_dir),
        "output_dir": str(args.output_dir),
        "groups": list(dict.fromkeys(args.groups)),
        "subjects": list(selected_subjects),
        "n_subjects": len(selected_subjects),
        "n_channels": len(source.channel_names),
        "n_times": len(source.times_s),
        "batch_size": args.batch_size,
    }
    print(json.dumps(plan, indent=2))
    if args.plan_only:
        print("Plan only: no feature file was written.")
        return 0

    runtime = FeatureExtractionConfig(
        sampling_rate_hz=args.sampling_rate,
        batch_size=args.batch_size,
    )
    extractors = []
    for group in dict.fromkeys(args.groups):
        if group == "temporal":
            extractors.append(
                TemporalFeatureExtractor(source.times_s, source.channel_names)
            )
        elif group == "wavelet":
            extractors.append(
                WaveletFeatureExtractor(
                    source.times_s,
                    source.channel_names,
                    config=WaveletConfig(
                        wavelet=args.wavelet,
                        frequency_step_hz=args.frequency_step_hz,
                        method=args.cwt_method,
                        signal_batch_size=args.wavelet_signal_batch_size,
                    ),
                )
            )

    manifest = extract_feature_groups(
        source,
        args.output_dir,
        extractors,
        runtime,
        subjects=selected_subjects,
        overwrite=args.overwrite,
    )
    written = sum(record["status"] == "written" for record in manifest["files"])
    reused = sum(record["status"] == "reused" for record in manifest["files"])
    print(f"Feature extraction complete: written={written}, reused={reused}")
    print(f"Manifest: {args.output_dir / 'feature_manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
