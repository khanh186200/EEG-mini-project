"""Validate all currently available group handover artifacts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.handover_validation import validate_handover_directory


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate Proposal section 3.5 artifacts.")
    parser.add_argument(
        "--handover-dir",
        type=Path,
        default=Path("results/handover_standard"),
        help=(
            "Root handover directory or a specific branch subfolder "
            "(e.g. results/handover_standard/DL2, results/handover_standard/Ml3)."
        ),
    )
    parser.add_argument(
        "--require-prediction-pair",
        action="store_true",
        help="Require every per-fold CSV to have a matching predictions NPZ and vice versa.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    report = validate_handover_directory(
        args.handover_dir,
        require_prediction_pair=args.require_prediction_pair,
    )
    if report.is_valid:
        print(f"Handover schema valid: {report.files_checked} file(s) checked.")
        return 0
    for issue in report.issues:
        print(f"ERROR {issue.path}: {issue.message}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
