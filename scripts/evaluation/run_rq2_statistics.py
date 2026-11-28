"""Future entrypoint for paired RQ2 tests.

Input aggregation and pairing live in ``src.evaluation.statistics_rq2``. The
choice of final model files and sensitivity cohorts remains pending.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run paired RQ2 statistical tests.")
    parser.add_argument(
        "--handover-dir",
        type=Path,
        default=Path("results/handover_standard"),
        help=(
            "Root handover directory containing branch subfolders "
            "(e.g. results/handover_standard/DL2, results/handover_standard/Ml3)."
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    build_arg_parser().parse_args(argv)
    raise NotImplementedError(
        "RQ2 execution is pending the final five-model result manifest."
    )


if __name__ == "__main__":
    raise SystemExit(main())
