"""CLI xuất trọn bộ 6 biểu đồ chuẩn bài báo (300 DPI PNG + Vector PDF).

Ví dụ:
    py -3.14 scripts/evaluation/generate_figures.py
    py -3.14 scripts/evaluation/generate_figures.py --external-results-dir results/handover_all
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.evaluation.visualization import generate_all_paper_figures


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Xuất trọn bộ 6 biểu đồ chuẩn bài báo (300 DPI) vào results/figures/paper/."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/processed"),
        help="Thư mục dữ liệu đã tiền xử lý (mặc định: data/processed).",
    )
    parser.add_argument(
        "--metrics-dir",
        type=Path,
        default=Path("results/metrics/dl2"),
        help="Thư mục chứa xai_loso_cache.npz, xai_rq3_summary.json & handover CSV.",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=Path("results/figures/paper"),
        help="Thư mục xuất 6 biểu đồ (mặc định: results/figures/paper).",
    )
    parser.add_argument(
        "--external-results-dir",
        type=Path,
        default=None,
        help="Thư mục tùy chọn chứa *_per_fold.csv & *_preds.npz của các nhánh khác cho Fig 3 & Fig 3b.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    generate_all_paper_figures(
        data_dir=args.data_dir,
        metrics_dir=args.metrics_dir,
        figures_dir=args.figures_dir,
        external_results_dir=args.external_results_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
