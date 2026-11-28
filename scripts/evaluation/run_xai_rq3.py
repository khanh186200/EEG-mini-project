"""CLI chạy Pipeline Explainable AI (RQ3 — 4 lớp kiểm chứng) & sinh báo cáo Bước 6.

Ví dụ:
    py -3.14 scripts/evaluation/run_xai_rq3.py --resume
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.evaluation.xai_rq3 import run_full_xai_rq3_pipeline, run_step6_5_addendum


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Chạy Pipeline XAI 4 lớp kiểm chứng (RQ3) cho P300-ATCNet."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/processed"),
        help="Thư mục dữ liệu đã tiền xử lý (mặc định: data/processed).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
        help="Thư mục gốc chứa checkpoints và metrics (mặc định: results).",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=Path("reports/STEP6_XAI_RQ3_REPORT.md"),
        help="Đường dẫn xuất báo cáo Markdown Bước 6.",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=64,
        help="Số bước tích phân Gauss-Legendre của Captum IntegratedGradients (mặc định: 64).",
    )
    parser.add_argument(
        "--internal-batch-size",
        type=int,
        default=32,
        help="Kích thước internal_batch_size của Captum IG (mặc định: 32).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed cố định lấy mẫu đối chứng Frequent (mặc định: 42).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Thiết bị chạy (vd: cuda, cpu); mặc định tự chọn CUDA nếu có.",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Dùng lại cache xai_loso_cache.npz và checkpoints cắt cửa sổ sớm nếu đã có.",
    )
    parser.add_argument(
        "--force-recompute",
        action="store_true",
        help="Buộc tính lại từ đầu kể cả khi đã tồn tại cache và kết quả cắt cửa sổ sớm.",
    )
    parser.add_argument(
        "--addendum-only",
        action="store_true",
        help="Chỉ chạy kiểm chứng bổ sung Bước 6.5 và cập nhật summary/báo cáo.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.addendum_only:
        run_step6_5_addendum(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            report_path=args.report_path,
            n_steps=args.n_steps,
            internal_batch_size=args.internal_batch_size,
            device=args.device,
        )
        return 0
    run_full_xai_rq3_pipeline(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        report_path=args.report_path,
        n_steps=args.n_steps,
        internal_batch_size=args.internal_batch_size,
        seed=args.seed,
        resume=args.resume,
        force_recompute=args.force_recompute,
        device=args.device,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
