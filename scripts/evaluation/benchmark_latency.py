"""Đo tốc độ suy luận single-trial (ms/epoch, Mục 3.6 Đề cương) cho DL2 trên GPU & CPU.

Chạy độc lập, không train lại mô hình:
    py -3.14 scripts/evaluation/benchmark_latency.py
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.models import P300ATCNet


def benchmark_model_on_device(
    model_name: str,
    model_factory: Callable[[], nn.Module],
    device_str: str,
    *,
    batch_size: int = 1,
    n_channels: int = 127,
    n_times: int = 251,
    n_warmup: int = 50,
    n_runs: int = 500,
    seed: int = 42,
) -> Dict[str, Any]:
    """Đo latency suy luận của một mô hình trên `cuda` hoặc `cpu`."""
    device = torch.device(device_str)
    torch.manual_seed(seed)
    model = model_factory().to(device)
    model.eval()
    n_params = int(sum(p.numel() for p in model.parameters() if p.requires_grad))

    x = torch.randn(batch_size, n_channels, n_times, dtype=torch.float32, device=device)
    is_cuda = device.type == "cuda"

    timings_ms: List[float] = []
    with torch.inference_mode():
        for _ in range(n_warmup):
            _ = model(x)
            if is_cuda:
                torch.cuda.synchronize(device)

        for _ in range(n_runs):
            if is_cuda:
                torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            out = model(x)
            if is_cuda:
                torch.cuda.synchronize(device)
            t1 = time.perf_counter()
            if out.shape != (batch_size, 1):
                raise RuntimeError(f"Shape đầu ra bất thường: {tuple(out.shape)}")
            timings_ms.append((t1 - t0) * 1000.0)

    arr = np.asarray(timings_ms, dtype=np.float64)
    q25, q75 = np.percentile(arr, [25.0, 75.0])
    mean_ms = float(np.mean(arr))
    return {
        "model_name": model_name,
        "device": device_str,
        "device_name": (
            torch.cuda.get_device_name(device) if is_cuda else (platform.processor() or platform.machine())
        ),
        "n_params": n_params,
        "batch_size": batch_size,
        "input_shape": [batch_size, n_channels, n_times],
        "n_warmup": n_warmup,
        "n_runs": n_runs,
        "mean_ms": mean_ms,
        "std_ms": float(np.std(arr, ddof=1)),
        "median_ms": float(np.median(arr)),
        "p95_ms": float(np.percentile(arr, 95.0)),
        "p99_ms": float(np.percentile(arr, 99.0)),
        "min_ms": float(np.min(arr)),
        "max_ms": float(np.max(arr)),
        "iqr_ms": float(q75 - q25),
        "throughput_epochs_per_sec": float(1000.0 / mean_ms) if mean_ms > 0 else 0.0,
    }


def run_latency_benchmark(
    output_path: Path = Path("results/metrics/dl2/inference_latency.json"),
    *,
    n_warmup: int = 50,
    n_runs: int = 500,
) -> Dict[str, Any]:
    """Chạy benchmark cho P300ATCNet (attn_pool) trên GPU và CPU."""
    devices: List[str] = []
    if torch.cuda.is_available():
        devices.append("cuda")
    devices.append("cpu")

    model_specs: List[Tuple[str, Callable[[], nn.Module]]] = [
        ("P300ATCNet (attn_pool)", lambda: P300ATCNet(n_channels=127, n_times=251, readout="attn_pool")),
    ]

    results: List[Dict[str, Any]] = []
    for device_str in devices:
        for model_name, factory in model_specs:
            res = benchmark_model_on_device(
                model_name=model_name,
                model_factory=factory,
                device_str=device_str,
                batch_size=1,
                n_channels=127,
                n_times=251,
                n_warmup=n_warmup,
                n_runs=n_runs,
            )
            results.append(res)

    payload: Dict[str, Any] = {
        "protocol": "single_trial_inference_latency",
        "batch_size": 1,
        "input_shape": [1, 127, 251],
        "n_warmup": n_warmup,
        "n_runs": n_runs,
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "cpu_name": platform.processor() or platform.machine(),
        "python_version": platform.python_version(),
        "benchmarks": results,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("\n=== Inference Latency Benchmark (batch_size=1, shape=(1, 127, 251), warmup=50, runs=500) ===")
    header = (
        f"{'Model':<24} | {'Device':<6} | {'Params':>7} | "
        f"{'Mean ± Std (ms/ep)':>20} | {'Median (ms)':>11} | {'p95 (ms)':>9} | {'Throughput (ep/s)':>17}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        mean_std = f"{r['mean_ms']:.3f} ± {r['std_ms']:.3f}"
        print(
            f"{r['model_name']:<24} | {r['device'].upper():<6} | {r['n_params']:>7,} | "
            f"{mean_std:>20} | {r['median_ms']:>11.3f} | {r['p95_ms']:>9.3f} | "
            f"{r['throughput_epochs_per_sec']:>17.1f}"
        )
    print(f"\n[Benchmark] Đã lưu kết quả tại: {output_path}")
    return payload


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Đo Inference Latency (ms/epoch) cho P300ATCNet (attn_pool).")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/metrics/dl2/inference_latency.json"),
        help="Đường dẫn file JSON kết quả (mặc định: results/metrics/dl2/inference_latency.json).",
    )
    parser.add_argument("--warmup", type=int, default=50, help="Số lần warm-up (mặc định: 50).")
    parser.add_argument("--runs", type=int, default=500, help="Số lần đo (mặc định: 500).")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    run_latency_benchmark(output_path=args.output, n_warmup=args.warmup, n_runs=args.runs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
