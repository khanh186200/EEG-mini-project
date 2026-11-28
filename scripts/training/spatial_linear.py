"""Standard ML Nhánh 1: Các Pipeline Phân loại EEG Tuyến tính & Không gian.

Module cấu hình này hiện thực hóa các kiến trúc giải mã P300 Oddball đơn thử nghiệm (Single-Trial):
1. Baseline_Vectorizer_LR: Mô hình cơ sở không dùng bộ lọc không gian.
2. xDAWN_ShrinkageLDA: Mô hình chủ đạo kết hợp lọc không gian xDAWN tối ưu SSNR
   với Phân tích biệt số tuyến tính co ngót Ledoit-Wolf (Shrinkage LDA).

Toàn bộ các bước trích chọn đặc trưng, biến đổi và phân loại được đóng gói bên trong
`sklearn.pipeline.Pipeline` nhằm đảm bảo hoàn toàn không xảy ra rò rỉ dữ liệu (zero leakage).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict

import numpy as np
from scipy.linalg import eigh
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from mne.decoding import Vectorizer


# ==============================================================================
# xDAWN an toàn thiếu hạng + tiết kiệm RAM
# ==============================================================================
class MemoryEfficientXdawn(BaseEstimator, TransformerMixin):
    """xDAWN (Rivet et al., 2009) tương đương `mne.decoding.XdawnTransformer` nhưng:

    * An toàn khi dữ liệu thiếu hạng (average reference, kênh nội suy): hiệp phương sai tổng R được chiếu
      vào không gian chính (các trị riêng > ``rank_rtol`` x trị riêng lớn nhất) trước khi giải eigh(S, R).
    * Tiết kiệm RAM: R và evoked từng lớp được cộng dồn theo lô, không tạo bản sao float64 của toàn bộ dữ liệu.

    Thuật toán giống MNE (đã kiểm chứng |cos| = 1.0000 trên cả 8 bộ lọc): với mỗi lớp c, giải bài toán trị riêng
    suy rộng S_c w = lambda R w, trong đó S_c = evoked_c @ evoked_c.T / T (không trừ trung bình, assume_centered)
    và R = hiệp phương sai (không trừ trung bình) của toàn bộ epoch. Lấy ``n_components`` bộ lọc có lambda lớn nhất
    cho mỗi lớp => đầu ra có n_classes x n_components kênh ảo.

    Tham số
    -------
    n_components : số bộ lọc không gian giữ lại cho mỗi lớp.
    rank_rtol    : ngưỡng tương đối để coi một trị riêng của R là "bằng 0".
    batch_size   : số epoch xử lý mỗi lô.
    """

    def __init__(self, n_components: int = 4, rank_rtol: float = 1e-6, batch_size: int = 512):
        self.n_components = n_components
        self.rank_rtol = rank_rtol
        self.batch_size = batch_size

    def fit(self, X, y):
        y = np.asarray(y)
        n, n_ch, n_t = X.shape
        classes = np.unique(y)
        if len(classes) < 2:
            raise ValueError("xDAWN cần ít nhất 2 lớp trong dữ liệu huấn luyện.")

        s2 = np.zeros((n_ch, n_ch))
        evo_sum = {c: np.zeros((n_ch, n_t)) for c in classes}
        n_cls = {c: 0 for c in classes}
        for i in range(0, n, self.batch_size):
            xb = np.asarray(X[i:i + self.batch_size], dtype=np.float64)
            yb = y[i:i + self.batch_size]
            flat = xb.transpose(1, 0, 2).reshape(n_ch, -1)
            s2 += flat @ flat.T
            for c in classes:
                m = yb == c
                if m.any():
                    evo_sum[c] += xb[m].sum(axis=0)
                    n_cls[c] += int(m.sum())

        R = s2 / (n * n_t)
        R = (R + R.T) / 2.0
        evals_r, V = np.linalg.eigh(R)
        keep = evals_r > self.rank_rtol * evals_r[-1]
        P = V[:, keep].T                      # (k, n_ch): cơ sở trực chuẩn của không gian chính của R
        R_r = P @ R @ P.T                     # xác định dương theo cách xây dựng

        filters = []
        for c in classes:
            evo = evo_sum[c] / max(n_cls[c], 1)
            S = evo @ evo.T / n_t
            S_r = P @ S @ P.T
            evals, evecs = eigh((S_r + S_r.T) / 2.0, R_r)
            order = np.argsort(evals)[::-1][: self.n_components]
            W = P.T @ evecs[:, order]         # đưa về không gian kênh gốc
            W /= np.linalg.norm(W, axis=0, keepdims=True)
            filters.append(W.T)

        self.filters_ = np.concatenate(filters, axis=0)   # (n_classes * n_components, n_ch)
        self.classes_ = classes
        self.n_rank_ = int(keep.sum())
        return self

    def transform(self, X):
        n = X.shape[0]
        W = self.filters_.astype(np.float32)
        out = np.empty((n, W.shape[0], X.shape[2]), dtype=np.float32)
        for i in range(0, n, self.batch_size):
            xb = np.asarray(X[i:i + self.batch_size], dtype=np.float32)
            out[i:i + self.batch_size] = np.einsum("fc,nct->nft", W, xb, optimize=True)
        return out


class TimeDecimator(BaseEstimator, TransformerMixin):
    """Giảm mẫu thời gian theo bước ``decim`` (dữ liệu đã lọc thông thấp 40 Hz nên an toàn với decim <= 3-5)."""

    def __init__(self, decim: int = 1):
        self.decim = decim

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.asarray(X[:, :, :: int(self.decim)])


# ==============================================================================
# Các pipeline
# ==============================================================================
def get_spatial_linear_models(
    n_components: int = 4,
    random_state: int = 42,
    baseline_decim: int = 1,
    xdawn_backend: str = "efficient",
) -> Dict[str, Pipeline]:
    """Khởi tạo và trả về từ điển chứa 2 pipeline phân loại tuyến tính và không gian.

    Pipeline xDAWN_ShrinkageLDA (Rivet et al., 2009):
    - xDAWN(n_components=4): 4 bộ lọc không gian mỗi lớp (tổng 8 thành phần), tối đa hóa tỷ số
      tín hiệu trên tín hiệu cộng nhiễu (SSNR) cho sóng P300. Chống thiếu hạng + tiết kiệm RAM.
    - Vectorizer: Trải phẳng mảng 3D (n_trials, 8, 251) thành 2D (n_trials, 2008).
    - StandardScaler: Chuẩn hóa trung bình=0, phương sai=1 trên từng đặc trưng.
    - Shrinkage LDA: co ngót Ledoit-Wolf tự động ổn định ma trận nghịch đảo hiệp phương sai khi p lớn.
      priors=[0.5, 0.5] đặt ngưỡng quyết định không thiên vị lớp đa số (dữ liệu thực ~86:14, tức ~6:1).

    Tham số
    -------
    n_components : số bộ lọc không gian giữ lại cho mỗi lớp.
    random_state : hạt giống ngẫu nhiên đảm bảo tái lập.
    baseline_decim : bước giảm mẫu thời gian cho Baseline (1 = giữ nguyên 250 Hz; 5 -> 50 Hz giảm RAM ~5 lần).
    xdawn_backend : "efficient" (mặc định, khuyến nghị) | "mne" (XdawnTransformer gốc + reg='ledoit_wolf';
                    tốn RAM rất lớn với Cross-Subject).

    Trả về
    -------
    dict: str -> sklearn.pipeline.Pipeline
        - 'Baseline_Vectorizer_LR': [TimeDecimator] -> Vectorizer -> StandardScaler -> PCA(50) -> LogisticRegression
        - 'xDAWN_ShrinkageLDA': xDAWN -> Vectorizer -> StandardScaler -> Shrinkage LDA
    """
    if xdawn_backend == "efficient":
        xdawn = MemoryEfficientXdawn(n_components=n_components)
    elif xdawn_backend == "mne":
        from mne.decoding import XdawnTransformer
        xdawn = XdawnTransformer(n_components=n_components, reg="ledoit_wolf")
    else:
        raise ValueError(f"xdawn_backend không hợp lệ: {xdawn_backend!r}")

    baseline_steps = []
    if baseline_decim and baseline_decim > 1:
        baseline_steps.append(("decim", TimeDecimator(decim=baseline_decim)))

    models: Dict[str, Pipeline] = {
        # Mô hình cơ sở đối chuẩn: (giảm mẫu) -> Trải phẳng -> Chuẩn hóa -> PCA(50) -> Hồi quy Logistic cân bằng
        "Baseline_Vectorizer_LR": Pipeline(
            baseline_steps
            + [
                ("vectorizer", Vectorizer()),
                ("scaler", StandardScaler()),
                ("pca", PCA(n_components=50, random_state=random_state)),
                (
                    "clf",
                    LogisticRegression(
                        C=0.1,
                        class_weight="balanced",
                        max_iter=1000,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        # Mô hình chủ đạo: xDAWN lọc không gian tối ưu SSNR -> Trải phẳng -> Chuẩn hóa -> Shrinkage LDA
        "xDAWN_ShrinkageLDA": Pipeline(
            [
                ("xdawn", xdawn),
                ("vectorizer", Vectorizer()),
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LinearDiscriminantAnalysis(
                        solver="lsqr",
                        shrinkage="auto",
                        priors=[0.5, 0.5],
                    ),
                ),
            ]
        ),
    }

    return models


if __name__ == "__main__":
    print("=" * 60)
    print("Cấu hình Pipeline Standard ML Nhánh 1 (Spatial & Linear)")
    print("=" * 60)
    models = get_spatial_linear_models()
    for name, pipe in models.items():
        print(f"\n[+] Mô hình: {name}")
        for step_name, step_obj in pipe.steps:
            print(f"    - {step_name}: {step_obj}")
    print("\n[OK] Khởi tạo thành công các pipeline.")
