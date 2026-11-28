"""Bộ công cụ khám phá toàn diện dữ liệu EEG (Oddball Paradigm - P300)
Tệp: scripts/data/explore_data.py

Quy trình 6 bước chuẩn mực trong EEG Signal Processing:
1. 01 - Dataset Overview: Quét file, kiểm tra sampling rate, số kênh, dung lượng, thời lượng ghi.
2. 02 - Channel Analysis: Khám phá hệ thống 127 kênh, kiểm tra kênh EOG trán cực (Fp1, Fp2),
   kênh tai (TP9, TP10), kiểm tra hoán đổi thứ tự kênh giữa 42 subjects.
3. 03 - Event Analysis: Trích xuất và phân tích trigger (Frequent S5, Target S6, Distractor S7),
   tính tỷ lệ mất cân bằng lớp và trực quan hóa phân bố kích thích.
4. 04 - Signal Quality Analysis: Kiểm tra toàn vẹn dữ liệu (NaN/Inf), tính toán thống kê biên độ (PtP, std),
   phát hiện tự động kênh phẳng (flat) và kênh nhiễu bất thường (noisy/outliers).
5. 05 - Signal Visualization & PSD:
   - Trực quan hóa các đoạn sóng EEG thô trên các kênh đại diện (Fp1, Cz, Pz, Oz).
   - Phân tích phổ mật độ công suất (Power Spectral Density - PSD), nhận diện đỉnh nhiễu điện lưới 50 Hz.
6. 06 - Inter-subject Analysis: Tổng hợp và so sánh thời lượng, tần số lấy mẫu và số lượng trial
   giữa toàn bộ 42 đối tượng đo, xuất báo cáo CSV và biểu đồ so sánh.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # Chế độ headless an toàn, tự động lưu ảnh không cần GUI
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd

# Thiết lập mức log MNE
mne.set_log_level("WARNING")


# ==============================================================================
# HÀM BỔ TRỢ CƠ BẢN
# ==============================================================================
def natural_sort_key(s: str) -> List[int | str]:
    """Sắp xếp tự nhiên theo thứ tự số (ví dụ sub-02 trước sub-10)."""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r"(\d+)", str(s))]


def extract_subject_id(vhdr_path: Path) -> str:
    """Trích xuất subject_id an toàn bằng regex."""
    match = re.search(r"(sub-\d+)", vhdr_path.stem)
    if match:
        return match.group(1)
    match_parent = re.search(r"(sub-\d+)", str(vhdr_path))
    if match_parent:
        return match_parent.group(1)
    return vhdr_path.parent.name


def normalize_event_name(name: str) -> str:
    """Chuẩn hóa tên sự kiện: loại bỏ khoảng trắng và chuyển chữ thường."""
    return re.sub(r"\s+", "", name).lower()


def default_project_root() -> Path:
    """SỬA: không phụ thuộc cứng vào __file__ (notebook/Colab không có __file__)."""
    env = os.getenv("EEG_ROOT")
    if env:
        return Path(env)
    try:
        return Path(__file__).resolve().parents[2]
    except (NameError, IndexError):
        return Path.cwd()


def line_noise_report(
    raw: mne.io.Raw,
    line_freq: float = 50.0,
    l_freq: float = 0.5,
    h_freq: float = 40.0,
    resample_freq: float = 250.0,
    residual_db: float = 3.0,
) -> Dict[str, Any]:
    """SỬA: thay nhận xét "cứng" bằng phép đo. Độ nổi của đỉnh lưới trên dữ liệu thô, trừ đi độ suy giảm
    của bộ bandpass sẽ dùng trong pipeline => còn nổi bao nhiêu dB sau lọc? Quyết định notch dựa trên số này."""
    sfreq = raw.info["sfreq"]
    tmax = min(120.0, raw.times[-1])
    n_fft = int(min(sfreq * 4, tmax * sfreq))
    psd = raw.compute_psd(method="welch", fmin=line_freq - 8, fmax=line_freq + 8, tmax=tmax,
                          picks="eeg", n_fft=n_fft, verbose=False)
    data, freqs = psd.get_data(return_freqs=True)
    db = (10 * np.log10(np.maximum(data, 1e-30))).mean(axis=0)
    d = np.abs(freqs - line_freq)
    prominence = float(db[d <= 0.5].max() - np.median(db[(d >= 2.0) & (d <= 6.0)]))
    from scipy.signal import freqz
    h = mne.filter.create_filter(None, resample_freq, l_freq, h_freq, method="fir", phase="zero",
                                 fir_design="firwin", verbose=False)
    _, resp = freqz(h, worN=[2 * np.pi * line_freq / resample_freq])
    atten = float(20 * np.log10(max(abs(resp[0]), 1e-12)))
    residual = prominence + atten
    return {
        "prominence_db": prominence,
        "bandpass_attenuation_db": atten,
        "residual_db": residual,
        "notch_needed": residual > residual_db,
    }


def check_montage_files(vhdr_files: List[Path], output_dir: Optional[Path] = None) -> pd.DataFrame:
    """MỚI: kiểm tra từng subject có file tọa độ điện cực thực tế không, và dữ liệu có hợp lý không
    (đơn vị, bán kính trung vị trên da đầu). Kết quả quyết định pipeline dùng tọa độ đo hay montage mẫu."""
    print(f"\n{'='*75}\n02b. KIỂM TRA FILE TỌA ĐỘ ĐIỆN CỰC THỰC TẾ\n{'='*75}")
    first_dir = vhdr_files[0].parent
    print(f" • Nội dung thư mục subject đầu tiên ({first_dir}):")
    for f in sorted(first_dir.iterdir()):
        print(f"     - {f.name}")
    rows = []
    for vhdr in vhdr_files:
        sub = extract_subject_id(vhdr)
        d = vhdr.parent
        tsvs = sorted(d.glob(f"*{sub}*electrodes.tsv")) or sorted(d.glob("*electrodes.tsv"))
        cs = sorted(d.glob(f"*{sub}*coordsystem.json")) or sorted(d.glob("*coordsystem.json"))
        bvct = sorted(d.glob("*.bvct"))
        row = {"subject_id": sub, "electrodes_tsv": bool(tsvs), "coordsystem_json": bool(cs),
               "bvct": bool(bvct), "n_electrodes": None, "unit": None, "median_radius_m": None, "plausible": None}
        if tsvs:
            try:
                df = pd.read_csv(tsvs[0], sep="\t")
                xyz = df[["x", "y", "z"]].apply(pd.to_numeric, errors="coerce").dropna().to_numpy(float)
                unit = "m"
                if cs:
                    meta = json.loads(cs[0].read_text(encoding="utf-8"))
                    unit = str(meta.get("EEGCoordinateUnits", "m")).lower()
                scale = {"m": 1.0, "cm": 1e-2, "mm": 1e-3}.get(unit, 1.0)
                r = float(np.median(np.linalg.norm(xyz * scale, axis=1)))
                row.update(n_electrodes=len(xyz), unit=unit, median_radius_m=round(r, 4), plausible=0.05 <= r <= 0.15)
            except Exception as e:  # noqa: BLE001
                row["unit"] = f"LỖI: {e}"
        rows.append(row)
    df_m = pd.DataFrame(rows)
    n_ok = int(df_m["plausible"].fillna(False).sum())
    print(f" • Subject có electrodes.tsv hợp lý: {n_ok}/{len(df_m)} | có .bvct: {int(df_m['bvct'].sum())}/{len(df_m)}")
    if n_ok == 0 and int(df_m["bvct"].sum()) == 0:
        print(" => KHÔNG có tọa độ thực tế dùng được; pipeline sẽ dùng montage mẫu 10-05 (cần ghi rõ trong báo cáo).")
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        df_m.to_csv(output_dir / "montage_files_report.csv", index=False)
    return df_m


def estimate_memory(df: pd.DataFrame, resample_freq: float = 250.0, tmin: float = -0.2, tmax: float = 0.8) -> None:
    """MỚI: ước lượng RAM để biết có cần ghi từng subject ra đĩa (pipeline V3 luôn làm vậy)."""
    n_ch = int(df["n_channels"].iloc[0])
    n_ep = int((df["n_frequent"] + df["n_target"]).sum())
    n_t = int(round((tmax - tmin) * resample_freq)) + 1
    per_set = n_ep * n_ch * n_t * 4 / 1e9
    sfreq = float(df["sampling_rate"].iloc[0])
    raw_gb = float(df["duration_min"].max()) * 60 * sfreq * n_ch * 8 / 1e9
    print(f"\n BỘ NHỚ (ước lượng): {n_ep:,} epoch x {n_ch} kênh x {n_t} mẫu (float32)")
    print(f" • Mỗi bộ epoch toàn dataset:                 {per_set:.2f} GB")
    print(f" • Cách cũ (2 bộ + np.concatenate, đỉnh):     ~{4 * per_set:.1f} GB (Colab thường 12-13 GB)")
    print(f" • 1 recording dài nhất @ {sfreq:g} Hz float64:   {raw_gb:.2f} GB (pipeline giữ ~2-3 bản mỗi subject)")
    print(f" • Cách mới (từng subject + memmap):          đỉnh ~{3 * raw_gb:.1f} GB bất kể số subject")


def scan_raw_files(raw_dir: str | Path) -> List[Path]:
    """Quét thư mục và trả về danh sách file .vhdr đã sắp xếp tự nhiên."""
    raw_path = Path(raw_dir)
    if not raw_path.exists():
        raise FileNotFoundError(f"Không tìm thấy thư mục dữ liệu thô tại: {raw_path.resolve()}")
    return sorted(list(raw_path.rglob("*.vhdr")), key=natural_sort_key)


# ==============================================================================
# BƯỚC 1 & 2: OVERVIEW & CHANNEL ANALYSIS
# ==============================================================================
def inspect_channel_types(ch_names: List[str]) -> Dict[str, Any]:
    """Phân loại các kênh đo để xác định kênh EOG chuyên dụng hoặc kênh trán cực (Fp1/Fp2)."""
    ch_names_upper = [ch.upper() for ch in ch_names]

    dedicated_eog = [
        ch for ch, ch_u in zip(ch_names, ch_names_upper)
        if any(tag in ch_u for tag in ["EOG", "VEOG", "HEOG"])
    ]

    frontal_surrogates = [
        ch for ch in ch_names
        if ch.upper() in ["FP1", "FP2", "AFP1", "AFP2", "FPZ"]
    ]

    mastoids_ears = [
        ch for ch in ch_names
        if any(tag in ch.upper() for tag in ["TP9", "TP10", "A1", "A2", "EAR", "MASTOID"])
    ]

    return {
        "total_channels": len(ch_names),
        "dedicated_eog": dedicated_eog,
        "frontal_surrogates": frontal_surrogates,
        "mastoids_ears": mastoids_ears,
    }


def check_channel_consistency(vhdr_files: List[Path]) -> Dict[str, Any]:
    """Kiểm tra tính nhất quán của danh sách và thứ tự kênh giữa tất cả các subject."""
    print(f"\n{'='*75}")
    print("02. KIỂM TRA TÍNH NHẤT QUÁN KÊNH ĐO (CHANNEL CONSISTENCY & ORDER)")
    print(f"{'='*75}")

    reference_channels = None
    ref_sub = None
    order_mismatches = []
    set_mismatches = []

    for vhdr in vhdr_files:
        sub = extract_subject_id(vhdr)
        with open(vhdr, "r", encoding="latin1") as f:
            lines = [l.strip() for l in f if re.match(r"^Ch\d+\s*=", l.strip())]
        names = [l.split("=")[1].split(",")[0].strip() for l in lines]

        # Chuẩn hóa tên kênh tai
        names = ["TP9" if n == "TP9 LEFT EAR" else "TP10" if n == "TP10 RIGHT EAR" else n for n in names]

        if reference_channels is None:
            reference_channels = names
            ref_sub = sub
        else:
            if set(names) != set(reference_channels):
                set_mismatches.append((sub, set(reference_channels) ^ set(names)))
            elif names != reference_channels:
                diffs = [
                    (i, c_ref, c_cur)
                    for i, (c_ref, c_cur) in enumerate(zip(reference_channels, names))
                    if c_ref != c_cur
                ]
                order_mismatches.append((sub, diffs))

    print(f" • Subject tham chiếu ({ref_sub}): Tổng cộng {len(reference_channels)} kênh.")
    ch_info = inspect_channel_types(reference_channels)
    print(f" • Kênh EOG chuyên dụng: {ch_info['dedicated_eog'] or 'Không có'}")
    print(f" • Kênh trán cực thay thế EOG (Surrogates): {ch_info['frontal_surrogates']}")
    print(f" • Kênh vùng tai/xương chũm (Mastoids): {ch_info['mastoids_ears']}")
    if set_mismatches:
        print(f"  CẢNH BÁO: Có {len(set_mismatches)} subjects khác biệt về danh sách kênh: {set_mismatches}")
    else:
        print(" 100% các subjects có cùng một tập hợp 127 kênh EEG.")

    if order_mismatches:
        print(f" PHÁT HIỆN: Có {len(order_mismatches)} subjects bị ĐẢO THỨ TỰ KÊNH so với {ref_sub}:")
        for sub, diffs in order_mismatches:
            print(f"    - {sub}: {len(diffs)} kênh hoán đổi vị trí (ví dụ index {diffs[0][0]}: {diffs[0][1]} vs {diffs[0][2]})")
        print("   Khuyến nghị: Pipeline bắt buộc phải gọi raw.reorder_channels() trước khi xuất mảng NumPy!")
    else:
        print(" Thứ tự kênh hoàn toàn đồng nhất trên tất cả subjects.")

    return {
        "reference_channels": reference_channels,
        "order_mismatches": order_mismatches,
        "set_mismatches": set_mismatches,
    }


# ==============================================================================
# BƯỚC 3: EVENT ANALYSIS & CLASS DISTRIBUTION
# ==============================================================================
def analyze_events(raw: mne.io.Raw) -> Tuple[np.ndarray, Dict[str, int], Dict[str, int]]:
    """Trích xuất và thống kê an toàn số lượng trial của từng sự kiện từ Annotations."""
    events, event_id = mne.events_from_annotations(raw, verbose="WARNING")

    if len(events) == 0:
        return events, event_id, {}

    unique_ids, counts = np.unique(events[:, 2], return_counts=True)
    id_to_annot = {v: k for k, v in event_id.items()}

    event_counts = {}
    for code, count in zip(unique_ids, counts):
        annot_name = id_to_annot.get(code, f"Code_{code}")
        event_counts[annot_name] = int(count)

    return events, event_id, event_counts


def plot_event_distribution(
    event_counts: Dict[str, int],
    subject_id: str,
    output_dir: Path,
) -> None:
    """Vẽ biểu đồ phân bố sự kiện và tỷ lệ mất cân bằng lớp Oddball."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Lọc lấy các stimulus chính trong bài toán Oddball
    stim_map = {
        "s5": "Frequent (Standard)",
        "s6": "Rare Target (P300)",
        "s7": "Rare Distractor",
        "s1": "Response (Button)",
    }

    labels = []
    values = []
    colors = ["#2b5c8f", "#d95f02", "#7570b3", "#1b9e77"]

    for key, name in stim_map.items():
        count = sum(v for k, v in event_counts.items() if normalize_event_name(k).endswith(key))
        labels.append(name)
        values.append(count)

    plt.figure(figsize=(9, 5))
    bars = plt.bar(labels, values, color=colors[:len(labels)], width=0.55, edgecolor="black", alpha=0.85)

    total_stim = values[0] + values[1] + values[2] if len(values) >= 3 else sum(values)
    for bar, val in zip(bars, values):
        pct = (val / total_stim * 100) if total_stim > 0 else 0
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(values) * 0.015,
            f"{val:,}\n({pct:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    plt.title(f"Phân bố kích thích Oddball Paradigm - {subject_id}", fontsize=13, fontweight="bold")
    plt.ylabel("Số lượng Trials", fontsize=11)
    plt.ylim(0, max(max(values), 1) * 1.18)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()

    save_path = output_dir / f"{subject_id}_event_distribution.png"
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f" Đã lưu biểu đồ phân bố sự kiện: {save_path.name}")


# ==============================================================================
# BƯỚC 4: SIGNAL QUALITY ANALYSIS (KIỂM TRA CHẤT LƯỢNG TÍN HIỆU THỰC TẾ)
# ==============================================================================
def analyze_signal_quality(
    raw: mne.io.Raw,
    subject_id: str,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Đánh giá chi tiết chất lượng tín hiệu EEG thực tế:

    1. Kiểm tra NaN / Inf.
    2. Đánh giá biên độ (Peak-to-Peak, Mean, Std) trên từng kênh.
    3. Tự động phát hiện Flat channels (< 0.1 µV) và Noisy channels (z-score > 5.0).
    """
    print(f"\n{'='*75}")
    print(f"04. ĐÁNH GIÁ CHẤT LƯỢNG TÍN HIỆU THỰC TẾ (SIGNAL QUALITY) - {subject_id}")
    print(f"{'='*75}")

    data = raw.get_data(picks="eeg")
    ch_names = [raw.ch_names[p] for p in mne.pick_types(raw.info, eeg=True)]

    # 1. Kiểm tra NaN / Inf
    has_nan = bool(np.isnan(data).any())
    has_inf = bool(np.isinf(data).any())
    print(f" • Kiểm tra tính toàn vẹn: NaN={has_nan}, Inf={has_inf}")
    if has_nan or has_inf:
        print(" CẢNH BÁO: Tín hiệu chứa giá trị NaN hoặc Inf!")
    else:
        print(" Dữ liệu hoàn toàn sạch, không có giá trị khuyết thiếu (NaN/Inf).")

    # 2. Thống kê biên độ theo microvolt (µV)
    data_uv = data * 1e6
    ptp_uv = np.ptp(data_uv, axis=1)  # Peak-to-peak amplitude
    std_uv = np.std(data_uv, axis=1)
    mean_uv = np.mean(data_uv, axis=1)

    median_std = np.median(std_uv)
    mad = np.median(np.abs(std_uv - median_std))
    mad = mad if mad > 1e-6 else 1e-6

    # Phát hiện Flat channels (< 0.1 µV std)
    flat_channels = [ch for ch, s in zip(ch_names, std_uv) if s < 0.1]
    # Phát hiện Noisy channels (z-score > 5.0)
    noisy_channels = []
    for ch, s in zip(ch_names, std_uv):
        z = 0.6745 * (s - median_std) / mad
        if z > 5.0:
            noisy_channels.append((ch, round(s, 2), round(z, 2)))

    print(f" • Biên độ Std trung vị toàn đầu: {median_std:.2f} µV")
    print(f" • Kênh có biên độ Peak-to-Peak nhỏ nhất: {ch_names[np.argmin(ptp_uv)]} ({np.min(ptp_uv):.1f} µV)")
    print(f" • Kênh có biên độ Peak-to-Peak lớn nhất: {ch_names[np.argmax(ptp_uv)]} ({np.max(ptp_uv):.1f} µV)")

    if flat_channels:
        print(f" CẢNH BÁO FLAT CHANNELS (std < 0.1 µV): {flat_channels}")
    else:
        print(" Không phát hiện kênh phẳng (Dead/Flat channels).")

    if noisy_channels:
        print(f" CẢNH BÁO NOISY CHANNELS (MAD z-score > 5.0): {len(noisy_channels)} kênh")
        for ch, s, z in noisy_channels[:5]:
            print(f"    - {ch}: std = {s} µV (z-score = {z})")
        if len(noisy_channels) > 5:
            print(f"    ... và {len(noisy_channels) - 5} kênh khác.")
    else:
        print(" Không có kênh nào bị nhiễu biên độ đột biến kéo dài.")

    # Vẽ biểu đồ phân bố biên độ Std của 127 kênh
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(10, 4))
        plt.plot(range(len(ch_names)), std_uv, color="#1f77b4", lw=1.2, label="Độ lệch chuẩn kênh (µV)")
        plt.axhline(median_std, color="green", linestyle="--", label=f"Median ({median_std:.1f} µV)")
        plt.axhline(median_std + 5 * mad / 0.6745, color="red", linestyle=":", label="Ngưỡng nhiễu (z=5)")
        plt.title(f"Phân bố độ lệch chuẩn biên độ các kênh - {subject_id}", fontsize=12, fontweight="bold")
        plt.xlabel("Chỉ số kênh (1 - 127)", fontsize=10)
        plt.ylabel("Standard Deviation (µV)", fontsize=10)
        plt.legend(loc="upper right")
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.tight_layout()

        save_path = output_dir / f"{subject_id}_channel_std_distribution.png"
        plt.savefig(save_path, dpi=200)
        plt.close()
        print(f" Đã lưu biểu đồ biên độ kênh: {save_path.name}")

    return {
        "has_nan": has_nan,
        "has_inf": has_inf,
        "median_std_uv": median_std,
        "flat_channels": flat_channels,
        "noisy_channels": [ch[0] for ch in noisy_channels],
    }


# ==============================================================================
# BƯỚC 5: SIGNAL VISUALIZATION & POWER SPECTRAL DENSITY (PSD)
# ==============================================================================
def visualize_signals_and_psd(
    raw: mne.io.Raw,
    subject_id: str,
    output_dir: Path,
) -> None:
    """Trực quan hóa đoạn sóng EEG thô và phân tích phổ tần số (PSD)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*75}")
    print(f"05. TRỰC QUAN HÓA TÍN HIỆU EEG & PHÂN TÍCH PHỔ TẦN SỐ (PSD) - {subject_id}")
    print(f"{'='*75}")

    # 1. Trực quan hóa sóng EEG thô trên một số kênh tiêu biểu
    # Kênh trán (Fp1, Fp2 - chớp mắt), Kênh đỉnh (Cz, Pz - P300), Kênh chẩm (Oz - Alpha)
    target_channels = ["Fp1", "Fp2", "Fz", "Cz", "Pz", "Oz"]
    selected_channels = [ch for ch in target_channels if ch in raw.ch_names]

    if selected_channels:
        sfreq = int(raw.info["sfreq"])
        duration_to_plot = 6.0  # Vẽ 6 giây đầu tiên sau 10s khởi động
        start_sec = min(15.0, max(0.0, raw.times[-1] - duration_to_plot - 1.0))
        start_sample = int(start_sec * sfreq)
        n_samples = int(duration_to_plot * sfreq)

        data_seg, times = raw[selected_channels, start_sample : start_sample + n_samples]
        times = times - times[0]  # Đưa trục thời gian về 0 - 6s

        fig, axes = plt.subplots(len(selected_channels), 1, figsize=(12, 8), sharex=True)
        if len(selected_channels) == 1:
            axes = [axes]

        for ax, ch, ch_data in zip(axes, selected_channels, data_seg):
            ch_uv = ch_data * 1e6
            ax.plot(times, ch_uv, lw=1.0, color="#1a365d")
            ax.set_ylabel(f"{ch}\n(µV)", fontsize=9, fontweight="bold", rotation=0, labelpad=25)
            ax.grid(True, linestyle="--", alpha=0.4)

        axes[-1].set_xlabel("Thời gian (giây)", fontsize=11)
        fig.suptitle(f"Dạng sóng EEG thô trên các kênh tiêu biểu (6s) - {subject_id}", fontsize=13, fontweight="bold")
        plt.tight_layout()

        raw_trace_path = output_dir / f"{subject_id}_raw_eeg_traces.png"
        fig.savefig(raw_trace_path, dpi=200)
        plt.close(fig)
        print(f"  Đã lưu sóng EEG thô tiêu biểu: {raw_trace_path.name}")

    # 2. Phân tích phổ mật độ công suất (Power Spectral Density - PSD)
    print(" • Đang tính toán Power Spectral Density (PSD) từ 1.0 đến 70.0 Hz...")
    psd = raw.compute_psd(fmin=1.0, fmax=70.0, picks="eeg", verbose=False)
    psd_data, freqs = psd.get_data(return_freqs=True)
    # Chuyển đổi công suất sang dB (10 * log10(µV²/Hz))
    psd_db = 10 * np.log10(np.maximum(psd_data * 1e12, 1e-12))
    mean_psd_db = np.mean(psd_db, axis=0)

    plt.figure(figsize=(10, 5))
    plt.plot(freqs, mean_psd_db, color="#2b5c8f", lw=2.0, label="PSD Trung bình toàn bộ điện cực")

    # Đánh dấu các vùng dải sóng EEG sinh học
    plt.axvspan(1, 4, color="gray", alpha=0.15, label="Delta (1-4 Hz)")
    plt.axvspan(4, 8, color="purple", alpha=0.12, label="Theta (4-8 Hz)")
    plt.axvspan(8, 13, color="orange", alpha=0.15, label="Alpha (8-13 Hz)")
    plt.axvspan(13, 30, color="green", alpha=0.10, label="Beta (13-30 Hz)")

    # Đánh dấu đỉnh nhiễu điện lưới 50 Hz
    plt.axvline(50.0, color="red", linestyle="--", lw=1.5, label="Nhiễu điện lưới (50 Hz)")

    plt.title(f"Phổ mật độ công suất EEG (Power Spectral Density) - {subject_id}", fontsize=13, fontweight="bold")
    plt.xlabel("Tần số (Hz)", fontsize=11)
    plt.ylabel("Công suất (dB: 10*log10(µV²/Hz))", fontsize=11)
    plt.legend(loc="upper right", fontsize=9)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    psd_path = output_dir / f"{subject_id}_psd_spectrum.png"
    plt.savefig(psd_path, dpi=200)
    plt.close()
    print(f" Đã lưu biểu đồ phổ PSD: {psd_path.name}")
    rep = line_noise_report(raw)
    print(f" • Đỉnh 50 Hz nổi {rep['prominence_db']:.1f} dB trên phổ lân cận (dữ liệu thô).")
    print(f" • Bandpass 0.5-40 Hz của pipeline suy giảm {rep['bandpass_attenuation_db']:.1f} dB tại 50 Hz "
          f"=> còn nổi {rep['residual_db']:.1f} dB sau lọc.")
    print("   KẾT LUẬN: " + ("CẦN thêm notch 50 Hz cho subject này." if rep["notch_needed"]
                           else "KHÔNG cần notch riêng (bandpass đã đủ)."))


# ==============================================================================
# BƯỚC 6: INTER-SUBJECT ANALYSIS (THỐNG KÊ TOÀN DIỆN 42 SUBJECTS)
# ==============================================================================
def analyze_inter_subject_statistics(
    vhdr_files: List[Path],
    output_dir: Path,
) -> pd.DataFrame:
    """Quét metadata của toàn bộ 42 subjects (không preload data, siêu nhanh):

    So sánh tần số lấy mẫu, số kênh, thời lượng ghi và phân bố trial của từng người.
    """
    print(f"\n{'='*75}")
    print("06. PHÂN TÍCH THỐNG KÊ LIÊN ĐỐI TƯỢNG (INTER-SUBJECT COMPARISON - 42 SUBJECTS)")
    print(f"{'='*75}")

    summary_rows = []
    print(f"[+] Đang quét metadata nhanh cho {len(vhdr_files)} đối tượng...")

    for vhdr in vhdr_files:
        sub_id = extract_subject_id(vhdr)
        try:
            raw = mne.io.read_raw_brainvision(vhdr, preload=False, verbose="WARNING")
            sfreq = raw.info["sfreq"]
            n_channels = len(raw.ch_names)
            duration_min = (raw.n_times / sfreq) / 60.0

            _, _, counts = analyze_events(raw)

            norm_counts = {normalize_event_name(k): v for k, v in counts.items()}
            n_frequent = sum(v for k, v in norm_counts.items() if k.endswith("s5"))
            n_target = sum(v for k, v in norm_counts.items() if k.endswith("s6"))
            n_distractor = sum(v for k, v in norm_counts.items() if k.endswith("s7"))
            n_response = sum(v for k, v in norm_counts.items() if k.endswith("s1"))
            total_oddball = n_frequent + n_target + n_distractor

            summary_rows.append({
                "subject_id": sub_id,
                "sampling_rate": int(sfreq),
                "n_channels": n_channels,
                "duration_min": round(duration_min, 2),
                "n_frequent": n_frequent,
                "n_target": n_target,
                "n_distractor": n_distractor,
                "total_stimuli": total_oddball,
                "n_responses": n_response,
            })
        except Exception as e:
            print(f"[!] Lỗi khi đọc metadata {sub_id}: {e}")

    df_summary = pd.DataFrame(summary_rows)

    # Lưu ra file CSV
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "inter_subject_summary.csv"
    df_summary.to_csv(csv_path, index=False)
    print(f"  Đã xuất bảng tổng hợp liên đối tượng: {csv_path.name}")

    # In bảng tóm tắt
    print(f"\n BẢNG TỔNG HỢP LIÊN ĐỐI TƯỢNG (5 SUBJECTS ĐẦU TIÊN):")
    print(df_summary.head().to_string(index=False))

    # Thống kê tổng hợp toàn dataset
    print(f"\n THỐNG KÊ TỔNG QUAN ({len(df_summary)} SUBJECTS):")
    print(f" • Tần số lấy mẫu (Sampling Rate):    Đồng nhất {df_summary['sampling_rate'].unique()} Hz")
    print(f" • Số lượng kênh (Channels):          Đồng nhất {df_summary['n_channels'].unique()} kênh")
    print(f" • Thời lượng ghi trung bình:         {df_summary['duration_min'].mean():.2f} ± {df_summary['duration_min'].std():.2f} phút")
    print(f" • Tổng số Standard (Frequent S5):    {df_summary['n_frequent'].sum():,} trials (TB: {df_summary['n_frequent'].mean():.1f}/sub)")
    print(f" • Tổng số Target (P300 S6):          {df_summary['n_target'].sum():,} trials (TB: {df_summary['n_target'].mean():.1f}/sub)")
    print(f" • Tổng số Distractor (S7):           {df_summary['n_distractor'].sum():,} trials (TB: {df_summary['n_distractor'].mean():.1f}/sub)")

    # Vẽ biểu đồ so sánh số lượng trial và thời lượng giữa các subject
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    subs = df_summary["subject_id"]
    ax1.bar(subs, df_summary["duration_min"], color="#2b5c8f", alpha=0.85, edgecolor="black")
    ax1.set_ylabel("Thời lượng (phút)", fontsize=11)
    ax1.set_title("Thời lượng ghi điện não đồ giữa các Subject", fontsize=12, fontweight="bold")
    ax1.grid(axis="y", linestyle="--", alpha=0.5)

    ax2.bar(subs, df_summary["n_frequent"], label="Frequent (S5)", color="#1f77b4", alpha=0.8)
    ax2.bar(subs, df_summary["n_target"], bottom=df_summary["n_frequent"], label="Target (S6)", color="#d95f02", alpha=0.9)
    ax2.bar(subs, df_summary["n_distractor"], bottom=df_summary["n_frequent"] + df_summary["n_target"], label="Distractor (S7)", color="#7570b3", alpha=0.8)
    ax2.set_ylabel("Số lượng Stimuli", fontsize=11)
    ax2.set_xlabel("Mã Đối tượng (Subject ID)", fontsize=11)
    ax2.set_title("Phân bố số lượng kích thích Oddball theo từng Subject", fontsize=12, fontweight="bold")
    ax2.legend(loc="upper right")
    ax2.tick_params(axis="x", rotation=90)
    ax2.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    chart_path = output_dir / "inter_subject_comparison.png"
    plt.savefig(chart_path, dpi=200)
    plt.close()
    print(f" Đã lưu biểu đồ so sánh liên đối tượng: {chart_path.name}")

    if len(df_summary):
        estimate_memory(df_summary)

    return df_summary


# ==============================================================================
# HÀM ĐIỀU PHỐI CHÍNH TOÀN BỘ QUY TRÌNH KHÁM PHÁ (PIPELINE)
# ==============================================================================
def run_exploration_suite(
    raw_data_dir: Optional[str | Path] = None,
    output_figures_dir: Optional[str | Path] = None,
    num_detailed_subjects: int = 2,
) -> None:
    """Thực thi đầy đủ 6 bước khám phá dữ liệu chuẩn mực."""
    project_root = default_project_root()
    if raw_data_dir is None:
        raw_data_dir = project_root / "data" / "raw" / "oddball_data"
    else:
        raw_data_dir = Path(raw_data_dir)
        if not raw_data_dir.is_absolute():
            raw_data_dir = project_root / raw_data_dir

    if output_figures_dir is None:
        output_figures_dir = project_root / "results" / "figures" / "exploration"
    else:
        output_figures_dir = Path(output_figures_dir)
        if not output_figures_dir.is_absolute():
            output_figures_dir = project_root / output_figures_dir

    print(" BẮT ĐẦU QUY TRÌNH KHÁM PHÁ DỮ LIỆU EEG (ODDBALL - P300)")

    output_dir = Path(output_figures_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Quét danh sách file
    vhdr_files = scan_raw_files(raw_data_dir)
    print(f"\n01. QUÉT DỮ LIỆU: Tìm thấy {len(vhdr_files)} file .vhdr trong '{Path(raw_data_dir).resolve()}'.")

    # 2. Kiểm tra tính đồng nhất của hệ thống kênh trên toàn bộ 42 subjects
    check_channel_consistency(vhdr_files)
    check_montage_files(vhdr_files, output_dir)

    # 3. Phân tích chi tiết và trực quan hóa cho các subject đại diện
    for vhdr in vhdr_files[:num_detailed_subjects]:
        sub_id = extract_subject_id(vhdr)
        print(f"* ĐANG PHÂN TÍCH CHI TIẾT TÍN HIỆU ĐỐI TƯỢNG: {sub_id}")

        # Đọc dữ liệu (preload=True để tính toán thống kê và vẽ biểu đồ)
        raw = mne.io.read_raw_brainvision(vhdr, preload=True, verbose=False)

        # Phân tích sự kiện
        _, _, event_counts = analyze_events(raw)
        plot_event_distribution(event_counts, sub_id, output_dir)

        # Phân tích chất lượng tín hiệu
        analyze_signal_quality(raw, sub_id, output_dir)

        # Trực quan hóa sóng thô và phân tích PSD
        visualize_signals_and_psd(raw, sub_id, output_dir)

    # 4. Phân tích thống kê liên đối tượng trên toàn bộ dataset
    analyze_inter_subject_statistics(vhdr_files, output_dir)

    print(f"\n{'='*75}")
    print("HOÀN TẤT TOÀN BỘ QUY TRÌNH KHÁM PHÁ DỮ LIỆU!")
    print(f"Tất cả biểu đồ và bảng thống kê đã được lưu tại: '{output_dir.resolve()}'.")
    print(f"{'='*75}\n")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--n-detailed", type=int, default=2)
    args, _ = ap.parse_known_args()  # bỏ qua tham số '-f' của Jupyter/Colab

    run_exploration_suite(
        raw_data_dir=Path(args.raw_dir).resolve() if args.raw_dir else None,   # tương đối theo thư mục hiện tại
        output_figures_dir=Path(args.out_dir).resolve() if args.out_dir else None,
        num_detailed_subjects=args.n_detailed,
    )
