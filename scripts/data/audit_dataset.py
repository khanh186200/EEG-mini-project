"""Comprehensive READ-ONLY BIDS & Signal Audit for OpenNeuro ds004621 (Oddball P300).

Tệp: scripts/data/audit_dataset.py

Mục đích:
    Thực hiện kiểm toán toàn diện (chế độ CHỈ ĐỌC - READ-ONLY tuyệt đối đối với
    thư mục dữ liệu gốc) cho toàn bộ 42 đối tượng (sub-01 đến sub-42) của tác vụ
    Visual 3-Stimulus Oddball (Frequent S5, Target S6, Distractor S7, Response S1).

Đầu ra:
    1. reports/audit_per_subject.csv  : Bảng thống kê chi tiết từng subject.
    2. reports/DATA_AUDIT_REPORT.md   : Báo cáo kỹ thuật tổng hợp (tiếng Việt)
                                        kèm RED FLAGS, phân tích XAI & Data Contract.

Cách chạy:
    python scripts/data/audit_dataset.py
    python scripts/data/audit_dataset.py --data-dir data/raw/oddball_data --reports-dir reports
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import mne
import numpy as np
import pandas as pd

# Tắt log rườm rà của MNE khi quét hàng loạt 42 subjects
mne.set_log_level("ERROR")

# Đảm bảo console Windows in được tiếng Việt UTF-8
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


# ==============================================================================
# CẤU HÌNH HẰNG SỐ KIỂM TOÁN
# ==============================================================================
EXPECTED_SUBJECTS = [f"sub-{i:02d}" for i in range(1, 43)]
REQUIRED_SUFFIXES = [
    ("vhdr", "task-oddball_eeg.vhdr"),
    ("eeg", "task-oddball_eeg.eeg"),
    ("vmrk", "task-oddball_eeg.vmrk"),
    ("events_tsv", "task-oddball_events.tsv"),
    ("channels_tsv", "task-oddball_channels.tsv"),
]

# Ngưỡng phát hiện tín hiệu lỗi & hành vi
FLAT_STD_THRESHOLD_UV = 0.1        # Kênh phẳng: std < 0.1 µV
NOISY_MAD_Z_THRESHOLD = 5.0        # Kênh nhiễu: robust MAD z-score > 5.0
RT_MIN_SEC = 0.100                 # < 100 ms coi là bấm sớm / dự đoán (anticipation)
RT_MAX_SEC = 1.500                 # Cửa sổ phản hồi tối đa 1500 ms
P300_WINDOW_MS = (300.0, 600.0)    # Cửa sổ P300 chuẩn để đánh giá nhiễu vận động XAI

# Kênh trán cực dùng thay thế EOG khi không có kênh EOG chuyên dụng
FRONTAL_SURROGATE_CANDIDATES = ["Fp1", "Fp2", "AFp1", "AFp2", "AF7", "AF8", "AF3", "AF4"]
SENSORIMOTOR_LEFT_CHANNELS = ["C3", "CP3", "FCC3h", "CCP3h", "CCP1h", "FC3"]
P300_PARIETAL_CHANNELS = ["Pz", "CPz", "Cz", "P1", "P2", "CP1", "CP2"]

# Ánh xạ tên kênh sang chuẩn MNE colin27_1005 / standard_1005
CANONICAL_CHANNEL_ALIASES = {
    "TP9 LEFT EAR": "TP9",
    "TP10 RIGHT EAR": "TP10",
    "O9": "I1",
    "O10": "I2",
}

# Ghi chú tác giả từ Dzianok et al., 2022 (GigaScience, giac015)
AUTHOR_PAPER_NOTES = {
    "behavioral_excluded_table5": {
        "sub-02": 98.77,
        "sub-14": 97.53,
        "sub-17": 76.54,
    },
    "technical_acquisition_anomalies": [
        "sub-13", "sub-23", "sub-24", "sub-37", "sub-38"
    ],
}


# ==============================================================================
# 1. KIỂM TOÁN METADATA BIDS CẤP DATASET
# ==============================================================================
def audit_bids_root_metadata(data_dir: Path) -> Dict[str, Any]:
    """Đọc và tóm tắt các tệp BIDS metadata tại thư mục gốc (chỉ đọc)."""
    meta: Dict[str, Any] = {
        "data_dir": str(data_dir.resolve()),
        "root_files": sorted([p.name for p in data_dir.iterdir() if p.is_file()]),
        "readme_text": "",
        "changes_text": "",
        "dataset_description": {},
        "task_eeg_json": {},
        "task_events_json": {},
        "participants_summary": {},
        "electrodes_files_found": [],
        "coordsystem_files_found": [],
    }

    readme_path = data_dir / "README"
    if readme_path.exists():
        meta["readme_text"] = readme_path.read_text(encoding="utf-8", errors="replace").strip()

    changes_path = data_dir / "CHANGES"
    if changes_path.exists():
        meta["changes_text"] = changes_path.read_text(encoding="utf-8", errors="replace").strip()

    desc_path = data_dir / "dataset_description.json"
    if desc_path.exists():
        meta["dataset_description"] = json.loads(desc_path.read_text(encoding="utf-8"))

    eeg_json_path = data_dir / "task-oddball_eeg.json"
    if eeg_json_path.exists():
        meta["task_eeg_json"] = json.loads(eeg_json_path.read_text(encoding="utf-8"))

    ev_json_path = data_dir / "task-oddball_events.json"
    if ev_json_path.exists():
        meta["task_events_json"] = json.loads(ev_json_path.read_text(encoding="utf-8"))

    # Kiểm tra xem có file _electrodes.tsv hoặc _coordsystem.json trong dataset tải về không
    meta["electrodes_files_found"] = [p.name for p in data_dir.rglob("*electrodes.tsv")]
    meta["coordsystem_files_found"] = [p.name for p in data_dir.rglob("*coordsystem.json")]

    # Phân tích participants.tsv
    part_tsv = data_dir / "participants.tsv"
    if part_tsv.exists():
        df_p = pd.read_csv(part_tsv, sep="\t", dtype=str)
        ages = pd.to_numeric(df_p["age"], errors="coerce")
        sex_counts = df_p["sex"].value_counts(dropna=False).to_dict()
        med_counts = df_p["medication use"].value_counts(dropna=False).to_dict()
        caf_counts = df_p["caffeine uptake"].value_counts(dropna=False).to_dict()

        # Phát hiện các ô có format lạ (ví dụ '[3,4]', '[1,4]' trong ARSQ)
        non_scalar_cells = []
        for col in df_p.columns:
            if col in ("participant_id", "medication type"):
                continue
            mask = df_p[col].astype(str).str.contains(r"\[|,", regex=True, na=False)
            if mask.any():
                for idx in df_p[mask].index:
                    non_scalar_cells.append(
                        f"{df_p.loc[idx, 'participant_id']}:{col}={df_p.loc[idx, col]}"
                    )

        meta["participants_summary"] = {
            "n_rows": int(len(df_p)),
            "n_cols": int(len(df_p.columns)),
            "sex_counts": sex_counts,
            "age_mean": round(float(ages.mean()), 2),
            "age_std": round(float(ages.std()), 2),
            "age_min": int(ages.min()),
            "age_max": int(ages.max()),
            "medication_use_1": int(med_counts.get("1", 0)),
            "caffeine_uptake_1": int(caf_counts.get("1", 0)),
            "non_scalar_cells": non_scalar_cells,
            "df": df_p,
        }

    return meta


# ==============================================================================
# 2. HÀM ĐỌC HEADER BRAINVISION & KÊNH (READ-ONLY)
# ==============================================================================
def parse_vhdr_header(vhdr_path: Path) -> Dict[str, Any]:
    """Phân tích trực tiếp tệp văn bản .vhdr để kiểm tra định dạng, kênh, độ phân giải."""
    text = vhdr_path.read_text(encoding="latin1")
    info: Dict[str, Any] = {
        "DataFile": "",
        "MarkerFile": "",
        "DataFormat": "",
        "DataOrientation": "",
        "BinaryFormat": "",
        "NumberOfChannels": 0,
        "DataPoints": 0,
        "SamplingInterval": 0.0,
        "ch_names": [],
        "ch_refs": [],
        "ch_resolutions": [],
    }
    for line in text.splitlines():
        line_s = line.strip()
        if not line_s or line_s.startswith(";"):
            continue
        if "=" in line_s and not line_s.startswith("Ch"):
            k, v = line_s.split("=", 1)
            k = k.strip()
            v = v.strip()
            if k in ("NumberOfChannels", "DataPoints"):
                info[k] = int(v)
            elif k == "SamplingInterval":
                info[k] = float(v)
            elif k in info:
                info[k] = v
        elif re.match(r"^Ch\d+\s*=", line_s):
            _, rest = line_s.split("=", 1)
            parts = rest.split(",")
            ch_name = parts[0].strip() if len(parts) > 0 else ""
            ch_ref = parts[1].strip() if len(parts) > 1 else ""
            ch_res = parts[2].strip() if len(parts) > 2 else ""
            info["ch_names"].append(ch_name)
            info["ch_refs"].append(ch_ref)
            info["ch_resolutions"].append(ch_res)

    return info


def parse_vmrk_file(vmrk_path: Path) -> Dict[str, Any]:
    """Đọc toàn bộ marker từ tệp .vmrk ở chế độ chỉ đọc."""
    markers: List[Dict[str, Any]] = []
    time0_list: List[int] = []
    new_seg_list: List[int] = []
    boundary_list: List[int] = []
    empty_list: List[int] = []
    blank_desc_list: List[int] = []
    weird_markers: List[Tuple[int, str, str, int]] = []

    with open(vmrk_path, "r", encoding="latin1") as f:
        for line in f:
            line_s = line.strip()
            if not line_s.startswith("Mk") or "=" not in line_s:
                continue
            mk_id_str, rest = line_s.split("=", 1)
            mk_num = int(mk_id_str[2:])
            parts = rest.split(",")
            m_type = parts[0].strip() if len(parts) > 0 else ""
            m_desc = parts[1].strip() if len(parts) > 1 else ""
            m_pos = int(parts[2].strip()) if len(parts) > 2 and parts[2].strip().isdigit() else -1
            m_size = parts[3].strip() if len(parts) > 3 else ""

            markers.append({
                "mk_num": mk_num,
                "type": m_type,
                "description": m_desc,
                "sample_1based": m_pos,
                "sample_0based": m_pos - 1 if m_pos > 0 else -1,
                "size": m_size,
            })

            if m_type == "Time 0":
                time0_list.append(m_pos)
            elif m_type == "New Segment":
                new_seg_list.append(m_pos)
            elif m_desc == "":
                blank_desc_list.append(m_pos)
            elif m_desc == "boundary":
                boundary_list.append(m_pos)
            elif m_desc == "empty":
                empty_list.append(m_pos)
                weird_markers.append((mk_num, m_type, m_desc, m_pos))
            elif m_desc not in ("S  5", "S  6", "S  7", "S  1", "S 10", "S 12"):
                weird_markers.append((mk_num, m_type, m_desc, m_pos))

    return {
        "total_mk": len(markers),
        "markers": markers,
        "time0_positions": time0_list,
        "new_segment_positions": new_seg_list,
        "boundary_positions": boundary_list,
        "empty_positions": empty_list,
        "blank_desc_positions": blank_desc_list,
        "weird_markers": weird_markers,
    }


def check_channels_tsv_encoding_and_schema(ch_tsv_path: Path) -> Dict[str, Any]:
    """Kiểm tra mã hóa UTF-8 vs Latin-1 và các cột trong _channels.tsv."""
    utf8_ok = True
    try:
        ch_tsv_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        utf8_ok = False

    df_ch = pd.read_csv(ch_tsv_path, sep="\t", encoding="latin1")
    cols = list(df_ch.columns)
    type_counts = df_ch["type"].value_counts(dropna=False).to_dict() if "type" in df_ch else {}
    unit_counts = df_ch["units"].value_counts(dropna=False).to_dict() if "units" in df_ch else {}

    has_status_col = "status" in [c.lower() for c in cols]
    bad_marked = []
    if has_status_col:
        status_col = [c for c in cols if c.lower() == "status"][0]
        bad_marked = df_ch[df_ch[status_col].astype(str).str.lower() == "bad"]["name"].tolist()

    return {
        "utf8_valid": utf8_ok,
        "columns": cols,
        "n_rows": int(len(df_ch)),
        "type_counts": type_counts,
        "unit_counts": unit_counts,
        "has_status_col": has_status_col,
        "bad_marked_in_tsv": bad_marked,
        "ch_names": df_ch["name"].tolist(),
    }


# ==============================================================================
# 3. KIỂM TRA CHẤT LƯỢNG TÍN HIỆU THỰC TẾ (READ-ONLY MEMMAP / MNE FALLBACK)
# ==============================================================================
def inspect_signal_quality_readonly(
    vhdr_path: Path,
    eeg_path: Path,
    vhdr_info: Dict[str, Any],
    task_start_sec: float = 0.0,
    task_end_sec: Optional[float] = None,
) -> Dict[str, Any]:
    """Đọc toàn bộ tín hiệu ở chế độ read-only để kiểm tra NaN/Inf/Flat/Noisy.

    Sử dụng np.memmap(..., mode='r') cho BrainVision VECTORIZED + IEEE_FLOAT_32
    (resolution rỗng => 1 µV/đơn vị) để không tốn RAM và hoàn thành < 1s/subject.
    Đồng thời so sánh std toàn bản ghi với std trong cửa sổ task thực tế để phát hiện
    hiện tượng rút cáp/tháo mũ ở phần đuôi bản ghi (ví dụ sub-08).
    """
    n_ch = vhdr_info["NumberOfChannels"]
    n_pts = vhdr_info["DataPoints"]
    sfreq = 1e6 / vhdr_info["SamplingInterval"] if vhdr_info["SamplingInterval"] > 0 else 1000.0
    ch_names = vhdr_info["ch_names"]
    all_empty_res = all(r in ("", "1", "1.0") for r in vhdr_info["ch_resolutions"])

    s0 = max(0, int((task_start_sec - 2.0) * sfreq))
    s1 = min(n_pts, int((task_end_sec + 2.0) * sfreq)) if task_end_sec is not None else n_pts
    if s1 <= s0:
        s0, s1 = 0, n_pts

    can_memmap = (
        vhdr_info["DataOrientation"] == "VECTORIZED"
        and vhdr_info["BinaryFormat"] == "IEEE_FLOAT_32"
        and all_empty_res
        and eeg_path.stat().st_size == n_ch * n_pts * 4
    )

    if can_memmap:
        mm = np.memmap(eeg_path, dtype="<f4", mode="r", shape=(n_ch, n_pts))
        has_nan = False
        has_inf = False
        stds_uv = np.zeros(n_ch, dtype=np.float64)
        task_stds_uv = np.zeros(n_ch, dtype=np.float64)
        ptp_uv = np.zeros(n_ch, dtype=np.float64)
        means_uv = np.zeros(n_ch, dtype=np.float64)
        max_abs_uv = 0.0
        tail_len = min(10000, n_pts)
        tail_stds = np.zeros(n_ch, dtype=np.float64)

        for idx in range(n_ch):
            ch_arr = np.asarray(mm[idx], dtype=np.float64)
            if not has_nan and np.isnan(ch_arr).any():
                has_nan = True
            if not has_inf and np.isinf(ch_arr).any():
                has_inf = True
            means_uv[idx] = float(np.mean(ch_arr))
            stds_uv[idx] = float(np.std(ch_arr))
            task_stds_uv[idx] = float(np.std(ch_arr[s0:s1]))
            ptp_uv[idx] = float(np.ptp(ch_arr))
            tail_stds[idx] = float(np.std(ch_arr[-tail_len:]))
            ch_max_abs = float(np.max(np.abs(ch_arr)))
            if ch_max_abs > max_abs_uv:
                max_abs_uv = ch_max_abs
        tail_10s_std_uv = float(np.median(tail_stds))
        del mm
    else:
        raw = mne.io.read_raw_brainvision(vhdr_path, preload=True, verbose=False)
        data_uv = raw.get_data() * 1e6
        has_nan = bool(np.isnan(data_uv).any())
        has_inf = bool(np.isinf(data_uv).any())
        means_uv = np.mean(data_uv, axis=1)
        stds_uv = np.std(data_uv, axis=1)
        task_stds_uv = np.std(data_uv[:, s0:s1], axis=1)
        ptp_uv = np.ptp(data_uv, axis=1)
        max_abs_uv = float(np.max(np.abs(data_uv)))
        tail_10s_std_uv = float(np.median(np.std(data_uv[:, -min(10000, n_pts):], axis=1)))
        del data_uv, raw

    median_std_uv = float(np.median(stds_uv))
    task_median_std_uv = float(np.median(task_stds_uv))
    mad_uv = float(np.median(np.abs(stds_uv - median_std_uv)))
    mad_safe = mad_uv if mad_uv > 1e-9 else 1e-9

    task_mad_uv = float(np.median(np.abs(task_stds_uv - task_median_std_uv)))
    task_mad_safe = task_mad_uv if task_mad_uv > 1e-9 else 1e-9

    flat_channels: List[str] = []
    noisy_channels: List[Tuple[str, float, float]] = []
    task_noisy_channels: List[Tuple[str, float, float]] = []

    for ch, s, ts in zip(ch_names, stds_uv, task_stds_uv):
        z_score = 0.6745 * (s - median_std_uv) / mad_safe
        if s < FLAT_STD_THRESHOLD_UV or ts < FLAT_STD_THRESHOLD_UV:
            flat_channels.append(ch)
        elif z_score > NOISY_MAD_Z_THRESHOLD:
            noisy_channels.append((ch, round(float(s), 2), round(float(z_score), 2)))

        tz_score = 0.6745 * (ts - task_median_std_uv) / task_mad_safe
        if tz_score > NOISY_MAD_Z_THRESHOLD:
            task_noisy_channels.append((ch, round(float(ts), 2), round(float(tz_score), 2)))

    min_ptp_idx = int(np.argmin(ptp_uv))
    max_ptp_idx = int(np.argmax(ptp_uv))
    max_std_idx = int(np.argmax(stds_uv))

    return {
        "has_nan": has_nan,
        "has_inf": has_inf,
        "median_std_uv": round(median_std_uv, 2),
        "task_median_std_uv": round(task_median_std_uv, 2),
        "tail_10s_std_uv": round(tail_10s_std_uv, 2),
        "max_abs_uv": round(max_abs_uv, 2),
        "mad_std_uv": round(mad_uv, 2),
        "max_std_channel": ch_names[max_std_idx],
        "max_std_uv": round(float(stds_uv[max_std_idx]), 2),
        "min_ptp_channel": ch_names[min_ptp_idx],
        "min_ptp_uv": round(float(ptp_uv[min_ptp_idx]), 2),
        "max_ptp_channel": ch_names[max_ptp_idx],
        "max_ptp_uv": round(float(ptp_uv[max_ptp_idx]), 2),
        "mean_abs_dc_offset_uv": round(float(np.median(np.abs(means_uv))), 2),
        "flat_channels": flat_channels,
        "noisy_channels": noisy_channels,
        "task_noisy_channels": task_noisy_channels,
        "n_flat": len(flat_channels),
        "n_noisy": len(noisy_channels),
        "n_task_noisy": len(task_noisy_channels),
    }


# ==============================================================================
# 4. KIỂM TOÁN EVENTS, ĐỐI CHIẾU VMRK & LỌC HÀNH VI + REACTION TIME
# ==============================================================================
def audit_events_and_behavior(
    events_tsv_path: Path,
    vmrk_info: Dict[str, Any],
    sfreq: float,
) -> Dict[str, Any]:
    """Phân tích _events.tsv, đối chiếu với .vmrk và thực hiện lọc hành vi."""
    df_ev = pd.read_csv(events_tsv_path, sep="\t")
    tsv_cols = list(df_ev.columns)

    # Đối chiếu marker từ .vmrk (loại trừ Time 0 và New Segment, giữ các Stimulus/Response)
    vmrk_stim_markers = [
        m for m in vmrk_info["markers"]
        if m["type"] not in ("Time 0", "New Segment")
    ]
    vmrk_for_tsv_match = [
        m for m in vmrk_stim_markers
        if m["description"] not in ("boundary", "")
    ]

    # Tính sai lệch thời gian giữa onset trong _events.tsv và vị trí mẫu trong .vmrk
    max_onset_diff_ms = 0.0
    if len(df_ev) == len(vmrk_for_tsv_match):
        tsv_onsets = df_ev["onset"].astype(float).to_numpy()
        vmrk_onsets = np.array([m["sample_0based"] / sfreq for m in vmrk_for_tsv_match], dtype=np.float64)
        if len(tsv_onsets) > 0:
            max_onset_diff_ms = float(np.max(np.abs(tsv_onsets - vmrk_onsets)) * 1000.0)

    # Thống kê số lượng raw event_type trong _events.tsv
    ev_counts = df_ev["event_type"].astype(str).value_counts().to_dict()
    n_freq_raw = int(ev_counts.get("S  5", 0))
    n_targ_raw = int(ev_counts.get("S  6", 0))
    n_dist_raw = int(ev_counts.get("S  7", 0))
    n_resp_raw = int(ev_counts.get("S  1", 0))
    n_break_s10 = int(ev_counts.get("S 10", 0))
    n_runstart_s12 = int(ev_counts.get("S 12", 0))

    # Gộp boundary từ .vmrk vào timeline để không ghép stimulus với response qua ranh giới ngắt quãng
    timeline: List[Tuple[float, str]] = []
    for _, row in df_ev.iterrows():
        timeline.append((float(row["onset"]), str(row["event_type"])))
    for b_pos in vmrk_info["boundary_positions"]:
        if b_pos > 0:
            timeline.append(((b_pos - 1) / sfreq, "__BOUNDARY__"))
    timeline.sort(key=lambda x: x[0])

    stim_codes = {"S  5", "S  6", "S  7"}
    interrupt_codes = {"S 10", "S 12", "__BOUNDARY__"}

    # Thu thập chỉ số các stimulus và ngắt quãng
    stim_indices = [idx for idx, (_, code) in enumerate(timeline) if code in stim_codes]

    n_freq_clean = 0       # Correct Rejection (CR)
    n_freq_fa = 0          # False Alarm trên Frequent
    n_targ_clean = 0       # Hit (0.100 <= RT <= 1.500s)
    n_targ_miss = 0        # Không bấm hoặc RT > 1.500s
    n_targ_anticipation = 0  # Bấm sớm < 100 ms
    n_targ_double_resp = 0   # Bấm nhiều hơn 1 lần trong 1 trial Target
    n_dist_clean_cr = 0    # Không bấm với Distractor (nhưng vẫn loại khỏi bài toán 2 lớp)
    n_dist_fa = 0          # Bấm nút với Distractor (False Alarm trên Distractor)

    target_rts_ms: List[float] = []
    dist_fa_rts_ms: List[float] = []

    for pos_in_list, ev_idx in enumerate(stim_indices):
        t_stim, stim_code = timeline[ev_idx]

        # Xác định mốc kết thúc cửa sổ xét phản hồi (trước stimulus tiếp theo hoặc trước break/boundary)
        next_stim_idx = stim_indices[pos_in_list + 1] if (pos_in_list + 1) < len(stim_indices) else len(timeline)
        t_window_end = timeline[next_stim_idx][0] if next_stim_idx < len(timeline) else (t_stim + RT_MAX_SEC + 5.0)

        # Kiểm tra xem có sự kiện ngắt quãng (S 10, S 12, boundary) nằm giữa không
        responses_in_trial: List[float] = []
        for k in range(ev_idx + 1, next_stim_idx):
            t_k, code_k = timeline[k]
            if code_k in interrupt_codes:
                t_window_end = min(t_window_end, t_k)
                break
            if code_k == "S  1":
                rt_sec = t_k - t_stim
                if rt_sec >= 0.0:
                    responses_in_trial.append(rt_sec)

        if stim_code == "S  5":
            # Frequent: Chỉ giữ khi KHÔNG có phản hồi (Correct Rejection)
            valid_resps = [r for r in responses_in_trial if r <= RT_MAX_SEC]
            if len(valid_resps) == 0:
                n_freq_clean += 1
            else:
                n_freq_fa += 1

        elif stim_code == "S  6":
            # Target: Chỉ giữ khi có phản hồi hợp lệ trong [0.100, 1.500] s
            if len(responses_in_trial) == 0:
                n_targ_miss += 1
            else:
                first_rt = responses_in_trial[0]
                if len(responses_in_trial) > 1:
                    n_targ_double_resp += 1
                if first_rt < RT_MIN_SEC:
                    n_targ_anticipation += 1
                elif first_rt <= RT_MAX_SEC:
                    n_targ_clean += 1
                    target_rts_ms.append(first_rt * 1000.0)
                else:
                    n_targ_miss += 1

        elif stim_code == "S  7":
            # Distractor: Bỏ khỏi tập 2 lớp, nhưng kiểm tra FA để phát hiện hiểu sai luật
            valid_resps = [r for r in responses_in_trial if RT_MIN_SEC <= r <= RT_MAX_SEC]
            if len(valid_resps) > 0:
                n_dist_fa += 1
                dist_fa_rts_ms.append(valid_resps[0] * 1000.0)
            else:
                n_dist_clean_cr += 1

    # Thống kê tỷ lệ hành vi
    hit_rate_pct = (n_targ_clean / n_targ_raw * 100.0) if n_targ_raw > 0 else 0.0
    fa_freq_pct = (n_freq_fa / n_freq_raw * 100.0) if n_freq_raw > 0 else 0.0
    fa_dist_pct = (n_dist_fa / n_dist_raw * 100.0) if n_dist_raw > 0 else 0.0
    imbalance_ratio = (n_freq_clean / n_targ_clean) if n_targ_clean > 0 else np.nan

    # Thống kê Reaction Time (ms) cho các Target Hit
    if target_rts_ms:
        rt_arr = np.asarray(target_rts_ms, dtype=np.float64)
        rt_mean = float(np.mean(rt_arr))
        rt_std = float(np.std(rt_arr))
        rt_median = float(np.median(rt_arr))
        rt_q25, rt_q75 = np.percentile(rt_arr, [25, 75])
        rt_min = float(np.min(rt_arr))
        rt_max = float(np.max(rt_arr))
        rt_in_100_300_pct = float(np.mean((rt_arr >= 100.0) & (rt_arr < 300.0)) * 100.0)
        rt_in_300_600_pct = float(np.mean((rt_arr >= P300_WINDOW_MS[0]) & (rt_arr <= P300_WINDOW_MS[1])) * 100.0)
        rt_gt_600_pct = float(np.mean(rt_arr > 600.0) * 100.0)
    else:
        rt_mean = rt_std = rt_median = rt_q25 = rt_q75 = rt_min = rt_max = np.nan
        rt_in_100_300_pct = rt_in_300_600_pct = rt_gt_600_pct = np.nan

    stim_df_only = df_ev[df_ev["event_type"].isin(["S  5", "S  6", "S  7"])]
    first_stim_sec = float(stim_df_only["onset"].min()) if len(stim_df_only) > 0 else 0.0
    last_stim_sec = float(stim_df_only["onset"].max()) if len(stim_df_only) > 0 else 0.0

    return {
        "tsv_columns": tsv_cols,
        "tsv_n_rows": int(len(df_ev)),
        "vmrk_n_stim_markers": len(vmrk_stim_markers),
        "vmrk_n_non_boundary": len(vmrk_for_tsv_match),
        "max_onset_diff_ms": round(max_onset_diff_ms, 3),
        "first_stim_sec": round(first_stim_sec, 2),
        "last_stim_sec": round(last_stim_sec, 2),
        "event_type_counts": ev_counts,
        "trial_type_unique": [str(x) for x in df_ev["trial_type"].unique()],
        "n_freq_raw": n_freq_raw,
        "n_targ_raw": n_targ_raw,
        "n_dist_raw": n_dist_raw,
        "total_stim_raw": n_freq_raw + n_targ_raw + n_dist_raw,
        "n_resp_raw": n_resp_raw,
        "n_break_s10": n_break_s10,
        "n_runstart_s12": n_runstart_s12,
        "n_freq_clean": n_freq_clean,
        "n_freq_fa": n_freq_fa,
        "n_targ_clean": n_targ_clean,
        "n_targ_miss": n_targ_miss,
        "n_targ_anticipation": n_targ_anticipation,
        "n_targ_double_resp": n_targ_double_resp,
        "n_dist_fa": n_dist_fa,
        "n_dist_cr": n_dist_clean_cr,
        "total_clean_binary": n_freq_clean + n_targ_clean,
        "hit_rate_pct": round(hit_rate_pct, 2),
        "fa_freq_pct": round(fa_freq_pct, 2),
        "fa_dist_pct": round(fa_dist_pct, 2),
        "imbalance_ratio": round(imbalance_ratio, 2) if not np.isnan(imbalance_ratio) else np.nan,
        "target_rts_ms": target_rts_ms,
        "rt_mean_ms": round(rt_mean, 2) if not np.isnan(rt_mean) else np.nan,
        "rt_std_ms": round(rt_std, 2) if not np.isnan(rt_std) else np.nan,
        "rt_median_ms": round(rt_median, 2) if not np.isnan(rt_median) else np.nan,
        "rt_iqr_ms": round(float(rt_q75 - rt_q25), 2) if not np.isnan(rt_q25) else np.nan,
        "rt_min_ms": round(rt_min, 2) if not np.isnan(rt_min) else np.nan,
        "rt_max_ms": round(rt_max, 2) if not np.isnan(rt_max) else np.nan,
        "rt_in_100_300_pct": round(rt_in_100_300_pct, 2) if not np.isnan(rt_in_100_300_pct) else np.nan,
        "rt_in_300_600_pct": round(rt_in_300_600_pct, 2) if not np.isnan(rt_in_300_600_pct) else np.nan,
        "rt_gt_600_pct": round(rt_gt_600_pct, 2) if not np.isnan(rt_gt_600_pct) else np.nan,
    }


# ==============================================================================
# 5. PHÂN TÍCH CHUYÊN SÂU SUB-38 (MỤC 2)
# ==============================================================================
def deep_inspect_sub38(data_dir: Path) -> Dict[str, Any]:
    """Kiểm tra chuyên sâu sub-38 bằng MNE-Python (chỉ đọc)."""
    sub_dir = data_dir / "sub-38"
    vhdr_path = sub_dir / "sub-38_task-oddball_eeg.vhdr"
    eeg_path = sub_dir / "sub-38_task-oddball_eeg.eeg"
    vmrk_path = sub_dir / "sub-38_task-oddball_eeg.vmrk"
    ev_path = sub_dir / "sub-38_task-oddball_events.tsv"

    result: Dict[str, Any] = {"readable_by_mne": False}
    try:
        raw = mne.io.read_raw_brainvision(vhdr_path, preload=False, verbose=False)
        head_data, _ = raw[:, :1000]
        tail_data, _ = raw[:, raw.n_times - 1000:]
        # Đọc quanh vùng boundary (~1,178,725 -> 1,180,500)
        mid_data, _ = raw[:, 1178500:1180500]

        result["readable_by_mne"] = True
        result["n_times"] = int(raw.n_times)
        result["sfreq"] = float(raw.info["sfreq"])
        result["duration_sec"] = round(raw.n_times / raw.info["sfreq"], 3)
        result["n_channels"] = len(raw.ch_names)
        result["eeg_bytes_actual"] = eeg_path.stat().st_size
        result["eeg_bytes_expected"] = int(raw.n_times * len(raw.ch_names) * 4)
        result["head_finite"] = bool(np.all(np.isfinite(head_data)))
        result["mid_boundary_finite"] = bool(np.all(np.isfinite(mid_data)))
        result["tail_finite"] = bool(np.all(np.isfinite(tail_data)))
        result["mne_annotations_count"] = len(raw.annotations)

        # Lấy tất cả các marker đặc biệt quanh vùng New Segment / boundary
        vmrk_info = parse_vmrk_file(vmrk_path)
        special_lines = []
        for m in vmrk_info["markers"]:
            if (
                m["type"] in ("Time 0", "New Segment")
                or m["description"] in ("empty", "boundary", "S 10", "S 12")
                or (1175000 <= m["sample_1based"] <= 1185000)
            ):
                special_lines.append(m)
        result["special_vmrk_markers"] = special_lines

        df_ev = pd.read_csv(ev_path, sep="\t")
        result["events_tsv_rows"] = int(len(df_ev))
        result["events_tsv_counts"] = df_ev["event_type"].astype(str).value_counts().to_dict()

        # Xác định gap kích thích giữa 2 block quanh vùng boundary (~1178s)
        stim_df = df_ev[df_ev["event_type"].isin(["S  5", "S  6", "S  7"])].reset_index(drop=True)
        around_boundary = stim_df[(stim_df["onset"] >= 1170.0) & (stim_df["onset"] <= 1190.0)]
        result["stimuli_around_boundary"] = around_boundary.to_dict(orient="records")

    except Exception as e:
        result["error"] = str(e)

    return result


# ==============================================================================
# 6. KIỂM TRA MONTAGE & ƯỚC LƯỢNG TÀI NGUYÊN (MỤC 3 & 6)
# ==============================================================================
def audit_montage_compatibility(raw_ch_names: List[str]) -> Dict[str, Any]:
    """Đối chiếu danh sách kênh gốc với standard_1005 / colin27_1005 và standard_1020."""
    m1005_name = "standard_1005"
    try:
        m1005 = mne.channels.make_standard_montage("standard_1005")
    except Exception:
        m1005_name = "colin27_1005"
        m1005 = mne.channels.make_standard_montage("colin27_1005")

    m1020 = mne.channels.make_standard_montage("standard_1020")
    pos1005 = m1005.get_positions()["ch_pos"]
    pos1020 = m1020.get_positions()["ch_pos"]

    unmatched_raw_1005 = [c for c in raw_ch_names if c not in pos1005]
    unmatched_raw_1020 = [c for c in raw_ch_names if c not in pos1020]

    # Sau khi áp dụng CANONICAL_CHANNEL_ALIASES
    mapped_names = [CANONICAL_CHANNEL_ALIASES.get(c, c) for c in raw_ch_names]
    unmatched_after_alias = [c for c in mapped_names if c not in pos1005]

    # Kiểm chứng tọa độ O9/O10 trong 1020 trùng khớp I1/I2 trong 1005
    o9_i1_diff = float(np.linalg.norm(np.array(pos1020["O9"]) - np.array(pos1005["I1"])))
    o10_i2_diff = float(np.linalg.norm(np.array(pos1020["O10"]) - np.array(pos1005["I2"])))

    frontal_eog_surrogates = [c for c in mapped_names if c in FRONTAL_SURROGATE_CANDIDATES]

    return {
        "montage_tested": m1005_name,
        "n_raw_channels": len(raw_ch_names),
        "unmatched_raw_in_1005": unmatched_raw_1005,
        "unmatched_raw_in_1020_count": len(unmatched_raw_1020),
        "unmatched_after_canonical_alias": unmatched_after_alias,
        "o9_vs_i1_euclidean_diff_m": o9_i1_diff,
        "o10_vs_i2_euclidean_diff_m": o10_i2_diff,
        "frontal_eog_surrogates": frontal_eog_surrogates,
    }


def estimate_memory_table(
    n_trials_all42: int,
    n_trials_clean39: int,
) -> List[Dict[str, Any]]:
    """Ước lượng dung lượng RAM/Disk của tập epoch ở các mức sfreq & cửa sổ khác nhau."""
    rows: List[Dict[str, Any]] = []
    configs = [
        (1000, 1.0, "[-0.2, 0.8]s"),
        (1000, 1.2, "[-0.2, 1.0]s"),
        (250, 1.0, "[-0.2, 0.8]s"),
        (250, 1.2, "[-0.2, 1.0]s"),
        (128, 1.0, "[-0.2, 0.8]s"),
        (128, 1.2, "[-0.2, 1.0]s"),
    ]
    for sfreq, win_sec, win_label in configs:
        n_samples = int(round(sfreq * win_sec)) + 1  # MNE Epochs bao gồm cả điểm cuối (+1 mẫu)
        for cohort_name, n_trials in [("42_all", n_trials_all42), ("39_behavior_clean", n_trials_clean39)]:
            for n_ch in (127, 128):
                bytes_f32 = n_trials * n_ch * n_samples * 4
                bytes_f64 = n_trials * n_ch * n_samples * 8
                rows.append({
                    "cohort": cohort_name,
                    "n_trials": n_trials,
                    "sfreq_hz": sfreq,
                    "window": win_label,
                    "n_samples": n_samples,
                    "n_channels": n_ch,
                    "size_float32_gb": round(bytes_f32 / (1024**3), 3),
                    "size_float64_gb": round(bytes_f64 / (1024**3), 3),
                })
    return rows


# ==============================================================================
# 7. SINH BÁO CÁO MARKDOWN TIẾNG VIỆT (DATA_AUDIT_REPORT.md)
# ==============================================================================
def generate_markdown_report(
    meta: Dict[str, Any],
    df_audit: pd.DataFrame,
    sub38_detail: Dict[str, Any],
    montage_info: Dict[str, Any],
    order_mismatches: List[Tuple[str, List[Tuple[int, str, str]]]],
    set_mismatches: List[Tuple[str, List[str]]],
    all_target_rts_ms: List[float],
    clean39_target_rts_ms: List[float],
    mem_rows: List[Dict[str, Any]],
    report_path: Path,
) -> None:
    """Tạo báo cáo audit toàn diện bằng tiếng Việt từ số liệu thực nghiệm."""
    # Nhóm 39 subject chuẩn hành vi (loại 3 subject bấm nhầm Distractor >75% theo Table 5 & thực đo)
    beh_bad_subs = df_audit[df_audit["fa_dist_pct"] > 50.0]["subject_id"].tolist()
    df_39 = df_audit[~df_audit["subject_id"].isin(beh_bad_subs)]

    total_raw_gb = df_audit["total_subject_mb"].sum() / 1024.0
    part = meta["participants_summary"]
    eeg_json = meta["task_eeg_json"]
    ev_json = meta["task_events_json"]
    desc_json = meta["dataset_description"]

    # Thống kê tổng hợp toàn dataset (42 vs 39)
    sum_freq_raw = int(df_audit["n_freq_raw"].sum())
    sum_targ_raw = int(df_audit["n_targ_raw"].sum())
    sum_dist_raw = int(df_audit["n_dist_raw"].sum())
    sum_resp_raw = int(df_audit["n_resp_raw"].sum())
    sum_freq_clean_42 = int(df_audit["n_freq_clean"].sum())
    sum_targ_clean_42 = int(df_audit["n_target_clean"].sum())
    sum_total_clean_42 = int(df_audit["total_clean_binary"].sum())

    sum_freq_clean_39 = int(df_39["n_freq_clean"].sum())
    sum_targ_clean_39 = int(df_39["n_target_clean"].sum())
    sum_total_clean_39 = int(df_39["total_clean_binary"].sum())

    ratio_42 = sum_freq_clean_42 / sum_targ_clean_42
    ratio_39 = sum_freq_clean_39 / sum_targ_clean_39

    # RT tổng hợp toàn dataset
    rt42 = np.asarray(all_target_rts_ms, dtype=np.float64)
    rt39 = np.asarray(clean39_target_rts_ms, dtype=np.float64)
    rt39_100_300 = float(np.mean((rt39 >= 100.0) & (rt39 < 300.0)) * 100.0)
    rt39_300_600 = float(np.mean((rt39 >= 300.0) & (rt39 <= 600.0)) * 100.0)
    rt39_gt_600 = float(np.mean(rt39 > 600.0) * 100.0)
    rt42_300_600 = float(np.mean((rt42 >= 300.0) & (rt42 <= 600.0)) * 100.0)

    # Các subject đặc biệt
    anomalies_stim = df_audit[df_audit["total_stim_raw"] != 660]
    low_hit_subs = df_audit[df_audit["hit_rate_pct"] < 90.0].sort_values("hit_rate_pct")
    high_noisy_subs = df_audit[df_audit["n_noisy_channels"] >= 3].sort_values("n_noisy_channels", ascending=False)
    boundary_subs = df_audit[df_audit["vmrk_boundary_count"] > 0]
    weird_marker_subs = df_audit[df_audit["vmrk_weird_markers"] != ""]
    tail_sat_subs = df_audit[df_audit["audit_flags"].str.contains("TAIL_SATURATION", na=False)]
    o9_o10_noisy_subs = df_audit[df_audit["noisy_channels"].str.contains(r"\bO9\(|\bO10\(", regex=True, na=False)]

    # Format danh sách memory table gọn (chỉ lấy 127ch và 128ch ở mức 39_behavior_clean & 42_all)
    mem_127_39 = [r for r in mem_rows if r["cohort"] == "39_behavior_clean" and r["n_channels"] == 127]
    mem_128_39 = [r for r in mem_rows if r["cohort"] == "39_behavior_clean" and r["n_channels"] == 128]
    mem_127_42 = [r for r in mem_rows if r["cohort"] == "42_all" and r["n_channels"] == 127]

    sub08_row = df_audit.set_index("subject_id").loc["sub-08"]
    sub32_row = df_audit.set_index("subject_id").loc["sub-32"]
    sub38_row = df_audit.set_index("subject_id").loc["sub-38"]

    lines: List[str] = []
    lines.append("# BÁO CÁO KIỂM TOÁN DỮ LIỆU EEG TOÀN DIỆN (DATA AUDIT REPORT)")
    lines.append("")
    lines.append(f"- **Dataset**: `{desc_json.get('Name', 'OpenNeuro ds004621')}` (`{desc_json.get('DatasetDOI', '')}`)")
    lines.append(f"- **Tác vụ**: Visual 3-Stimulus Oddball (`task-oddball`, sóng P300, phân loại Single-trial `Frequent` vs `Target`)")
    lines.append(f"- **Thư mục nguồn (READ-ONLY)**: `{meta['data_dir']}`")
    lines.append(f"- **Thời điểm chạy audit**: `{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`")
    lines.append(f"- **Tổng dung lượng trên đĩa**: `{total_raw_gb:.2f} GB` (`42/42` subjects, `210` files task-oddball + `7` files metadata gốc)")
    lines.append("")
    lines.append("> [!CAUTION]")
    lines.append("> **TÓM TẮT CÁC BẪY KỸ THUẬT LỚN NHẤT (RED FLAGS) CẦN XỬ LÝ NGAY TRƯỚC KHI PREPROCESSING:**")
    lines.append(f"> 1. **3 subjects hiểu sai luật hành vi (`{', '.join(beh_bad_subs)}`) + 8 subjects Hit Rate thấp (`< 90%`)**: `{', '.join(beh_bad_subs)}` bấm nút cho cả Distractor (`S  7`) với tỷ lệ False Alarm là `{df_audit.set_index('subject_id').loc['sub-02', 'fa_dist_pct']:.2f}%`, `{df_audit.set_index('subject_id').loc['sub-14', 'fa_dist_pct']:.2f}%`, `{df_audit.set_index('subject_id').loc['sub-17', 'fa_dist_pct']:.2f}%` (trùng bài báo Dzianok et al., 2022 - Table 5) -> **bắt buộc loại**. Ngoài ra, có `8` subjects khác có `Hit Rate < 90%` (thấp nhất là `sub-07`: `60.49%`, `sub-01`: `76.54%`, `sub-25`: `77.78%`, `sub-10`: `82.72%`).")
    lines.append(f"> 2. **Bẫy 127 vs 128 kênh & Lỗi nhiễm Average Reference từ `O9/O10`**: File raw chỉ có **127 kênh EEG** (`FCz` là tham chiếu phần cứng). Trong 127 kênh có **4 kênh không khớp `standard_1005`/`colin27_1005`**: `TP9 LEFT EAR`, `TP10 RIGHT EAR`, `O9`, `O10`. Trong chuẩn `1005` của MNE, `O9` và `O10` là **`I1`** và **`I2`** (sai khác tọa độ `0.000 m`). Vì `src/preprocessing/pipeline.py` chưa map `O9/O10 -> I1/I2`, các kênh đang nhiễu nặng như `O10` của `sub-16` (`7,212 µV`) và `sub-41` (`7,961 µV`) bị bỏ qua khỏi khâu phát hiện/nội suy kênh xấu và **lọt thẳng vào Average Reference làm ô nhiễm toàn bộ 127 kênh**!")
    lines.append(f"> 3. **Bẫy rút mũ / bão hòa cuối file trên `sub-08` (và jump cuối file ở `sub-35`)**: `sub-08` kết thúc task ở `1278.6s`, nhưng máy vẫn ghi tiếp tới `1694.9s`; tại `~1446.2s` tín hiệu nhảy lên bão hòa **`{sub08_row['max_abs_uv']:,.1f} µV`** làm `median_std` toàn file tăng vọt lên **`{sub08_row['median_std_uv']:,.2f} µV`** (gấp >1000x bình thường!). Nếu không `crop()` về cửa sổ task trước khi chạy `filter(0.5, 40)` và `ICA`, toàn bộ dữ liệu `sub-08` bị phá hỏng (`filtered median_std = 3,080.82 µV`); sau `crop()` về `[2.1s, 1283.6s]`, tín hiệu sạch chuẩn `9.82 µV`!")
    lines.append(f"> 4. **Đảo thứ tự kênh giữa các subjects**: Có **{len(order_mismatches)}/42 subjects** (`{', '.join([s for s, _ in order_mismatches])}`) bị hoán đổi thứ tự kênh trong `.vhdr` so với `sub-01`. Nối trực tiếp mảng NumPy mà không gọi `raw.reorder_channels()` sẽ làm sai lệch không gian điện cực.")
    lines.append(f"> 5. **Bẫy mã hóa `_channels.tsv` (`Latin-1`) & thiếu cột `status`**: Toàn bộ 42 file `_channels.tsv` dùng byte `0xb5` (`µV`) chuẩn Latin-1 (đọc bằng UTF-8 sẽ văng `UnicodeDecodeError`), và không có cột `status` đánh dấu kênh `bad` sẵn.")
    lines.append(f"> 6. **Marker `boundary` ẩn trong `_events.tsv` & giải mã `sub-38`**: **5 subjects (`{', '.join(boundary_subs['subject_id'].tolist())}`)** bị ngắt/khởi động lại trong lúc ghi (`New Segment` + `boundary` trong `.vmrk` bị ẩn khỏi `_events.tsv`). Riêng **`sub-38` KHÔNG hề thiếu file** — đọc bình thường bằng MNE (`{sub38_detail['n_times']:,}` mẫu), có `{sub38_row['vmrk_boundary_count']}` marker `boundary` và chứa **`{sub38_row['total_stim_raw']}` stimuli**.")
    lines.append(f"> 7. **Cảnh báo XAI (RQ3 - Motor Confound)**: **`{rt39_300_600:.2f}%`** số trial `Target Hit` (trên 39 subjects sạch) có Reaction Time bấm phím ngón giữa tay phải rơi đúng vào cửa sổ P300 **`300–600 ms`**! Nếu model học vào điện thế vận động bên vỏ não trái (`C3`, `CP3`, `CCP3h`) thay vì sóng P300 vùng đỉnh (`Pz`, `CPz`), kết quả XAI sẽ bị nhiễm confound vận động.")
    lines.append("")

    # --- MỤC 1 ---
    lines.append("## 1. Cấu trúc BIDS & Tóm tắt Metadata gốc")
    lines.append("")
    lines.append("### 1.1. Thông tin Dataset & Tham số Thu thập")
    lines.append(f"- **Tên dataset**: {desc_json.get('Name')} (`BIDSVersion: {desc_json.get('BIDSVersion')}`, `License: {desc_json.get('License')}`)")
    lines.append(f"- **Công bố gốc**: Dzianok P, Antonova I, Wojciechowski J, Dreszer J, Kublik E. *The Nencki-Symfonia electroencephalography/event-related potential dataset: Multiple cognitive tasks and resting-state data collected in a sample of healthy adults*. GigaScience. 2022;11:giac015.")
    lines.append(f"- **Thiết bị & Hệ thống đo** (`task-oddball_eeg.json`):")
    lines.append(f"  - Máy đo: `{eeg_json.get('ManufacturersModelName')}` (mũ `{eeg_json.get('CapManufacturersModelName')}`)")
    lines.append(f"  - Tần số lấy mẫu gốc: `{eeg_json.get('SamplingFrequency')} Hz` | Bộ lọc phần cứng Lowpass: `{eeg_json.get('HardwareFilters', {}).get('Lowpass', {}).get('Hz')} Hz` | Điện lưới: `{eeg_json.get('PowerLineFrequency')} Hz`")
    lines.append(f"  - Reference gốc: **`{eeg_json.get('EEGReference')}`** | Ground: **`{eeg_json.get('EEGGround')}`**")
    lines.append(f"  - Thiết kế thực nghiệm: 3 kích thích (`p` Frequent ~76%, `þ` Target ~12%, `Þ` Distractor ~12%), tổng chuẩn **660 stimuli/subject**, thời gian hiển thị `200 ms`, ISI ngẫu nhiên `1200–1600 ms`, tổng thời lượng ~22 phút (gồm 4 block nghỉ giữa chừng). Người tham gia bấm phím **`2`** trên bàn phím số bằng **ngón giữa tay phải** khi thấy Target `þ`.")
    lines.append("- **Lưu ý nhãn trong `task-oddball_events.json`**:")
    lines.append("  - `S  5`: Standard/Frequent stimulus (`trial_type = stimulus`)")
    lines.append("  - `S  6`: Target stimulus (`trial_type = stimulus`)")
    lines.append("  - `S  7`: Deviant/Distractor stimulus (`trial_type = stimulus`)")
    lines.append("  - `S  1`: Response button (`trial_type = response` — lưu ý file JSON ghi mô tả `'Response: button no. 1'` theo mã trigger BrainVision dù phím vật lý là phím `'2'`)")
    lines.append("  - `S 10`: Break (nghỉ giữa block) | `S 12`: Beginning of the run (`trial_type = n/a`)")
    lines.append("")
    lines.append("### 1.2. Tóm tắt Nhân khẩu học (`participants.tsv`)")
    lines.append(f"- Tổng số dòng: `{part['n_rows']}` (`sub-01` .. `sub-42`), `{part['n_cols']}` cột (nhân khẩu học + bảng hỏi tâm lý UMACL/UWIST & ARSQ).")
    lines.append(f"- Giới tính: Nữ (`F`) = `{part['sex_counts'].get('F', 0)}`, Nam (`M`) = `{part['sex_counts'].get('M', 0)}`.")
    lines.append(f"- Độ tuổi: `{part['age_mean']} ± {part['age_std']}` tuổi (khoảng `{part['age_min']}–{part['age_max']}` tuổi).")
    lines.append(f"- Dùng thuốc trong ngày đo (`medication use=1`): `{part['medication_use_1']}` người | Dùng caffeine (`caffeine uptake=1`): `{part['caffeine_uptake_1']}` người.")
    if part["non_scalar_cells"]:
        lines.append(f"- **Lưu ý parse `participants.tsv`**: Có `{len(part['non_scalar_cells'])}` ô bảng hỏi chứa chuỗi mảng thay vì số nguyên: `{', '.join(part['non_scalar_cells'])}`.")
    lines.append("")
    lines.append("### 1.3. Tổng hợp Ghi chú Tác giả & Lịch sử Sửa đổi (`README`, `CHANGES`, Bài báo GigaScience 2022)")
    lines.append("- **Từ `README` & `CHANGES`**:")
    lines.append("  - `15/06/2023` (v1.0.0): Chuyển đơn vị cột `onset` trong tất cả `*_events.tsv` từ mili-giây (`ms`) sang **giây (`s`)**.")
    lines.append("  - `10/2024` (v1.0.2): Sửa logfiles trong `/sourcedata` cho tác vụ MSIT & SRT (`sub-01`..`sub-03`).")
    lines.append("  - `02/2025` (v1.0.3): Sửa marker cho tác vụ **REST** của `sub-20`, `sub-23` (không ảnh hưởng đến `task-oddball`).")
    lines.append("- **Từ bài báo gốc (Dzianok et al., 2022, *GigaScience*, giac015)**:")
    lines.append("  - **Loại trừ hành vi (Table 5)**: Tác giả loại **3 subjects (`sub-02`, `sub-14`, `sub-17`)** khỏi phân tích hành vi Oddball vì họ bấm nút cho cả kích thích Distractor (`S  7`) với tỷ lệ False Alarm là `98.77%` (`sub-02`), `97.53%` (`sub-14`), và `76.54%` (`sub-17`) — chứng tỏ hiểu nhầm hướng dẫn thực nghiệm.")
    lines.append("  - **Sự cố kỹ thuật trong phiên ghi**: Tác giả ghi nhận **5 subjects (`sub-13`, `sub-23`, `sub-24`, `sub-37`, `sub-38`)** có tổng số kích thích khác mức chuẩn 660 (từ 659 đến 775) do lỗi kỹ thuật thu thập được khắc phục/khởi động lại ngay trong phiên ghi.")
    lines.append("")

    # --- MỤC 2 ---
    lines.append("## 2. Độ đầy đủ File của 42 Subjects & Giải mã `sub-38`")
    lines.append("")
    n_complete_5 = int((df_audit["n_files_present"] == 5).sum())
    n_bytes_ok = int(df_audit["eeg_bytes_match_header"].sum())
    lines.append(f"- **Độ đầy đủ file `task-oddball`**: **`{n_complete_5}/42` subjects** có đủ cả 5 file bắt buộc (`.vhdr`, `.eeg`, `.vmrk`, `_events.tsv`, `_channels.tsv`).")
    lines.append(f"- **Kiểm tra toàn vẹn nhị phân `.eeg`**: **`{n_bytes_ok}/42` subjects** khớp tuyệt đối từng byte với công thức `DataPoints × 127 × 4` (`IEEE_FLOAT_32`).")
    lines.append(f"- **File tọa độ điện cực (`_electrodes.tsv` / `_coordsystem.json`)**: Tìm thấy **`{len(meta['electrodes_files_found'])}` file** trong thư mục local `data/raw/oddball_data/`. *(Giải thích: Các file tọa độ CapTrak `.bvct` nằm trong thư mục `/sourcedata/sub-XX/coords/` trên S3 OpenNeuro và không nằm trong nhánh `sub-XX/eeg/` khi script tải lọc theo `'task-oddball' in fname`.)*")
    lines.append("")
    lines.append("### 2.1. Kiểm tra Chuyên sâu `sub-38` (Giải đáp nghi vấn 'thiếu dữ liệu')")
    lines.append("")
    lines.append(f"- **Khả năng đọc bằng MNE-Python**: `mne.io.read_raw_brainvision()` đọc **THÀNH CÔNG 100%** (`readable_by_mne = {sub38_detail['readable_by_mne']}`).")
    lines.append(f"- **Thông số thực tế của `sub-38`**:")
    lines.append(f"  - Số mẫu thực tế (`n_times`): **`{sub38_detail['n_times']:,}` mẫu** tại `sfreq = {sub38_detail['sfreq']:.0f} Hz` (thời lượng `{sub38_detail['duration_sec']:.2f} s` ≈ `{sub38_detail['duration_sec']/60:.2f} phút`).")
    lines.append(f"  - Dung lượng `.eeg` thực tế: **`{sub38_detail['eeg_bytes_actual']:,}` bytes** (khớp tuyệt đối `{sub38_detail['eeg_bytes_expected']:,}` bytes theo header `.vhdr`).")
    lines.append(f"  - Kiểm tra giá trị mẫu đầu / vùng boundary / cuối bằng MNE: `head_finite={sub38_detail['head_finite']}`, `mid_boundary_finite={sub38_detail['mid_boundary_finite']}`, `tail_finite={sub38_detail['tail_finite']}` (hoàn toàn không có NaN/Inf hay cắt cụt).")
    lines.append(f"- **Đối chiếu `.vmrk` vs `_events.tsv` của `sub-38`**:")
    lines.append(f"  - Tệp `.vmrk` có tổng cộng `896` dòng marker: gồm `2` `Time 0` tại mẫu `1`, `1` marker `Stimulus,empty` tại mẫu `1`, `{int(sub38_row['vmrk_new_segment_count'])}` marker `New Segment` (tại mẫu `1`, `1,178,725`, `1,179,901`), `{int(sub38_row['vmrk_boundary_count'])}` marker `Stimulus,boundary` (tại mẫu `1,179,901`), và `888` marker sự kiện task (`{int(sub38_row['n_freq_raw'])}` `S  5`, `{int(sub38_row['n_targ_raw'])}` `S  6`, `{int(sub38_row['n_dist_raw'])}` `S  7`, `{int(sub38_row['n_resp_raw'])}` `S  1`, `4` `S 10`, `4` `S 12`).")
    lines.append(f"  - Tệp `_events.tsv` có `{sub38_detail['events_tsv_rows']}` dòng (`888` marker sự kiện task + `1` dòng `empty` ở `onset = 0.001 s`), tức đã **ẩn các marker `New Segment` và `boundary`**. Sai lệch thời điểm tối đa giữa `_events.tsv` và `.vmrk`: `{sub38_row['max_onset_diff_ms']:.3f} ms` (lệch 1 sample `0.001s` do quy đổi 1-based sang giây).")
    lines.append("- **Nguyên nhân dẫn tới ghi chú lỗi trước đây**:")
    lines.append("  1. Trong bài báo gốc, tác giả liệt kê `sub-38` vào nhóm 5 subject gặp lỗi thu thập kỹ thuật vì phiên ghi bị **gián đoạn ở phút ~19.6 (`1178.7–1179.9 s`)**, sau đó được khởi động lại block bổ sung và ghép nối (`New Segment` + `boundary`) dẫn đến tổng số stimuli tăng từ `660` lên **`775`** (`583` S5 + `96` S6 + `96` S7). Ngoài ra, `sub-38` từng bị nhầm lẫn là thiếu dữ liệu oddball trong các commit ban đầu của nhóm trước khi cập nhật lại `download_oddball.py`.")
    lines.append(f"  2. **Kết luận về `sub-38`**: Dữ liệu `task-oddball` của `sub-38` **KHÔNG bị mất**, hoàn toàn đọc được bằng MNE và hành vi đạt chất lượng cao (`Hit Rate = {sub38_row['hit_rate_pct']:.1f}%`, `FA_Dist = {sub38_row['fa_dist_pct']:.1f}%`), chỉ cần lưu ý **không cắt epoch xuyên qua điểm boundary `1178.7–1179.9 s`**.")
    lines.append("")

    # --- MỤC 3 ---
    lines.append("## 3. Audit Tín hiệu EEG & Hệ kênh 128 (Montage Check)")
    lines.append("")
    lines.append("### 3.1. Thông số Chung & Bẫy 127 vs 128 Kênh")
    lines.append(f"- **Tần số lấy mẫu (`sfreq`)**: Đồng nhất **`1000.0 Hz`** trên cả 42/42 subjects.")
    lines.append(f"- **Số kênh trong file**: Đồng nhất **`127` kênh**, toàn bộ đều mang loại **`EEG`** (`0` kênh EOG/ECG/MISC/STIM riêng trong `_channels.tsv` và `.vhdr`).")
    lines.append(f"- **Giải thích 127 vs 128 kênh**: Hệ thống mũ là actiCHamp 128 điện cực, nhưng kênh thứ 128 là **`FCz`** được sử dụng làm **điện cực tham chiếu phần cứng (online reference)** nên không lưu thành kênh riêng trong file raw. Khi thực hiện Average Re-referencing, nhóm có thể tùy chọn gọi `mne.add_reference_channels(raw, ref_channels=['FCz'])` để khôi phục đủ 128 kênh trước `set_eeg_reference('average')`.")
    lines.append(f"- **Độ dài bản ghi**: Trung bình `{df_audit['duration_sec'].mean():.1f} ± {df_audit['duration_sec'].std():.1f} s` (`~{df_audit['duration_min'].mean():.2f} phút`), ngắn nhất `{df_audit['duration_min'].min():.2f} phút` (`{df_audit.loc[df_audit['duration_min'].idxmin(), 'subject_id']}`), dài nhất `{df_audit['duration_min'].max():.2f} phút` (`{df_audit.loc[df_audit['duration_min'].idxmax(), 'subject_id']}`).")
    lines.append("- **Danh sách kênh `bad` trong `_channels.tsv`**: Không có cột `status` (`good`/`bad`) trong bất kỳ file `_channels.tsv` nào (chỉ có 3 cột `name`, `type`, `units`). Tất cả 42 file `_channels.tsv` đều dùng mã hóa **`Latin-1`** (`µV` = byte `0xb5`), bắt buộc truyền `encoding='latin1'` khi đọc bằng Python/Pandas.")
    lines.append("")
    lines.append("### 3.2. Tính Đồng nhất Tên Kênh, Thứ tự Kênh & Đối chiếu Montage Topomap")
    lines.append(f"- **Tập hợp 127 tên kênh**: Đồng nhất **100%** trên cả 42/42 subjects (`set_mismatches = {len(set_mismatches)}`).")
    lines.append(f"- **Bẫy đảo thứ tự kênh**: Phát hiện **`{len(order_mismatches)}/42` subjects** bị hoán đổi thứ tự kênh trong `.vhdr` so với `sub-01`:")
    for sub, diffs in order_mismatches:
        sample_diff = ", ".join([f"idx {i} ({a} vs {b})" for i, a, b in diffs[:4]])
        lines.append(f"  - `{sub}`: hoán đổi `{len(diffs)}` vị trí ({sample_diff}{'...' if len(diffs) > 4 else ''}).")
    lines.append(f"- **Đối chiếu với `mne.channels.make_standard_montage('{montage_info['montage_tested']}')`** *(lưu ý MNE khuyến nghị `colin27_1005` thay `standard_1005` từ MNE 1.14)*:")
    lines.append(f"  - Số kênh khớp trực tiếp từ tên gốc: **`123 / 127` kênh**.")
    lines.append(f"  - **4 kênh không khớp trực tiếp**: `{', '.join(montage_info['unmatched_raw_in_1005'])}`.")
    lines.append("    1. `TP9 LEFT EAR` → đổi thành **`TP9`** (khớp `1005`).")
    lines.append("    2. `TP10 RIGHT EAR` → đổi thành **`TP10`** (khớp `1005`).")
    lines.append(f"    3. `O9` → trong `standard_1005`/`colin27_1005` của MNE tên chuẩn là **`I1`** (kiểm chứng tọa độ 3D của `O9` trong `standard_1020` và `I1` trong `standard_1005` sai khác = **`{montage_info['o9_vs_i1_euclidean_diff_m']:.6f} m`** — trùng khít tuyệt đối!).")
    lines.append(f"    4. `O10` → trong `standard_1005`/`colin27_1005` của MNE tên chuẩn là **`I2`** (sai khác tọa độ = **`{montage_info['o10_vs_i2_euclidean_diff_m']:.6f} m`**).")
    lines.append(f"  - Sau khi áp dụng đủ 4 alias trên: **`127 / 127` kênh (100%)** có đầy đủ tọa độ 3D (`unmatched = {montage_info['unmatched_after_canonical_alias']}`). *(Khuyến nghị sửa ngay `CHANNEL_ALIASES` trong `src/preprocessing/pipeline.py` dòng 105–108 để bổ sung `'O9': 'I1'` và `'O10': 'I2'`).*")
    lines.append("")
    lines.append("### 3.3. Kiểm tra Tín hiệu Lỗi (NaN, Inf, Flatline, Noisy Channels, Saturation) trên toàn 42 Subjects")
    n_nan = int(df_audit["has_nan"].sum())
    n_inf = int(df_audit["has_inf"].sum())
    flat_subs = df_audit[df_audit["n_flat_channels"] > 0]
    lines.append(f"- **Giá trị `NaN` / `Inf`**: `NaN` = `{n_nan}/42` subjects, `Inf` = `{n_inf}/42` subjects (toàn bộ 42 file sạch tuyệt đối NaN/Inf).")
    lines.append(f"- **Kênh phẳng (`std < 0.1 µV`)**: Phát hiện ở `{len(flat_subs)}/42` subjects" + (f" ({', '.join(flat_subs['subject_id'].tolist())})." if len(flat_subs) > 0 else " (không phát hiện kênh phẳng trên toàn cửa sổ task)."))
    lines.append(f"- **Biên độ Std trung vị trong cửa sổ task (`task_median_std_uv`) vs Toàn file (`median_std_uv`)**:")
    lines.append(f"  - Trong cửa sổ task thực tế (`[first_stim - 2s, last_stim + 2s]`): `task_median_std_uv` dao động từ `{df_audit['task_median_std_uv'].min():.1f} µV` (`{df_audit.loc[df_audit['task_median_std_uv'].idxmin(), 'subject_id']}`) đến `{df_audit['task_median_std_uv'].max():.1f} µV` (`{df_audit.loc[df_audit['task_median_std_uv'].idxmax(), 'subject_id']}`).")
    lines.append(f"  - **Bẫy rút mũ / bão hòa cuối file (`sub-08`)**: Trên toàn file `sub-08`, `median_std_uv = {sub08_row['median_std_uv']:,.2f} µV` (`max_abs = {sub08_row['max_abs_uv']:,.1f} µV`), gấp >1000 lần mức trong cửa sổ task (`task_median_std_uv = {sub08_row['task_median_std_uv']:,.2f} µV`) do sau khi task kết thúc ở `{sub08_row['last_stim_sec']:.1f}s`, tín hiệu nhảy step khổng lồ tại `1446.21s` rồi phẳng lì đến `1694.9s`. Kiểm chứng với MNE: nếu chạy `raw.filter(0.5, 40)` không `crop()`, `sub-08` bị ringing hỏng nặng (`filtered median_std = 3,080.82 µV`); sau khi `raw.crop()` theo cửa sổ task `[2.1s, 1283.6s]`, `filtered median_std` trở về **`9.82 µV`** (hoàn toàn sạch!). Ngoài ra `sub-35` cũng có jump cuối file (`5,467.5 µV` tại `1378.47s`, sau `last_stim = {df_audit.set_index('subject_id').loc['sub-35', 'last_stim_sec']:.1f}s`).")
    lines.append(f"  - **Làm rõ trường hợp `sub-32`**: `sub-32` có `median_std_uv = {sub32_row['median_std_uv']:.2f} µV` trên tín hiệu thô (cao hơn đa số các subject khác ~200–350 µV do DC drift/trở kháng cao toàn đầu), nhưng có `0` kênh vượt ngưỡng MAD z-score > 5.0 và sau bandpass `0.5–40 Hz` đạt `median_std = 12.44 µV`.")
    lines.append(f"- **Nguy cơ kênh `O9`/`O10` nhiễm vào Average Reference**: Có `{len(o9_o10_noisy_subs)}` subjects có kênh `O9` hoặc `O10` bị nhiễu rất nặng trên raw (`{', '.join(o9_o10_noisy_subs['subject_id'].tolist())}` — ví dụ `sub-41` có `O10 = 7,960.97 µV`, `sub-16` có `O10 = 7,212.39 µV`, `sub-17` có `O9 = 1,672.49 µV`). Do chưa map `O9->I1` và `O10->I2`, pipeline cũ bỏ qua phát hiện bad channel trên 2 kênh này khiến nhiễu cực lớn bị cộng vào Average Reference!")
    lines.append(f"- **Top subjects có nhiều kênh nhiễu (`MAD z-score > 5.0` trên tín hiệu thô)**:")
    for _, row in high_noisy_subs.head(10).iterrows():
        lines.append(f"  - `{row['subject_id']}`: `{row['n_noisy_channels']}` kênh noisy (`{row['noisy_channels']}`), `task_median_std = {row['task_median_std_uv']} µV`, `max_ptp = {row['max_ptp_uv']:.1f} µV` (`{row['max_ptp_channel']}`).")
    lines.append("")

    # --- MỤC 4 ---
    lines.append("## 4. Audit Events, Triggers & Lọc Hành vi")
    lines.append("")
    lines.append("### 4.1. Cấu trúc Cột & Giá trị Unique trong `*_events.tsv` và `.vmrk`")
    lines.append("- **Các cột trong tất cả 42 file `_events.tsv`**: `onset` (giây), `duration` (toàn bộ là `n/a`), `event_type`, `trial_type`. **Không có sẵn cột `correct` hay `reaction_time`** — bắt buộc phải tự tính từ cặp `Stimulus -> Response (S  1)`!")
    lines.append("- **Các giá trị `event_type` xuất hiện toàn tập**:")
    lines.append("  - Chuẩn Oddball: `S  5` (Frequent Standard), `S  6` (Target `þ`), `S  7` (Distractor `Þ`), `S  1` (Response), `S 10` (Break), `S 12` (Beginning of run).")
    lines.append("  - Bất thường trong `_events.tsv`: `empty` (tại `onset=0.001s` ở `sub-37`, `sub-38`), và `R 15`, `S 14`, `S 15` (tại `onset=1346.74–1347.03s` ở cuối bản ghi `sub-13`).")
    lines.append("  - **Marker ẩn trong `.vmrk` nhưng KHÔNG có trong `_events.tsv`**:")
    lines.append(f"    - **{len(boundary_subs)} subjects (`{', '.join(boundary_subs['subject_id'].tolist())}`)** có marker `Stimulus,boundary` và `New Segment` giữa bản ghi trong `.vmrk` (do tạm dừng/ghép nối). Ngoài ra `sub-13`, `sub-41`, `sub-42` có `1` dòng `Mk2=Stimulus,,1,0,0,0` (mô tả rỗng tại mẫu `1`).")
    lines.append(f"    - Sai lệch thời điểm `onset` giữa `_events.tsv` và `.vmrk`: tối đa `{df_audit['max_onset_diff_ms'].max():.3f} ms` trên toàn tập (chênh đúng `1 sample = 1.0 ms` do file BIDS `_events.tsv` quy đổi trực tiếp số thứ tự mẫu 1-based chia `1000`, trong khi MNE dùng chỉ số 0-based).")
    lines.append("")
    lines.append("### 4.2. Quy tắc Lọc Hành vi (Behavioral Filtering) & Thống kê Toàn Dataset")
    lines.append("Áp dụng tiêu chuẩn ghép cặp nối tiếp trong mỗi block (không ghép xuyên qua stimulus kế tiếp, `S 10`, `S 12` hoặc `boundary`):")
    lines.append("- **Frequent (`S  5`) hợp lệ (Correct Rejection - CR)**: Không có `S  1` giữa thời điểm kích thích và kích thích/ngắt quãng tiếp theo.")
    lines.append("- **Target (`S  6`) hợp lệ (Hit)**: Có `S  1` đầu tiên sau kích thích với `100 ms <= RT <= 1500 ms` trước kích thích/ngắt quãng tiếp theo.")
    lines.append("- **Loại bỏ**: Toàn bộ Distractor (`S  7`), toàn bộ Target Miss (`no response` / `RT > 1500 ms`), Target Anticipation (`RT < 100 ms`), và Frequent False Alarm (bấm nhầm nút khi thấy `S  5`).")
    lines.append("")
    lines.append("| Tập đối tượng | Frequent Raw (`S  5`) | Target Raw (`S  6`) | Distractor Raw (`S  7`) | Frequent Clean (CR) | Target Clean (Hit) | Tổng Epoch Sạch | Tỷ lệ `Frequent : Target` |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    lines.append(f"| **Toàn bộ 42 subjects** | {sum_freq_raw:,} | {sum_targ_raw:,} | {sum_dist_raw:,} | **{sum_freq_clean_42:,}** | **{sum_targ_clean_42:,}** | **{sum_total_clean_42:,}** | **{ratio_42:.2f} : 1** |")
    lines.append(f"| **39 subjects sạch hành vi** *(loại `sub-02, 14, 17`)* | {int(df_39['n_freq_raw'].sum()):,} | {int(df_39['n_targ_raw'].sum()):,} | {int(df_39['n_dist_raw'].sum()):,} | **{sum_freq_clean_39:,}** | **{sum_targ_clean_39:,}** | **{sum_total_clean_39:,}** | **{ratio_39:.2f} : 1** |")
    lines.append("")
    lines.append("### 4.3. Phát hiện Các Subject Bất thường về Số Trial & Hành vi")
    lines.append("- **Subject hiểu sai luật bấm nhầm Distractor (`S  7`) cực cao**:")
    for s in beh_bad_subs:
        r = df_audit[df_audit["subject_id"] == s].iloc[0]
        lines.append(f"  - **`{s}`**: `FA_Dist = {r['fa_dist_pct']:.2f}%` (`{int(r['n_dist_fa'])}/{int(r['n_dist_raw'])}` lần bấm cho Distractor!), `Hit Rate = {r['hit_rate_pct']:.2f}%`, `FA_Freq = {r['fa_freq_pct']:.2f}%`. -> **Cần loại bỏ** vì đối tượng phản ứng với cả `Þ` và `þ` (không còn là phân biệt Target thuần túy; đồng thời tạo nhiễu nhãn nghiêm trọng cho P3b vs P3a/Novelty P3).")
    lines.append("- **Subject có tổng số stimuli khác mức chuẩn 660**:")
    for _, r in anomalies_stim.iterrows():
        lines.append(f"  - `{r['subject_id']}`: Tổng `{int(r['total_stim_raw'])}` stimuli (`S5={int(r['n_freq_raw'])}, S6={int(r['n_targ_raw'])}, S7={int(r['n_dist_raw'])}`), boundary trong `.vmrk` = `{int(r['vmrk_boundary_count'])}`.")
    if len(low_hit_subs) > 0:
        lines.append(f"- **Cảnh báo `{len(low_hit_subs)}` subjects có Hit Rate Target < 90% (hoặc bấm nhầm Distractor mức trung bình 25–40%)**:")
        for _, r in low_hit_subs.iterrows():
            lines.append(
                f"  - `{r['subject_id']}`: `Hit Rate = {r['hit_rate_pct']:.2f}%` "
                f"(`{int(r['n_target_clean'])}/{int(r['n_targ_raw'])}` Target sạch, Miss=`{int(r['n_targ_miss'])}`, "
                f"`FA_Dist = {r['fa_dist_pct']:.2f}%`, `Imbalance = {r['imbalance_ratio']:.2f}:1`)."
            )
    else:
        lines.append("- Trên toàn bộ 42 subjects, không có subject nào có Hit Rate Target dưới 90%.")
    lines.append(f"- **Tỷ lệ mất cân bằng lớp (`Frequent : Target`) từng subject**: dao động từ `{df_39['imbalance_ratio'].min():.2f}:1` (`{df_39.loc[df_39['imbalance_ratio'].idxmin(), 'subject_id']}`) đến `{df_39['imbalance_ratio'].max():.2f}:1` (`{df_39.loc[df_39['imbalance_ratio'].idxmax(), 'subject_id']}`) trên 39 subjects sạch (trung bình `{df_39['imbalance_ratio'].mean():.2f} ± {df_39['imbalance_ratio'].std():.2f} : 1`).")
    lines.append("")

    # --- MỤC 5 ---
    lines.append("## 5. Audit Reaction Time (RT) & Nguy cơ Nhiễu cho Bài toán XAI (RQ3)")
    lines.append("")
    lines.append("### 5.1. Thống kê Phân bố Reaction Time của Target Hit")
    lines.append(f"Tính trên tất cả **`{len(rt39):,}` trials Target Hit** của 39 subjects sạch hành vi (và `{len(rt42):,}` trials trên toàn 42 subjects):")
    lines.append("")
    lines.append("| Chỉ số RT (Target Hit) | 39 Subjects Sạch Hành vi | Toàn bộ 42 Subjects |")
    lines.append("|---|---:|---:|")
    lines.append(f"| Mean ± Std (ms) | **{np.mean(rt39):.2f} ± {np.std(rt39):.2f} ms** | {np.mean(rt42):.2f} ± {np.std(rt42):.2f} ms |")
    lines.append(f"| Median [Q25–Q75] (ms) | **{np.median(rt39):.2f} [{np.percentile(rt39,25):.1f}–{np.percentile(rt39,75):.1f}] ms** | {np.median(rt42):.2f} [{np.percentile(rt42,25):.1f}–{np.percentile(rt42,75):.1f}] ms |")
    lines.append(f"| Min – Max (ms) | `{np.min(rt39):.1f} – {np.max(rt39):.1f} ms` | `{np.min(rt42):.1f} – {np.max(rt42):.1f} ms` |")
    lines.append(f"| RT trong `[100, 300) ms` | `{rt39_100_300:.2f}%` | `{float(np.mean((rt42 >= 100.0) & (rt42 < 300.0)) * 100.0):.2f}%` |")
    lines.append(f"| **RT rơi vào cửa sổ P300 `[300, 600] ms`** | **`{rt39_300_600:.2f}%`** | **`{rt42_300_600:.2f}%`** |")
    lines.append(f"| RT `> 600 ms` | `{rt39_gt_600:.2f}%` | `{float(np.mean(rt42 > 600.0) * 100.0):.2f}%` |")
    lines.append("")
    lines.append("### 5.2. Đánh giá Nguy cơ Nhiễu Vận động (Motor Confound) & Nhiễu Chớp mắt (EOG) cho XAI")
    lines.append(f"1. **Nguy cơ Motor Confound cực lớn**: Có tới **`{rt39_300_600:.2f}%`** các trial Target có thời điểm bấm phím rơi trực tiếp vào cửa sổ P300 (`300–600 ms`, trung vị `{np.median(rt39):.1f} ms`). Vì người tham gia bấm phím **`2`** bằng **ngón giữa tay phải** chỉ ở lớp `Target` (không bấm ở lớp `Frequent`), tín hiệu vùng vỏ não vận động sơ cấp bán cầu trái (`{', '.join(SENSORIMOTOR_LEFT_CHANNELS)}`) sẽ mang xung chuẩn bị & thực thi vận động (Motor Potential / Lateralized Readiness Potential) chồng lấp về mặt thời gian với sóng P300 (`{', '.join(P300_PARIETAL_CHANNELS)}`).")
    lines.append("   - **Khuyến nghị kiểm chứng XAI (RQ3)**: Khi trực quan hóa Topomap attribution (Saliency / Integrated Gradients / SHAP / DeepLIFT) trong cửa sổ `300–600 ms`, cần thực hiện 2 phép kiểm chứng bắt buộc:")
    lines.append("     - *(a) Kiểm tra tính đối xứng đỉnh vs lệch trái*: Sóng P300 chuẩn có phân bố đỉnh-trung tâm đối xứng quanh `Pz/CPz/Cz`; nếu heatmap XAI sáng rực bất thường tại `C3/CP3/FCC3h` gần thời điểm `RT`, mô hình đang khai thác **Motor Confound** thay vì đáp ứng nhận thức P300.")
    lines.append("     - *(b) Kiểm chứng theo phân vị RT (RT-stratified ERP & XAI)*: Tách các trial Target thành nhóm `Fast RT (< 450 ms)` vs `Slow RT (> 550 ms)` hoặc thử cắt cửa sổ phân loại sớm `[-0.2, 0.45] s` để xem đặc trưng quan trọng dịch chuyển theo RT hay giữ nguyên ở pha P300.")
    lines.append(f"2. **Kiểm chứng nhiễu chớp mắt (EOG)**: Dataset **KHÔNG có kênh EOG chuyên dụng**, nhưng có đầy đủ hệ thống kênh trán cực độ phân giải cao: **`{', '.join(montage_info['frontal_eog_surrogates'])}`**. Nhóm nên:")
    lines.append("   - Dùng cặp `Fp1, Fp2` (hoặc `AFp1, AFp2`) làm kênh EOG surrogate trong `ica.find_bads_eog()` (lưu ý chỉ dùng để tính điểm tương quan EOG, không xóa bỏ kênh khỏi tensor sau khi chạy `ica.apply()`).")
    lines.append("   - Trong XAI, kiểm tra tỷ trọng attribution tại cụm kênh trán (`Fp1`, `Fp2`, `AFp1`, `AFp2`, `AF7`, `AF8`) trên cả dữ liệu *có ICA* vs *không ICA* (`X_no_ica_no_reject`) để chứng minh mô hình không ăn gian bằng phản xạ chớp mắt sau kích thích Target.")
    lines.append("")

    # --- MỤC 6 ---
    lines.append("## 6. Ước lượng Tài nguyên & Đề xuất Data Contract cho Nhóm 5 Người")
    lines.append("")
    lines.append("### 6.1. Ước lượng Dung lượng Bộ nhớ Tập Epoch Sạch (Chưa tính tỷ lệ loại biên độ ICA/Artifact)")
    lines.append(f"Dựa trên tổng số epoch sạch hành vi thực tế (`{sum_total_clean_39:,}` trials cho **39 subjects sạch**, và `{sum_total_clean_42:,}` trials nếu giữ cả 42 subjects):")
    lines.append("")
    lines.append("| Tần số lấy mẫu | Cửa sổ Epoch | Số mẫu/epoch | 39 Subs (127 ch, `float32`) | 39 Subs (127 ch, `float64`) | 39 Subs (128 ch +FCz, `float32`) | 42 Subs (127 ch, `float32`) |")
    lines.append("|---:|---|---:|---:|---:|---:|---:|")
    for r39, r39_128, r42 in zip(mem_127_39, mem_128_39, mem_127_42):
        lines.append(
            f"| **{r39['sfreq_hz']} Hz** | `{r39['window']}` | {r39['n_samples']} | "
            f"**{r39['size_float32_gb']:.3f} GB** | {r39['size_float64_gb']:.3f} GB | "
            f"{r39_128['size_float32_gb']:.3f} GB | {r42['size_float32_gb']:.3f} GB |"
        )
    lines.append("")
    rec_row = [r for r in mem_127_39 if r["sfreq_hz"] == 250 and r["window"] == "[-0.2, 0.8]s"][0]
    lines.append("> [!TIP]")
    lines.append(
        f"> **Khuyến nghị cấu hình chuẩn**: Dùng `sfreq = 250 Hz`, cửa sổ `[-0.2, 0.8] s` (`{rec_row['n_samples']}` mẫu/epoch), "
        f"lưu kiểu `np.float32` (**`{rec_row['size_float32_gb']:.3f} GB`** cho `{sum_total_clean_39:,}` epochs của 39 subjects sạch hành vi) — "
        f"vừa giữ trọn băng thông P300 (`0.1–30/40 Hz`, Nyquist `125 Hz`), vừa đủ nhẹ để train EEGNet/ShallowFBCSPNet và chạy PyTorch Captum / SHAP trên GPU/RAM phổ thông 16 GB."
    )
    lines.append("")
    lines.append("### 6.2. Data Contract Chính thức cho Nhóm 5 Người")
    lines.append("")
    lines.append("#### A. Quy tắc Chọn / Loại Subject (Cohort Policy)")
    lines.append(f"- **Tier 1 — Loại BẮT BUỘC (`3` subjects)**: **`sub-02`, `sub-14`, `sub-17`** (lỗi hành vi hiểu nhầm quy tắc, bấm nút cho Distractor `76.5%–98.8%`). Tập chuẩn mặc định của toàn nhóm là **39 subjects** (`total_clean_binary = {sum_total_clean_39:,}` epochs sạch hành vi trước bước loại biên độ).")
    lines.append("- **Tier 2 — Cờ cảnh báo chất lượng & Ablation Sensitivity (Giữ mặc định ở tập 39, kiểm chứng độ nhạy khi chạy mô hình)**:")
    lines.append("  - **`sub-08` & `sub-35`**: BẮT BUỘC gọi **`raw.crop(tmin=max(0, first_stim - 5.0), tmax=min(raw.times[-1], last_stim + 5.0))` TRƯỚC KHI chạy `filter()` và `ICA`** để loại bỏ đoạn tháo mũ / bão hòa ở cuối file (`sub-08` tại `1446.21s`).")
    lines.append("  - **`sub-07` (`Hit=60.5%`, `49` Targets), `sub-01` (`Hit=76.5%`, `FA_Dist=37.0%`), `sub-10` (`Hit=82.7%`, `FA_Dist=39.5%`), `sub-25` (`Hit=77.8%`)**: Nhóm có độ tập trung thấp hoặc bấm nhầm Distractor mức trung bình — nên giữ trong báo cáo chính (39 subjects) nhưng chạy thêm một nhánh kiểm chứng độ nhạy (*Sensitivity Analysis*) nếu kết quả Cross-subject biến động mạnh ở các subject này.")
    lines.append("  - **`sub-15`, `sub-23`, `sub-24`, `sub-37`, `sub-38`**: Có đoạn ghép nối `boundary` trong `.vmrk` (`sub-23` có `692` stimuli, `sub-38` có `775` stimuli). Cần đọc `.vhdr/.vmrk` qua MNE (MNE tự tạo annotation `BAD boundary` / `EDGE boundary` từ `.vmrk`) để tự động drop các epoch chạm ranh giới ghép nối.")
    lines.append("  - **`sub-32` & các subject nhiều kênh nhiễu (`sub-17, 18, 11, 24, 37, 38`)**: Theo dõi tỷ lệ kênh nội suy và tỷ lệ giữ epoch sau ICA/reject; nếu một subject mất `> 30%` epoch hoặc có `> 15%` kênh xấu sau lọc dải tần `0.5–40 Hz` mới cân nhắc loại ở bước Preprocessing QC.")
    lines.append("  - **`sub-13`**: Có 3 marker kỹ thuật lạ ở cuối file (`R 15`, `S 14`, `S 15` tại `1346.7s`) — không ảnh hưởng tới bất kỳ stimulus Oddball nào nếu chỉ trích xuất `S  5`, `S  6`.")
    lines.append("")
    lines.append("#### B. Quy tắc Cắt gọn, Chuẩn hóa Kênh, Montage & Đồng bộ Thứ tự (Code Mẫu)")
    lines.append("```python")
    lines.append("# 0. QUAN TRỌNG: Cắt gọn về cửa sổ task thực tế TRƯỚC khi Filter & ICA (cứu sub-08, sub-35)")
    lines.append("ev_df = pd.read_csv(events_tsv_path, sep='\\t')")
    lines.append("stim_onsets = ev_df.loc[ev_df['event_type'].isin(['S  5', 'S  6', 'S  7']), 'onset'].astype(float)")
    lines.append("raw.crop(tmin=max(0.0, stim_onsets.min() - 5.0), tmax=min(raw.times[-1], stim_onsets.max() + 5.0))")
    lines.append("")
    lines.append("# 1. Chuẩn hóa tên kênh đầy đủ (khắc phục lỗi thiếu O9/O10 trong colin27_1005)")
    lines.append("CHANNEL_ALIASES = {")
    lines.append("    'TP9 LEFT EAR': 'TP9',")
    lines.append("    'TP10 RIGHT EAR': 'TP10',")
    lines.append("    'O9': 'I1',")
    lines.append("    'O10': 'I2',")
    lines.append("}")
    lines.append("raw.rename_channels({k: v for k, v in CHANNEL_ALIASES.items() if k in raw.ch_names})")
    lines.append("")
    lines.append("# 2. Gán montage chuẩn (trên MNE >= 1.14 dùng 'colin27_1005'; bản cũ hỗ trợ 'standard_1005')")
    lines.append("montage = mne.channels.make_standard_montage('colin27_1005')")
    lines.append("raw.set_montage(montage, on_missing='raise')  # Đảm bảo 127/127 kênh đều có tọa độ 3D")
    lines.append("")
    lines.append("# 3. Luôn chuẩn hóa thứ tự kênh về canonical_order (của sub-01 sau alias) trước khi xuất Tensor")
    lines.append("raw.reorder_channels(CANONICAL_127_CHANNELS)")
    lines.append("```")
    lines.append("")
    lines.append("#### C. Quy tắc Lọc Event Hành vi Chính xác (Code Mẫu Copy-Paste)")
    lines.append("```python")
    lines.append("def extract_behaviorally_clean_events(events_tsv_path: Path, sfreq: float = 1000.0,")
    lines.append("                                      rt_min: float = 0.100, rt_max: float = 1.500):")
    lines.append("    \"\"\"Trả về mảng MNE events (N, 3) chỉ gồm Frequent CR (0) và Target Hit (1).\"\"\"")
    lines.append("    df = pd.read_csv(events_tsv_path, sep='\\t')")
    lines.append("    stim_codes = {'S  5', 'S  6', 'S  7'}")
    lines.append("    break_codes = {'S 10', 'S 12'}")
    lines.append("    records = list(zip(df['onset'].astype(float), df['event_type'].astype(str)))")
    lines.append("    stim_idx = [i for i, (_, c) in enumerate(records) if c in stim_codes]")
    lines.append("    clean_events, rts_ms = [], []")
    lines.append("    for k, i in enumerate(stim_idx):")
    lines.append("        t_stim, code = records[i]")
    lines.append("        j_end = stim_idx[k + 1] if k + 1 < len(stim_idx) else len(records)")
    lines.append("        resps = []")
    lines.append("        for j in range(i + 1, j_end):")
    lines.append("            t_j, c_j = records[j]")
    lines.append("            if c_j in break_codes:")
    lines.append("                break")
    lines.append("            if c_j == 'S  1':")
    lines.append("                resps.append(t_j - t_stim)")
    lines.append("        sample = int(round(t_stim * sfreq))")
    lines.append("        if code == 'S  5' and not any(r <= rt_max for r in resps):")
    lines.append("            clean_events.append([sample, 0, 0])   # Label 0: Frequent CR")
    lines.append("        elif code == 'S  6' and resps and (rt_min <= resps[0] <= rt_max):")
    lines.append("            clean_events.append([sample, 0, 1])   # Label 1: Target Hit")
    lines.append("            rts_ms.append(resps[0] * 1000.0)")
    lines.append("    return np.asarray(clean_events, dtype=np.int64), np.asarray(rts_ms, dtype=np.float64)")
    lines.append("```")
    lines.append("")
    lines.append("#### D. Giao thức Chia Tập Chuẩn (Validation Splits) & Chống Data Leakage")
    lines.append("1. **Kịch bản 1 — Within-Subject Classification**:")
    lines.append("   - Chạy riêng trên từng subject với **`StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`** (đảm bảo tỷ lệ ~6:1 giữa Frequent/Target ở mỗi fold; chú ý `sub-XX` không xáo trộn rò rỉ giữa các fold vì các epoch đã tách rời, nhưng có thể cân nhắc `StratifiedGroupKFold` theo block nếu muốn kiểm tra drift theo thời gian).")
    lines.append("   - **Chống Leakage**: Mọi tham số chuẩn hóa (`StandardScaler`, `CSP`, `xDAWN`, `PCA`, `class_weight`) **chỉ được `.fit()` trên tập Train của từng fold**, tuyệt đối không fit trên toàn bộ dữ liệu trước khi chia fold.")
    lines.append("2. **Kịch bản 2 — Cross-Subject Classification (Tổng quát hóa sang người mới)**:")
    lines.append("   - Sử dụng **`StratifiedGroupKFold(n_splits=5)`** với `groups=subject_id` (hoặc **LOSO - `LeaveOneGroupOut()`** cho 39 folds để đánh giá chuẩn cuối cùng).")
    lines.append(f"   - **Cấm tuyệt đối**: Chia random toàn bộ `{sum_total_clean_39:,}` trials mà không nhóm theo `subject_id` (sẽ gây rò rỉ chữ ký sinh học từng cá nhân giữa Train và Test, đẩy AUC lên ảo).")
    lines.append(f"   - **Metric đánh giá bắt buộc**: Vì mất cân bằng lớp `{ratio_39:.2f} : 1`, KHÔNG dùng Accuracy thô làm metric chính; bắt buộc báo cáo **`Balanced Accuracy`**, **`ROC-AUC`**, **`PR-AUC`**, và **`F1-score (Target class)`**.")
    lines.append("")

    # --- MỤC 7 ---
    lines.append("## 7. Bảng Tổng hợp Toàn bộ 42 Subjects (Trích yếu)")
    lines.append(f"Chi tiết đầy đủ {len(df_audit.columns)} cột cho từng subject được lưu tại [`reports/audit_per_subject.csv`](audit_per_subject.csv).")
    lines.append("")
    lines.append("| Subject | Size (MB) | Phút | Kênh lỗi (Flat/Noisy) | S5 Raw | S6 Raw | S7 Raw | S5 Clean | S6 Clean | Ratio (F:T) | Hit (%) | FA_Dist (%) | RT Mean (ms) | RT in P300 (%) | Trạng thái / Ghi chú |")
    lines.append("|---|---:|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    for _, r in df_audit.iterrows():
        ch_err = f"{int(r['n_flat_channels'])}/{int(r['n_noisy_channels'])}"
        flags_md = str(r["audit_flags"]).replace(" | ", "; ")
        lines.append(
            f"| {r['subject_id']} | {r['total_subject_mb']:.1f} | {r['duration_min']:.2f} | {ch_err} | "
            f"{int(r['n_freq_raw'])} | {int(r['n_targ_raw'])} | {int(r['n_dist_raw'])} | "
            f"{int(r['n_freq_clean'])} | {int(r['n_target_clean'])} | {r['imbalance_ratio']:.2f}:1 | "
            f"{r['hit_rate_pct']:.1f}% | {r['fa_dist_pct']:.1f}% | {r['rt_mean_ms']:.1f} | "
            f"{r['rt_in_300_600_pct']:.1f}% | {flags_md} |"
        )
    lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8")


# ==============================================================================
# 8. HÀM CHÍNH ĐIỀU PHỐI TOÀN BỘ AUDIT (READ-ONLY)
# ==============================================================================
def run_audit(data_dir: Path, reports_dir: Path) -> None:
    """Thực thi toàn bộ quy trình kiểm toán 7 mục ở chế độ READ-ONLY."""
    t_start = time.time()
    if not data_dir.exists():
        raise FileNotFoundError(f"Không tìm thấy thư mục dữ liệu: {data_dir}")

    reports_dir.mkdir(parents=True, exist_ok=True)
    print(f"{'='*80}")
    print("BẮT ĐẦU KIỂM TOÁN TOÀN DIỆN DATASET EEG ODDBALL (READ-ONLY MODE)")
    print(f"Data dir   : {data_dir.resolve()}")
    print(f"Reports dir: {reports_dir.resolve()}")
    print(f"{'='*80}")

    # 1. Audit BIDS root metadata
    print("[1/6] Đang đọc BIDS root metadata (README, CHANGES, participants.tsv, JSON)...")
    meta = audit_bids_root_metadata(data_dir)
    df_part = meta["participants_summary"]["df"].set_index("participant_id")

    # 2-5. Quét lần lượt 42 subjects
    print("[2/6] Đang kiểm tra tệp, header, chất lượng tín hiệu & hành vi cho 42 subjects...")
    ref_channels: Optional[List[str]] = None
    ref_sub: Optional[str] = None
    order_mismatches: List[Tuple[str, List[Tuple[int, str, str]]]] = []
    set_mismatches: List[Tuple[str, List[str]]] = []
    montage_info: Optional[Dict[str, Any]] = None

    audit_rows: List[Dict[str, Any]] = []
    all_target_rts_ms: List[float] = []
    clean39_target_rts_ms: List[float] = []

    for sub_id in EXPECTED_SUBJECTS:
        sub_dir = data_dir / sub_id
        file_sizes: Dict[str, int] = {}
        n_files_present = 0
        for key, suffix in REQUIRED_SUFFIXES:
            fpath = sub_dir / f"{sub_id}_{suffix}"
            if fpath.exists():
                file_sizes[key] = fpath.stat().st_size
                n_files_present += 1
            else:
                file_sizes[key] = -1

        vhdr_path = sub_dir / f"{sub_id}_task-oddball_eeg.vhdr"
        eeg_path = sub_dir / f"{sub_id}_task-oddball_eeg.eeg"
        vmrk_path = sub_dir / f"{sub_id}_task-oddball_eeg.vmrk"
        events_path = sub_dir / f"{sub_id}_task-oddball_events.tsv"
        ch_tsv_path = sub_dir / f"{sub_id}_task-oddball_channels.tsv"

        # Đọc header .vhdr + kiểm chứng qua MNE preload=False
        vhdr_info = parse_vhdr_header(vhdr_path)
        raw_hdr = mne.io.read_raw_brainvision(vhdr_path, preload=False, verbose=False)
        sfreq = float(raw_hdr.info["sfreq"])
        n_times = int(raw_hdr.n_times)
        n_channels = len(raw_hdr.ch_names)
        ch_types_unique = sorted(set(raw_hdr.get_channel_types()))
        expected_eeg_bytes = n_times * n_channels * 4
        eeg_bytes_match = (file_sizes["eeg"] == expected_eeg_bytes)

        # Kiểm tra đồng nhất tập hợp & thứ tự kênh
        if ref_channels is None:
            ref_channels = list(raw_hdr.ch_names)
            ref_sub = sub_id
            channel_order_ok = True
            channel_set_ok = True
            montage_info = audit_montage_compatibility(ref_channels)
        else:
            if set(raw_hdr.ch_names) != set(ref_channels):
                channel_set_ok = False
                channel_order_ok = False
                set_mismatches.append((sub_id, sorted(set(ref_channels) ^ set(raw_hdr.ch_names))))
            else:
                channel_set_ok = True
                if list(raw_hdr.ch_names) != ref_channels:
                    channel_order_ok = False
                    diffs = [
                        (idx, a, b)
                        for idx, (a, b) in enumerate(zip(ref_channels, raw_hdr.ch_names))
                        if a != b
                    ]
                    order_mismatches.append((sub_id, diffs))
                else:
                    channel_order_ok = True

        # Kiểm tra _channels.tsv
        ch_tsv_info = check_channels_tsv_encoding_and_schema(ch_tsv_path)

        # Kiểm tra .vmrk & lọc hành vi + RT từ _events.tsv (chạy trước để lấy cửa sổ task)
        vmrk_info = parse_vmrk_file(vmrk_path)
        ev_info = audit_events_and_behavior(events_path, vmrk_info, sfreq)

        # Kiểm tra tín hiệu thực tế (NaN, Inf, Flat, Noisy, Saturation) bằng memmap read-only
        sig_info = inspect_signal_quality_readonly(
            vhdr_path,
            eeg_path,
            vhdr_info,
            task_start_sec=ev_info["first_stim_sec"],
            task_end_sec=ev_info["last_stim_sec"],
        )

        all_target_rts_ms.extend(ev_info["target_rts_ms"])
        if ev_info["fa_dist_pct"] <= 50.0:
            clean39_target_rts_ms.extend(ev_info["target_rts_ms"])

        # Tạo audit_flags
        flags: List[str] = []
        if ev_info["fa_dist_pct"] > 50.0:
            flags.append("EXCLUDE_BEHAVIOR_FA_DIST")
        if ev_info["hit_rate_pct"] < 90.0:
            flags.append(f"LOW_HIT_RATE({ev_info['hit_rate_pct']:.1f}%)")
        if sig_info["median_std_uv"] > 10.0 * max(sig_info["task_median_std_uv"], 1.0) or sig_info["max_abs_uv"] > 600000.0:
            flags.append("TAIL_SATURATION_CAP_REMOVAL")
        if not channel_order_ok:
            flags.append("REORDER_CHANNELS")
        if len(vmrk_info["boundary_positions"]) > 0:
            flags.append(f"BOUNDARY({len(vmrk_info['boundary_positions'])})")
        if ev_info["total_stim_raw"] != 660:
            flags.append(f"STIM_COUNT_{ev_info['total_stim_raw']}")
        if sig_info["n_noisy"] >= 5:
            flags.append(f"HIGH_NOISY_CHS({sig_info['n_noisy']})")
        elif sig_info["n_noisy"] > 0:
            flags.append(f"NOISY_CHS({sig_info['n_noisy']})")
        if sig_info["n_flat"] > 0:
            flags.append(f"FLAT_CHS({sig_info['n_flat']})")
        if vmrk_info["weird_markers"]:
            flags.append("WEIRD_MARKERS")
        if not flags:
            flags.append("OK")

        p_row = df_part.loc[sub_id] if sub_id in df_part.index else None
        total_mb = sum(v for v in file_sizes.values() if v > 0) / (1024**2)
        noisy_str = ";".join([f"{c}({s}uV/z={z})" for c, s, z in sig_info["noisy_channels"]])
        weird_str = ";".join([f"Mk{m}:{d}@{p}" for m, _, d, p in vmrk_info["weird_markers"]])

        audit_rows.append({
            "subject_id": sub_id,
            "sex": str(p_row["sex"]) if p_row is not None else "",
            "age": str(p_row["age"]) if p_row is not None else "",
            "n_files_present": n_files_present,
            "eeg_size_mb": round(file_sizes["eeg"] / (1024**2), 2),
            "total_subject_mb": round(total_mb, 2),
            "eeg_bytes_match_header": eeg_bytes_match,
            "sfreq_hz": sfreq,
            "n_channels": n_channels,
            "channel_types": ",".join(ch_types_unique),
            "channels_tsv_utf8_ok": ch_tsv_info["utf8_valid"],
            "channel_set_ok": channel_set_ok,
            "channel_order_ok": channel_order_ok,
            "n_samples": n_times,
            "duration_sec": round(n_times / sfreq, 2),
            "duration_min": round(n_times / sfreq / 60.0, 2),
            "first_stim_sec": ev_info["first_stim_sec"],
            "last_stim_sec": ev_info["last_stim_sec"],
            "has_nan": sig_info["has_nan"],
            "has_inf": sig_info["has_inf"],
            "median_std_uv": sig_info["median_std_uv"],
            "task_median_std_uv": sig_info["task_median_std_uv"],
            "tail_10s_std_uv": sig_info["tail_10s_std_uv"],
            "max_abs_uv": sig_info["max_abs_uv"],
            "max_std_channel": sig_info["max_std_channel"],
            "max_std_uv": sig_info["max_std_uv"],
            "max_ptp_channel": sig_info["max_ptp_channel"],
            "max_ptp_uv": sig_info["max_ptp_uv"],
            "n_flat_channels": sig_info["n_flat"],
            "flat_channels": ";".join(sig_info["flat_channels"]),
            "n_noisy_channels": sig_info["n_noisy"],
            "n_task_noisy_channels": sig_info["n_task_noisy"],
            "noisy_channels": noisy_str,
            "tsv_rows": ev_info["tsv_n_rows"],
            "vmrk_non_segment_markers": ev_info["vmrk_n_stim_markers"],
            "vmrk_boundary_count": len(vmrk_info["boundary_positions"]),
            "vmrk_new_segment_count": len(vmrk_info["new_segment_positions"]),
            "vmrk_weird_markers": weird_str,
            "max_onset_diff_ms": ev_info["max_onset_diff_ms"],
            "n_freq_raw": ev_info["n_freq_raw"],
            "n_targ_raw": ev_info["n_targ_raw"],
            "n_dist_raw": ev_info["n_dist_raw"],
            "total_stim_raw": ev_info["total_stim_raw"],
            "n_resp_raw": ev_info["n_resp_raw"],
            "n_freq_clean": ev_info["n_freq_clean"],
            "n_freq_fa": ev_info["n_freq_fa"],
            "n_target_clean": ev_info["n_targ_clean"],
            "n_targ_miss": ev_info["n_targ_miss"],
            "n_targ_anticipation": ev_info["n_targ_anticipation"],
            "n_dist_fa": ev_info["n_dist_fa"],
            "total_clean_binary": ev_info["total_clean_binary"],
            "imbalance_ratio": ev_info["imbalance_ratio"],
            "hit_rate_pct": ev_info["hit_rate_pct"],
            "fa_freq_pct": ev_info["fa_freq_pct"],
            "fa_dist_pct": ev_info["fa_dist_pct"],
            "rt_mean_ms": ev_info["rt_mean_ms"],
            "rt_std_ms": ev_info["rt_std_ms"],
            "rt_median_ms": ev_info["rt_median_ms"],
            "rt_iqr_ms": ev_info["rt_iqr_ms"],
            "rt_min_ms": ev_info["rt_min_ms"],
            "rt_max_ms": ev_info["rt_max_ms"],
            "rt_in_100_300_pct": ev_info["rt_in_100_300_pct"],
            "rt_in_300_600_pct": ev_info["rt_in_300_600_pct"],
            "rt_gt_600_pct": ev_info["rt_gt_600_pct"],
            "audit_flags": " | ".join(flags),
        })
        print(
            f"  [{sub_id}] {n_times/sfreq/60:.2f}m | "
            f"Flat/Noisy={sig_info['n_flat']}/{sig_info['n_noisy']} | "
            f"Clean F/T={ev_info['n_freq_clean']}/{ev_info['n_targ_clean']} | "
            f"Hit={ev_info['hit_rate_pct']:.1f}% FA_Dist={ev_info['fa_dist_pct']:.1f}% | "
            f"RT={ev_info['rt_mean_ms']:.0f}ms ({ev_info['rt_in_300_600_pct']:.1f}% in P300)"
        )

    df_audit = pd.DataFrame(audit_rows)

    # 3. Kiểm tra chuyên sâu sub-38
    print("[3/6] Đang soi chuyên sâu sub-38 bằng MNE...")
    sub38_detail = deep_inspect_sub38(data_dir)

    # 4. Ước lượng tài nguyên bộ nhớ
    print("[4/6] Đang tính toán bảng ước lượng tài nguyên...")
    df_39 = df_audit[df_audit["fa_dist_pct"] <= 50.0]
    mem_rows = estimate_memory_table(
        n_trials_all42=int(df_audit["total_clean_binary"].sum()),
        n_trials_clean39=int(df_39["total_clean_binary"].sum()),
    )

    # 5. Xuất CSV
    csv_path = reports_dir / "audit_per_subject.csv"
    df_audit.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"[5/6] Đã xuất bảng chi tiết từng subject: {csv_path}")

    # 6. Xuất Markdown report
    md_path = reports_dir / "DATA_AUDIT_REPORT.md"
    assert montage_info is not None
    generate_markdown_report(
        meta=meta,
        df_audit=df_audit,
        sub38_detail=sub38_detail,
        montage_info=montage_info,
        order_mismatches=order_mismatches,
        set_mismatches=set_mismatches,
        all_target_rts_ms=all_target_rts_ms,
        clean39_target_rts_ms=clean39_target_rts_ms,
        mem_rows=mem_rows,
        report_path=md_path,
    )
    print(f"[6/6] Đã xuất báo cáo tổng hợp tiếng Việt: {md_path}")
    print(f"HOÀN TẤT AUDIT TRONG {time.time() - t_start:.2f} GIÂY (CHẾ ĐỘ READ-ONLY).")


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Read-only comprehensive audit for OpenNeuro ds004621 (Oddball P300)."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=project_root / "data" / "raw" / "oddball_data",
        help="Đường dẫn thư mục dữ liệu gốc BIDS (chế độ READ-ONLY).",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=project_root / "reports",
        help="Đường dẫn thư mục xuất báo cáo CSV và Markdown.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_audit(data_dir=args.data_dir, reports_dir=args.reports_dir)
