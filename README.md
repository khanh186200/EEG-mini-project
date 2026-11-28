# EEG Single-Trial Classification (Oddball Paradigm - P300)

Dự án nghiên cứu và phân loại tín hiệu điện não đồ (EEG) theo từng thử nghiệm đơn lẻ (single-trial classification) trên bài toán **Oddball Paradigm (P300)** thuộc học phần **Tính toán thần kinh**.

---

## 📌 1. Giới thiệu bài toán & Dữ liệu

- **Bài toán**: Nhận diện và phân loại đáp ứng điện thế liên quan sự kiện (Event-Related Potential - ERP / sóng **P300**) đối với kích thích mục tiêu (Target) và kích thích chuẩn (Non-target/Standard).
- **Dataset**: [OpenNeuro ds004621](https://openneuro.org/datasets/ds004621).
  - Tác vụ: `task-oddball`
  - Đối tượng: 42 đối tượng (`sub-01` đến `sub-42`), tất cả đều có đủ dữ liệu task-oddball. Một số subject (vd: `sub-32` trở kháng cao) có thể được loại ở bước tiền xử lý.
  - Tín hiệu: Dữ liệu EEG đa kênh kèm tọa độ điện cực và nhãn sự kiện.

---

## 📂 2. Cấu trúc dự án

```text
EEG-mini-project/
│
├── data/
│   ├── raw/
│   │   └── oddball_data/               # 42 subjects + metadata (ignored by git)
│   └── processed/                      # Dữ liệu sau tiền xử lý (epochs, .fif, .npy) (ignored)
│
├── scripts/
│   ├── data/                            # Tải, audit, khám phá, tiền xử lý và chia tập
│   ├── training/                        # Standard ML, DL1, DL2 và full-training orchestration
│   └── evaluation/                      # Latency, XAI, biểu đồ và file bàn giao
│
├── notebooks/                          # Notebook phẳng: 01_exploration, 02_preprocessing, 03_erp, 04_ml, 05_dl
│
├── src/                                # Code tái sử dụng (notebook/script import vào)
│   ├── preprocessing/                  # Pipeline tiền xử lý (pipeline.py)
│   ├── models/                         # Kiến trúc mô hình ML & DL
│   ├── training/                       # Pipeline huấn luyện, tối ưu siêu tham số
│   └── evaluation/                     # Đánh giá (ROC-AUC, Confusion Matrix, Cross-Validation)
│
├── results/                            # (nội dung bị ignore, chỉ giữ .gitkeep)
│   ├── figures/                        # Biểu đồ, sóng ERP, phổ công suất
│   ├── metrics/                        # Accuracy, F1-score, AUC
│   └── checkpoints/                    # Trọng số mô hình (.pt, .pth, .pkl)
│
├── reports/                            # Báo cáo đồ án / thuyết trình
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 🚀 3. Hướng dẫn cài đặt & Khởi tạo

### 3.1. Thiết lập môi trường

Khuyến nghị sử dụng Python 3.9 - 3.11 trong môi trường ảo:

```powershell
# Tạo môi trường ảo
python -m venv .venv

# Kích hoạt môi trường (Windows PowerShell)
.venv\Scripts\Activate.ps1

# Cài đặt các thư viện cần thiết
pip install -r requirements.txt
```

*(Các thư viện cốt lõi thường dùng: `mne`, `boto3`, `tqdm`, `numpy`, `scipy`, `scikit-learn`, `matplotlib`, `seaborn`, `torch` / `braindecode`)*.

---

## 📥 4. Tải dữ liệu thô (Raw Data)

Dữ liệu được tải tự động đa luồng trực tiếp từ S3 Bucket của OpenNeuro qua script:

```powershell
python scripts/data/download_oddball.py
```

- Toàn bộ dữ liệu 42 subjects và metadata đi kèm sẽ được tải và lưu tập trung tại thư mục `data/raw/oddball_data/`.
- Script hỗ trợ tiếp tục tải (skip file đã hoàn tất) và kiểm tra toàn vẹn dung lượng file.
- Thư mục dữ liệu `oddball_data/` đã được cấu hình trong `.gitignore`, không lo bị vô tình commit lên Git.

---

## 🔄 5. Quy trình thực hiện (Pipeline)

1. **Khám phá dữ liệu (`scripts/data/explore_data.py` / `notebooks/01_exploration.ipynb`)**:
   - Kiểm tra thông tin kênh đo, tỷ lệ lấy mẫu, sự kiện (Events/Triggers: Target vs. Standard).
2. **Tiền xử lý (`notebooks/02_preprocessing.ipynb` & `src/preprocessing/`)**:
   - Lọc dải tần (Band-pass filter, e.g., 0.1 - 30 Hz), lọc Notching 50Hz.
   - Loại bỏ artifacts bằng ICA hoặc ngưỡng biên độ.
   - Cắt epoch (-200ms đến 800ms quanh kích thích) và hiệu chỉnh baseline.
3. **Phân tích ERP (`notebooks/03_erp_analysis.ipynb`)**:
   - Trực quan hóa hình thái sóng P300 (đỉnh dương xuất hiện khoảng 300 - 500ms sau kích thích mục tiêu, rõ nhất tại các điện cực đỉnh/trung tâm như Pz, Cz).
4. **Phân loại Single-Trial**:
   - **Machine Learning (`notebooks/04_standard_ml.ipynb`)**: Trích xuất đặc trưng thời gian - tần số, CSP (Common Spatial Pattern) kết hợp SVM / LDA / Random Forest.
   - **Deep Learning (`notebooks/05_deep_learning.ipynb`)**: Huấn luyện các kiến trúc chuyên biệt cho EEG như EEGNet, ShallowFBCSPNet.
5. **Đánh giá (`results/` & `reports/`)**:
   - Cross-Validation (Within-subject và Cross-subject).
   - Đánh giá qua các chỉ số: Balanced Accuracy, F1-Score, ROC-AUC.

### Huấn luyện DL1

```powershell
python scripts/train_dl1.py --data-dir data/processed --results-dir results
```

DL1 chạy lần lượt **Within-subject** (195 fold), **LOSO** (39 fold) và **Cross-subject 5-fold** (5 fold cho `clean`, sau đó 5 fold cho `no_ica`). Dùng `--mode pilot` để kiểm tra nhanh cùng thứ tự, mỗi lượt 1 fold và 1 epoch. Khi chạy lại, script dùng lại các fold đã hoàn tất.
