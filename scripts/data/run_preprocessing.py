"""Script CLI tiện ích chạy pipeline tiền xử lý EEG Oddball P300.

Tệp: scripts/data/run_preprocessing.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.pipeline import main

if __name__ == "__main__":
    main()
