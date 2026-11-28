"""Repository-level schema smoke test for currently available handover files."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.handover_validation import validate_handover_directory


def run_smoke_test() -> int:
    report = validate_handover_directory(PROJECT_ROOT / "results/handover_standard")
    report.raise_for_errors()
    print(f"Handover schema valid: {report.files_checked} file(s) checked.")
    return 0


def test_current_handover_files_follow_section_35_schema() -> None:
    assert run_smoke_test() == 0


if __name__ == "__main__":
    raise SystemExit(run_smoke_test())
