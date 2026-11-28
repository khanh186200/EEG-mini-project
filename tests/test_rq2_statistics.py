import pandas as pd

from src.evaluation.statistics_rq2 import build_paired_subject_table


def test_within_folds_are_averaged_before_pairing_models() -> None:
    model_a = pd.DataFrame(
        {
            "subject_id": ["sub-01", "sub-01", "sub-02", "sub-02"],
            "roc_auc": [0.6, 0.8, 0.7, 0.9],
        }
    )
    model_b = pd.DataFrame(
        {
            "subject_id": ["sub-01", "sub-01", "sub-03", "sub-03"],
            "roc_auc": [0.5, 0.7, 0.8, 1.0],
        }
    )
    paired = build_paired_subject_table({"a": model_a, "b": model_b}, "roc_auc")
    assert paired.index.tolist() == ["sub-01"]
    assert paired.loc["sub-01"].to_dict() == {"a": 0.7, "b": 0.6}
