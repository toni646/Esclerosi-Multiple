import pytest

from ms_seg.folds import DATA, check_folds, make_folds


@pytest.mark.skipif(not DATA.exists(), reason="MSLesSeg data not available")
def test_folds_patient_level_and_balanced():
    df = make_folds()
    check_folds(df)  # no leakage, every patient validated once
    assert len(df) == 93 and df.patient.nunique() == 53
    studies = df.groupby("fold").size()
    assert studies.max() - studies.min() <= 2  # balanced
    assert make_folds().equals(df)  # deterministic
