"""Patient-level, balanced 5-fold cross-validation split of the MSLesSeg training set.

Rules
-----
* Patient-level: all timepoints of a patient go to the same fold, so no patient is
  ever in both training and validation (no data leakage).
* Balanced: each fold gets a similar number of studies and a similar total lesion
  volume. Patients are sorted by number of studies and lesion volume (largest
  first) and each one is assigned to the fold that currently has the fewest
  studies (ties: least lesion volume). The procedure is deterministic.
* The test set (P54-P75) is never used.

Why not the paper's folds? The reference paper (Guarnera et al., 2025, Table 4)
uses contiguous patient blocks (44-53, 34-43, 24-33, 14-23, 1-10). Its table has
two typos (patient 23 in training and validation of fold 4; patients 11-13 never
validated) and, once corrected, the folds are very unbalanced (10 to 36 studies)
because the first patients have more timepoints. Comparison with the paper is done
on the fixed test set, which does not depend on the folds.

Run:  python -m ms_seg.folds   ->  writes configs/folds.json
"""

import json

import pandas as pd
from loguru import logger

from ms_seg.config import INTERIM_DATA_DIR, PROJ_ROOT, RAW_DATA_DIR

DATA = RAW_DATA_DIR / "MSLesSeg"
FOLDS_FILE = PROJ_ROOT / "configs" / "folds.json"
N_FOLDS = 5


def list_train_studies() -> pd.DataFrame:
    """One row per training study (patient, timepoint, study_id) found on disk."""
    rows = [
        {"patient": d.parent.name, "timepoint": d.name, "study_id": f"{d.parent.name}_{d.name}"}
        for d in DATA.glob("train/P*/T*")
        if d.is_dir()
    ]
    df = pd.DataFrame(rows)
    df["num"] = df.patient.str[1:].astype(int)
    return df.sort_values(["num", "timepoint"]).drop(columns="num").reset_index(drop=True)


def lesion_volume_per_study() -> pd.Series:
    """Lesion voxels per study from the EDA output (data/interim/study_stats.csv)."""
    stats = pd.read_csv(INTERIM_DATA_DIR / "study_stats.csv")
    return stats.set_index("study_id").lesion_voxels


def make_folds(n_folds: int = N_FOLDS) -> pd.DataFrame:
    """Assign each training study to a validation fold (1..n_folds), balanced and patient-level."""
    df = list_train_studies()
    df["lesion_voxels"] = df.study_id.map(lesion_volume_per_study())
    patients = (
        df.groupby("patient")
        .agg(studies=("study_id", "size"), lesion_voxels=("lesion_voxels", "sum"))
        .sort_values(["studies", "lesion_voxels"], ascending=False)
    )
    load = {f: [0, 0.0] for f in range(1, n_folds + 1)}  # fold -> [studies, lesion voxels]
    assignment = {}
    for patient, row in patients.iterrows():
        fold = min(load, key=lambda f: (load[f][0], load[f][1], f))
        assignment[patient] = fold
        load[fold][0] += row.studies
        load[fold][1] += row.lesion_voxels
    df["fold"] = df.patient.map(assignment)
    check_folds(df, n_folds)
    return df


def check_folds(df: pd.DataFrame, n_folds: int = N_FOLDS) -> None:
    """Raise if the split has leakage or leaves a training patient out."""
    assert (df.groupby("patient").fold.nunique() == 1).all(), "patient split across folds"
    assert sorted(df.patient.str[1:].astype(int).unique()) == list(range(1, 54)), "missing patients"
    for f in range(1, n_folds + 1):
        val = set(df.loc[df.fold == f, "patient"])
        train = set(df.loc[df.fold != f, "patient"])
        assert val and not val & train, f"leakage or empty fold {f}"


def to_json(df: pd.DataFrame) -> dict:
    out = {"description": __doc__.strip().splitlines()[0], "folds": []}
    for f in sorted(df.fold.unique()):
        val, train = df[df.fold == f], df[df.fold != f]
        out["folds"].append({
            "fold": int(f),
            "val_patients": sorted(val.patient.unique(), key=lambda p: int(p[1:])),
            "train": train.study_id.tolist(),
            "val": val.study_id.tolist(),
        })
    return out


if __name__ == "__main__":
    df = make_folds()
    FOLDS_FILE.parent.mkdir(parents=True, exist_ok=True)
    FOLDS_FILE.write_text(json.dumps(to_json(df), indent=2))
    summary = df.groupby("fold").agg(
        patients=("patient", "nunique"), studies=("study_id", "size"), lesion_voxels=("lesion_voxels", "sum")
    )
    logger.info(f"Folds written to {FOLDS_FILE}\n{summary}")
