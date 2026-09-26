"""Exploratory data analysis helpers for the MSLesSeg dataset.

Computes per-study and per-lesion statistics from the NIfTI files and caches
them as CSV files in ``data/interim`` so the EDA notebook runs quickly.
Lesions are connected components of the mask with 6-connectivity.

Usage (from the repository root)::

    python -m ms_seg.eda
"""

from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from scipy import ndimage

from ms_seg.config import INTERIM_DATA_DIR, RAW_DATA_DIR

DATA_DIR = RAW_DATA_DIR / "MSLesSeg"
MODALITIES = ["FLAIR", "T1", "T2"]
# Lesions are connected components of the mask. We use 6-connectivity (voxels belong to the
# same lesion only if they share a face) because it reproduces exactly the lesion counts
# reported by the MSLesSeg authors (see compare_connectivity).
CONNECTIVITY = 6
_RANK = {6: 1, 18: 2, 26: 3}
STRUCTURE = ndimage.generate_binary_structure(3, _RANK[CONNECTIVITY])


def list_studies(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Return one row per study with its split, patient, timepoint and folder."""
    rows = []
    for d in sorted((data_dir / "train").glob("P*/T*")):
        rows.append({"split": "train", "patient": d.parent.name, "timepoint": d.name, "path": d})
    for d in sorted((data_dir / "test").glob("P*")):
        rows.append({"split": "test", "patient": d.name, "timepoint": "T1", "path": d})
    df = pd.DataFrame(rows)
    df["study_id"] = df["patient"] + "_" + df["timepoint"]
    return df


def load_volume(study_dir: Path, modality: str) -> np.ndarray:
    f = next(study_dir.glob(f"*_{modality}.nii.gz"))
    return nib.load(f).get_fdata(dtype=np.float32)


def study_stats(study_dir: Path) -> tuple[dict, list[dict]]:
    """Statistics for one study and a list with the size of each lesion."""
    mask = load_volume(study_dir, "MASK") > 0.5
    flair = load_volume(study_dir, "FLAIR")
    brain = flair > 0  # images are skull-stripped: background is exactly 0

    labels, n_lesions = ndimage.label(mask, structure=STRUCTURE)
    sizes = np.bincount(labels.ravel())[1:] if n_lesions else np.array([], dtype=int)

    stats = {
        "brain_voxels": int(brain.sum()),
        "lesion_voxels": int(mask.sum()),
        "lesion_fraction_pct": 100 * mask.sum() / max(brain.sum(), 1),
        "n_lesions": int(n_lesions),
        "median_lesion_size": float(np.median(sizes)) if n_lesions else 0.0,
        "max_lesion_size": int(sizes.max()) if n_lesions else 0,
    }
    for mod in MODALITIES:
        img = flair if mod == "FLAIR" else load_volume(study_dir, mod)
        vals = img[brain]
        p1, p50, p99 = np.percentile(vals, [1, 50, 99])
        stats.update({f"{mod}_p1": p1, f"{mod}_p50": p50, f"{mod}_p99": p99})
        if n_lesions:
            stats[f"{mod}_lesion_median"] = float(np.median(img[mask]))
    lesions = [{"lesion_id": i + 1, "size_voxels": int(s)} for i, s in enumerate(sizes)]
    return stats, lesions


def compute_all(out_dir: Path = INTERIM_DATA_DIR, overwrite: bool = False) -> None:
    """Compute statistics for every study, caching each one so the run can be resumed."""
    cache = out_dir / f"eda_cache_c{CONNECTIVITY}"
    cache.mkdir(parents=True, exist_ok=True)
    studies = list_studies()
    for _, s in studies.iterrows():
        f = cache / f"{s.study_id}.csv"
        if f.exists() and not overwrite:
            continue
        stats, lesions = study_stats(s.path)
        pd.DataFrame([stats]).to_csv(f, index=False)
        pd.DataFrame(lesions).to_csv(cache / f"{s.study_id}_lesions.csv", index=False)

    study_rows, lesion_rows = [], []
    for _, s in studies.iterrows():
        st = pd.read_csv(cache / f"{s.study_id}.csv").iloc[0].to_dict()
        study_rows.append({**s.drop("path").to_dict(), **st})
        les = pd.read_csv(cache / f"{s.study_id}_lesions.csv") if st["n_lesions"] else pd.DataFrame()
        for _, l in les.iterrows():
            lesion_rows.append({"split": s.split, "study_id": s.study_id, **l.to_dict()})
    pd.DataFrame(study_rows).to_csv(out_dir / "study_stats.csv", index=False)
    pd.DataFrame(lesion_rows).to_csv(out_dir / "lesions.csv", index=False)


def load_clinical(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Clinical data (age, sex, MS type, EDSS) plus the lesion volume/number reported by the authors.

    The CSV header is shifted by one column: the "Lesion Number" column is empty and the
    lesion count is stored in an unnamed last column, so columns are renamed explicitly.
    """
    df = pd.read_csv(data_dir / "info_dataset/clinical_data.csv", sep=";", encoding="utf-8-sig",
                     decimal=",")
    df.columns = ["patient", "timepoint", "age", "sex", "ms_type", "edss", "_empty",
                  "reported_lesion_volume", "reported_lesion_number"]
    df = df.drop(columns="_empty")
    df["reported_lesion_volume"] = pd.to_numeric(df["reported_lesion_volume"], errors="coerce")
    df["ms_type"] = df["ms_type"].map({"SMRR": "RRMS", "SMSP": "SPMS", "SMPP": "PPMS"})
    return df


def load_scanners(data_dir: Path = DATA_DIR, modality: str = "FLAIR") -> pd.DataFrame:
    f = data_dir / f"info_dataset/patient_scanners_info/patient_scanners_info_{modality}.csv"
    df = pd.read_csv(f, sep=";", encoding="utf-8-sig")
    df = df.rename(columns={"Patient": "patient", "Timepoint": "timepoint"})
    df["manufacturer"] = df["MANUFACTURER"].str.split().str[0].str.upper().replace("NONE", "unknown")
    df["scanner"] = (df["manufacturer"] + " " + df["SCANNER"].astype(str)).replace("unknown NONE", "unknown")
    return df[["patient", "timepoint", "manufacturer", "scanner", "SliceThickness[mm]"]]


def compare_connectivity(out_dir: Path = INTERIM_DATA_DIR, min_sizes=(1, 2, 3, 5, 10)) -> pd.DataFrame:
    """Lesion counts per study for 6-, 18- and 26-connectivity and several minimum sizes.

    Used to find which counting rule reproduces the lesion numbers reported by the authors.
    Results are cached in ``connectivity_counts.csv``.
    """
    f = out_dir / "connectivity_counts.csv"
    if f.exists():
        return pd.read_csv(f)
    rows = []
    for _, s in list_studies().iterrows():
        mask = load_volume(s.path, "MASK") > 0.5
        row = {"study_id": s.study_id, "split": s.split}
        for conn, rank in _RANK.items():
            labels, n = ndimage.label(mask, structure=ndimage.generate_binary_structure(3, rank))
            sizes = np.bincount(labels.ravel())[1:]
            for m in min_sizes:
                row[f"c{conn}_min{m}"] = int((sizes >= m).sum())
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(f, index=False)
    return df


def log_summary_to_mlflow() -> None:
    """Log the main dataset statistics as an MLflow run (experiment "eda")."""
    from ms_seg.tracking import start_run

    studies = pd.read_csv(INTERIM_DATA_DIR / "study_stats.csv")
    lesions = pd.read_csv(INTERIM_DATA_DIR / "lesions.csv")
    sizes = lesions.size_voxels
    with start_run("eda", run_name="dataset-statistics", params={"connectivity": CONNECTIVITY}):
        mlflow_metrics = {
            "n_studies": len(studies),
            "n_patients": studies.patient.nunique(),
            "n_lesions": len(lesions),
            "median_lesion_fraction_pct": float(studies.lesion_fraction_pct.median()),
            "median_lesion_size_mm3": float(sizes.median()),
            "pct_lesions_le_100mm3": float(100 * (sizes <= 100).mean()),
            "pct_volume_in_lesions_le_100mm3": float(100 * sizes[sizes <= 100].sum() / sizes.sum()),
        }
        import mlflow

        mlflow.log_metrics(mlflow_metrics)
        mlflow.log_artifact(str(INTERIM_DATA_DIR / "study_stats.csv"))


if __name__ == "__main__":
    compute_all()
    log_summary_to_mlflow()
