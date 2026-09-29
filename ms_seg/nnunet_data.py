"""Convert MSLesSeg to the nnU-Net v2 raw-data format and write our cross-validation folds.

Output (under data/processed/, not tracked by Git):

    nnUNet_raw/Dataset101_MSLesSeg/
        dataset.json
        imagesTr/P1_T1_0000.nii.gz   FLAIR   (channel 0)
                 P1_T1_0001.nii.gz   T1-w    (channel 1)
                 P1_T1_0002.nii.gz   T2-w    (channel 2)
        labelsTr/P1_T1.nii.gz        lesion mask (uint8: 0 background, 1 lesion)
        imagesTs/P54_0000.nii.gz ... test images (same channels)
        labelsTs/P54.nii.gz          test masks (only for our own evaluation)
    nnUNet_preprocessed/Dataset101_MSLesSeg/splits_final.json
        our patient-level, balanced 5 folds (configs/folds.json) instead of
        nnU-Net's default random split by case, which would put visits of the
        same patient in training and validation (data leakage).

Images are copied unchanged (nnU-Net does its own z-score normalisation per image
for MRI channels). Masks are stored as uint8. The script checks that the four
volumes of every study share the same shape and affine, and it can be re-run:
files that already exist are skipped.

Run:  python -m ms_seg.nnunet_data
"""

import json
import shutil

import nibabel as nib
import numpy as np
from loguru import logger
from tqdm import tqdm

from ms_seg.config import PROCESSED_DATA_DIR, PROJ_ROOT, RAW_DATA_DIR

DATA = RAW_DATA_DIR / "MSLesSeg"
DATASET_NAME = "Dataset101_MSLesSeg"
RAW_OUT = PROCESSED_DATA_DIR / "nnUNet_raw" / DATASET_NAME
PREP_OUT = PROCESSED_DATA_DIR / "nnUNet_preprocessed" / DATASET_NAME
FOLDS_FILE = PROJ_ROOT / "configs" / "folds.json"
CHANNELS = {"FLAIR": "0000", "T1": "0001", "T2": "0002"}


def studies():
    """Yield (subset, case_id, study_dir) for all 115 studies."""
    for d in sorted(DATA.glob("train/P*/T*"), key=lambda p: (int(p.parent.name[1:]), p.name)):
        yield "Tr", f"{d.parent.name}_{d.name}", d
    for d in sorted(DATA.glob("test/P*"), key=lambda p: int(p.name[1:])):
        yield "Ts", d.name, d


def convert_study(subset: str, case: str, study_dir, overwrite: bool = False) -> None:
    files = {m: next(study_dir.glob(f"*_{m}.nii.gz")) for m in [*CHANNELS, "MASK"]}
    ref = nib.load(files["MASK"])
    for mod in CHANNELS:
        img = nib.load(files[mod])
        if img.shape != ref.shape or not np.allclose(img.affine, ref.affine, atol=1e-4):
            raise ValueError(f"{case}: {mod} geometry differs from the mask")
    for mod, ch in CHANNELS.items():
        dst = RAW_OUT / f"images{subset}" / f"{case}_{ch}.nii.gz"
        if overwrite or not dst.exists():
            tmp = dst.with_name("tmp_" + dst.name)
            shutil.copyfile(files[mod], tmp)
            tmp.replace(dst)  # atomic: an interrupted run never leaves a half-written file
    dst = RAW_OUT / f"labels{subset}" / f"{case}.nii.gz"
    if overwrite or not dst.exists():
        mask = (np.asanyarray(ref.dataobj) > 0).astype(np.uint8)
        out = nib.Nifti1Image(mask, ref.affine, ref.header)
        out.set_data_dtype(np.uint8)
        tmp = dst.with_name("tmp_" + dst.name)
        nib.save(out, tmp)
        tmp.replace(dst)


def write_dataset_json(n_train: int) -> None:
    meta = {
        "name": "MSLesSeg",
        "description": "Multiple sclerosis lesion segmentation (Guarnera et al., Sci Data 2025)",
        "channel_names": {"0": "FLAIR", "1": "T1", "2": "T2"},  # non-CT names -> z-score per image
        "labels": {"background": 0, "lesion": 1},
        "numTraining": n_train,
        "file_ending": ".nii.gz",
    }
    (RAW_OUT / "dataset.json").write_text(json.dumps(meta, indent=2))


def write_splits() -> list:
    folds = json.loads(FOLDS_FILE.read_text())["folds"]
    splits = [{"train": f["train"], "val": f["val"]} for f in sorted(folds, key=lambda f: f["fold"])]
    PREP_OUT.mkdir(parents=True, exist_ok=True)
    (PREP_OUT / "splits_final.json").write_text(json.dumps(splits, indent=2))
    return splits


def check_output(splits) -> None:
    train_cases = sorted(p.name[: -len(".nii.gz")] for p in (RAW_OUT / "labelsTr").glob("*.nii.gz"))
    for case in train_cases:
        for ch in CHANNELS.values():
            assert (RAW_OUT / "imagesTr" / f"{case}_{ch}.nii.gz").exists(), f"missing {case}_{ch}"
    assert len(train_cases) == 93, len(train_cases)
    assert len(list((RAW_OUT / "labelsTs").glob("*.nii.gz"))) == 22
    for s in splits:  # every case in exactly one validation set, no patient overlap
        tr = {c.split("_")[0] for c in s["train"]}
        va = {c.split("_")[0] for c in s["val"]}
        assert not tr & va, "patient in train and val"
    assert sorted(c for s in splits for c in s["val"]) == sorted(train_cases)


def main() -> None:
    for sub in ["imagesTr", "labelsTr", "imagesTs", "labelsTs"]:
        (RAW_OUT / sub).mkdir(parents=True, exist_ok=True)
    all_studies = list(studies())
    for subset, case, d in tqdm(all_studies, desc="converting"):
        convert_study(subset, case, d)
    write_dataset_json(sum(s[0] == "Tr" for s in all_studies))
    splits = write_splits()
    check_output(splits)
    logger.success(f"nnU-Net dataset ready in {RAW_OUT} ({len(all_studies)} studies, {len(splits)} folds)")


if __name__ == "__main__":
    main()
