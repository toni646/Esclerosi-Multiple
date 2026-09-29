"""Evaluation metrics for MS lesion segmentation (issue #7).

Same seven metrics as the reference paper (Guarnera et al., 2025, Table 5):

Voxel-wise
    DSC   Dice similarity coefficient      2|P∩G| / (|P|+|G|)                 higher is better
    TPR   true positive rate (sensitivity) |P∩G| / |G|                         higher is better
    PPV   positive predictive value        |P∩G| / |P|                         higher is better
    AVD   absolute volume difference (%)   100 · | |P| − |G| | / |G|           lower is better
    ASSD  average symmetric surface distance (mm)                              lower is better

Lesion-wise (lesions = connected components with 6-connectivity, the rule that
reproduces the paper's lesion counts, see notebooks/2.0-tlb-eda.ipynb §3.3)
    LTPR  reference lesions overlapped by the prediction / reference lesions     higher is better
    LFPR  predicted lesions not overlapping the reference / predicted lesions   lower is better

A lesion "overlaps" if it shares at least one voxel with the other mask.
P = predicted mask, G = ground-truth (reference) mask; both binary 3D arrays.
Metrics are computed per study and then averaged.
"""

import numpy as np
from scipy import ndimage

CONNECTIVITY_STRUCTURE = ndimage.generate_binary_structure(3, 1)  # 6-connectivity


def _binary(x) -> np.ndarray:
    return np.asarray(x) > 0


def dice(pred, gt) -> float:
    p, g = _binary(pred), _binary(gt)
    denom = p.sum() + g.sum()
    return 1.0 if denom == 0 else 2.0 * (p & g).sum() / denom


def tpr(pred, gt) -> float:
    p, g = _binary(pred), _binary(gt)
    return np.nan if g.sum() == 0 else (p & g).sum() / g.sum()


def ppv(pred, gt) -> float:
    p, g = _binary(pred), _binary(gt)
    return np.nan if p.sum() == 0 else (p & g).sum() / p.sum()


def avd(pred, gt, voxel_volume: float = 1.0) -> float:
    """Absolute volume difference in % of the reference volume."""
    vp, vg = _binary(pred).sum() * voxel_volume, _binary(gt).sum() * voxel_volume
    return np.nan if vg == 0 else 100.0 * abs(vp - vg) / vg


def _surface(mask: np.ndarray) -> np.ndarray:
    """Boundary voxels: mask voxels with at least one 6-neighbour outside the mask."""
    return mask & ~ndimage.binary_erosion(mask, structure=CONNECTIVITY_STRUCTURE, border_value=0)


def assd(pred, gt, spacing=(1.0, 1.0, 1.0)) -> float:
    """Average symmetric surface distance in mm (NaN if either mask is empty)."""
    p, g = _binary(pred), _binary(gt)
    if p.sum() == 0 or g.sum() == 0:
        return np.nan
    sp, sg = _surface(p), _surface(g)
    dist_to_g = ndimage.distance_transform_edt(~sg, sampling=spacing)
    dist_to_p = ndimage.distance_transform_edt(~sp, sampling=spacing)
    d = np.concatenate([dist_to_g[sp], dist_to_p[sg]])
    return float(d.mean())


def label_lesions(mask) -> tuple[np.ndarray, int]:
    """Split a binary mask into lesions (6-connected components)."""
    return ndimage.label(_binary(mask), structure=CONNECTIVITY_STRUCTURE)


def _overlapping_fraction(a, b) -> tuple[int, int]:
    """(# lesions of a that overlap b, # lesions of a)."""
    labels, n = label_lesions(a)
    if n == 0:
        return 0, 0
    hit = np.unique(labels[_binary(b) & (labels > 0)])
    return len(hit), n


def ltpr(pred, gt) -> float:
    hit, n = _overlapping_fraction(gt, pred)
    return np.nan if n == 0 else hit / n


def lfpr(pred, gt) -> float:
    hit, n = _overlapping_fraction(pred, gt)
    return np.nan if n == 0 else (n - hit) / n


METRICS = {"DSC": dice, "TPR": tpr, "PPV": ppv, "LTPR": ltpr, "LFPR": lfpr, "AVD": avd, "ASSD": assd}


def evaluate(pred, gt, spacing=(1.0, 1.0, 1.0)) -> dict:
    """All seven metrics for one study, in the order of the paper's Table 5."""
    out = {name: fn(pred, gt) for name, fn in METRICS.items() if name not in ("AVD", "ASSD")}
    out["AVD"] = avd(pred, gt, voxel_volume=float(np.prod(spacing)))
    out["ASSD"] = assd(pred, gt, spacing)
    return {k: float(v) for k, v in out.items()}
