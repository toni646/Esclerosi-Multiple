import numpy as np
import pytest

from ms_seg.metrics import assd, avd, dice, evaluate, lfpr, ltpr, ppv, tpr


def cube(shape=(20, 20, 20), start=(5, 5, 5), size=4):
    m = np.zeros(shape, bool)
    x, y, z = start
    m[x:x + size, y:y + size, z:z + size] = True
    return m


def test_perfect_prediction():
    g = cube()
    r = evaluate(g, g)
    assert r["DSC"] == r["TPR"] == r["PPV"] == r["LTPR"] == 1.0
    assert r["LFPR"] == 0.0 and r["AVD"] == 0.0 and r["ASSD"] == 0.0


def test_half_overlap():
    g = cube(size=4)                      # 64 voxels
    p = cube(start=(7, 5, 5), size=4)     # shifted by 2 -> 32 shared voxels
    assert dice(p, g) == pytest.approx(0.5)
    assert tpr(p, g) == pytest.approx(0.5)
    assert ppv(p, g) == pytest.approx(0.5)
    assert avd(p, g) == 0.0
    assert assd(p, g) > 0


def test_lesion_wise_and_six_connectivity():
    g = cube(size=3) | cube(start=(14, 14, 14), size=3)   # 2 reference lesions
    p = cube(size=3) | cube(start=(1, 1, 1), size=2)      # hits lesion 1, plus a separate FP lesion
    p[0, 0, 0] = True                                     # touches (1,1,1) only by a corner -> own lesion
    assert ltpr(p, g) == pytest.approx(0.5)               # 1 of 2 reference lesions found
    # predicted lesions: big cube, small cube (touches big one? no: 1..2 vs 5..7), corner voxel -> 3 lesions, 2 FP
    assert lfpr(p, g) == pytest.approx(2 / 3)


def test_empty_masks():
    e = np.zeros((10, 10, 10), bool)
    assert dice(e, e) == 1.0
    assert np.isnan(assd(e, cube((10, 10, 10), (2, 2, 2), 3)))
