# 5. Experimental framework

This chapter describes the common framework shared by all experiments: the tools that make the work reproducible, the cross-validation folds, the evaluation metrics and the analyses that will be used to compare models. Sections 5.1–5.3 are implemented; Sections 5.4 and 5.5 describe the planned analysis.

## 5.1 Reproducible workflow

The project follows the principle that every result must be regenerated from the public repository with documented commands (objective O5). Table 5.1 lists the tools used.

| Need | Tool | Use in this project |
|---|---|---|
| Project structure | Cookiecutter Data Science | Standard folders: `data/`, `ms_seg/` (code), `notebooks/`, `reports/`, `configs/`, `tests/` |
| Code versioning and planning | Git and GitHub | Public repository; work organised in five milestones and issues on a project board |
| Software environment | conda (`environment.yml`) | Same Python version and libraries on every machine |
| Data versioning and pipeline | DVC | The raw dataset is versioned by its checksum (`MSLesSeg.dvc`); `dvc repro` re-runs only the stages whose inputs changed |
| Experiment tracking | MLflow | Parameters, metrics, artefacts and code version of every run |
| Testing | pytest | Automatic tests for the folds and the metrics |
| Computing | Kaggle notebooks | Free GPU (about 30 hours per week) for training |

*Table 5.1. Tools of the reproducible workflow.*

The data are not stored in Git: DVC keeps a small pointer file with the checksum of the dataset, so that each commit records exactly which version of the data was used. The pipeline currently has two stages, defined in `dvc.yaml`:

- **eda**: computes the statistics of every study and every lesion (`data/interim/study_stats.csv`, `lesions.csv`) used in Chapter 4.
- **folds**: builds the cross-validation folds from these statistics (`configs/folds.json`, Section 5.2).

## 5.2 Cross-validation folds

Models are developed with **5-fold cross-validation** on the 53 training patients: each fold uses about four fifths of the patients for training and the remaining fifth for validation, and every patient is used for validation exactly once. The test set (P54–P75) is used only for the final comparison of the selected models.

**Folds used in the reference study.** Guarnera et al. [7] split the training patients into contiguous blocks (Table 5.2). Their table contains two typos: patient 23 appears in both the training and the validation set of fold 4 (data leakage), and patients 11–13 are never in a validation set. Moreover, once corrected, the blocks are very unbalanced, because the first patients have more timepoints: the validation sets would contain between 10 and 36 studies.

| Fold | Training patients | Validation patients | Issue |
|---|---|---|---|
| 1 | 1–43 | 44–53 | — |
| 2 | 1–33, 44–53 | 34–43 | — |
| 3 | 1–23, 34–53 | 24–33 | — |
| 4 | 1–13, 23–53 | 14–23 | Patient 23 in training and validation |
| 5 | 11–53 | 1–10 | Patients 11–13 never validated |

*Table 5.2. Cross-validation folds reported by Guarnera et al. [7] (Table 4 of the original publication).*

**Folds used in this work.** New folds were built with two rules:

- **Patient level:** all timepoints of a patient belong to the same fold, so no patient appears in both training and validation.
- **Balanced:** patients are sorted by number of studies and lesion volume (largest first), and each one is assigned to the fold that currently has the fewest studies (ties: least lesion volume). The procedure is deterministic.

The resulting folds are balanced in number of studies and lesion volume (Table 5.3). The implementation (`ms_seg/folds.py`) checks automatically that no patient is in two folds and that every training patient is validated once.

| Fold | Patients | Studies | Lesion volume (ml) |
|---|---|---|---|
| 1 | 11 | 19 | 230.9 |
| 2 | 11 | 19 | 214.0 |
| 3 | 11 | 19 | 214.7 |
| 4 | 10 | 18 | 241.4 |
| 5 | 10 | 18 | 230.6 |

*Table 5.3. Validation sets of the five folds used in this work (`configs/folds.json`).*

Because the reference study evaluates its models on the test set, which is fixed, using different cross-validation folds does not prevent the comparison of test results.

## 5.3 Evaluation metrics

Models are evaluated with the seven metrics of the reference study [7], implemented in `ms_seg/metrics.py`. Let P be the predicted mask and G the reference (ground-truth) mask of a study, both binary, and |·| the number of voxels.

**Voxel-wise metrics:**

- **Dice similarity coefficient:** DSC = 2 |P ∩ G| / (|P| + |G|). Overall overlap (higher is better).
- **True positive rate** (sensitivity): TPR = |P ∩ G| / |G|. Fraction of the lesion volume that is detected (higher is better).
- **Positive predictive value** (precision): PPV = |P ∩ G| / |P|. Fraction of the predicted volume that is lesion (higher is better).
- **Absolute volume difference:** AVD = 100 · | |P| − |G| | / |G|, in % of the reference volume (lower is better).
- **Average symmetric surface distance** (ASSD, mm): mean distance from each boundary voxel of P to the boundary of G and from each boundary voxel of G to the boundary of P (lower is better).

**Lesion-wise metrics.** Lesions are the connected components of each mask with **6-connectivity**, the rule that reproduces the authors' lesion counts (Section 4.3.3). A lesion is detected if it shares at least one voxel with the other mask.

- **Lesion true positive rate:** LTPR = reference lesions overlapped by P / reference lesions (higher is better).
- **Lesion false positive rate:** LFPR = predicted lesions that do not overlap G / predicted lesions (lower is better).

Metrics are computed per study and averaged over studies. The implementation was verified with automatic tests on synthetic masks with known results (for example, two cubes overlapping by half give DSC = 0.5, and two voxels touching only at a corner are counted as two lesions) and on the real masks: comparing each mask with itself gives perfect scores, and the lesion counts match the values released by the authors.

## 5.4 Statistical comparison (planned)

Mean metrics alone are not enough to decide whether one model is better than another: in the reference study, the two best models differ by 0.007 in Dice, less than the variation between folds. Models will therefore be compared **per patient** with the **Wilcoxon signed-rank test** for paired samples, and differences will be reported with 95 % bootstrap confidence intervals. The test set is used only once, for the final models.

## 5.5 Stratified and sensitivity analyses (planned)

The exploratory analysis showed that a global mean hides important differences. In addition to the overall metrics, results will be reported:

- **by lesion size** (≤ 10, 11–100, 101–1,000 and > 1,000 mm³), using lesion-wise detection rates, because small lesions are the majority but barely affect the Dice coefficient;
- **by lesion load** of the study (low, medium, high), because metrics are expected to be lower for studies with little lesion;
- **by scanner manufacturer** (Philips versus others), because the test set contains many more non-Philips studies than the training set;
- **without the four flagged studies** (P44_T1, P45_T1, P53_T1 and P49_T2), as a sensitivity analysis of the data-quality issues found in Section 4.3.
