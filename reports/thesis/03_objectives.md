# 3. Objectives

## 3.1 Research question

The open questions identified in Section 2.3 lead to the research question of this thesis:

*Starting from a strong, reproducible baseline, can the detection of small multiple sclerosis lesions be improved and the number of false-positive lesions reduced, and how robust are these models across scanners and datasets?*

## 3.2 General objective

The general objective is to develop and rigorously evaluate a deep-learning pipeline for MS lesion segmentation on the MSLesSeg dataset that improves the detection of small lesions with respect to a strong baseline, while remaining fully reproducible.

## 3.3 Specific objectives

The general objective is divided into five specific objectives, each building on the previous one (Table 3.1).

| | Objective | Success criterion |
|---|---|---|
| O1 | Build a reproducible training and evaluation pipeline and **reproduce the reference benchmark** with SwinUNETR [11]. | Voxel-wise and lesion-wise metrics of the same order as those reported in [7]. |
| O2 | Train **nnU-Net** [16] as a strong baseline, a comparison missing from the reference study. | Complete cross-validation and test-set results with the same metrics as [7]. |
| O3 | Improve the **detection of small lesions and reduce false-positive lesions** through targeted modifications of the baseline (e.g. lesion-aware loss functions, post-processing, uncertainty estimation). | Statistically significant improvement in LTPR and/or LFPR over the baseline without a significant loss of Dice. |
| O4 | Assess **generalisation and robustness**: performance by scanner manufacturer and on an external dataset without retraining. | Quantified performance gap between domains and analysis of its likely causes. |
| O5 | Ensure **reproducibility** by versioning code, data and experiments. | All results can be regenerated from the public repository with documented commands. |

*Table 3.1. Specific objectives and success criteria.*

## 3.4 Scope

To keep the work focused, the following are outside the scope of this thesis:

- **Designing a new architecture from scratch.** The contribution consists of targeted, measurable modifications of an established architecture.
- **Clinical deployment.** The pipeline is a research tool and has not been validated for clinical use.
- **Longitudinal analysis.** Detection of new or enlarging lesions between visits is left as future work, although the dataset contains longitudinal data.

## 3.5 Expected contributions

- The first evaluation of **nnU-Net on MSLesSeg**, providing a strong reference for future work.
- A **measured improvement in small-lesion detection**, supported by paired statistical tests.
- An analysis of performance **by lesion size and by scanner manufacturer**, which the reference study does not report.
- A **reproducible, public pipeline**, including the patient-level cross-validation folds.
- Findings from the exploratory analysis of the dataset: the identification of the **lesion counting rule used by the authors** (6-connectivity) and of **studies with data-quality issues** not reported in the original publication (Chapter 5).

## 3.6 Evaluation principles

All experiments follow the same principles, so that results are comparable and conclusions are sound:

- **Patient-level cross-validation:** timepoints of the same patient never appear in both training and validation.
- **Same metrics as the reference study**, computed with the same lesion counting rule, complemented by lesion-wise metrics stratified by lesion size.
- **Paired statistical tests per patient** (Wilcoxon signed-rank test) and confidence intervals to compare models, instead of comparing means alone.
- **The test set is used only once**, for the final comparison of the selected models.
- **Every experiment is logged** (configuration, code version and metrics) and linked to an issue in the project repository.

## 3.7 Work plan

The work is organised in five milestones (Table 3.2), tracked as milestones and issues in the project repository.

| Milestone | Due date | Main outputs |
|---|---|---|
| M1 · Data and exploratory analysis | 11 Oct 2026 | Environment, dataset verification, exploratory analysis, patient-level folds |
| M2 · Reproduction and nnU-Net baseline | 1 Nov 2026 | Metrics, SwinUNETR reproduction, nnU-Net baseline, statistical comparison |
| M3 · Own contribution | 6 Dec 2026 | Lesion-aware loss, post-processing, uncertainty estimation |
| M4 · External validation and analysis | 27 Dec 2026 | External dataset, robustness by scanner, error analysis |
| M5 · Thesis and defence | 24 Jan 2027 | Written thesis, reproducible repository, presentation |

*Table 3.2. Work plan.*
