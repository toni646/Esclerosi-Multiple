# 2. Starting point: the MSLesSeg study

This thesis builds on the dataset and benchmark published by Guarnera et al. [7]. This chapter summarises what the study provides, how its baseline models were evaluated, and the open questions it leaves, which motivate the objectives of this work (Chapter 3).

## 2.1 The MSLesSeg dataset

MSLesSeg is a public dataset for MS lesion segmentation released on figshare [8]. It contains MRI data from **75 patients** (48 female, 27 male; age 18–59 years, mean 37 ± 10.3 years) acquired at an Italian hospital and private clinics on **1.5 T and 3 T scanners**. Some patients were scanned at up to four timepoints, giving **115 studies** in total. Each study includes three sequences (**FLAIR, T1-weighted and T2-weighted**) and a binary lesion mask.

All images were preprocessed with the same pipeline: conversion from DICOM to NIfTI, affine registration (12 degrees of freedom) to the MNI152 1 mm template with FLIRT [9], and skull stripping with BET [10]. The released volumes are 182 × 218 × 182 voxels of 1 mm³. No intensity normalisation is described.

Lesion masks were drawn by a junior rater trained by two senior experts (a neuroradiologist and a neurologist specialised in MS). Lesions were identified on FLAIR, with T1-w and T2-w used to resolve doubtful cases, and the masks were reviewed in periodic meetings with the senior raters. Unlike datasets that provide several annotations per image, such as ISBI 2015 [5], MSLesSeg provides **a single expert-validated mask per study**.

The dataset is divided into a **training set of 53 patients** (93 studies, one to four timepoints per patient) and a **test set of 22 patients** (one study each). It was also used in the ICPR 2024 Competition on Multiple Sclerosis Lesion Segmentation.

## 2.2 Baseline models and evaluation

The authors benchmarked four 3D deep-learning models: their own diffusion-based model, **MSSegDiff** [7, 15], which iteratively denoises a noisy mask conditioned on the MRI; **SwinUNETR** [11]; **UNETR** [12]; and **TransBTS** [13]. Models were trained with 5-fold cross-validation on the training patients and evaluated with voxel-wise metrics (Dice similarity coefficient, true positive rate and positive predictive value), lesion-wise metrics (lesion true positive rate, LTPR, and lesion false positive rate, LFPR), the absolute volume difference (AVD) and the average symmetric surface distance (ASSD). Table 2.1 summarises the reported results.

| Model | Dice ↑ | TPR ↑ | PPV ↑ | LTPR ↑ | LFPR ↓ | AVD ↓ | ASSD ↓ |
|---|---|---|---|---|---|---|---|
| MSSegDiff | **0.685** | 0.672 | **0.724** | 0.625 | **0.223** | **24.5** | **3.47** |
| SwinUNETR | 0.679 | 0.668 | 0.716 | 0.702 | 0.296 | 25.1 | 3.62 |
| UNETR | 0.642 | **0.675** | 0.650 | **0.733** | 0.433 | 90.2 | 4.52 |
| TransBTS | 0.492 | 0.545 | 0.503 | 0.511 | 0.541 | 68.0 | 7.55 |

*Table 2.1. Results reported by Guarnera et al. [7] (mean over five folds). Best value per column in bold.*

In addition, the authors compared the masks submitted to the ICPR 2024 competition with a consensus mask obtained with STAPLE [14]. The automatic methods made similar errors, mainly on the contours of large or unusual lesions, and errors were larger for methods that used only FLAIR.

## 2.3 Open questions

The study provides a valuable public resource, but several aspects of its benchmark leave room for further work:

- **The best model wins by a small margin.** MSSegDiff and SwinUNETR differ by 0.007 in mean Dice, a difference within the variation between folds, and no statistical test is reported.
- **Voxel-wise and lesion-wise metrics disagree.** MSSegDiff has the highest Dice but the lowest LTPR of the three competitive models (0.625 versus 0.702 and 0.733): it detects fewer lesions. Because the Dice coefficient is dominated by large lesions, a model can obtain a good Dice while missing many small ones.
- **No nnU-Net baseline.** nnU-Net [16], the de-facto reference method in medical image segmentation, is not included in the comparison.
- **Limited reproducibility.** Training hyper-parameters and the assignment of patients to folds are not reported, the released code [15] targets a different dataset (ISBI 2015), and the rule used to count lesions (which determines LTPR and LFPR) is not described. The paper also mentions test patients 54–77, whereas the released test set contains patients 54–75.
- **Acquisition heterogeneity is not analysed.** Scanner information is released with the data, but performance is not stratified by scanner.

These observations define the starting point of this thesis: a strong reference model, a focus on the detection of small lesions, rigorous statistical comparison, full reproducibility and an analysis of robustness across scanners.
