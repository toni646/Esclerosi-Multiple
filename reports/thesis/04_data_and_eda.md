# 4. Data and exploratory analysis

Before training any model, the whole MSLesSeg dataset was analysed to understand its content, to check its quality and to derive the design decisions of the following chapters. Every figure and number in this chapter is produced by `ms_seg/eda.py` and the notebook `notebooks/2.0-tlb-eda.ipynb`, and can be regenerated with a single command (`dvc repro`, Section 5.1).

The analysis is organised around four questions: what counts as a lesion (Section 4.3.3), what the data look like (Sections 4.1, 4.2 and 4.4–4.8), whether the data can be trusted (Section 4.3) and what all this implies for the models (Section 4.9).

## 4.1 Dataset content

Each study (one visit of one patient) contains three co-registered MRI sequences (FLAIR, T1-w and T2-w) and one binary lesion mask, all with 182 × 218 × 182 voxels of 1 mm³ in the MNI152 space (Figure 4.1). Table 4.1 summarises the size of the two official subsets.

| | Patients | Studies | Lesions |
|---|---|---|---|
| Training set | 53 | 93 | 2,688 |
| Test set | 22 | 22 | 907 |
| Total | 75 | 115 | 3,595 |

*Table 4.1. Size of the MSLesSeg subsets. Lesions are counted as 6-connected components (Section 4.3.3).*

In the training set, 28 patients have one timepoint, 15 have two, 5 have three and 5 have four; test patients have a single timepoint. There is **one mask per study**: it was drawn by a junior rater and reviewed and validated by two senior experts (Section 2.1), so it is a consensus annotation, but inter-rater variability cannot be measured on this dataset.

![Figure 4.1. Study P1_T1: the three MRI sequences and the lesion mask (red) on the axial slice with the most lesion.](first_look_P1_T1_axial.png)

**Implication.** With 93 training studies the dataset is small for deep learning. This supports building on a well-established, self-configuring architecture with extensive data augmentation (nnU-Net [16]) rather than designing a new one. Because several studies belong to the same patient, all data splits must be made **at patient level** (Section 5.2).

## 4.2 Patients

Table 4.2 compares the demographics of the two subsets (first visit of each patient).

| | Training (53) | Test (22) |
|---|---|---|
| Age (years), mean ± SD | 36.4 ± 10.2 | 38.5 ± 10.5 |
| Female | 32 (60 %) | 16 (73 %) |
| MS type | 50 RRMS, 3 SPMS | 21 RRMS, 1 PPMS |
| EDSS, median [IQR] | 2.0 [0.0–3.0] | 1.0 [0.25–3.0] |

*Table 4.2. Demographics of the training and test patients. RRMS: relapsing-remitting; SPMS: secondary progressive; PPMS: primary progressive; EDSS: Expanded Disability Status Scale (0 = normal, 10 = death due to MS).*

The two subsets are comparable. The cohort is dominated by relapsing-remitting MS with mild disability, and the female majority is consistent with the known prevalence of the disease.

**Implication.** The test set is representative of the training population. As a limitation, models trained on this dataset may not generalise to progressive forms or to more severe disease, which are barely represented.

## 4.3 Data quality checks

### 4.3.1 Registration

After registration to MNI152 all brains should have a similar volume. The brain volume (non-zero FLAIR voxels) has a median of 1.90 million voxels, and 112 of the 115 studies lie between 1.63 and 2.33 million (Figure 4.2). Three studies, **P44_T1, P45_T1 and P53_T1**, have brains 1.73–1.84 times larger than the median and are cropped at the borders of the image: their registration failed to scale them to the template. In addition, the **T1-w image of P49_T2 is empty** (all voxels are zero).

![Figure 4.2. Brain volume of each study after registration. Three studies are far above the rest (dashed line: 1.4 × median).](eda_brain_volume.png)

### 4.3.2 Are the flagged studies usable?

For the four flagged studies the lesion masks were checked against their own images. No lesion voxel lies outside the brain, and the median FLAIR intensity inside the lesions is 1.29–1.54 times that of the whole brain, within the range of the other studies, as expected for MS lesions. The masks are therefore consistent with the images; the badly registered studies are only at a different scale.

The reference study does not mention excluded studies, missing images or failed registrations, and its cross-validation covers every training patient from 1 to 53 [7], so these studies were almost certainly used. They are **kept** for comparability, and a **sensitivity analysis** without them will be reported (Section 5.5).

The check also revealed two inconsistencies in the reference study. First, its technical validation mentions 24 test patients (P54–P77), whereas the dataset description and the released data contain 22 (P54–P75); this work uses the released split. Second, its table of cross-validation folds contains typos (Section 5.2).

### 4.3.3 How lesions are counted

The reference study reports the number of lesions per study and uses lesion-wise metrics (LTPR and LFPR), but it does not describe how a binary mask is split into individual lesions. Lesions are normally defined as **connected components**, and in 3D there are three possible neighbourhoods: two lesion voxels p = (x, y, z) and q = (x′, y′, z′) are neighbours if they share a face (6-connectivity, |x − x′| + |y − y′| + |z − z′| = 1), a face or an edge (18-connectivity) or a face, an edge or a corner (26-connectivity).

The three rules were applied to all 115 masks and compared with the number of lesions released by the authors for each study (Table 4.3).

| Rule | Studies matching the authors' count | Mean lesions (train / test) |
|---|---|---|
| **6-connectivity** | **115 / 115 (100 %)** | **28.9 / 41.2** |
| 18-connectivity | 49 / 115 (42.6 %) | 27.4 / 40.0 |
| 26-connectivity | 43 / 115 (37.4 %) | 27.2 / 39.7 |
| Reference study [7] | — | 28.9 / 41.2 |

*Table 4.3. Lesion counts obtained with each connectivity rule, compared with the counts released by the authors.*

Only **6-connectivity** reproduces the authors' numbers, in every study and without any minimum lesion size. The total lesion volume computed from the masks also matches the released values in all studies. This rule is adopted for all lesion counts and lesion-wise metrics in this thesis; otherwise LTPR and LFPR would not be comparable with the reference study. The voxel-wise Dice coefficient does not depend on it.

## 4.4 Lesion load

Lesion load varies enormously between studies (Figure 4.3): the total lesion volume ranges from 0.7 to 73 ml (median 6.0 ml) and the number of lesions from 3 to 193 (median 25). Lesion volume is only weakly correlated with disability (Spearman ρ = 0.33 with EDSS), in line with the well-known clinico-radiological paradox of MS.

![Figure 4.3. Total lesion volume (log scale) and number of lesions per study, training and test sets.](eda_lesion_load.png)

**Implication.** Segmentation metrics are expected to be lower and more variable for studies with little lesion, where every error weighs more. Results will also be reported by lesion-load group, not only as a global mean.

## 4.5 Class imbalance

Lesions occupy a median of **0.30 %** of the brain volume (range 0.04–3.9 %): for every lesion voxel there are about 330 healthy voxels.

**Implication.** Accuracy is meaningless (predicting "no lesion" everywhere would be 99.7 % accurate). Training requires overlap-based losses (Dice combined with cross-entropy) and oversampling of patches that contain lesions, and evaluation relies on the overlap and lesion-wise metrics of Section 5.3.

## 4.6 Lesion size

The 3,595 lesions have a median size of 63 mm³, but their size spans five orders of magnitude (Figure 4.4 and Table 4.4).

| Lesion size | Share of lesions | Share of lesion volume |
|---|---|---|
| ≤ 10 mm³ | 10.7 % | 0.1 % |
| 11–27 mm³ | 12.8 % | 0.7 % |
| 28–100 mm³ | 42.3 % | 6.4 % |
| 101–1,000 mm³ | 29.3 % | 21.3 % |
| > 1,000 mm³ | 5.0 % | 71.5 % |

*Table 4.4. Distribution of lesion sizes (all studies).*

![Figure 4.4. Size of each lesion (log scale). Two out of three lesions are 100 mm³ or smaller.](eda_lesion_sizes.png)

**Two out of three lesions are small (≤ 100 mm³), yet together they account for only 7 % of the lesion volume**, while 5 % of the lesions account for 72 % of it.

**Implication.** The voxel-wise Dice coefficient is dominated by a few large lesions: a model can miss most small lesions and still obtain a good Dice. Lesion-wise metrics and an analysis stratified by lesion size are therefore essential, and small-lesion detection is the main target for improvement (objective O3).

## 4.7 Scanners and intensities

The studies were acquired on eight scanner models from three manufacturers (Table 4.5), at 1.5 T and 3 T, with FLAIR slice thicknesses between 2 and 5 mm.

| Manufacturer | Training studies | Test studies |
|---|---|---|
| Philips | 84 (90 %) | 16 (73 %) |
| GE | 4 (4 %) | 5 (23 %) |
| Siemens | 0 | 1 (5 %) |
| Unknown | 5 (5 %) | 0 |

*Table 4.5. Scanner manufacturer by subset (FLAIR metadata).*

Image intensities are not comparable between studies: the 99th percentile of the FLAIR brain intensity ranges from 130 to 1,675 arbitrary units and depends on the manufacturer (median 806 for GE versus 273 for Philips), but also varies widely within the same manufacturer (Figure 4.5). Lesions are clearly hyperintense on FLAIR (the median lesion intensity is 1.45 times the median brain intensity, median over studies) and much less so on T1-w (1.12) and T2-w (1.16). Moreover, **non-Philips studies are much more frequent in the test set (27 %) than in the training set (4 %)**. The reference study only registers and skull-strips the images; it does not describe any intensity normalisation or harmonisation, nor results by scanner.

![Figure 4.5. 99th percentile of the brain intensity by manufacturer and sequence.](eda_intensities_by_manufacturer.png)

**Implication.** Each image must be intensity-normalised (z-score within the brain) and intensity augmentation should be used during training. Results will be reported by manufacturer (Philips versus others) to measure robustness to the acquisition. FLAIR is the most informative sequence.

## 4.8 Longitudinal patients

For the 25 training patients with more than one timepoint, lesion volume is usually stable between consecutive visits (Figure 4.6): the median ratio between visits is 1.03 and 22 of the 40 pairs change by less than 20 %. Four pairs change by more than a factor of three (Table 4.6).

| Patient | Visits | Lesion volume (mm³) | Ratio | EDSS |
|---|---|---|---|---|
| P20 | T1 → T2 | 744 → 5,807 | 7.81 | 3.0 → 0.0 |
| P20 | T2 → T3 | 5,807 → 1,537 | 0.26 | 0.0 → 1.0 |
| P49 | T1 → T2 | 34,709 → 7,763 | 0.22 | 2.0 → 2.5 |
| P50 | T1 → T2 | 9,144 → 36,435 | 3.98 | 2.5 → 3.5 |

*Table 4.6. Pairs of consecutive visits with a lesion-volume change larger than a factor of three.*

![Figure 4.6. Total lesion volume over time for the 25 patients with more than one timepoint.](eda_longitudinal.png)

The increase in P50 is consistent with its clinical worsening. The changes in P20 and P49 are less clear: they may reflect real disease activity (active lesions swell and later shrink) or differences in annotation between visits. With a single annotation per study this cannot be resolved; these studies are kept and flagged for review.

**Implication.** Visits of the same patient are strongly correlated, which confirms that validation must be done at patient level. Detection of new lesions between visits is a possible extension (Section 3.4).

## 4.9 Summary and design decisions

Table 4.7 summarises the findings of the exploratory analysis and the decisions derived from them.

| Finding | Decision |
|---|---|
| Small dataset; several visits per patient | Established architecture (nnU-Net) with data augmentation; patient-level folds |
| Lesions are 0.30 % of the brain | Dice + cross-entropy loss and lesion-oversampling; no accuracy |
| 2/3 of lesions are small but 7 % of the volume | Lesion-wise metrics; results by lesion size; focus of the contribution |
| Large variability of lesion load | Results by lesion-load group |
| Intensities depend on scanner; more non-Philips in test | Per-image z-score normalisation; intensity augmentation; results by manufacturer |
| Lesions are counted with 6-connectivity | Same rule for all lesion-wise metrics |
| 3 badly registered studies, 1 missing T1 | Kept (as in the reference study) + sensitivity analysis |
| 22 released test patients (paper mentions 24) | Released split used; noted when comparing with [7] |

*Table 4.7. Main findings of the exploratory analysis and resulting design decisions.*
