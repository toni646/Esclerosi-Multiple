# 1. Introduction

## 1.1 Multiple sclerosis

Multiple sclerosis (MS) is a chronic, immune-mediated disease of the central nervous system and one of the most common causes of non-traumatic neurological disability in young adults. It affects an estimated 2.8 million people worldwide [1]. In MS, the immune system attacks the myelin sheath that insulates nerve fibres, causing inflammation, demyelination and, over time, axonal loss. These processes produce focal areas of damage, known as **lesions** or **plaques**, found mainly in the white matter of the brain and spinal cord.

## 1.2 The role of MRI

Magnetic resonance imaging (MRI) is the key imaging tool for diagnosing and monitoring MS. White-matter lesions appear hyperintense on T2-weighted sequences and, most clearly, on **FLAIR** (Fluid-Attenuated Inversion Recovery) images. FLAIR suppresses the signal of the cerebrospinal fluid, so lesions next to the ventricles stand out. The number, location and volume of lesions are part of the diagnostic criteria for MS [2]. The appearance of new or enlarging lesions over time is used to assess disease activity and response to treatment [3].

## 1.3 Lesion segmentation

Using these measurements in practice requires **segmenting** the lesions, that is, deciding for every voxel of the image whether it belongs to a lesion or not. Manual segmentation by experts is the reference standard, but it is time-consuming (it can take hours per patient) and subject to considerable inter- and intra-rater variability [4]. These limitations have motivated the development of automatic segmentation methods, which in recent years have been dominated by deep learning [5, 6].

## 1.4 Challenges

Automatic MS lesion segmentation remains difficult for several reasons:

- **Class imbalance:** lesions occupy a very small fraction of the brain.
- **Size variability:** lesion size ranges from a few voxels to large confluent regions, and small lesions are easily missed.
- **Acquisition heterogeneity:** image appearance depends on the scanner, field strength and acquisition protocol.
- **Reference uncertainty:** the annotations themselves are uncertain, especially at lesion boundaries.

Progress in the field therefore depends on public, well-annotated datasets that allow methods to be trained and compared under the same conditions. **MSLesSeg** [7] is the dataset this thesis builds on.
