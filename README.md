# Esclerosi-Multiple

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Master's thesis: **automatic segmentation of Multiple Sclerosis lesions in magnetic resonance images** (FLAIR, T1, T2).

- **Dataset:** MSLesSeg (75 patients, 115 studies) — [figshare](https://doi.org/10.6084/m9.figshare.27919209)
- **Reference paper:** Guarnera et al., *MSLesSeg: baseline and benchmarking of a new Multiple Sclerosis Lesion Segmentation dataset*, Scientific Data (2025) — [link](https://www.nature.com/articles/s41597-025-05250-y)
- **Supervisor:** Manel Frigola

## Planning

Progress is tracked with [Issues](../../issues), [Milestones](../../milestones) and the [project board](https://github.com/users/toni646/projects/1).

| Phase | Goal |
|---|---|
| M1 | Data and exploratory analysis |
| M2 | Reproduction of the paper and nnU-Net baseline |
| M3 | Own contribution (small lesions, false positives, uncertainty) |
| M4 | External validation and analysis |
| M5 | Thesis and defense |

## Installation

```bash
conda env create -f environment.yml
conda activate esclerosi-multiple
```

Data is not stored on GitHub. Download MSLesSeg and place it in `data/raw/`.

## Project Organization

```
├── Makefile           <- Convenience commands (`make requirements`, `make lint`, `make test`...)
├── README.md          <- This file
├── environment.yml    <- Conda environment to reproduce the project
├── pyproject.toml     <- Package metadata for ms_seg and tool configuration (ruff)
│
├── configs            <- YAML configuration files for each experiment
│
├── data               <- (not tracked by Git)
│   ├── external       <- External datasets for validation (MSSEG, ISBI...)
│   ├── interim        <- Intermediate data
│   ├── processed      <- Final data ready for training
│   └── raw            <- Original, immutable MSLesSeg data
│
├── docs               <- Documentation (mkdocs)
├── models             <- Trained models and predictions (not tracked by Git)
│
├── notebooks          <- Jupyter notebooks. Naming convention: number, initials and description,
│                         e.g. `1.0-tlb-eda-mslesseg.ipynb`
│
├── references         <- Papers, manuals and other reference material
├── reports            <- Generated reports
│   └── figures        <- Figures for the thesis
│
├── tests              <- Tests (pytest)
│
└── ms_seg             <- Source code for this project
    ├── __init__.py
    ├── config.py      <- Paths and global variables
    ├── dataset.py     <- Data loading and preparation
    ├── features.py    <- Preprocessing
    ├── modeling
    │   ├── __init__.py
    │   ├── train.py   <- Model training
    │   └── predict.py <- Inference
    └── plots.py       <- Visualizations
```
