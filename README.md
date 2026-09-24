# Esclerosi-Multiple

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Treball de Fi de Màster: **segmentació automàtica de lesions d'Esclerosi Múltiple en imatges de ressonància magnètica** (FLAIR, T1, T2).

- **Dataset:** MSLesSeg (75 pacients, 115 estudis) — [figshare](https://doi.org/10.6084/m9.figshare.27919209)
- **Article de referència:** Guarnera et al., *MSLesSeg: baseline and benchmarking of a new Multiple Sclerosis Lesion Segmentation dataset*, Scientific Data (2025) — [enllaç](https://www.nature.com/articles/s41597-025-05250-y)
- **Tutor:** Manel Frigola

## Planificació

El seguiment es fa amb [Issues](../../issues), [Milestones](../../milestones) i el [tauler del projecte](https://github.com/users/toni646/projects/1).

| Fase | Objectiu |
|---|---|
| M1 | Dades i anàlisi exploratòria |
| M2 | Reproducció de l'article i baseline nnU-Net |
| M3 | Aportació pròpia (lesions petites, falsos positius, incertesa) |
| M4 | Validació externa i anàlisi |
| M5 | Memòria i defensa |

## Instal·lació

```bash
conda env create -f environment.yml
conda activate esclerosi-multiple
```

Les dades no es pugen a GitHub. Descarrega MSLesSeg i deixa-les a `data/raw/`.

## Organització del projecte

```
├── Makefile           <- Comandes útils (`make requirements`, `make lint`, `make test`...)
├── README.md          <- Aquest fitxer
├── environment.yml    <- Entorn conda per reproduir el projecte
├── pyproject.toml     <- Configuració del paquet ms_seg i d'eines (ruff)
│
├── configs            <- Fitxers YAML amb la configuració de cada experiment
│
├── data               <- (no versionat a Git)
│   ├── external       <- Datasets externs per a validació (MSSEG, ISBI...)
│   ├── interim        <- Dades intermèdies
│   ├── processed      <- Dades finals preparades per entrenar
│   └── raw            <- MSLesSeg original, sense modificar
│
├── docs               <- Documentació (mkdocs)
├── models             <- Models entrenats i prediccions (no versionat a Git)
│
├── notebooks          <- Notebooks. Convenció: número, inicials i descripció,
│                         p. ex. `1.0-tlb-eda-mslesseg.ipynb`
│
├── references         <- Articles, manuals i material de referència
├── reports            <- Informes generats
│   └── figures        <- Figures per a la memòria
│
├── tests              <- Tests (pytest)
│
└── ms_seg             <- Codi font del projecte
    ├── __init__.py
    ├── config.py      <- Rutes i variables globals
    ├── dataset.py     <- Càrrega i preparació de les dades
    ├── features.py    <- Preprocessament
    ├── modeling
    │   ├── __init__.py
    │   ├── train.py   <- Entrenament
    │   └── predict.py <- Inferència
    └── plots.py       <- Visualitzacions
```
