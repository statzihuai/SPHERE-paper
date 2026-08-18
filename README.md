# Unlocking Sensitive Data with SPHERE in the Age of AI

## Requirements

Python ≥ 3.11. Install dependencies:

```bash
pip install -r requirements.txt
```

Anonymeter (privacy panels) requires a separate Python 3.11 virtualenv with
`anonymeter==1.0.0` pinned. Deep-learning baselines and Table 1 each need
additional virtualenvs; see comments in `requirements.txt`.

## SPHERE implementation

The SPHERE implementation is available as a
[macOS desktop application](https://github.com/statzihuai/SPHERE) and a
[command-line tool](https://github.com/statzihuai/sphere-cli).
The computation scripts (`step*.py`, `panel_*.py`) call `sphere()` from this
implementation; install it to regenerate synthetic data from raw inputs.

## Repository structure

```
requirements.txt
code/
├── figure1/                         # SPHERE benchmark (33 public datasets)
│   ├── render.py                    #   draw figure from panel CSVs
│   ├── panel_{b..f}_*.py            #   per-panel computation (5 scripts)
│   ├── _common.py                   #   shared loaders and methods
│   └── _anonymeter_worker.py        #   privacy subprocess
├── figure2/                         # NHANES virtual lab
│   ├── analyze.py                   #   data processing and metrics
│   └── render.py                    #   panel rendering
├── figure2_virtuallab.py            #   unified entry point
├── figure3/                         # UK Biobank WGS + Olink proteomics
│   ├── step{1..10}_*.py             #   computation pipeline
│   └── figure3_plot.py              #   draw figure
├── figure4/                         # Stanford ADRC multi-modal synthetic twin
│   ├── step{1..8}_*.py              #   computation pipeline
│   ├── adrc_common.py               #   shared constants and loaders
│   └── render.py                    #   draw figure
├── figure5/                         # Organ-aging clocks, pTau217, GNPC
│   ├── step{1..4}_*.py              #   Python computation steps
│   ├── step{5,6}_*.R                #   R computation steps
│   └── render.py                    #   draw figure
└── table1/                          # Deep-learning modalities (public data)
    ├── step{1..7}_*.py              #   end-to-end from public sources
    └── _common.py                   #   shared constants

data/                                # raw inputs (public only)
├── benchmark/                       #   33 OpenML datasets (Figure 1)
```

## Data availability

| Dataset | Access | Included |
|---|---|---|
| 33 OpenML benchmark datasets (Figure 1) | public | `data/benchmark/` |
| NHANES 2017–2018 (Figure 2) | public ([CDC](https://wwwn.cdc.gov/nchs/nhanes/continuousnhanes/default.aspx?BeginYear=2017)) | downloaded by `analyze.py` |
| Dreaddit, ImageNet-100, PhysioNet PTB (Table 1) | public | downloaded by steps 1–3 |
| UK Biobank WGS and Olink (Figures 3, 5) | controlled access | not included |
| Stanford ADRC (Figures 4, 5) | controlled access | not included |
| GNPC (Figure 5) | controlled access | not included |

For controlled-access datasets, aggregate derivatives are not included. Access to UK Biobank is via
[ukbiobank.ac.uk](https://www.ukbiobank.ac.uk); Stanford ADRC data is
available through the Stanford ADRC. GNPC data is available upon request to
qualified researchers through
[www.neuroproteome.org/harmonized-data-set-hds](https://www.neuroproteome.org/harmonized-data-set-hds).

## Correspondence

Zihuai He (zihuai@stanford.edu)
