# RetailGuard

Phase 1: Foundation & Data Acquisition

## Recommended project structure

```text
RetailGuard_DS/
├─ data/
│  ├─ raw/
│  ├─ interim/
│  └─ processed/
├─ notebooks/
├─ src/
├─ reports/
│  └─ figures/
├─ requirements.txt
└─ README.md
```

## What goes where

- `data/raw/`: original downloaded CSV files from UCI or Kaggle
- `data/interim/`: intermediate cleaned files used during exploration
- `data/processed/`: final model-ready datasets later in the project
- `notebooks/`: Jupyter notebooks like `01_exploration.ipynb`
- `src/`: reusable Python modules once Phase 2 starts
- `reports/figures/`: charts and exported visuals

## Phase 1 notebook

The first notebook only verifies that the dataset can be loaded correctly and that the local environment can handle the file in memory.

