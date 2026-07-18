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

## Phase 2: Data Cleaning & Preprocessing

Goal: transform the raw transaction log into a per-purchase classification dataset ready for modeling. The preprocessing step:

- Drops rows with missing `CustomerID` (customer profiling is required for modeling).
- Removes exact duplicate rows.
- Filters out rows with `UnitPrice <= 0` (administrative adjustments / bad data).
- Separates purchases (Quantity > 0) from returns (InvoiceNo starting with `C`) and maps returns back to prior purchases by `CustomerID` + `StockCode` using a FIFO allocation. Purchases that receive any returned quantity are labeled `IsReturned = 1` (partial returns count as returned); the rest are `0`.

Output: the cleaned per-purchase dataset `data/preprocessed_retail.csv` is created and ready for Phase 3 modeling.

How to run (local Jupyter / notebook):

1. Open `notebooks/02_preprocessing.ipynb` (or create it using the provided Phase 2 code) in PyCharm/Jupyter.
2. Execute the cells to load `data/raw/online_retail.csv`, perform cleaning and target engineering, and export `data/preprocessed_retail.csv`.

Notes:
- The preprocessing is implemented to be memory-efficient and uses group-wise allocation to match returns to purchases. If you prefer alternative labeling (e.g., only fully returned purchases marked as returned), update the allocation logic accordingly in the notebook.
- After running, inspect the `IsReturned` distribution in the resulting CSV to understand class imbalance before modeling.


