"""
01_exploration.py
Simple exploration script for the RetailGuard project. Intended to be run inside the project venv.
"""
from pathlib import Path
import pandas as pd

DATA_RAW = Path("data/raw/online_retail.csv")

if __name__ == "__main__":
    df = pd.read_csv(DATA_RAW, encoding="ISO-8859-1", parse_dates=["InvoiceDate"], low_memory=False)
    print("Rows:", len(df))
    print(df.head().to_string())
    print('\nColumns:', df.columns.tolist())
    print('\nMissing values per column:')
    print(df.isna().sum())

