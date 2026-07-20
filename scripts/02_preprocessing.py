"""
02_preprocessing.py
Phase 2 preprocessing: load raw Online Retail dataset and create data/preprocessed_retail.csv
"""
# pylint: disable=invalid-name,too-many-locals,too-many-statements
from collections import defaultdict
from pathlib import Path
import pandas as pd
import numpy as np

if "__file__" in globals():
    BASE = Path(__file__).resolve().parent.parent
else:
    BASE = Path().resolve()
    if BASE.name in ("notebooks", "scripts"):
        BASE = BASE.parent
RAW_PATH = BASE / "data" / "raw" / "online_retail.csv"
OUT_PATH = BASE / "data" / "preprocessed_retail.csv"


def main():
    """Execute the preprocessing phase."""
    print("Loading raw data from", RAW_PATH)
    df = pd.read_csv(
        RAW_PATH, encoding="ISO-8859-1", parse_dates=["InvoiceDate"], low_memory=False
    )
    print(f"Loaded {len(df):,} rows")

    # Drop missing CustomerID
    df = df.dropna(subset=["CustomerID"]).copy()
    df = df.drop_duplicates().copy()

    df["InvoiceNo"] = df["InvoiceNo"].astype(str)
    df["StockCode"] = df["StockCode"].astype(str)
    df["CustomerID"] = df["CustomerID"].astype(str)

    # Remove UnitPrice <= 0
    df = df[df["UnitPrice"] > 0].copy()

    # Separate purchases and returns
    purchases = df[df["Quantity"] > 0].copy()
    returns = df[df["InvoiceNo"].str.startswith("C", na=False)].copy()

    # Prepare purchases
    purchases.sort_values(by=["CustomerID", "StockCode", "InvoiceDate"], inplace=True)
    purchases.reset_index(inplace=True)
    purchases.rename(columns={"index": "_orig_index"}, inplace=True)

    # Arrays
    p_idx = purchases.index.to_numpy()
    p_customer = purchases["CustomerID"].to_numpy()
    p_stock = purchases["StockCode"].to_numpy()
    p_qty_remaining = purchases["Quantity"].to_numpy().astype(float)
    is_returned_flag = np.zeros(len(purchases), dtype=np.int8)

    group_to_positions = defaultdict(list)
    for pos, cust, stock in zip(p_idx, p_customer, p_stock):
        group_to_positions[(cust, stock)].append(pos)

    returns.sort_values(by=["CustomerID", "StockCode", "InvoiceDate"], inplace=True)

    for (cust, stock), grp in returns.groupby(["CustomerID", "StockCode"]):
        key = (str(cust), str(stock))
        if key not in group_to_positions:
            continue
        positions = group_to_positions[key]
        if not positions:
            continue
        ptr = 0
        for _, ret_row in grp.iterrows():
            ret_qty = abs(float(ret_row["Quantity"]))
            while ret_qty > 0 and ptr < len(positions):
                pos = positions[ptr]
                available = p_qty_remaining[pos]
                if available <= 0:
                    ptr += 1
                    continue
                allocate = min(available, ret_qty)
                if allocate > 0:
                    is_returned_flag[pos] = 1
                    p_qty_remaining[pos] -= allocate
                    ret_qty -= allocate
                    if p_qty_remaining[pos] <= 0:
                        ptr += 1
                else:
                    ptr += 1

    purchases["IsReturned"] = is_returned_flag
    processed = purchases.drop(columns=["_orig_index"]).copy()

    cols = [
        "InvoiceNo",
        "InvoiceDate",
        "CustomerID",
        "StockCode",
        "Description",
        "Quantity",
        "UnitPrice",
        "IsReturned",
    ]
    existing_cols = [c for c in cols if c in processed.columns]
    remaining = [c for c in processed.columns if c not in existing_cols]
    processed = processed[existing_cols + remaining]

    processed.to_csv(OUT_PATH, index=False)
    print(f"Saved processed dataset to {OUT_PATH} with {len(processed):,} rows")


if __name__ == "__main__":
    main()
