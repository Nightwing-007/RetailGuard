"""
04_feature_engineering.py
Phase 4: feature engineering, train/test split, SMOTE oversampling on train, scaling, and persistence of scaler/imputer.
"""
from pathlib import Path
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

BASE = Path(__file__).resolve().parent.parent
DATA_IN = BASE / "data" / "preprocessed_retail.csv"
OUT_DIR = BASE / "data"
OUT_DIR.mkdir(parents=True, exist_ok=True)
RANDOM_STATE = 42


def main():
    df = pd.read_csv(DATA_IN, parse_dates=["InvoiceDate"], low_memory=False)
    df["IsReturned"] = df["IsReturned"].astype(int)
    df["CustomerID"] = df["CustomerID"].astype(str)
    df["StockCode"] = df["StockCode"].astype(str)
    df["OrderValue"] = df["Quantity"].astype(float) * df["UnitPrice"].astype(float)

    # Temporal
    df["Hour"] = df["InvoiceDate"].dt.hour.astype(np.int8)
    df["DayOfWeek"] = df["InvoiceDate"].dt.dayofweek.astype(np.int8)
    df["Month"] = df["InvoiceDate"].dt.month.astype(np.int8)

    # Keep subset
    cols_keep = ["InvoiceNo","InvoiceDate","CustomerID","StockCode","Quantity","UnitPrice","OrderValue","Hour","DayOfWeek","Month","IsReturned"]
    df = df[[c for c in cols_keep if c in df.columns]]

    # Train/test split
    train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['IsReturned'], random_state=RANDOM_STATE)

    # Aggregates on train
    cust_grp = train_df.groupby('CustomerID', observed=True)
    customer_agg = cust_grp.agg(
        Customer_Total_Orders=('InvoiceNo', lambda s: s.nunique()),
        Customer_Total_Items=('Quantity','sum'),
        Customer_Average_Spend=('OrderValue','mean'),
        Customer_Return_Rate=('IsReturned','mean')
    ).reset_index()

    prod_grp = train_df.groupby('StockCode', observed=True)
    product_agg = prod_grp.agg(
        Product_Total_Sales=('Quantity','sum'),
        Product_Total_Orders=('InvoiceNo', lambda s: s.nunique()),
        Product_Return_Rate=('IsReturned','mean')
    ).reset_index()

    # Merge into train and test
    train = train_df.merge(customer_agg, on='CustomerID', how='left')
    test = test_df.merge(customer_agg, on='CustomerID', how='left')
    train = train.merge(product_agg, on='StockCode', how='left')
    test = test.merge(product_agg, on='StockCode', how='left')

    # Defaults
    cust_defaults = {
        'Customer_Total_Orders': int(customer_agg['Customer_Total_Orders'].median()) if not customer_agg.empty else 0,
        'Customer_Total_Items': float(customer_agg['Customer_Total_Items'].median()) if not customer_agg.empty else 0.0,
        'Customer_Average_Spend': float(customer_agg['Customer_Average_Spend'].median()) if not customer_agg.empty else 0.0,
        'Customer_Return_Rate': float(customer_agg['Customer_Return_Rate'].median()) if not customer_agg.empty else 0.0
    }
    prod_defaults = {
        'Product_Total_Sales': float(product_agg['Product_Total_Sales'].median()) if not product_agg.empty else 0.0,
        'Product_Total_Orders': int(product_agg['Product_Total_Orders'].median()) if not product_agg.empty else 0,
        'Product_Return_Rate': float(product_agg['Product_Return_Rate'].median()) if not product_agg.empty else 0.0
    }

    for c, default in {**cust_defaults, **prod_defaults}.items():
        if c in train.columns:
            train[c] = train[c].fillna(default)
        if c in test.columns:
            test[c] = test[c].fillna(default)

    FEATURE_COLUMNS = [
        'Quantity','UnitPrice','OrderValue','Hour','DayOfWeek','Month',
        'Customer_Total_Orders','Customer_Total_Items','Customer_Average_Spend','Customer_Return_Rate',
        'Product_Total_Sales','Product_Total_Orders','Product_Return_Rate'
    ]
    FEATURE_COLUMNS = [c for c in FEATURE_COLUMNS if c in train.columns]

    X_train = train[FEATURE_COLUMNS].copy()
    y_train = train['IsReturned'].copy()
    X_test = test[FEATURE_COLUMNS].copy()
    y_test = test['IsReturned'].copy()

    # Impute
    imputer = SimpleImputer(strategy='median')
    imputer.fit(X_train)
    X_train_imputed = pd.DataFrame(imputer.transform(X_train), columns=FEATURE_COLUMNS)
    X_test_imputed = pd.DataFrame(imputer.transform(X_test), columns=FEATURE_COLUMNS)

    # Persist imputer
    joblib.dump(imputer, OUT_DIR / 'imputer.joblib')

    # SMOTE on training
    smote = SMOTE(random_state=RANDOM_STATE)
    X_resampled, y_resampled = smote.fit_resample(X_train_imputed, y_train)

    # Scale
    scaler = StandardScaler()
    scaler.fit(X_resampled)
    X_train_scaled = pd.DataFrame(scaler.transform(X_resampled), columns=FEATURE_COLUMNS)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test_imputed), columns=FEATURE_COLUMNS)

    # Persist scaler
    joblib.dump(scaler, OUT_DIR / 'scaler.joblib')

    # Save outputs
    X_train_scaled.to_csv(OUT_DIR / 'X_train_scaled.csv', index=False)
    X_test_scaled.to_csv(OUT_DIR / 'X_test_scaled.csv', index=False)
    pd.Series(y_resampled, name='IsReturned').to_csv(OUT_DIR / 'y_train_resampled.csv', index=False)
    y_test.reset_index(drop=True).to_csv(OUT_DIR / 'y_test.csv', index=False)

    print('Saved processed partitions and imputer/scaler to', OUT_DIR)


if __name__ == '__main__':
    main()


