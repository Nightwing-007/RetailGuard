# 🛠️ RetailGuard: Technical Architecture & Developer Guide

This document provides in-depth technical specifications, pipeline architecture documentation, mathematical/algorithmic details, and local development setup instructions for the **RetailGuard** machine learning system.

---

## 🏛️ System Architecture & Pipeline Design

RetailGuard is structured as a modular, 6-phase machine learning lifecycle pipeline:

```
┌────────────────────────┐      ┌─────────────────────────┐      ┌──────────────────────────┐
│  Phase 1: Exploration  │ ───► │  Phase 2: Preprocessing │ ───► │      Phase 3: EDA        │
│  Schema validation &   │      │  FIFO return mapping &  │      │  Temporal & geographical │
│  missing value audits  │      │  noise/anomaly cleaning │      │  visual analysis         │
└────────────────────────┘      └─────────────────────────┘      └──────────────────────────┘
                                                                               │
┌────────────────────────┐      ┌─────────────────────────┐                    ▼
│  Phase 6: Deployment   │ ◄─── │  Phase 5: ML Modeling   │ ◄─── ┌──────────────────────────┐
│  Streamlit real-time   │      │  XGBoost tuning via     │      │ Phase 4: Feature Eng.    │
│  inference dashboard   │      │  RandomizedSearchCV     │      │ SMOTE balancing, scaling │
└────────────────────────┘      └─────────────────────────┘      └──────────────────────────┘
```

---

### Phase-by-Phase Technical Breakdown

#### 1. Phase 1: Data Acquisition & Initial Exploration
- **Source Dataset:** Raw Online Retail transactional dataset (`data/raw/online_retail.csv`, 541,909 rows).
- **Encoding & Parsing:** Parsed using `ISO-8859-1` character encoding with low-memory streaming and ISO timestamp parsing on `InvoiceDate`.
- **Key Findings:** 135,080 missing `CustomerID` records, canceled transactions designated by prefix `C`, and zero/negative `UnitPrice` adjustments.

#### 2. Phase 2: Preprocessing & FIFO Return Allocation
- **Integrity Filters:** Drops rows with missing `CustomerID` and non-positive `UnitPrice` ($\le 0$).
- **Cancellation Reconciliation (FIFO Algorithm):**
  - Canceled orders (e.g., `InvoiceNo` starting with `C` and negative `Quantity`) are separated from standard purchases.
  - For every return record $(C, S, Q_{\text{ret}})$, the algorithm searches the customer's prior purchases for matching `StockCode` $S$ in first-in-first-out order, marking the original purchase as `IsReturned = 1`.
- **Output:** Cleaned dataset persisted to `data/preprocessed_retail.csv` (392,692 rows).

#### 3. Phase 3: Exploratory Data Analysis (EDA)
- Generates statistical aggregations and visual charts in `reports/figures/`:
  - `isreturned_counts.png`: Quantifies the severe class imbalance (~1.94% positive return rate).
  - `monthly_return_rate.png`: Analyzes seasonal spikes (e.g., post-holiday return waves).
  - `weekday_return_rate.png` & `price_bucket_return_rate.png`: Examines day-of-week and price elasticity correlations.
  - `country_return_rate.png`: Evaluates geographic variance across domestic vs. international shipments.

#### 4. Phase 4: Feature Engineering & Imbalance Handling
- **Temporal Extraction:** `Hour` ($0-23$), `DayOfWeek` ($0-6$), and `Month` ($1-12$).
- **Monetary Transformation:** `OrderValue = Quantity * UnitPrice`.
- **Customer Historical Profiling:** Rolling aggregations per customer:
  - `Customer_Total_Orders`, `Customer_Total_Items`, `Customer_Average_Spend`, `Customer_Return_Rate`.
- **Product Velocity Profiling:** Rolling aggregations per SKU / StockCode:
  - `Product_Total_Sales`, `Product_Total_Orders`, `Product_Return_Rate`.
- **13-Feature Schema:**
  ```python
  FEATURE_COLUMNS = [
      "Quantity", "UnitPrice", "OrderValue", "Hour", "DayOfWeek", "Month",
      "Customer_Total_Orders", "Customer_Total_Items", "Customer_Average_Spend",
      "Customer_Return_Rate", "Product_Total_Sales", "Product_Total_Orders",
      "Product_Return_Rate"
  ]
  ```
- **Train/Test Splitting:** 80/20 stratified split (`random_state=42`).
- **Median Imputation & StandardScaler:** Missing historical customer/product medians are imputed via `SimpleImputer(strategy='median')` and scaled via `StandardScaler()`. Both transformers are serialized to `data/imputer.joblib` and `data/scaler.joblib`.
- **Synthetic Minority Over-sampling (SMOTE):** Applied **exclusively to the training split** to synthesize minority return samples, expanding training volume from 314,153 to 616,140 balanced observations without data leakage into the test set.

#### 5. Phase 5 & 5.5: Model Training, Evaluation, and Tuning
- **Baseline Models:** Evaluated `LogisticRegression` ($F_1 = 0.1576$) and `RandomForestClassifier`.
- **XGBoost Hyperparameter Optimization:** Configured `RandomizedSearchCV` optimizing for minority-class $F_1$ (`scoring='f1'`, 3-fold CV, 10 iterations):
  - `max_depth`: `[3, 4, 5]` (constrained to prevent overfitting on SMOTE samples)
  - `learning_rate`: `[0.01, 0.05, 0.1]`
  - `n_estimators`: `[100, 200]`
  - `subsample`: `[0.8, 1.0]`
- **Best Hyperparameters:** `{'subsample': 1.0, 'n_estimators': 200, 'max_depth': 5, 'learning_rate': 0.1}`
- **Persistence:** Serialized top-performing classifier to `data/best_model.joblib`.

#### 6. Phase 6: Real-Time Web Application (Streamlit)
- Production UI in `app.py` with `@st.cache_resource` artifact caching, scenario presets, risk gauge meters, and automated operational recommendations.

---

## 📊 Model Evaluation & Benchmark Results

Evaluated on the **untouched test set** ($N = 78,539$ samples, $1,521$ actual returns):

### Classification Report (Tuned XGBoost)
```text
              precision    recall  f1-score   support

    Keep (0)       0.98      0.99      0.99     77,018
  Return (1)       0.37      0.17      0.23      1,521

    accuracy                           0.98     78,539
   macro avg       0.68      0.58      0.61     78,539
weighted avg       0.97      0.98      0.97     78,539
```

### Confusion Matrix Breakdown
```text
                Predicted Keep (0)    Predicted Return (1)
Actual Keep (0)       76,582                 436
Actual Return (1)      1,268                 253
```

### Model Performance Comparison

| Model | Minority Class Precision | Minority Class Recall | Minority Class $F_1$-Score | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Baseline Logistic Regression** | 0.11 | 0.28 | `0.1576` | Baseline |
| **Random Forest Classifier** | 0.31 | 0.15 | `0.2021` | Benchmark |
| **Tuned XGBoost Classifier** | **0.37** | **0.17** | **`0.2290`** | **Production Best (+45.3% vs Baseline)** |

---

## 🧰 Technology Stack & Dependencies

- **Language:** Python `3.10` / `3.11` / `3.12`
- **Data Manipulation:** `pandas>=2.0.0`, `numpy>=1.24.0`
- **Machine Learning:** `scikit-learn>=1.3.0`, `xgboost>=2.0.0`, `imbalanced-learn>=0.11.0`, `joblib>=1.3.0`
- **Visualization:** `matplotlib>=3.7.0`, `seaborn>=0.12.0`
- **Interactive Computing:** `jupyter>=1.0.0`, `ipykernel>=6.25.0`, `nbformat>=5.9.0`
- **Web Application & UI:** `streamlit>=1.30.0`

---

## 💻 Local Setup & Environment Installation

### 1. Clone the Repository
```bash
git clone https://github.com/Nightwing-007/RetailGuard.git
cd RetailGuard
```

### 2. Create and Activate Virtual Environment

**On Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**On macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Upgrade Pip and Install Dependencies
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## ⚡ Running the Pipeline

You can reproduce the entire machine learning pipeline either through automated Python scripts or step-by-step Jupyter notebooks.

### Option A: Sequential Script Execution (Recommended)

Run the following commands from the project root:

```bash
# Step 1: Run basic data exploration and schema validation
python scripts/01_exploration.py

# Step 2: Preprocess raw data (clean anomalies & perform FIFO return matching)
python scripts/02_preprocessing.py

# Step 3: Generate Exploratory Data Analysis figures in reports/figures/
python scripts/03_eda.py

# Step 4: Engineer features, apply SMOTE, scale data, and export transformers
python scripts/04_feature_engineering.py

# Step 5: Execute XGBoost hyperparameter tuning and save data/best_model.joblib
python scripts/train_and_tune_xgb.py
```

### Option B: Interactive Jupyter Notebook Execution

Launch the notebook server:
```bash
jupyter notebook
```

Navigate to `notebooks/` and execute the notebooks in chronological order:
1. `01_exploration.ipynb`
2. `02_preprocessing.ipynb`
3. `03_eda.ipynb`
4. `04_feature_engineering.ipynb`
5. `05_model_training.ipynb`
6. `05_model_tuning_xgb.ipynb`

---

## 🌐 Launching the Streamlit Web Application

To start the local inference dashboard:

```bash
streamlit run app.py
```

*(Or via explicit virtual environment path on Windows)*:
```powershell
.\.venv\Scripts\streamlit.exe run app.py
```

Once started, open your browser to **`http://localhost:8501`**.

---

## 📁 Repository Directory Structure

```text
RetailGuard/
├── data/
│   ├── raw/
│   │   ├── .gitkeep
│   │   └── online_retail.csv          # Raw source transactional data
│   ├── preprocessed_retail.csv        # Cleaned dataset post-FIFO allocation
│   ├── X_train_scaled.csv             # Scaled training features (SMOTE balanced)
│   ├── X_test_scaled.csv              # Scaled test features (untouched)
│   ├── y_train_resampled.csv          # Resampled training labels
│   ├── y_test.csv                     # Test labels
│   ├── imputer.joblib                 # Serialized median SimpleImputer
│   ├── scaler.joblib                  # Serialized StandardScaler
│   └── best_model.joblib              # Serialized optimal XGBoost classifier
├── notebooks/
│   ├── 01_exploration.ipynb           # Interactive exploration
│   ├── 02_preprocessing.ipynb         # Data cleaning & FIFO reconciliation
│   ├── 03_eda.ipynb                   # Exploratory data visualization
│   ├── 04_feature_engineering.ipynb   # Feature generation & SMOTE
│   ├── 05_model_training.ipynb        # Baseline Logistic Regression & RF
│   └── 05_model_tuning_xgb.ipynb      # XGBoost tuning & evaluation
├── reports/
│   └── figures/                       # Generated EDA & Confusion Matrix plots
├── scripts/
│   ├── 01_exploration.py              # CLI exploration
│   ├── 02_preprocessing.py            # CLI preprocessing
│   ├── 03_eda.py                      # CLI visualization generator
│   ├── 04_feature_engineering.py      # CLI feature pipeline
│   ├── create_xgb_notebook.py         # Notebook generator helper
│   └── train_and_tune_xgb.py          # Automated XGBoost tuning script
├── app.py                             # Streamlit web deployment application
├── DEVELOPER.md                       # Technical & architecture documentation
├── README.md                          # Business & executive documentation
└── requirements.txt                   # Production package dependencies
```
