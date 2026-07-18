# RetailGuard

RetailGuard is an end-to-end Machine Learning project to predict retail returns. By identifying transactions that are likely to be returned, the business can take proactive measures for loss prevention and optimize inventory.

## Project Structure

```text
RetailGuard_DS/
├─ data/
│  ├─ raw/             # Original downloaded CSV files
│  ├─ interim/         # Intermediate cleaned files
│  └─ processed/       # Final model-ready datasets and saved models
├─ notebooks/          # Jupyter notebooks for each phase of the project
├─ scripts/            # Python scripts for utility tasks
├─ reports/
│  └─ figures/         # Charts and exported visuals
├─ requirements.txt    # Project dependencies
└─ README.md
```

## Phases Completed

### Phase 1: Foundation & Data Acquisition
**Notebook**: `01_exploration.ipynb`
- Verified that the dataset can be loaded correctly and the local environment handles the memory requirements.

### Phase 2: Data Cleaning & Preprocessing
**Notebook**: `02_preprocessing.ipynb`
- Dropped rows with missing `CustomerID` and exact duplicates.
- Filtered out rows with `UnitPrice <= 0`.
- Mapped returns (InvoiceNo starting with `C`) to prior purchases using a FIFO allocation.
- Labeled purchases with returned quantities as `IsReturned = 1`, and the rest as `0`.
- Output: `data/preprocessed_retail.csv`

### Phase 3: Exploratory Data Analysis (EDA)
**Notebook**: `03_eda.ipynb`
- Explored the distribution of features, class imbalance of `IsReturned`, and relationships between customer purchasing behavior and return likelihood.

### Phase 4: Feature Engineering
**Notebook**: `04_feature_engineering.ipynb`
- Created new predictive features from the cleaned dataset.
- Addressed class imbalance using SMOTE (Synthetic Minority Over-sampling Technique).
- Scaled features and split data into train/test sets.
- Output: `X_train_scaled.csv`, `X_test_scaled.csv`, `y_train_resampled.csv`, `y_test.csv`

### Phase 5: Model Training
**Notebook**: `05_model_training.ipynb`
- Trained a baseline Logistic Regression model on the balanced dataset.
- Evaluated performance and established a baseline F1 score for the minority class (`IsReturned = 1`).

### Phase 5.5: XGBoost Optimization
**Notebook**: `05_model_tuning_xgb.ipynb`
- Initialized an `XGBClassifier` and tuned hyperparameters (`max_depth`, `learning_rate`, `n_estimators`, `subsample`) using `RandomizedSearchCV`.
- Optimized strictly for the `f1` score on the minority class.
- The tuned XGBoost model achieved a significant improvement over the baseline Logistic Regression model (F1 score of 0.2290 vs baseline 0.1576).
- The best performing model was saved to `data/best_model.joblib`.

## Getting Started

1. Ensure your virtual environment is activated.
2. Install dependencies (e.g., `pip install pandas numpy scikit-learn xgboost`).
3. Run the notebooks in the `notebooks/` directory sequentially from 01 to 05.
