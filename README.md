# RetailGuard

## Abstract / About
RetailGuard is an end-to-end Machine Learning project designed to predict retail product returns based on historical transaction data. Product returns present a significant financial challenge for the retail industry, leading to direct revenue loss, increased logistical overhead, and inventory mismanagement. By identifying specific transactions or customers that are highly likely to result in a return, businesses can take proactive measures such as targeted customer service, adjusted return policies, or improved product descriptions. This project demonstrates a complete data science lifecycle, progressing from raw data exploration and robust preprocessing to advanced feature engineering and hyperparameter optimization using XGBoost.

## Use of Project
The predictive models and analytical insights developed in RetailGuard serve several key business use cases:
- **Loss Prevention:** High-risk transactions can be flagged in real-time, allowing businesses to adjust shipping promotions or issue warnings before fulfilling the order.
- **Inventory Optimization:** Anticipating return volumes enables warehouse managers to better predict restock needs and manage reverse logistics operations efficiently.
- **Customer Segmentation:** The insights gathered from the exploratory data analysis (EDA) help identify problematic purchasing behaviors and allow for more targeted marketing or policy enforcement.
- **Financial Forecasting:** Accurate predictions of return rates provide better estimations of net revenue for financial planning.

## Architecture
The project is built on a modular data pipeline architecture, with clear separations between data extraction, transformation, exploratory analysis, and model training.

1. **Data Acquisition and Storage (`data/raw/`)**: Raw transactional data is stored and loaded. The dataset is large and requires efficient memory management during initial parsing.
2. **Preprocessing Pipeline (`scripts/02_preprocessing.py`)**: Cleans the raw data by handling missing customer IDs, filtering invalid prices, and employing a FIFO allocation algorithm to map subsequent return invoices back to their original purchase records. The output is a structured dataset ready for analysis.
3. **Exploratory Data Analysis (`scripts/03_eda.py`)**: A statistical and visual evaluation phase that investigates class imbalances, temporal trends (e.g., return rates by month or day of the week), and geographical distributions. Output figures are generated automatically.
4. **Feature Engineering Pipeline (`scripts/04_feature_engineering.py`)**: Transforms temporal and monetary data into predictive features. Computes aggregate metrics per customer and per product. Missing values are imputed, data is scaled, and the Synthetic Minority Over-sampling Technique (SMOTE) is applied to address the severe class imbalance in returns.
5. **Model Training and Evaluation (`notebooks/`)**: Models are trained and compared. A baseline Logistic Regression model is established first. Subsequently, an XGBoost classifier is trained with hyperparameter tuning via RandomizedSearchCV, optimizing specifically for the F1 score on the minority class. The best performing model is serialized and saved for downstream inference.

## Project Structure
```text
RetailGuard_DS/
├─ data/
│  ├─ raw/             # Original downloaded CSV files
│  ├─ interim/         # Intermediate cleaned files
│  └─ processed/       # Final model-ready datasets and saved models
├─ notebooks/          # Jupyter notebooks for interactive analysis and modeling
├─ scripts/            # Python scripts for data processing and EDA generation
├─ reports/
│  └─ figures/         # Charts and exported visuals
├─ requirements.txt    # Project dependencies
└─ README.md           # Project documentation
```

## How to Run

Follow these instructions to reproduce the environment and execute the pipeline from scratch.

### 1. Environment Setup
It is highly recommended to use a virtual environment to manage dependencies.

```bash
# Create a virtual environment
python -m venv .venv

# Activate the virtual environment
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate

# Install the required packages
pip install -r requirements.txt
```
*(Note: If `requirements.txt` is not fully populated, you may manually install `pandas`, `numpy`, `scikit-learn`, `xgboost`, `matplotlib`, `seaborn`, `imbalanced-learn`, and `jupyter`.)*

### 2. Executing the Pipeline via Scripts
The project provides automated scripts to run the heavy data processing and engineering tasks sequentially. From the root of the project, execute the following commands:

```bash
# Step 1: Run basic data exploration and validation
python scripts/01_exploration.py

# Step 2: Preprocess the raw data (handles missing values and maps returns)
python scripts/02_preprocessing.py

# Step 3: Generate Exploratory Data Analysis (EDA) visualizations in reports/figures/
python scripts/03_eda.py

# Step 4: Engineer features, apply SMOTE, and perform train/test splits
python scripts/04_feature_engineering.py
```

### 3. Training the Models via Notebooks
After the data is preprocessed and engineered, you can run the Jupyter notebooks to train and evaluate the machine learning models.

```bash
# Launch Jupyter Notebook
jupyter notebook
```
Navigate to the `notebooks/` directory and execute the following notebooks in order:
- `05_model_training.ipynb`: Establishes the Logistic Regression baseline.
- `05_model_tuning_xgb.ipynb`: Performs hyperparameter tuning on the XGBoost classifier and saves the final optimal model to `data/best_model.joblib`.
```
