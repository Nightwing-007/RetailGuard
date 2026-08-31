"""
Train and tune XGBoost model and execute/generate the 05_model_tuning_xgb.ipynb notebook.
"""
from pathlib import Path
import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor

BASE = Path(__file__).resolve().parent.parent
NOTEBOOK_PATH = BASE / "notebooks" / "05_model_tuning_xgb.ipynb"

TEXT_INTRO = """\
# Phase 5.5: XGBoost Optimization
In this notebook, we load our finalized datasets, train and tune an XGBoost model to optimize the F1 score for the minority class, and evaluate it on the test set. If it outperforms the baseline Logistic Regression model (F1 = 0.1576), we save it as our new best model.
"""

CODE_IMPORTS = """\
from pathlib import Path
import warnings
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, f1_score
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')

BASE = Path.cwd()
if BASE.name == 'notebooks':
    BASE = BASE.parent
DATA = BASE / 'data'
FIG_DIR = BASE / 'reports' / 'figures'
FIG_DIR.mkdir(parents=True, exist_ok=True)
"""

CODE_LOAD = """\
# Load finalized datasets
X_train_scaled = pd.read_csv(DATA / 'X_train_scaled.csv')
X_test_scaled = pd.read_csv(DATA / 'X_test_scaled.csv')
y_train_resampled = pd.read_csv(DATA / 'y_train_resampled.csv').iloc[:, 0].values
y_test = pd.read_csv(DATA / 'y_test.csv').iloc[:, 0].values

print("Data loaded successfully.")
print("X_train shape:", X_train_scaled.shape)
print("X_test shape:", X_test_scaled.shape)
print("y_train_resampled shape:", y_train_resampled.shape)
print("y_test shape:", y_test.shape)
"""

CODE_TUNE = """\
# Initialize XGBClassifier
xgb = XGBClassifier(random_state=42, eval_metric='logloss', n_jobs=-1)

# Define search space
param_distributions = {
    'max_depth': [3, 4, 5],
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [100, 200],
    'subsample': [0.8, 1.0]
}

# Set up RandomizedSearchCV to optimize for minority class F1
random_search = RandomizedSearchCV(
    estimator=xgb,
    param_distributions=param_distributions,
    n_iter=10,
    scoring='f1',
    cv=3,
    verbose=2,
    random_state=42,
    n_jobs=-1
)

# Fit RandomizedSearchCV
print("Starting XGBoost hyperparameter tuning...")
random_search.fit(X_train_scaled, y_train_resampled)
print("Tuning complete.")
print("Best Parameters:", random_search.best_params_)
print(f"Best CV F1 Score: {random_search.best_score_:.4f}")
"""

CODE_EVAL = """\
# Evaluate the tuned model on the test set
best_xgb = random_search.best_estimator_
y_pred = best_xgb.predict(X_test_scaled)

print("Classification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Plot and save confusion matrix
fig, ax = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Kept (0)", "Returned (1)"]).plot(ax=ax, cmap="Blues")
plt.title("XGBoost Confusion Matrix")
plt.tight_layout()
plt.savefig(FIG_DIR / "cm_xgboost.png", dpi=120)
plt.close(fig)

# Get F1 score for minority class
test_f1 = f1_score(y_test, y_pred)
print(f"Test F1 Score (Minority Class): {test_f1:.4f}")
"""

CODE_SAVE = """\
# Compare with baseline and save
baseline_f1 = 0.1576

print(f"Baseline Logistic Regression F1: {baseline_f1:.4f}")
print(f"Tuned XGBoost Test F1:          {test_f1:.4f}")

if test_f1 > baseline_f1:
    print(f"\\nNew model outperforms baseline! ({test_f1:.4f} > {baseline_f1:.4f})")
    joblib.dump(best_xgb, DATA / 'best_model.joblib')
    print(f"New best model saved to {DATA / 'best_model.joblib'}")
else:
    print(f"\\nNew model did not outperform baseline. ({test_f1:.4f} <= {baseline_f1:.4f})")
    print("Keeping the previous best model.")
"""

def main():
    nb = nbf.v4.new_notebook()
    nb["cells"] = [
        nbf.v4.new_markdown_cell(TEXT_INTRO),
        nbf.v4.new_code_cell(CODE_IMPORTS),
        nbf.v4.new_code_cell(CODE_LOAD),
        nbf.v4.new_code_cell(CODE_TUNE),
        nbf.v4.new_code_cell(CODE_EVAL),
        nbf.v4.new_code_cell(CODE_SAVE),
    ]

    print("Executing notebook...")
    ep = ExecutePreprocessor(timeout=600, kernel_name="python3")
    ep.preprocess(nb, {"metadata": {"path": str(BASE / "notebooks")}})

    with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"Notebook successfully executed and saved to {NOTEBOOK_PATH}")

if __name__ == "__main__":
    main()
