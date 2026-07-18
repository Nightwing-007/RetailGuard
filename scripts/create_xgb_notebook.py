import nbformat as nbf

nb = nbf.v4.new_notebook()

text = """\
# Phase 5.5: XGBoost Optimization
In this notebook, we load our finalized datasets, train and tune an XGBoost model to optimize the F1 score for the minority class, and evaluate it on the test set. If it outperforms the baseline Logistic Regression model, we save it as our new best model.
"""

code_imports = """\
import pandas as pd
import numpy as np
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import warnings
warnings.filterwarnings('ignore')
"""

code_load = """\
# Load finalized datasets
X_train_scaled = pd.read_csv('../data/X_train_scaled.csv')
X_test_scaled = pd.read_csv('../data/X_test_scaled.csv')
y_train_resampled = pd.read_csv('../data/y_train_resampled.csv').values.ravel()
y_test = pd.read_csv('../data/y_test.csv').values.ravel()

print("Data loaded successfully.")
print("X_train shape:", X_train_scaled.shape)
print("X_test shape:", X_test_scaled.shape)
"""

code_tune = """\
# Initialize XGBClassifier
xgb = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')

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
"""

code_eval = """\
# Evaluate the tuned model on the test set
best_xgb = random_search.best_estimator_
y_pred = best_xgb.predict(X_test_scaled)

print("Classification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Get F1 score for minority class (assuming minority class is 1)
test_f1 = f1_score(y_test, y_pred)
print(f"Test F1 Score (Minority Class): {test_f1:.4f}")
"""

code_save = """\
# Compare with baseline and save
baseline_f1 = 0.1576

if test_f1 > baseline_f1:
    print(f"New model outperforms baseline! ({test_f1:.4f} > {baseline_f1:.4f})")
    joblib.dump(best_xgb, '../data/best_model.joblib')
    print("New best model saved to data/best_model.joblib")
else:
    print(f"New model did not outperform baseline. ({test_f1:.4f} <= {baseline_f1:.4f})")
    print("Keeping the previous best model.")
"""

nb['cells'] = [
    nbf.v4.new_markdown_cell(text),
    nbf.v4.new_code_cell(code_imports),
    nbf.v4.new_code_cell(code_load),
    nbf.v4.new_code_cell(code_tune),
    nbf.v4.new_code_cell(code_eval),
    nbf.v4.new_code_cell(code_save)
]

with open('../notebooks/05_model_tuning_xgb.ipynb', 'w') as f:
    nbf.write(nb, f)
