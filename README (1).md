# CART Classification - Bank Marketing Dataset

## Assignment
Implementation of the Classification and Regression Tree (CART) algorithm in Python to predict whether a bank customer subscribes to a term deposit.

## Files
- `CART_Bank_Marketing_Assignment.ipynb` — complete, step-by-step Jupyter Notebook with explanations, code, executed outputs, visualizations, model evaluation, and conclusion.
- `bank-full.csv` — input dataset used by the notebook.
- `requirements.txt` — Python dependencies.

## Dataset
- 45,211 rows and 17 columns.
- Target column: `y`
  - `no` = 0 (did not subscribe)
  - `yes` = 1 (subscribed)
- CSV delimiter: semicolon (`;`).

## Algorithm and methodology
- CART classification using `sklearn.tree.DecisionTreeClassifier`.
- Splitting criterion: Gini impurity.
- Train/test split: 80:20, stratified, `random_state=42`.
- Tree controls: `max_depth=5`, `min_samples_leaf=50`.
- `class_weight='balanced'` accounts for the minority `yes` class.
- Numerical values are median-imputed; categorical values are most-frequent-imputed and one-hot encoded.
- Evaluation: accuracy, precision, recall, F1-score, confusion matrix, and ROC-AUC.

## Important modelling decision
The `duration` column is intentionally excluded. It represents the length of the contact and is only known after the contact occurs. Excluding it avoids using post-contact information when the intended task is to predict subscription before contacting the customer.

## How to run
1. Keep `CART_Bank_Marketing_Assignment.ipynb` and `bank-full.csv` in the same directory.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Open the notebook in Jupyter Notebook, JupyterLab, or VS Code.
4. Run all cells from top to bottom.

## Notes
- The notebook uses the exact uploaded `bank-full.csv` dataset.
- The notebook follows the supplied Titanic CART reference notebook's step-by-step structure, adapted to the Bank Marketing dataset.
- Manual Gini calculations demonstrate selected candidate splits. The Scikit-learn CART estimator performs its own split search during model fitting.
- The notebook was executed successfully and includes output cells.
- Metrics can vary if the dataset, split, preprocessing, or tree parameters are changed.
