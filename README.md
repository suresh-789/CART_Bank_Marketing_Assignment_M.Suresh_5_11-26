# CART Classification for Bank Marketing

> **Machine Learning Assignment | Supervised Learning | Decision Tree Classification**

## 1. Project Overview

This project implements the **Classification and Regression Tree (CART)** algorithm in Python to predict whether a bank customer will subscribe to a term deposit following a marketing campaign.

The implementation uses the **Bank Marketing dataset** (`bank-full.csv`) and follows a complete machine-learning workflow: data inspection, preprocessing, Gini impurity analysis, model training, prediction, evaluation, and interpretation.

The notebook is designed as an academic assignment, with explanations before the relevant code cells and visualizations of the key results.

## 2. Problem Statement

A bank contacts customers as part of a term-deposit marketing campaign. The objective is to build a classification model that estimates whether a customer will subscribe to the term deposit.

This is a **binary classification** problem:

| Target value | Meaning | Encoded value |
|---|---|---:|
| `no` | Customer did not subscribe | 0 |
| `yes` | Customer subscribed | 1 |

The model learns patterns from historical customer and campaign information and predicts the target class for previously unseen records.

## 3. Project Objectives

- Understand and inspect the Bank Marketing dataset.
- Identify the independent variables and the target variable.
- Examine data quality, missing values, duplicate records, and target-class distribution.
- Understand **Gini impurity** and **weighted Gini impurity**.
- Demonstrate how candidate binary splits can reduce impurity.
- Preprocess numerical and categorical features appropriately.
- Implement a CART classifier using Scikit-learn.
- Visualize the trained decision tree.
- Evaluate the classifier using suitable classification metrics.
- Interpret feature importance and summarize the findings.

## 4. Dataset Information

**Dataset file:** `bank-full.csv`

| Property | Description |
|---|---|
| Dataset | Bank Marketing (`bank-full.csv`) |
| Number of records | 45,211 |
| Number of columns | 17 |
| Target column | `y` |
| Problem type | Binary classification |
| File delimiter | Semicolon (`;`) |

The dataset contains customer attributes and information related to a bank's direct-marketing campaign. The target column `y` indicates whether the customer subscribed to a term deposit.

### Target variable

- `no` — the customer did not subscribe.
- `yes` — the customer subscribed.

The notebook checks the actual class distribution before training. The classes are not equally represented, so accuracy should not be interpreted on its own.

## 5. Important Modelling Decision: Avoiding Data Leakage

The feature **`duration` is intentionally excluded** from model training.

`duration` represents the length of the last contact with the customer. It is only known after the contact has taken place. If the intended use is to estimate subscription before contacting a customer, using this column would introduce information that would not yet be available at prediction time.

Excluding it makes the modelling setup more appropriate for a pre-contact prediction task. The target column `y` is also excluded from the input features.

## 6. What Is CART?

**CART** stands for **Classification and Regression Trees**. It is a supervised-learning algorithm that builds a decision tree by repeatedly splitting the data into two child groups.

For classification, CART can use Gini impurity to select splits. A good split creates child nodes that contain more homogeneous target classes than the parent node.

### Gini impurity

\[
Gini = 1 - \sum_{i=1}^{k} p_i^2
\]

where:

- \(k\) is the number of classes.
- \(p_i\) is the proportion of observations belonging to class \(i\).

For this binary classification task:

\[
Gini = 1 - (p_{no}^{2} + p_{yes}^{2})
\]

### Weighted Gini impurity after a split

\[
Gini_{split} =
\frac{N_L}{N}Gini_L +
\frac{N_R}{N}Gini_R
\]

where:

- \(N\) is the number of observations in the parent node.
- \(N_L\) and \(N_R\) are the observations in the left and right child nodes.
- \(Gini_L\) and \(Gini_R\) are the impurity values of those child nodes.

The algorithm prefers a split with a lower weighted Gini impurity, which means a greater reduction in impurity relative to the parent node.

## 7. Tools and Technologies

| Tool / Library | Purpose |
|---|---|
| Python | Programming language |
| Jupyter Notebook | Interactive development and presentation |
| Pandas | Data loading, inspection, and manipulation |
| NumPy | Numerical calculations |
| Matplotlib | Plotting and visualizations |
| Seaborn | Statistical visualizations |
| Scikit-learn | Preprocessing, CART model, and evaluation |

## 8. Methodology and Workflow

The notebook is organized into the following stages:

1. **Import libraries** — load the Python packages required for data analysis, modelling, and visualization.
2. **Load the dataset** — read `bank-full.csv` using the semicolon delimiter.
3. **Explore the data** — inspect sample records, dimensions, data types, and descriptive statistics.
4. **Check data quality** — examine missing values and duplicate records.
5. **Analyze the target** — inspect the number and percentage of `yes` and `no` records.
6. **Prepare features and target** — separate `X` and `y`, encode the target as 0 and 1, and exclude `duration`.
7. **Identify feature types** — separate numerical and categorical columns.
8. **Split the data** — create stratified training and testing sets.
9. **Understand Gini impurity** — calculate parent impurity and demonstrate weighted Gini for selected candidate splits.
10. **Preprocess features** — impute missing values and one-hot encode categorical variables.
11. **Train CART** — fit a decision tree classifier using the Gini criterion.
12. **Visualize the tree** — display the learned decision rules.
13. **Generate predictions** — predict classes and subscription probabilities for test records.
14. **Evaluate the model** — calculate classification metrics and plot the confusion matrix and ROC curve.
15. **Interpret the model** — inspect feature importance and summarize the results.

## 9. Model Configuration

The notebook uses the following CART configuration:

| Parameter | Value | Purpose |
|---|---:|---|
| `criterion` | `gini` | Select splits using Gini impurity |
| `max_depth` | `5` | Limit the maximum tree depth |
| `min_samples_leaf` | `50` | Require at least 50 training records in a leaf |
| `class_weight` | `balanced` | Adjust class weights for the imbalanced target |
| `random_state` | `42` | Make the model's randomized behaviour reproducible |

The model is implemented with Scikit-learn's `DecisionTreeClassifier`, which uses a binary CART-style decision tree.

### Data preprocessing

- **Numerical features:** missing values are filled using the median.
- **Categorical features:** missing values are filled using the most frequent category, then one-hot encoded.
- **Unseen categories:** `handle_unknown="ignore"` allows the encoder to handle categories that were not present in the training data.

Preprocessing is placed inside a Scikit-learn pipeline. This ensures that imputation and encoding are fitted on the training data rather than on the complete dataset.

### Train-test split

- Training data: 80%
- Testing data: 20%
- `random_state=42`
- `stratify=y`

Stratification helps preserve the target-class proportions in both subsets.

## 10. Model Evaluation

The notebook evaluates the model on the held-out test set using the following measures:

| Metric | What it tells us |
|---|---|
| Accuracy | Overall proportion of correct predictions |
| Precision | Of the customers predicted as `yes`, the proportion who actually subscribed |
| Recall | Of all customers who actually subscribed, the proportion identified by the model |
| F1-score | Harmonic mean of precision and recall |
| Confusion matrix | Counts of correct and incorrect predictions for each class |
| ROC-AUC | How well the model ranks positive cases above negative cases across classification thresholds |

For this project, the positive class is **`yes` (subscribed)**.

Because the target classes are imbalanced, precision, recall, F1-score, and the confusion matrix should be considered alongside accuracy. In a marketing context, recall can indicate how many potential subscribers are identified, while precision indicates how many predicted subscribers actually subscribed.

The notebook calculates the actual scores when its cells are executed. Results should be reported from the executed notebook rather than assumed in advance.

## 11. Repository Structure

Keep the files in the following structure:

```text
CART-Bank-Marketing/
│
├── CART_Bank_Marketing_Assignment.ipynb
├── bank-full.csv
├── README.md
└── requirements.txt
```

**Important:** The notebook expects `bank-full.csv` to be in the same directory as the notebook.

## 12. Installation and Execution

### Prerequisites

- Python 3.9 or later
- Jupyter Notebook, JupyterLab, or Visual Studio Code with the Jupyter extension

### Step 1: Open the project directory

Open a terminal in the directory containing the notebook and dataset.

### Step 2: Install the dependencies

```bash
pip install -r requirements.txt
```

Alternatively, install the packages directly:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter ipykernel
```

### Step 3: Launch Jupyter Notebook

```bash
jupyter notebook
```

### Step 4: Open and run the notebook

Open `CART_Bank_Marketing_Assignment.ipynb` and run the cells from top to bottom.

Make sure `bank-full.csv` is present in the same folder. Running the notebook in order will load the data, train the CART classifier, display the visualizations, and calculate the evaluation metrics.

## 13. Expected Outputs

After running all cells, the notebook produces:

- Dataset previews, dimensions, data types, and statistical summaries.
- Missing-value and duplicate-record checks.
- Target-class counts and a target-distribution chart.
- Parent Gini impurity and selected weighted-Gini split calculations.
- Training and testing set information.
- A fitted CART classification model.
- A decision-tree visualization.
- Actual-versus-predicted sample results.
- Accuracy, precision, recall, F1-score, and ROC-AUC.
- A confusion matrix and ROC curve.
- A ranked feature-importance table and chart.
- A final model-performance summary and conclusion.

## 14. Limitations and Considerations

- The dataset represents historical campaign records; patterns in historical data may not remain unchanged.
- The target classes are imbalanced. A high accuracy score does not necessarily mean that the model identifies subscribers well.
- Excluding `duration` is appropriate for the stated pre-contact prediction objective, but results may differ from models that include it.
- The manually calculated Gini examples inspect selected candidate splits for learning purposes. They are not an exhaustive search of every possible split. The Scikit-learn CART estimator performs its own split search during training.
- Feature importance describes how the fitted tree uses features to reduce impurity. It does not establish that a feature causes a customer to subscribe.
- This is an educational machine-learning assignment, not a validated system for making real banking or customer-contact decisions.

## 15. Conclusion

This project demonstrates an end-to-end implementation of CART classification using the Bank Marketing dataset. It covers data exploration, preprocessing, Gini impurity, binary decision-tree training, prediction, evaluation, and feature-importance analysis.

The trained model predicts whether a customer subscribed to a term deposit. Its performance is assessed using multiple classification metrics so that both overall correctness and the identification of the minority `yes` class can be understood.

The project also demonstrates two important machine-learning practices: using a training-only preprocessing pipeline to reduce data leakage, and excluding the post-contact `duration` feature when the prediction is intended to be made before contacting the customer.

---

## Author / Submission Details

Fill in these details before submitting:

- **Student Name:** M Suresh
- **Student ID:** 2300030390
- **Assignment Title:** Implementation of CART Using Python
- **Dataset:** Bank Marketing Dataset
- **Programming Language:** Python
