# Credit Card Fraud Detection using Logistic Regression

This project aims to detect fraudulent credit card transactions using a logistic regression model. The goal is to minimize false positives while maintaining a high recall for fraud detection.

## Objective

- Classify credit card transactions as fraudulent or non-fraudulent.
- Use logistic regression on real-world anonymized data.
- Visualize data imbalance, model performance, and confusion matrix.

## Dataset

- **Source**: [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- 284,807 transactions
- 492 fraud cases
- Features: V1–V28 (PCA components), Time, Amount
- Target: Class (0 = non-fraud, 1 = fraud)

## Tech Stack

- Python 3
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn
- Jupyter Notebook
- Google Colab

## Workflow

### 1. Data Loading & Preprocessing

- Mounted Google Drive and loaded dataset.
- Normalized 'Amount' and 'Time' features.
- Detected and handled data imbalance.

### 2. Exploratory Data Analysis (EDA)

- Scatter Plot Amount Fraud:

  ![Class Distribution](https://github.com/user-attachments/assets/6edb1c99-cc24-4243-b949-b0554d2ce562)

### 3. Model Training

- Split dataset using `train_test_split`.
- Applied Logistic Regression with `class_weight='balanced'`.

### 4. Evaluation

- Confusion matrix and classification report:

  ![Confusion Matrix](https://github.com/user-attachments/assets/b927bbb2-949d-4e8f-a527-0252da3c5352)

## Results

| Metric       | Value |
|--------------|-------|
| Accuracy     | 99.2% |
| Precision    | 1.00  |
| Recall       | 0.96  |
| F1 Score     | 0.98  |
| ROC AUC      | 95%   |

## How to Run

1. Clone the repo or open in Google Colab.
2. Mount Google Drive (already included in notebook).
3. Install requirements (if running locally):
4. pip install pandas numpy matplotlib seaborn scikit-learn
5. Run the notebook step by step.

## Future Work

- Test with SMOTE or ADASYN for oversampling
- Explore tree-based models (Random Forest, XGBoost)
- Deploy with Streamlit or Flask
