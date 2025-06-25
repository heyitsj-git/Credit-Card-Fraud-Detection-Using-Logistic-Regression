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

- Checked class distribution:

  ![Class Distribution](images/output_6_0.png)

- Plotted correlations and boxplots:

  ![Correlation Heatmap](images/output_13_0.png)

### 3. Model Training

- Split dataset using `train_test_split`.
- Applied Logistic Regression with `class_weight='balanced'`.

### 4. Evaluation

- Confusion matrix and classification report:

  ![Confusion Matrix](images/output_29_0.png)

- ROC-AUC Curve:

  ![ROC Curve](images/output_40_0.png)

- Precision-Recall curve:

  ![Precision Recall](images/output_45_0.png)

## Results

| Metric       | Value (sample) |
|--------------|----------------|
| Accuracy     | ~99.2%         |
| Precision    | ~81.5%         |
| Recall       | ~90.2%         |
| F1 Score     | ~85.6%         |
| ROC AUC      | ~95%           |

*(Replace with exact numbers from your report)*

## How to Run

1. Clone the repo or open in Google Colab.
2. Mount Google Drive (already included in notebook).
3. Install requirements (if running locally):
4. pip install pandas numpy matplotlib seaborn scikit-learn
5. Run the notebook step by step.

## Folder Structure
project/
│
├── Credit_Card_Fraud_Detection_Using_Logistic_Regression.ipynb
├── README.md
└── images/
├── output_6_0.png
├── output_13_0.png
├── output_29_0.png
├── output_40_0.png
└── output_45_0.png


## Future Work

- Test with SMOTE or ADASYN for oversampling
- Explore tree-based models (Random Forest, XGBoost)
- Deploy with Streamlit or Flask
