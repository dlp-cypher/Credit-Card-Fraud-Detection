# Credit-Card-Fraud-Detection

# 🚀 Enhanced Credit Card Fraud Detection with SHAP Analysis

## 📌 Overview
This project is a **credit card fraud detection system** that uses:
- **Random Forest Classifier** for robust classification
- **SMOTE** to handle severe class imbalance
- **SHAP (SHapley Additive exPlanations)** for explainable AI

It not only detects fraudulent transactions but also **explains** why the model made its predictions, providing **actionable insights** for analysts and stakeholders.

---

## ✨ Features
- **Data Preprocessing**: Loads and prepares credit card transaction data
- **Class Imbalance Handling**: Uses SMOTE to oversample minority fraud cases
- **Custom Thresholding**: Adjusts decision boundaries for improved fraud detection
- **Model Evaluation**: Outputs confusion matrix, classification report, and ROC AUC score
- **Feature Importance**: Visualizes the top features driving fraud predictions
- **SHAP Analysis**: Offers global and local model interpretability
- **Business Insights**: Generates recommendations based on model behavior

---

## 🛠️ Tech Stack
- Python 3.8+
- Pandas / NumPy for data manipulation
- Scikit-learn for model training and evaluation
- Imbalanced-learn (SMOTE) for class balancing
- SHAP for model interpretability
- Matplotlib / Seaborn for visualization

---

## 📂 Project Structure


---

## 📊 Dataset
**Source**: [Credit Card Fraud Detection Dataset - Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud)  
**Description**:
- Transactions made by European cardholders in September 2013
- 284,807 transactions, 492 of which are fraud (0.172%)
- Features are numerical and PCA-transformed for confidentiality

---

## 🚀 How to Run

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/fraud-detection-shap.git
   cd fraud-detection-shap

