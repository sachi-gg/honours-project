# Anemia Detection with Machine Learning

## Overview
This repository contains a comprehensive machine learning pipeline and interactive web application for the detection and diagnosis of anemia using Complete Blood Count (CBC) clinical laboratory attributes.

The dataset comprises 1,421 patient records with five diagnostic blood profile attributes and a target classification label:
- **Gender**: 0 (Male), 1 (Female)
- **Hemoglobin**: Hemoglobin level (g/dL)
- **MCH**: Mean Corpuscular Hemoglobin (pg)
- **MCHC**: Mean Corpuscular Hemoglobin Concentration (g/dL)
- **MCV**: Mean Corpuscular Volume (fL)
- **Result**: Binary diagnostic outcome (`0` = Non-Anemic, `1` = Anemic)

---

## Clinical Background
Anemia is a medical condition characterized by a deficiency of healthy red blood cells or a reduction in blood hemoglobin levels below normal thresholds. Hemoglobin is the vital iron-rich protein responsible for transporting oxygen from the respiratory system throughout bodily tissues.

Common symptoms include fatigue, weakness, dyspnea, pallor, dizziness, and cognitive impairment. Early, automated screening using routine hematological parameters can assist clinicians with rapid triage and timely interventions.

---

## Machine Learning Pipeline
1. **Exploratory Data Analysis (EDA)**: Distribution inspection, outlier detection, and correlation structure.
2. **Statistical Hypothesis Testing**: Two-sample t-tests, odds ratio evaluations, and Chi-square association tests.
3. **Feature Selection**:
   - Correlation analysis
   - SelectKBest (ANOVA F-value)
   - Extra Trees feature importance
4. **Feature Scaling & Transformation**:
   - Log transformation
   - StandardScaler (Z-score standardization)
   - MinMaxScaler (Normalization)
5. **Class Imbalance Mitigation**:
   - Random Undersampling
   - Random Oversampling
   - SMOTE (Synthetic Minority Over-sampling Technique)
   - ADASYN (Adaptive Synthetic)
6. **Data Leakage Prevention**: Pipeline segregation with strict cross-validation splitting.
7. **Supervised Classification Models**:
   - Random Forest (RF)
   - Decision Tree (DT)
   - Logistic Regression (LR)
   - K-Nearest Neighbors (KNN)
   - Support Vector Machine (SVM)
   - Gaussian Naive Bayes (GNB)
8. **Evaluation Metrics**:
   - Classification Accuracy
   - ROC-AUC (Area Under the Receiver Operating Characteristic Curve)
   - Precision, Recall, and F1-Score
   - Cohen's Kappa Coefficient
9. **Hyperparameter Optimization**: Systematic Grid Search with Stratified K-Fold Cross-Validation.

---

## Installation & Setup

### Prerequisites
- Python 3.9+
- Recommended: Virtual environment (`venv` or `conda`)

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train / Retrain Models
```bash
python train.py
```

### 3. Launch Interactive Web Application
```bash
streamlit run app.py
```
