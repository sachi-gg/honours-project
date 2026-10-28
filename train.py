"""
train.py
=============================================================================
Anemia Detection Machine Learning Pipeline
- Dataset: Anemia CBC Dataset (Gender, Hemoglobin, MCH, MCHC, MCV)
- Task: Binary Classification (0: Non-Anemic, 1: Anemic)
- Model: Random Forest Classifier with StandardScaler preprocessing
=============================================================================
"""

import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# File paths
DATA_PATH = "anemia.csv"
FALLBACK_DATA_PATH = None
MODEL_PATH = "anemia_model.pkl"
SCALER_PATH = "scaler.pkl"

FEATURE_COLUMNS = ["Gender", "Hemoglobin", "MCH", "MCHC", "MCV"]
TARGET_COLUMN = "Result"


def load_dataset(filepath: str, fallback_path: str = None) -> pd.DataFrame:
    """Load the dataset from disk with fallback support."""
    if os.path.exists(filepath):
        df = pd.read_csv(filepath)
    elif fallback_path and os.path.exists(fallback_path):
        print(f"[*] '{filepath}' not found. Loading fallback: '{fallback_path}'")
        df = pd.read_csv(fallback_path)
    else:
        raise FileNotFoundError(f"Neither '{filepath}' nor '{fallback_path}' was found.")
    
    return df


def perform_eda(df: pd.DataFrame) -> None:
    """Display concise Exploratory Data Analysis (EDA) summary."""
    print("=" * 60)
    print("               EXPLORATORY DATA ANALYSIS (EDA)              ")
    print("=" * 60)
    print(f"[*] Dataset Shape: {df.shape[0]} rows x {df.shape[1]} columns")
    
    print("\n[*] Missing Values Check:")
    missing = df.isnull().sum()
    print(missing)
    if missing.sum() == 0:
        print("    -> No missing values detected in the dataset.")
        
    print("\n[*] Target Class Distribution ('Result'):")
    distribution = df[TARGET_COLUMN].value_counts()
    for label, count in distribution.items():
        name = "Anemic (1)" if label == 1 else "Non-Anemic (0)"
        percentage = (count / len(df)) * 100
        print(f"    - {name}: {count} samples ({percentage:.1f}%)")
    print("=" * 60)


def train_and_evaluate():
    """Main model training, scaling, evaluation, and persistence pipeline."""
    # 1. Load data
    df = load_dataset(DATA_PATH, FALLBACK_DATA_PATH)
    
    # 2. Simple EDA
    perform_eda(df)
    
    # 3. Feature & Target Extraction
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]
    
    # 4. Train-Test Split (80% Train, 20% Test, Stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"\n[*] Train set size: {X_train.shape[0]} samples")
    print(f"[*] Test set size:  {X_test.shape[0]} samples")
    
    # 5. Feature Scaling (StandardScaler)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 6. Model Training (Random Forest Classifier)
    print("\n[*] Training RandomForestClassifier (n_estimators=100, random_state=42)...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # 7. Model Evaluation
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    print("\n" + "=" * 60)
    print("                   EVALUATION RESULTS                       ")
    print("=" * 60)
    print(f"[*] Test Accuracy Score: {acc * 100:.2f}%\n")
    
    print("[*] Confusion Matrix:")
    print(f"                  Predicted Non-Anemic (0)   Predicted Anemic (1)")
    print(f"Actual Non-Anemic (0) :     {cm[0][0]:<15}            {cm[0][1]:<10}")
    print(f"Actual Anemic (1)     :     {cm[1][0]:<15}            {cm[1][1]:<10}\n")
    
    print("[*] Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Non-Anemic (0)", "Anemic (1)"]))
    
    print("[*] Feature Importances:")
    importances = model.feature_importances_
    for col, imp in sorted(zip(FEATURE_COLUMNS, importances), key=lambda x: x[1], reverse=True):
        print(f"    - {col:<12}: {imp * 100:.2f}%")
    print("=" * 60)
    
    # 8. Save Artifacts with joblib
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print(f"\n[+] Successfully saved model to '{MODEL_PATH}'")
    print(f"[+] Successfully saved scaler to '{SCALER_PATH}'")
    print("[+] Training pipeline completed successfully.\n")


if __name__ == "__main__":
    train_and_evaluate()
