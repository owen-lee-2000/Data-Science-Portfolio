# ============================================================================
# SUPERVISED MACHINE LEARNING & PREDICTIVE MODELING PIPELINE
# Python / Google Colab
# Focus: Tree-Based Ensemble Classification and Performance Metrics
# ============================================================================

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, roc_curve, auc

# Setting chart styling to look clean
sns.set_theme(style="whitegrid")

# ============================================================================
# 1. DATA INGESTION & PREPARATION
# ============================================================================
# Define the path to the wine quality dataset
DATA_PATH = "winequality.csv"

# Verify file existence and load into the main data frame
if os.path.exists(DATA_PATH):
    df = pd.read_csv(DATA_PATH)
    print(f"Dataset loaded successfully from: '{DATA_PATH}'")
    print(f"Dimensions: {df.shape[0]} rows, {df.shape[1]} features.\n")
else:
    raise FileNotFoundError(f"Target data resource not detected at path: '{DATA_PATH}'")

# Split the dataset into training (75%) and testing (25%) sets with a fixed seed for reproducibility
train_data, test_data = train_test_split(df, test_size=0.25, random_state=42)

# Isolate features (X) and the target outcome variable (y)
X_train = train_data.drop('quality', axis=1)
y_train = train_data['quality']
X_test = test_data.drop('quality', axis=1)
y_test = test_data['quality']


# ============================================================================
# 2. RANDOM FOREST CLASSIFIER
# ============================================================================
# Initialize and fit the Random Forest model
rf = RandomForestClassifier(random_state=1)
rf.fit(X_train, y_train)

# Extract and format feature importance metrics
rf_importance = pd.DataFrame({'Importance': rf.feature_importances_ * 100}, index=X_train.columns)
rf_importance = rf_importance.sort_values('Importance', ascending=False)[0:20]

# Plot Random Forest feature importance
plt.figure(figsize=(8, 5))
rf_importance.plot(kind='barh', color='steelblue', legend=False).invert_yaxis()
plt.title('Feature Importance Top 20: Random Forest', fontsize=13, fontweight='bold', pad=12)
plt.xlabel('Relative Importance Score')
plt.tight_layout()
plt.show()

# Predict probabilities on the test set for the positive class
pred_rf = rf.predict_proba(X_test)[:, 1]

# Calculate ROC curve metrics and the Area Under the Curve (AUC)
fpr_rf, tpr_rf, _ = roc_curve(y_test, pred_rf)
roc_auc_rf = auc(fpr_rf, tpr_rf)

# Generate predictions using a standard 50% classification threshold
rf_class_preds = (pred_rf > 0.5).astype(int)
print("--- Random Forest Confusion Matrix (50% Threshold) ---")
print(confusion_matrix(y_test, rf_class_preds))


# ============================================================================
# 3. GRADIENT BOOSTING CLASSIFIER
# ============================================================================
# Initialize and fit the Gradient Boosting model
boost = GradientBoostingClassifier(random_state=1)
boost.fit(X_train, y_train)

# Extract and sort feature importance metrics
boost_importance = pd.Series(boost.feature_importances_ * 100, index=X_train.columns).sort_values(ascending=False)
boost_importance = boost_importance[0:20]

# Plot Gradient Boosting feature importance
plt.figure(figsize=(8, 5))
boost_importance.plot(kind='barh', color='seagreen').invert_yaxis()
plt.title('Feature Importance Top 20: Gradient Boosting', fontsize=13, fontweight='bold', pad=12)
plt.xlabel('Relative Importance Score')
plt.tight_layout()
plt.show()

# Predict probabilities on the test set for the positive class
pred_boost = boost.predict_proba(X_test)[:, 1]

# Calculate ROC curve metrics and the Area Under the Curve (AUC)
fpr_boost, tpr_boost, _ = roc_curve(y_test, pred_boost)
roc_auc_boost = auc(fpr_boost, tpr_boost)

# Generate predictions using a standard 50% classification threshold
boost_class_preds = (pred_boost > 0.5).astype(int)
print("\n--- Gradient Boosting Confusion Matrix (50% Threshold) ---")
print(confusion_matrix(y_test, boost_class_preds))


# ============================================================================
# 4. MODEL COMPARISON VISUALIZATION
# ============================================================================
# Overlay both ROC curves onto a single chart for direct performance analysis
plt.figure(figsize=(8, 6))
plt.title('Model Performance Comparison: ROC Curves', fontsize=14, fontweight='bold', pad=15)

plt.plot(fpr_rf, tpr_rf, color='steelblue', lw=2, label=f'Random Forest (AUC = {roc_auc_rf:.2f})')
plt.plot(fpr_boost, tpr_boost, color='seagreen', lw=2, label=f'Gradient Boosting (AUC = {roc_auc_boost:.2f})')

# Reference line representing a random-guess baseline
plt.plot([0, 1], [0, 1], color='crimson', linestyle='--', label='Baseline (Random Guess)')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False... Positive Rate (1 - Specificity)', fontsize=11)
plt.ylabel('True Positive Rate (Sensitivity)', fontsize=11)
plt.legend(loc='lower right', frameon=True)
plt.tight_layout()
plt.show()
