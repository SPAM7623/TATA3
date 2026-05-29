"""
========================================================
PHASE 10: OVERFITTING ANALYSIS - Measure Generalization Gap
========================================================
Objective: Measure generalization gap.

COMPARE: Train vs OOF

KEY FINDING: Recall survives generalization

FEEDS INTO: → Ensemble Analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import roc_auc_score, average_precision_score, recall_score
import pickle
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("PHASE 10: OVERFITTING ANALYSIS - STARTING")
print("="*80)

train_df = pd.read_csv('train.csv')
with open('05_engineering_insights.pkl', 'rb') as f:
    eng = pickle.load(f)
with open('06_imbalance_insights.pkl', 'rb') as f:
    imb = pickle.load(f)
with open('08_threshold_insights.pkl', 'rb') as f:
    thresh = pickle.load(f)

X_engineered = eng['X_engineered']
y = train_df['Y']
selected_params = imb['selected_params']
selected_threshold = thresh['selected_threshold']

# Train on full set, evaluate OOF
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
train_metrics = []
oof_metrics = []

for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_engineered, y)):
    X_train, X_val = X_engineered.iloc[train_idx], X_engineered.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    model = xgb.XGBClassifier(**selected_params)
    model.fit(X_train, y_train, verbose=False)

    # Train predictions
    y_train_pred_proba = model.predict_proba(X_train)[:, 1]
    y_train_pred = (y_train_pred_proba >= selected_threshold).astype(int)

    train_roc = roc_auc_score(y_train, y_train_pred_proba)
    train_recall = recall_score(y_train, y_train_pred, zero_division=0)

    # OOF predictions
    y_val_pred_proba = model.predict_proba(X_val)[:, 1]
    y_val_pred = (y_val_pred_proba >= selected_threshold).astype(int)

    oof_roc = roc_auc_score(y_val, y_val_pred_proba)
    oof_recall = recall_score(y_val, y_val_pred, zero_division=0)

    train_metrics.append({'ROC_AUC': train_roc, 'Recall': train_recall})
    oof_metrics.append({'ROC_AUC': oof_roc, 'Recall': oof_recall})

    print(f"  Fold {fold_idx+1}: Train ROC={train_roc:.4f}, OOF ROC={oof_roc:.4f} (gap: {train_roc-oof_roc:.4f})")

train_df_metrics = pd.DataFrame(train_metrics)
oof_df_metrics = pd.DataFrame(oof_metrics)

print(f"\n✓ Generalization Gap Analysis:")
print(f"\n  ROC-AUC:")
print(f"    Train: {train_df_metrics['ROC_AUC'].mean():.4f}")
print(f"    OOF:   {oof_df_metrics['ROC_AUC'].mean():.4f}")
print(f"    Gap:   {(train_df_metrics['ROC_AUC'].mean() - oof_df_metrics['ROC_AUC'].mean()):.4f}")

print(f"\n  Recall:")
print(f"    Train: {train_df_metrics['Recall'].mean():.4f}")
print(f"    OOF:   {oof_df_metrics['Recall'].mean():.4f}")
print(f"    Gap:   {(train_df_metrics['Recall'].mean() - oof_df_metrics['Recall'].mean()):.4f}")

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

folds = list(range(1, 6))
axes[0].plot(folds, train_df_metrics['ROC_AUC'], 'o-', label='Train', linewidth=2, markersize=8)
axes[0].plot(folds, oof_df_metrics['ROC_AUC'], 's-', label='OOF', linewidth=2, markersize=8)
axes[0].fill_between(folds, train_df_metrics['ROC_AUC'], oof_df_metrics['ROC_AUC'], alpha=0.2)
axes[0].set_xlabel('Fold')
axes[0].set_ylabel('ROC-AUC')
axes[0].set_title('ROC-AUC: Train vs OOF (Overfitting)')
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].plot(folds, train_df_metrics['Recall'], 'o-', label='Train', linewidth=2, markersize=8)
axes[1].plot(folds, oof_df_metrics['Recall'], 's-', label='OOF', linewidth=2, markersize=8)
axes[1].fill_between(folds, train_df_metrics['Recall'], oof_df_metrics['Recall'], alpha=0.2)
axes[1].set_xlabel('Fold')
axes[1].set_ylabel('Recall')
axes[1].set_title('Recall: Train vs OOF')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('10_overfitting_analysis.png', dpi=150, bbox_inches='tight')
print("\n✓ Saved: 10_overfitting_analysis.png")
plt.close()

with open('10_overfitting_insights.pkl', 'wb') as f:
    pickle.dump({'train_metrics': train_df_metrics, 'oof_metrics': oof_df_metrics}, f)

print("✓ Saved: 10_overfitting_insights.pkl")
print("\n" + "="*80)
print("✓ PHASE 10 COMPLETE")
print("="*80)
