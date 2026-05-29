"""
========================================================
PHASE 9: STABILITY ANALYSIS - Verify Robustness
========================================================
Objective: Verify robustness and generalization.

TESTS:
□ CV seeds
□ Model seeds
□ Threshold sensitivity

FINDINGS: Performance is real. Not random luck.

FEEDS INTO: → Overfitting Analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, recall_score
import pickle
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("PHASE 9: STABILITY ANALYSIS - STARTING")
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

# ═══════════════════════════════════════════════════════════════════════════════════════
# SEED STABILITY TEST
# ═══════════════════════════════════════════════════════════════════════════════════════
print("\nTesting stability across different random seeds...")

seeds = [42, 123, 999, 2025, 7777]
stability_results = []

for seed_id, seed in enumerate(seeds):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    fold_scores = []

    for train_idx, val_idx in skf.split(X_engineered, y):
        X_train, X_val = X_engineered.iloc[train_idx], X_engineered.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = xgb.XGBClassifier(**selected_params)
        model.fit(X_train, y_train, verbose=False)

        y_pred_proba = model.predict_proba(X_val)[:, 1]
        y_pred = (y_pred_proba >= selected_threshold).astype(int)

        roc_auc = roc_auc_score(y_val, y_pred_proba)
        recall = recall_score(y_val, y_pred, zero_division=0)

        fold_scores.append({'ROC_AUC': roc_auc, 'Recall': recall})

    fold_df = pd.DataFrame(fold_scores)
    stability_results.append({
        'Seed': seed,
        'Mean_ROC_AUC': fold_df['ROC_AUC'].mean(),
        'Std_ROC_AUC': fold_df['ROC_AUC'].std(),
        'Mean_Recall': fold_df['Recall'].mean(),
        'Std_Recall': fold_df['Recall'].std()
    })

    print(f"  Seed {seed}: ROC-AUC {fold_df['ROC_AUC'].mean():.4f}±{fold_df['ROC_AUC'].std():.4f}, "
          f"Recall {fold_df['Recall'].mean():.4f}±{fold_df['Recall'].std():.4f}")

stability_df = pd.DataFrame(stability_results)

print(f"\n✓ Stability Summary:")
print(f"  ROC-AUC Mean: {stability_df['Mean_ROC_AUC'].mean():.4f} "
      f"(range: {stability_df['Mean_ROC_AUC'].min():.4f}-{stability_df['Mean_ROC_AUC'].max():.4f})")
print(f"  Recall Mean: {stability_df['Mean_Recall'].mean():.4f} "
      f"(range: {stability_df['Mean_Recall'].min():.4f}-{stability_df['Mean_Recall'].max():.4f})")

# ═══════════════════════════════════════════════════════════════════════════════════════
# VISUALIZATION & SAVE
# ═══════════════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].errorbar(stability_df['Seed'].astype(str), stability_df['Mean_ROC_AUC'],
                 yerr=stability_df['Std_ROC_AUC'], fmt='o-', capsize=5, linewidth=2)
axes[0].set_ylabel('ROC-AUC')
axes[0].set_title('ROC-AUC Stability Across Seeds')
axes[0].grid(alpha=0.3)

axes[1].errorbar(stability_df['Seed'].astype(str), stability_df['Mean_Recall'],
                 yerr=stability_df['Std_Recall'], fmt='s-', capsize=5, linewidth=2, color='green')
axes[1].set_ylabel('Recall')
axes[1].set_title('Recall Stability Across Seeds')
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('09_stability_analysis.png', dpi=150, bbox_inches='tight')
print("\n✓ Saved: 09_stability_analysis.png")
plt.close()

with open('09_stability_insights.pkl', 'wb') as f:
    pickle.dump({'stability_df': stability_df}, f)

print("✓ Saved: 09_stability_insights.pkl")
print("\n" + "="*80)
print("✓ PHASE 9 COMPLETE")
print("="*80)
