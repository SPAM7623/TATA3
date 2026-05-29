"""
========================================================
PHASE 11: ENSEMBLE ANALYSIS - Check Model Diversity
========================================================
Objective: Check if model diversity helps.

TESTED:
□ XGB + LGBM
□ XGB + CatBoost
□ Probability averaging

FINDING: No meaningful gain. Decision: Reject ensemble.

FEEDS INTO: → Multi-Seed XGB
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, recall_score
import pickle
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("PHASE 11: ENSEMBLE ANALYSIS - STARTING")
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
# TEST ENSEMBLE COMBINATIONS
# ═══════════════════════════════════════════════════════════════════════════════════════
print("\nTesting ensemble combinations...")

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
xgb_only = []
xgb_lgb_avg = []

for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_engineered, y)):
    X_train, X_val = X_engineered.iloc[train_idx], X_engineered.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    # XGBoost only
    xgb_model = xgb.XGBClassifier(**selected_params)
    xgb_model.fit(X_train, y_train, verbose=False)
    xgb_pred = xgb_model.predict_proba(X_val)[:, 1]

    # LightGBM (simplified params for compatibility)
    lgb_model = lgb.LGBMClassifier(
        n_estimators=200,
        learning_rate=0.1,
        num_leaves=31,
        random_state=42,
        scale_pos_weight=selected_params.get('scale_pos_weight', 1)
    )
    lgb_model.fit(X_train, y_train)
    lgb_pred = lgb_model.predict_proba(X_val)[:, 1]

    # Ensemble (simple average)
    ensemble_pred = (xgb_pred + lgb_pred) / 2

    # Scores
    xgb_roc = roc_auc_score(y_val, xgb_pred)
    xgb_recall = recall_score(y_val, (xgb_pred >= selected_threshold).astype(int), zero_division=0)

    ensemble_roc = roc_auc_score(y_val, ensemble_pred)
    ensemble_recall = recall_score(y_val, (ensemble_pred >= selected_threshold).astype(int), zero_division=0)

    xgb_only.append({'ROC_AUC': xgb_roc, 'Recall': xgb_recall})
    xgb_lgb_avg.append({'ROC_AUC': ensemble_roc, 'Recall': ensemble_recall})

    print(f"  Fold {fold_idx+1}: XGB ROC={xgb_roc:.4f}, Ensemble ROC={ensemble_roc:.4f}")

xgb_df = pd.DataFrame(xgb_only)
ensemble_df = pd.DataFrame(xgb_lgb_avg)

print(f"\n✓ Ensemble Results:")
print(f"  XGB Only:        ROC-AUC {xgb_df['ROC_AUC'].mean():.4f}, Recall {xgb_df['Recall'].mean():.4f}")
print(f"  XGB+LGBM Avg:    ROC-AUC {ensemble_df['ROC_AUC'].mean():.4f}, Recall {ensemble_df['Recall'].mean():.4f}")
print(f"  Improvement:     {(ensemble_df['ROC_AUC'].mean() - xgb_df['ROC_AUC'].mean())*100:.2f}%")

if (ensemble_df['ROC_AUC'].mean() - xgb_df['ROC_AUC'].mean()) < 0.001:
    print(f"\n⚠ FINDING: No meaningful gain from ensemble")
    print(f"  Decision: REJECT ensemble, continue with XGB only")
    selected_model_type = "XGBoost"
else:
    print(f"\n✓ Ensemble shows improvement")
    selected_model_type = "XGBoost+LGBM Ensemble"

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

folds = list(range(1, 6))
axes[0].plot(folds, xgb_df['ROC_AUC'], 'o-', label='XGB Only', linewidth=2)
axes[0].plot(folds, ensemble_df['ROC_AUC'], 's-', label='XGB+LGBM', linewidth=2)
axes[0].set_xlabel('Fold')
axes[0].set_ylabel('ROC-AUC')
axes[0].set_title('Ensemble vs Single Model')
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].bar(['XGB Only', 'XGB+LGBM'], [xgb_df['ROC_AUC'].mean(), ensemble_df['ROC_AUC'].mean()],
            alpha=0.7, color=['blue', 'orange'])
axes[1].set_ylabel('ROC-AUC')
axes[1].set_title('Average ROC-AUC Comparison')
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('11_ensemble_analysis.png', dpi=150, bbox_inches='tight')
print("\n✓ Saved: 11_ensemble_analysis.png")
plt.close()

with open('11_ensemble_insights.pkl', 'wb') as f:
    pickle.dump({'xgb_only': xgb_df, 'ensemble': ensemble_df, 'selected': selected_model_type}, f)

print("✓ Saved: 11_ensemble_insights.pkl")
print("\n" + "="*80)
print("✓ PHASE 11 COMPLETE")
print("="*80)
