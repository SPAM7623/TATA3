"""
========================================================
PHASE 12: MULTI-SEED XGB - Reduce Variance
========================================================
Objective: Reduce variance through multi-seed averaging.

SEEDS:
□ 42 □ 123 □ 999 □ 2025 □ 7777

METHOD: Average probabilities

FEEDS INTO: → Final Threshold Re-optimization
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
print("PHASE 12: MULTI-SEED XGB - STARTING")
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
# MULTI-SEED TRAINING
# ═══════════════════════════════════════════════════════════════════════════════════════
print("\nTraining with multiple seeds and averaging predictions...")

seeds = [42, 123, 999, 2025, 7777]
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

single_seed_scores = []
multi_seed_scores = []

for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_engineered, y)):
    X_train, X_val = X_engineered.iloc[train_idx], X_engineered.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    # Single seed (42)
    params_42 = selected_params.copy()
    params_42['random_state'] = 42
    model_42 = xgb.XGBClassifier(**params_42)
    model_42.fit(X_train, y_train, verbose=False)
    pred_42 = model_42.predict_proba(X_val)[:, 1]

    roc_42 = roc_auc_score(y_val, pred_42)
    recall_42 = recall_score(y_val, (pred_42 >= selected_threshold).astype(int), zero_division=0)
    single_seed_scores.append({'ROC_AUC': roc_42, 'Recall': recall_42})

    # Multi-seed average
    all_preds = [pred_42]

    for seed in seeds[1:]:
        params_seed = selected_params.copy()
        params_seed['random_state'] = seed
        model_seed = xgb.XGBClassifier(**params_seed)
        model_seed.fit(X_train, y_train, verbose=False)
        pred_seed = model_seed.predict_proba(X_val)[:, 1]
        all_preds.append(pred_seed)

    avg_pred = np.mean(all_preds, axis=0)

    roc_avg = roc_auc_score(y_val, avg_pred)
    recall_avg = recall_score(y_val, (avg_pred >= selected_threshold).astype(int), zero_division=0)
    multi_seed_scores.append({'ROC_AUC': roc_avg, 'Recall': recall_avg})

    print(f"  Fold {fold_idx+1}: Single-seed ROC={roc_42:.4f}, Multi-seed ROC={roc_avg:.4f}")

single_df = pd.DataFrame(single_seed_scores)
multi_df = pd.DataFrame(multi_seed_scores)

print(f"\n✓ Multi-Seed Results:")
print(f"  Single-seed (42):   ROC-AUC {single_df['ROC_AUC'].mean():.4f}, Recall {single_df['Recall'].mean():.4f}")
print(f"  Multi-seed avg:     ROC-AUC {multi_df['ROC_AUC'].mean():.4f}, Recall {multi_df['Recall'].mean():.4f}")
print(f"  Improvement:        {(multi_df['ROC_AUC'].mean() - single_df['ROC_AUC'].mean())*100:+.2f}%")
print(f"  Variance reduction: {(single_df['ROC_AUC'].std() - multi_df['ROC_AUC'].std())*100:+.2f}%")

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

folds = list(range(1, 6))
axes[0].plot(folds, single_df['ROC_AUC'], 'o-', label='Single Seed', linewidth=2, markersize=8)
axes[0].plot(folds, multi_df['ROC_AUC'], 's-', label='Multi-Seed Avg', linewidth=2, markersize=8)
axes[0].fill_between(folds, single_df['ROC_AUC'], multi_df['ROC_AUC'], alpha=0.2)
axes[0].set_xlabel('Fold')
axes[0].set_ylabel('ROC-AUC')
axes[0].set_title('Single vs Multi-Seed ROC-AUC')
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].bar(['Single-Seed', 'Multi-Seed'], [single_df['ROC_AUC'].mean(), multi_df['ROC_AUC'].mean()],
            alpha=0.7, color=['blue', 'green'], yerr=[single_df['ROC_AUC'].std(), multi_df['ROC_AUC'].std()],
            capsize=5)
axes[1].set_ylabel('ROC-AUC')
axes[1].set_title('Variance Reduction via Multi-Seed')
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('12_multiseed_xgb.png', dpi=150, bbox_inches='tight')
print("\n✓ Saved: 12_multiseed_xgb.png")
plt.close()

with open('12_multiseed_insights.pkl', 'wb') as f:
    pickle.dump({'single_seed': single_df, 'multi_seed': multi_df, 'seeds': seeds}, f)

print("✓ Saved: 12_multiseed_insights.pkl")
print("\n" + "="*80)
print("✓ PHASE 12 COMPLETE")
print("="*80)
