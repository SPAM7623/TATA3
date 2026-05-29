"""
========================================================
PHASE 6: IMBALANCE HANDLING - Protect Rare Defects
========================================================
Objective: Prevent model from ignoring defects.

CHECKLIST - STRATEGIES:
□ scale_pos_weight (already applied)
□ SMOTE (Synthetic Minority Oversampling)
□ Class weights tuning
□ Threshold-aware sampling

INFERENCE: Defect class extremely rare (4.88%) → needs special care

OUTPUT:
□ scale_pos_weight = 19.48 (baseline already)
□ SMOTE comparison
□ Optimal imbalance strategy

FEEDS INTO: → Final XGB → Calibration Analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTE
import xgboost as xgb
from sklearn.metrics import roc_auc_score, average_precision_score, recall_score, precision_score, f1_score
import pickle
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("PHASE 6: IMBALANCE HANDLING - STARTING")
print("="*80)

train_df = pd.read_csv('train.csv')
with open('05_engineering_insights.pkl', 'rb') as f:
    eng_insights = pickle.load(f)

X_engineered = eng_insights['X_engineered']
y = train_df['Y']
imbalance_ratio = 19.48

print(f"\n✓ DATA LOADED")
print(f"  Class distribution: {(y==0).sum()} normal, {(y==1).sum()} defects")
print(f"  Imbalance ratio: {imbalance_ratio:.2f}:1")

# ═══════════════════════════════════════════════════════════════════════════════════════
# STRATEGY 1: BASELINE (scale_pos_weight)
# ═══════════════════════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("STRATEGY 1: BASELINE (scale_pos_weight)")
print("="*80)

xgb_params_baseline = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'random_state': 42,
    'n_estimators': 200,
    'learning_rate': 0.1,
    'max_depth': 6,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'scale_pos_weight': imbalance_ratio,
    'tree_method': 'hist',
    'verbosity': 0
}

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
baseline_scores = {'roc_auc': [], 'pr_auc': [], 'recall': [], 'precision': [], 'f1': []}

for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_engineered, y)):
    X_train, X_val = X_engineered.iloc[train_idx], X_engineered.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    model = xgb.XGBClassifier(**xgb_params_baseline)
    model.fit(X_train, y_train, verbose=False)

    y_pred_proba = model.predict_proba(X_val)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)

    baseline_scores['roc_auc'].append(roc_auc_score(y_val, y_pred_proba))
    baseline_scores['pr_auc'].append(average_precision_score(y_val, y_pred_proba))
    baseline_scores['recall'].append(recall_score(y_val, y_pred, zero_division=0))
    baseline_scores['precision'].append(precision_score(y_val, y_pred, zero_division=0))
    baseline_scores['f1'].append(f1_score(y_val, y_pred, zero_division=0))

print(f"\n✓ Baseline (scale_pos_weight={imbalance_ratio:.2f}):")
print(f"  ROC-AUC: {np.mean(baseline_scores['roc_auc']):.4f}")
print(f"  Recall: {np.mean(baseline_scores['recall']):.4f}")

# ═══════════════════════════════════════════════════════════════════════════════════════
# STRATEGY 2: SMOTE
# ═══════════════════════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("STRATEGY 2: SMOTE (Oversampling)")
print("="*80)

smote_scores = {'roc_auc': [], 'pr_auc': [], 'recall': [], 'precision': [], 'f1': []}

for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_engineered, y)):
    X_train, X_val = X_engineered.iloc[train_idx], X_engineered.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    # Apply SMOTE
    smote = SMOTE(random_state=42, k_neighbors=min(3, (y_train==1).sum()-1))
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

    model = xgb.XGBClassifier(**xgb_params_baseline)
    model.fit(X_train_smote, y_train_smote, verbose=False)

    y_pred_proba = model.predict_proba(X_val)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)

    smote_scores['roc_auc'].append(roc_auc_score(y_val, y_pred_proba))
    smote_scores['pr_auc'].append(average_precision_score(y_val, y_pred_proba))
    smote_scores['recall'].append(recall_score(y_val, y_pred, zero_division=0))
    smote_scores['precision'].append(precision_score(y_val, y_pred, zero_division=0))
    smote_scores['f1'].append(f1_score(y_val, y_pred, zero_division=0))

print(f"\n✓ SMOTE Results:")
print(f"  ROC-AUC: {np.mean(smote_scores['roc_auc']):.4f}")
print(f"  Recall: {np.mean(smote_scores['recall']):.4f}")

# ═══════════════════════════════════════════════════════════════════════════════════════
# STRATEGY 3: Increased scale_pos_weight
# ═══════════════════════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("STRATEGY 3: Increased scale_pos_weight")
print("="*80)

increased_weight_scores = {'roc_auc': [], 'pr_auc': [], 'recall': [], 'precision': [], 'f1': []}
increased_weight = imbalance_ratio * 1.5  # 29.22

xgb_params_high_weight = xgb_params_baseline.copy()
xgb_params_high_weight['scale_pos_weight'] = increased_weight

for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_engineered, y)):
    X_train, X_val = X_engineered.iloc[train_idx], X_engineered.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    model = xgb.XGBClassifier(**xgb_params_high_weight)
    model.fit(X_train, y_train, verbose=False)

    y_pred_proba = model.predict_proba(X_val)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)

    increased_weight_scores['roc_auc'].append(roc_auc_score(y_val, y_pred_proba))
    increased_weight_scores['pr_auc'].append(average_precision_score(y_val, y_pred_proba))
    increased_weight_scores['recall'].append(recall_score(y_val, y_pred, zero_division=0))
    increased_weight_scores['precision'].append(precision_score(y_val, y_pred, zero_division=0))
    increased_weight_scores['f1'].append(f1_score(y_val, y_pred, zero_division=0))

print(f"\n✓ Increased weight (scale_pos_weight={increased_weight:.2f}):")
print(f"  ROC-AUC: {np.mean(increased_weight_scores['roc_auc']):.4f}")
print(f"  Recall: {np.mean(increased_weight_scores['recall']):.4f}")

# ═══════════════════════════════════════════════════════════════════════════════════════
# COMPARISON
# ═══════════════════════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("IMBALANCE HANDLING STRATEGY COMPARISON")
print("="*80)

comparison = pd.DataFrame({
    'Strategy': ['scale_pos_weight=19.48', 'SMOTE', 'scale_pos_weight=29.22'],
    'ROC-AUC': [np.mean(baseline_scores['roc_auc']), np.mean(smote_scores['roc_auc']), np.mean(increased_weight_scores['roc_auc'])],
    'PR-AUC': [np.mean(baseline_scores['pr_auc']), np.mean(smote_scores['pr_auc']), np.mean(increased_weight_scores['pr_auc'])],
    'Recall': [np.mean(baseline_scores['recall']), np.mean(smote_scores['recall']), np.mean(increased_weight_scores['recall'])],
    'Precision': [np.mean(baseline_scores['precision']), np.mean(smote_scores['precision']), np.mean(increased_weight_scores['precision'])],
    'F1': [np.mean(baseline_scores['f1']), np.mean(smote_scores['f1']), np.mean(increased_weight_scores['f1'])]
})

print(f"\n{comparison.to_string(index=False)}")

best_strategy_idx = comparison['Recall'].idxmax()
best_strategy = comparison.iloc[best_strategy_idx]

print(f"\n✓ BEST STRATEGY: {comparison.iloc[best_strategy_idx]['Strategy']}")
print(f"  Recall: {best_strategy['Recall']:.4f}")
print(f"  ROC-AUC: {best_strategy['ROC-AUC']:.4f}")

# Select best strategy for downstream
if best_strategy_idx == 1:  # SMOTE
    selected_scores = smote_scores
    selected_params = xgb_params_baseline
    selected_strategy = "SMOTE"
    print(f"\n→ SELECTED: SMOTE for downstream phases")
elif best_strategy_idx == 2:  # Increased weight
    selected_scores = increased_weight_scores
    selected_params = xgb_params_high_weight
    selected_strategy = "Increased scale_pos_weight"
    print(f"\n→ SELECTED: Increased scale_pos_weight for downstream phases")
else:  # Baseline
    selected_scores = baseline_scores
    selected_params = xgb_params_baseline
    selected_strategy = "Baseline scale_pos_weight"
    print(f"\n→ SELECTED: Baseline strategy (already working well)")

# ═══════════════════════════════════════════════════════════════════════════════════════
# VISUALIZATION
# ═══════════════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

metrics = ['ROC-AUC', 'PR-AUC', 'Recall', 'Precision']
x_pos = np.arange(len(comparison))
width = 0.2

for idx, metric in enumerate(['ROC-AUC', 'PR-AUC', 'Recall', 'Precision']):
    ax = axes[idx // 2, idx % 2]
    values = comparison[metric].values
    bars = ax.bar(x_pos, values, width=0.6, alpha=0.7, color=['blue', 'orange', 'green'])
    ax.set_ylabel(metric)
    ax.set_title(f'{metric} Comparison')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(['Baseline', 'SMOTE', 'Inc. Weight'], rotation=0)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 1])

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('06_imbalance_handling.png', dpi=150, bbox_inches='tight')
print("\n✓ Saved: 06_imbalance_handling.png")
plt.close()

# ═══════════════════════════════════════════════════════════════════════════════════════
# SAVE OUTPUTS
# ═══════════════════════════════════════════════════════════════════════════════════════
imbalance_insights = {
    'baseline_scores': baseline_scores,
    'smote_scores': smote_scores,
    'increased_weight_scores': increased_weight_scores,
    'comparison': comparison,
    'selected_strategy': selected_strategy,
    'selected_params': selected_params,
    'selected_scores': selected_scores
}

with open('06_imbalance_insights.pkl', 'wb') as f:
    pickle.dump(imbalance_insights, f)

print(f"✓ Saved: 06_imbalance_insights.pkl")

print("\n" + "="*80)
print("✓ PHASE 6 COMPLETE")
print("="*80)
