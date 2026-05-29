"""
========================================================
PHASE 3: BASELINE MODELING - Find Strongest Model Family
========================================================
Objective: Find strongest model family.

CHECKLIST - MODELS:
□ XGBoost
□ LightGBM
□ Random Forest

CHECKLIST - METRICS:
□ ROC-AUC
□ PR-AUC
□ Recall
□ Precision
□ F1
□ MCC

INFERENCE: Which model learns defect signal best?

OUTPUT:
□ XGBoost selected (expected winner)

FEEDS INTO: → SHAP → Threshold Optimization → Stability Analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, average_precision_score, recall_score,
                             precision_score, f1_score, matthews_corrcoef,
                             roc_curve, precision_recall_curve, confusion_matrix)
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
import pickle
import warnings
warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════════════════════════════
# 1. LOAD DATA
# ═══════════════════════════════════════════════════════════════════════════════════════
print("="*80)
print("PHASE 3: BASELINE MODELING - STARTING")
print("="*80)

train_df = pd.read_csv('train.csv')

with open('01_eda_insights.pkl', 'rb') as f:
    eda_insights = pickle.load(f)

with open('02_correlation_insights.pkl', 'rb') as f:
    correlation_insights = pickle.load(f)

X_features = [col for col in train_df.columns if col.startswith('X')]
imbalance_ratio = eda_insights['imbalance_ratio']

# Prepare data
X = train_df[X_features].fillna(train_df[X_features].median())
y = train_df['Y']

print(f"\n✓ DATA LOADED")
print(f"  Features: {len(X_features)}")
print(f"  Samples: {len(X)}")
print(f"  Imbalance ratio: {imbalance_ratio:.2f}:1")
print(f"  Defect samples: {y.sum()}")

# ═══════════════════════════════════════════════════════════════════════════════════════
# 2. SETUP CROSS-VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("2. SETTING UP CROSS-VALIDATION")
print("="*80)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print(f"\n✓ 5-Fold Stratified K-Fold setup")
print(f"  Fold sizes:")

for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    train_defects = y.iloc[train_idx].sum()
    val_defects = y.iloc[val_idx].sum()
    print(f"    Fold {fold_idx+1}: Train={len(train_idx)} ({train_defects} defects), "
          f"Val={len(val_idx)} ({val_defects} defects)")

# ═══════════════════════════════════════════════════════════════════════════════════════
# 3. XGBOOST BASELINE
# ═══════════════════════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("3. XGBOOST BASELINE")
print("="*80)

xgb_params = {
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

xgb_oof_pred = np.zeros(len(X))
xgb_scores = {'roc_auc': [], 'pr_auc': [], 'recall': [], 'precision': [],
              'f1': [], 'mcc': []}

print(f"\nTraining XGBoost (5-fold CV)...")
for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    xgb_model = xgb.XGBClassifier(**xgb_params)
    xgb_model.fit(X_train, y_train, verbose=False)

    # Predictions
    y_pred_proba = xgb_model.predict_proba(X_val)[:, 1]
    xgb_oof_pred[val_idx] = y_pred_proba

    # Metrics at threshold 0.5
    y_pred = (y_pred_proba >= 0.5).astype(int)

    roc_auc = roc_auc_score(y_val, y_pred_proba)
    pr_auc = average_precision_score(y_val, y_pred_proba)
    recall = recall_score(y_val, y_pred, zero_division=0)
    precision = precision_score(y_val, y_pred, zero_division=0)
    f1 = f1_score(y_val, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_val, y_pred)

    xgb_scores['roc_auc'].append(roc_auc)
    xgb_scores['pr_auc'].append(pr_auc)
    xgb_scores['recall'].append(recall)
    xgb_scores['precision'].append(precision)
    xgb_scores['f1'].append(f1)
    xgb_scores['mcc'].append(mcc)

    print(f"  Fold {fold_idx+1}: ROC-AUC={roc_auc:.4f}, PR-AUC={pr_auc:.4f}, "
          f"Recall={recall:.4f}, F1={f1:.4f}")

print(f"\n✓ XGBoost CV Results:")
for metric in xgb_scores.keys():
    mean_val = np.mean(xgb_scores[metric])
    std_val = np.std(xgb_scores[metric])
    print(f"  {metric.upper():<12}: {mean_val:.4f} ± {std_val:.4f}")

# ═══════════════════════════════════════════════════════════════════════════════════════
# 4. LIGHTGBM BASELINE
# ═══════════════════════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("4. LIGHTGBM BASELINE")
print("="*80)

lgb_params = {
    'objective': 'binary',
    'metric': 'auc',
    'random_state': 42,
    'num_leaves': 31,
    'learning_rate': 0.1,
    'n_estimators': 200,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'scale_pos_weight': imbalance_ratio,
    'verbosity': -1
}

lgb_oof_pred = np.zeros(len(X))
lgb_scores = {'roc_auc': [], 'pr_auc': [], 'recall': [], 'precision': [],
              'f1': [], 'mcc': []}

print(f"\nTraining LightGBM (5-fold CV)...")
for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    lgb_model = lgb.LGBMClassifier(**lgb_params)
    lgb_model.fit(X_train, y_train)

    # Predictions
    y_pred_proba = lgb_model.predict_proba(X_val)[:, 1]
    lgb_oof_pred[val_idx] = y_pred_proba

    # Metrics
    y_pred = (y_pred_proba >= 0.5).astype(int)

    roc_auc = roc_auc_score(y_val, y_pred_proba)
    pr_auc = average_precision_score(y_val, y_pred_proba)
    recall = recall_score(y_val, y_pred, zero_division=0)
    precision = precision_score(y_val, y_pred, zero_division=0)
    f1 = f1_score(y_val, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_val, y_pred)

    lgb_scores['roc_auc'].append(roc_auc)
    lgb_scores['pr_auc'].append(pr_auc)
    lgb_scores['recall'].append(recall)
    lgb_scores['precision'].append(precision)
    lgb_scores['f1'].append(f1)
    lgb_scores['mcc'].append(mcc)

    print(f"  Fold {fold_idx+1}: ROC-AUC={roc_auc:.4f}, PR-AUC={pr_auc:.4f}, "
          f"Recall={recall:.4f}, F1={f1:.4f}")

print(f"\n✓ LightGBM CV Results:")
for metric in lgb_scores.keys():
    mean_val = np.mean(lgb_scores[metric])
    std_val = np.std(lgb_scores[metric])
    print(f"  {metric.upper():<12}: {mean_val:.4f} ± {std_val:.4f}")

# ═══════════════════════════════════════════════════════════════════════════════════════
# 5. RANDOM FOREST BASELINE
# ═══════════════════════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("5. RANDOM FOREST BASELINE")
print("="*80)

rf_oof_pred = np.zeros(len(X))
rf_scores = {'roc_auc': [], 'pr_auc': [], 'recall': [], 'precision': [],
             'f1': [], 'mcc': []}

# Calculate class weights for imbalance
class_weight = {0: 1, 1: imbalance_ratio}

print(f"\nTraining Random Forest (5-fold CV)...")
for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    rf_model = RandomForestClassifier(n_estimators=200, max_depth=15,
                                      max_samples=0.8,
                                      random_state=42,
                                      class_weight='balanced',
                                      n_jobs=-1)
    rf_model.fit(X_train, y_train)

    # Predictions
    y_pred_proba = rf_model.predict_proba(X_val)[:, 1]
    rf_oof_pred[val_idx] = y_pred_proba

    # Metrics
    y_pred = (y_pred_proba >= 0.5).astype(int)

    roc_auc = roc_auc_score(y_val, y_pred_proba)
    pr_auc = average_precision_score(y_val, y_pred_proba)
    recall = recall_score(y_val, y_pred, zero_division=0)
    precision = precision_score(y_val, y_pred, zero_division=0)
    f1 = f1_score(y_val, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_val, y_pred)

    rf_scores['roc_auc'].append(roc_auc)
    rf_scores['pr_auc'].append(pr_auc)
    rf_scores['recall'].append(recall)
    rf_scores['precision'].append(precision)
    rf_scores['f1'].append(f1)
    rf_scores['mcc'].append(mcc)

    print(f"  Fold {fold_idx+1}: ROC-AUC={roc_auc:.4f}, PR-AUC={pr_auc:.4f}, "
          f"Recall={recall:.4f}, F1={f1:.4f}")

print(f"\n✓ Random Forest CV Results:")
for metric in rf_scores.keys():
    mean_val = np.mean(rf_scores[metric])
    std_val = np.std(rf_scores[metric])
    print(f"  {metric.upper():<12}: {mean_val:.4f} ± {std_val:.4f}")

# ═══════════════════════════════════════════════════════════════════════════════════════
# 6. MODEL COMPARISON
# ═══════════════════════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("6. MODEL COMPARISON")
print("="*80)

comparison_df = pd.DataFrame({
    'XGBoost': {k: np.mean(v) for k, v in xgb_scores.items()},
    'LightGBM': {k: np.mean(v) for k, v in lgb_scores.items()},
    'RandomForest': {k: np.mean(v) for k, v in rf_scores.items()}
})

print(f"\n{'Metric':<15} {'XGBoost':<15} {'LightGBM':<15} {'RandomForest':<15}")
print("-" * 60)

for metric in comparison_df.index:
    xgb_val = comparison_df.loc[metric, 'XGBoost']
    lgb_val = comparison_df.loc[metric, 'LightGBM']
    rf_val = comparison_df.loc[metric, 'RandomForest']

    # Mark best
    values = [xgb_val, lgb_val, rf_val]
    best_idx = np.argmax(values)

    models = ['XGBoost', 'LightGBM', 'RandomForest']
    line = f"{metric:<15}"

    for i, model in enumerate(models):
        val = values[i]
        marker = "★" if i == best_idx else " "
        line += f" {marker} {val:.4f}"

    print(line)

# ═══════════════════════════════════════════════════════════════════════════════════════
# 7. VISUALIZATION
# ═══════════════════════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("7. CREATING VISUALIZATIONS")
print("="*80)

fig = plt.figure(figsize=(18, 10))

# ROC curves
ax1 = plt.subplot(2, 3, 1)
fpr_xgb, tpr_xgb, _ = roc_curve(y, xgb_oof_pred)
fpr_lgb, tpr_lgb, _ = roc_curve(y, lgb_oof_pred)
fpr_rf, tpr_rf, _ = roc_curve(y, rf_oof_pred)

ax1.plot(fpr_xgb, tpr_xgb, label=f'XGBoost (AUC={roc_auc_score(y, xgb_oof_pred):.4f})', linewidth=2)
ax1.plot(fpr_lgb, tpr_lgb, label=f'LightGBM (AUC={roc_auc_score(y, lgb_oof_pred):.4f})', linewidth=2)
ax1.plot(fpr_rf, tpr_rf, label=f'RF (AUC={roc_auc_score(y, rf_oof_pred):.4f})', linewidth=2)
ax1.plot([0, 1], [0, 1], 'k--', alpha=0.3)
ax1.set_xlabel('False Positive Rate')
ax1.set_ylabel('True Positive Rate')
ax1.set_title('ROC Curves')
ax1.legend()
ax1.grid(alpha=0.3)

# PR curves
ax2 = plt.subplot(2, 3, 2)
prec_xgb, rec_xgb, _ = precision_recall_curve(y, xgb_oof_pred)
prec_lgb, rec_lgb, _ = precision_recall_curve(y, lgb_oof_pred)
prec_rf, rec_rf, _ = precision_recall_curve(y, rf_oof_pred)

ax2.plot(rec_xgb, prec_xgb, label=f'XGBoost (AP={average_precision_score(y, xgb_oof_pred):.4f})', linewidth=2)
ax2.plot(rec_lgb, prec_lgb, label=f'LightGBM (AP={average_precision_score(y, lgb_oof_pred):.4f})', linewidth=2)
ax2.plot(rec_rf, prec_rf, label=f'RF (AP={average_precision_score(y, rf_oof_pred):.4f})', linewidth=2)
ax2.set_xlabel('Recall')
ax2.set_ylabel('Precision')
ax2.set_title('Precision-Recall Curves')
ax2.legend()
ax2.grid(alpha=0.3)

# Metric comparison
ax3 = plt.subplot(2, 3, 3)
x_pos = np.arange(len(comparison_df.index))
width = 0.25

for i, model in enumerate(['XGBoost', 'LightGBM', 'RandomForest']):
    values = comparison_df[model].values
    ax3.bar(x_pos + i*width, values, width, label=model, alpha=0.8)

ax3.set_ylabel('Score')
ax3.set_title('Metric Comparison (CV Average)')
ax3.set_xticks(x_pos + width)
ax3.set_xticklabels(comparison_df.index, rotation=45, ha='right')
ax3.legend()
ax3.grid(axis='y', alpha=0.3)
ax3.set_ylim([0, 1])

# Probability distributions
ax4 = plt.subplot(2, 3, 4)
ax4.hist(xgb_oof_pred[y == 0], bins=30, alpha=0.6, label='Normal (XGBoost)', color='blue')
ax4.hist(xgb_oof_pred[y == 1], bins=10, alpha=0.6, label='Defect (XGBoost)', color='red')
ax4.set_xlabel('Probability')
ax4.set_ylabel('Frequency')
ax4.set_title('XGBoost Prediction Distribution')
ax4.legend()
ax4.set_yscale('log')
ax4.grid(alpha=0.3)

# LightGBM probability distributions
ax5 = plt.subplot(2, 3, 5)
ax5.hist(lgb_oof_pred[y == 0], bins=30, alpha=0.6, label='Normal (LightGBM)', color='blue')
ax5.hist(lgb_oof_pred[y == 1], bins=10, alpha=0.6, label='Defect (LightGBM)', color='red')
ax5.set_xlabel('Probability')
ax5.set_ylabel('Frequency')
ax5.set_title('LightGBM Prediction Distribution')
ax5.legend()
ax5.set_yscale('log')
ax5.grid(alpha=0.3)

# RF probability distributions
ax6 = plt.subplot(2, 3, 6)
ax6.hist(rf_oof_pred[y == 0], bins=30, alpha=0.6, label='Normal (RF)', color='blue')
ax6.hist(rf_oof_pred[y == 1], bins=10, alpha=0.6, label='Defect (RF)', color='red')
ax6.set_xlabel('Probability')
ax6.set_ylabel('Frequency')
ax6.set_title('Random Forest Prediction Distribution')
ax6.legend()
ax6.set_yscale('log')
ax6.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('03_baseline_modeling.png', dpi=150, bbox_inches='tight')
print("\n✓ Saved: 03_baseline_modeling.png")
plt.close()

# ═══════════════════════════════════════════════════════════════════════════════════════
# 8. SAVE OUTPUTS
# ═══════════════════════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("8. SAVING OUTPUTS FOR DOWNSTREAM PHASES")
print("="*80)

modeling_insights = {
    'xgb_oof_pred': xgb_oof_pred,
    'lgb_oof_pred': lgb_oof_pred,
    'rf_oof_pred': rf_oof_pred,
    'xgb_scores': xgb_scores,
    'lgb_scores': lgb_scores,
    'rf_scores': rf_scores,
    'comparison_df': comparison_df,
    'y_true': y.values
}

with open('03_modeling_insights.pkl', 'wb') as f:
    pickle.dump(modeling_insights, f)

print(f"\n✓ Saved: 03_modeling_insights.pkl")

# ═══════════════════════════════════════════════════════════════════════════════════════
# 9. SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("PHASE 3 SUMMARY")
print("="*80)

xgb_roc = np.mean(xgb_scores['roc_auc'])
lgb_roc = np.mean(lgb_scores['roc_auc'])
rf_roc = np.mean(rf_scores['roc_auc'])

winner = 'XGBoost' if xgb_roc >= max(lgb_roc, rf_roc) else ('LightGBM' if lgb_roc > rf_roc else 'RandomForest')

print(f"""
BASELINE MODEL PERFORMANCE (5-Fold CV):

1. ROC-AUC SCORES:
   - XGBoost: {xgb_roc:.4f}
   - LightGBM: {lgb_roc:.4f}
   - RandomForest: {rf_roc:.4f}
   → WINNER: {winner}

2. PR-AUC SCORES:
   - XGBoost: {np.mean(xgb_scores['pr_auc']):.4f}
   - LightGBM: {np.mean(lgb_scores['pr_auc']):.4f}
   - RandomForest: {np.mean(rf_scores['pr_auc']):.4f}

3. RECALL (at 0.5 threshold):
   - XGBoost: {np.mean(xgb_scores['recall']):.4f}
   - LightGBM: {np.mean(lgb_scores['recall']):.4f}
   - RandomForest: {np.mean(rf_scores['recall']):.4f}

4. KEY OBSERVATIONS:
   - {winner} shows strongest ROC-AUC performance
   - Probability distributions are heavily skewed toward 0
   - Default threshold 0.5 is inappropriate for this task
   - Will need custom threshold optimization
   - Signal is present and learnable

NEXT PHASE (Phase 4: SHAP Interpretation):
→ Validate EDA findings with SHAP
→ Confirm important variables
→ Identify key interactions
→ Prepare for feature engineering
""")

print("\n" + "="*80)
print("✓ PHASE 3 COMPLETE")
print("="*80)
