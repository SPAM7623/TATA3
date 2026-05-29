"""
========================================================
PHASE 8: THRESHOLD OPTIMIZATION
========================================================
Objective: Find optimal decision threshold.

TASKS:
□ Test multiple thresholds
□ Evaluate on validation set
□ Select best threshold

INFERENCE: Model ranking quality matters more than raw probabilities.

OUTPUT:
□ Optimal threshold
□ Performance at optimal threshold

FEEDS INTO: → Final Submission
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb
from sklearn.metrics import roc_auc_score, average_precision_score, recall_score, precision_score, f1_score, matthews_corrcoef
import pickle
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("PHASE 8: THRESHOLD OPTIMIZATION - STARTING")
print("="*80)

train_df = pd.read_csv('train.csv')
with open('05_engineering_insights.pkl', 'rb') as f:
    eng_insights = pickle.load(f)
with open('06_imbalance_insights.pkl', 'rb') as f:
    imbalance_insights = pickle.load(f)
with open('07_calibration_insights.pkl', 'rb') as f:
    cal_insights = pickle.load(f)

X_engineered = eng_insights['X_engineered']
y = train_df['Y']
selected_params = imbalance_insights['selected_params']
all_y_pred = cal_insights['all_y_pred']
all_y_true = cal_insights['all_y_true']

# ═══════════════════════════════════════════════════════════════════════════════════════
# THRESHOLD SEARCH
# ═══════════════════════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("THRESHOLD SEARCH")
print("="*80)

# Test thresholds from 0.001 to 0.5
thresholds = np.linspace(0.001, 0.5, 100)
threshold_results = []

for threshold in thresholds:
    y_pred = (all_y_pred >= threshold).astype(int)

    tp = ((y_pred == 1) & (all_y_true == 1)).sum()
    fp = ((y_pred == 1) & (all_y_true == 0)).sum()
    tn = ((y_pred == 0) & (all_y_true == 0)).sum()
    fn = ((y_pred == 0) & (all_y_true == 1)).sum()

    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    mcc = matthews_corrcoef(all_y_true, y_pred)

    threshold_results.append({
        'Threshold': threshold,
        'Recall': recall,
        'Precision': precision,
        'F1': f1,
        'MCC': mcc,
        'TP': tp,
        'FP': fp,
        'TN': tn,
        'FN': fn
    })

results_df = pd.DataFrame(threshold_results)

# Find best thresholds for different objectives
best_f1_idx = results_df['F1'].idxmax()
best_recall_idx = results_df['Recall'].idxmax()
best_precision_idx = results_df['Precision'].idxmax()
best_mcc_idx = results_df['MCC'].idxmax()

print(f"\n✓ Optimal Thresholds by Metric:")
print(f"\n  Best F1:")
best_f1_row = results_df.iloc[best_f1_idx]
print(f"    Threshold: {best_f1_row['Threshold']:.6f}")
print(f"    F1: {best_f1_row['F1']:.4f}, Recall: {best_f1_row['Recall']:.4f}, Precision: {best_f1_row['Precision']:.4f}")

print(f"\n  Best MCC:")
best_mcc_row = results_df.iloc[best_mcc_idx]
print(f"    Threshold: {best_mcc_row['Threshold']:.6f}")
print(f"    MCC: {best_mcc_row['MCC']:.4f}, Recall: {best_mcc_row['Recall']:.4f}, Precision: {best_mcc_row['Precision']:.4f}")

print(f"\n  Best Recall (>=90%):")
good_recall = results_df[results_df['Recall'] >= 0.9]
if len(good_recall) > 0:
    best_high_recall_idx = good_recall['Precision'].idxmax()
    best_high_recall_row = results_df.iloc[best_high_recall_idx]
    print(f"    Threshold: {best_high_recall_row['Threshold']:.6f}")
    print(f"    Recall: {best_high_recall_row['Recall']:.4f}, Precision: {best_high_recall_row['Precision']:.4f}")
    selected_threshold = best_high_recall_row['Threshold']
else:
    # Use best F1 if no 90% recall available
    selected_threshold = best_f1_row['Threshold']
    print(f"    90% Recall not achievable. Using best F1 threshold.")

print(f"\n→ SELECTED THRESHOLD: {selected_threshold:.6f}")

# ═══════════════════════════════════════════════════════════════════════════════════════
# VERIFICATION ON HOLD-OUT SET
# ═══════════════════════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("THRESHOLD VERIFICATION")
print("="*80)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold_results = []

for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_engineered, y)):
    X_train, X_val = X_engineered.iloc[train_idx], X_engineered.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    model = xgb.XGBClassifier(**selected_params)
    model.fit(X_train, y_train, verbose=False)

    y_pred_proba = model.predict_proba(X_val)[:, 1]
    y_pred = (y_pred_proba >= selected_threshold).astype(int)

    recall = recall_score(y_val, y_pred, zero_division=0)
    precision = precision_score(y_val, y_pred, zero_division=0)
    f1 = f1_score(y_val, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_val, y_pred)

    fold_results.append({
        'Fold': fold_idx+1,
        'Recall': recall,
        'Precision': precision,
        'F1': f1,
        'MCC': mcc
    })
    print(f"  Fold {fold_idx+1}: Recall={recall:.4f}, Precision={precision:.4f}, F1={f1:.4f}")

fold_df = pd.DataFrame(fold_results)
print(f"\n✓ Average performance at threshold {selected_threshold:.6f}:")
print(f"  Recall: {fold_df['Recall'].mean():.4f} ± {fold_df['Recall'].std():.4f}")
print(f"  Precision: {fold_df['Precision'].mean():.4f} ± {fold_df['Precision'].std():.4f}")
print(f"  F1: {fold_df['F1'].mean():.4f} ± {fold_df['F1'].std():.4f}")
print(f"  MCC: {fold_df['MCC'].mean():.4f} ± {fold_df['MCC'].std():.4f}")

# ═══════════════════════════════════════════════════════════════════════════════════════
# VISUALIZATION
# ═══════════════════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(16, 10))

# 1. Metrics vs Threshold
ax1 = plt.subplot(2, 3, 1)
ax1.plot(results_df['Threshold'], results_df['Recall'], label='Recall', linewidth=2)
ax1.plot(results_df['Threshold'], results_df['Precision'], label='Precision', linewidth=2)
ax1.plot(results_df['Threshold'], results_df['F1'], label='F1', linewidth=2)
ax1.axvline(x=selected_threshold, color='r', linestyle='--', label=f'Selected: {selected_threshold:.4f}')
ax1.set_xlabel('Threshold')
ax1.set_ylabel('Score')
ax1.set_title('Performance Metrics vs Threshold')
ax1.legend()
ax1.grid(alpha=0.3)

# 2. MCC vs Threshold
ax2 = plt.subplot(2, 3, 2)
ax2.plot(results_df['Threshold'], results_df['MCC'], color='purple', linewidth=2)
ax2.axvline(x=selected_threshold, color='r', linestyle='--')
ax2.set_xlabel('Threshold')
ax2.set_ylabel('MCC')
ax2.set_title('Matthews Correlation Coefficient vs Threshold')
ax2.grid(alpha=0.3)

# 3. F1 detail
ax3 = plt.subplot(2, 3, 3)
ax3.plot(results_df['Threshold'], results_df['F1'], color='green', linewidth=2)
ax3.scatter([best_f1_row['Threshold']], [best_f1_row['F1']], color='red', s=100, label='Best F1')
ax3.axvline(x=selected_threshold, color='orange', linestyle='--', label=f'Selected')
ax3.set_xlabel('Threshold')
ax3.set_ylabel('F1 Score')
ax3.set_title('F1 Score (Detail)')
ax3.legend()
ax3.grid(alpha=0.3)

# 4. Confusion matrix at selected threshold
ax4 = plt.subplot(2, 3, 4)
y_pred_selected = (all_y_pred >= selected_threshold).astype(int)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(all_y_true, y_pred_selected)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax4, cbar=False)
ax4.set_xlabel('Predicted')
ax4.set_ylabel('Actual')
ax4.set_title(f'Confusion Matrix (Threshold={selected_threshold:.6f})')
ax4.set_xticklabels(['Normal', 'Defect'])
ax4.set_yticklabels(['Normal', 'Defect'])

# 5. Recall-Precision tradeoff
ax5 = plt.subplot(2, 3, 5)
ax5.plot(results_df['Recall'], results_df['Precision'], linewidth=2, color='blue')
ax5.scatter([best_f1_row['Recall']], [best_f1_row['Precision']], color='green', s=100, label='Best F1')
selected_row = results_df.iloc[(results_df['Threshold'] - selected_threshold).abs().argmin()]
ax5.scatter([selected_row['Recall']], [selected_row['Precision']], color='red', s=100, label='Selected')
ax5.set_xlabel('Recall')
ax5.set_ylabel('Precision')
ax5.set_title('Recall-Precision Tradeoff')
ax5.legend()
ax5.grid(alpha=0.3)

# 6. TP, FP, FN, TN at selected threshold
ax6 = plt.subplot(2, 3, 6)
selected_row = results_df.iloc[(results_df['Threshold'] - selected_threshold).abs().argmin()]
counts = [selected_row['TP'], selected_row['FP'], selected_row['FN'], selected_row['TN']]
labels = ['TP', 'FP', 'FN', 'TN']
colors_pie = ['green', 'red', 'orange', 'blue']
ax6.pie(counts, labels=labels, autopct='%1.1f%%', colors=colors_pie, startangle=90)
ax6.set_title(f'Prediction Breakdown at Threshold {selected_threshold:.6f}')

plt.tight_layout()
plt.savefig('08_threshold_optimization.png', dpi=150, bbox_inches='tight')
print("\n✓ Saved: 08_threshold_optimization.png")
plt.close()

# ═══════════════════════════════════════════════════════════════════════════════════════
# SAVE OUTPUTS
# ═══════════════════════════════════════════════════════════════════════════════════════
threshold_insights = {
    'threshold_results_df': results_df,
    'selected_threshold': selected_threshold,
    'fold_results_df': fold_df,
    'best_f1_threshold': best_f1_row['Threshold'],
    'best_mcc_threshold': best_mcc_row['Threshold']
}

with open('08_threshold_insights.pkl', 'wb') as f:
    pickle.dump(threshold_insights, f)

print(f"✓ Saved: 08_threshold_insights.pkl")

print("\n" + "="*80)
print("✓ PHASE 8 COMPLETE")
print("="*80)
