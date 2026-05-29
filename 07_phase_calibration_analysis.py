"""
========================================================
PHASE 7: CALIBRATION & PROBABILITY ANALYSIS
========================================================
Objective: Understand probability behavior.

TASKS:
□ Probability histogram
□ Calibration testing
□ Probability compression analysis

FINDING: Probabilities compressed near zero

INFERENCE: Threshold 0.5 invalid. Need ultra-low threshold.

FEEDS INTO: → Threshold Optimization
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
import xgboost as xgb
from sklearn.metrics import roc_auc_score, average_precision_score
import pickle
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("PHASE 7: CALIBRATION & PROBABILITY ANALYSIS - STARTING")
print("="*80)

train_df = pd.read_csv('train.csv')
with open('05_engineering_insights.pkl', 'rb') as f:
    eng_insights = pickle.load(f)
with open('06_imbalance_insights.pkl', 'rb') as f:
    imbalance_insights = pickle.load(f)

X_engineered = eng_insights['X_engineered']
y = train_df['Y']
selected_params = imbalance_insights['selected_params']

# ═══════════════════════════════════════════════════════════════════════════════════════
# PROBABILITY ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("PROBABILITY DISTRIBUTION ANALYSIS")
print("="*80)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
all_y_true = []
all_y_pred = []

for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_engineered, y)):
    X_train, X_val = X_engineered.iloc[train_idx], X_engineered.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    model = xgb.XGBClassifier(**selected_params)
    model.fit(X_train, y_train, verbose=False)

    y_pred_proba = model.predict_proba(X_val)[:, 1]
    all_y_true.extend(y_val)
    all_y_pred.extend(y_pred_proba)

all_y_true = np.array(all_y_true)
all_y_pred = np.array(all_y_pred)

# Statistics
print(f"\nProbability Statistics:")
print(f"  Min: {all_y_pred.min():.6f}")
print(f"  Max: {all_y_pred.max():.6f}")
print(f"  Mean: {all_y_pred.mean():.6f}")
print(f"  Median: {np.median(all_y_pred):.6f}")
print(f"  Std: {all_y_pred.std():.6f}")

# Percentiles
percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
print(f"\n  Percentiles:")
for p in percentiles:
    val = np.percentile(all_y_pred, p)
    print(f"    {p}th: {val:.6f}")

# Count by threshold
thresholds_to_test = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5]
print(f"\n  Predictions above threshold:")
for t in thresholds_to_test:
    count = (all_y_pred >= t).sum()
    pct = (count / len(all_y_pred)) * 100
    defect_pct = (all_y_pred[all_y_true==1] >= t).sum() / (all_y_true==1).sum() * 100
    print(f"    >= {t}: {count:4d} ({pct:5.2f}%), defects: {defect_pct:5.2f}%")

# ═══════════════════════════════════════════════════════════════════════════════════════
# CALIBRATION CURVE
# ═══════════════════════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("CALIBRATION CURVE ANALYSIS")
print("="*80)

prob_true, prob_pred = calibration_curve(all_y_true, all_y_pred, n_bins=10)

print(f"\nCalibration Error Analysis:")
print(f"  Mean calibration error: {np.mean(np.abs(prob_pred - prob_true)):.4f}")
print(f"  Max calibration error: {np.max(np.abs(prob_pred - prob_true)):.4f}")

# ═══════════════════════════════════════════════════════════════════════════════════════
# CALIBRATED MODEL
# ═══════════════════════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("CALIBRATION WITH ISOTONIC REGRESSION")
print("="*80)

# Train calibrated model
model = xgb.XGBClassifier(**selected_params)
calibrated_model = CalibratedClassifierCV(model, method='isotonic', cv=5)

calibrated_all_y_pred = []
for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_engineered, y)):
    X_train, X_val = X_engineered.iloc[train_idx], X_engineered.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    cal_model = CalibratedClassifierCV(xgb.XGBClassifier(**selected_params), method='isotonic', cv=5)
    cal_model.fit(X_train, y_train)
    y_pred_proba = cal_model.predict_proba(X_val)[:, 1]
    calibrated_all_y_pred.extend(y_pred_proba)

calibrated_all_y_pred = np.array(calibrated_all_y_pred)

prob_true_cal, prob_pred_cal = calibration_curve(all_y_true, calibrated_all_y_pred, n_bins=10)
print(f"\n✓ Calibrated model:")
print(f"  Mean calibration error (after): {np.mean(np.abs(prob_pred_cal - prob_true_cal)):.4f}")

# ═══════════════════════════════════════════════════════════════════════════════════════
# VISUALIZATION
# ═══════════════════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(16, 10))

# 1. Probability distribution
ax1 = plt.subplot(2, 3, 1)
ax1.hist(all_y_pred[all_y_true==0], bins=50, alpha=0.6, label='Normal', color='blue')
ax1.hist(all_y_pred[all_y_true==1], bins=20, alpha=0.6, label='Defects', color='red')
ax1.set_xlabel('Predicted Probability')
ax1.set_ylabel('Frequency')
ax1.set_title('Probability Distribution (Raw)')
ax1.set_yscale('log')
ax1.legend()
ax1.grid(alpha=0.3)

# 2. Cumulative distribution
ax2 = plt.subplot(2, 3, 2)
sorted_probs = np.sort(all_y_pred)
cumsum = np.arange(1, len(sorted_probs)+1) / len(sorted_probs)
ax2.plot(sorted_probs, cumsum, linewidth=2)
ax2.axvline(x=0.5, color='r', linestyle='--', label='0.5 threshold')
ax2.set_xlabel('Probability Threshold')
ax2.set_ylabel('Cumulative %')
ax2.set_title('Cumulative Distribution of Predictions')
ax2.legend()
ax2.grid(alpha=0.3)

# 3. Calibration curve (before)
ax3 = plt.subplot(2, 3, 3)
ax3.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
ax3.plot(prob_pred, prob_true, 'o-', label='Model (before)', linewidth=2, markersize=8)
ax3.plot(prob_pred_cal, prob_true_cal, 's-', label='Model (calibrated)', linewidth=2, markersize=8)
ax3.set_xlabel('Mean Predicted Probability')
ax3.set_ylabel('Mean Observed Probability')
ax3.set_title('Calibration Curve')
ax3.legend()
ax3.grid(alpha=0.3)

# 4. Defect capture by threshold
ax4 = plt.subplot(2, 3, 4)
thresholds = np.linspace(0, 1, 100)
defect_capture = [(all_y_pred[all_y_true==1] >= t).sum() / (all_y_true==1).sum() for t in thresholds]
ax4.plot(thresholds, defect_capture, linewidth=2, color='red')
ax4.axhline(y=0.9, color='g', linestyle='--', label='90% recall target')
ax4.set_xlabel('Probability Threshold')
ax4.set_ylabel('Defect Capture Rate (Recall)')
ax4.set_title('Recall vs Threshold')
ax4.legend()
ax4.grid(alpha=0.3)

# 5. False positive rate by threshold
ax5 = plt.subplot(2, 3, 5)
fp_rate = [(all_y_pred[all_y_true==0] >= t).sum() / (all_y_true==0).sum() for t in thresholds]
ax5.plot(thresholds, fp_rate, linewidth=2, color='orange')
ax5.set_xlabel('Probability Threshold')
ax5.set_ylabel('False Positive Rate')
ax5.set_title('False Positives vs Threshold')
ax5.grid(alpha=0.3)

# 6. Precision-Recall by threshold
ax6 = plt.subplot(2, 3, 6)
precision_vals = []
recall_vals = []
for t in thresholds:
    tp = ((all_y_pred >= t) & (all_y_true == 1)).sum()
    fp = ((all_y_pred >= t) & (all_y_true == 0)).sum()
    fn = ((all_y_pred < t) & (all_y_true == 1)).sum()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    precision_vals.append(precision)
    recall_vals.append(recall)

ax6.plot(recall_vals, precision_vals, linewidth=2, color='purple')
ax6.set_xlabel('Recall')
ax6.set_ylabel('Precision')
ax6.set_title('Precision-Recall vs Threshold')
ax6.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('07_calibration_analysis.png', dpi=150, bbox_inches='tight')
print("\n✓ Saved: 07_calibration_analysis.png")
plt.close()

# ═══════════════════════════════════════════════════════════════════════════════════════
# SAVE OUTPUTS
# ═══════════════════════════════════════════════════════════════════════════════════════
calibration_insights = {
    'all_y_true': all_y_true,
    'all_y_pred': all_y_pred,
    'calibrated_all_y_pred': calibrated_all_y_pred,
    'prob_true': prob_true,
    'prob_pred': prob_pred,
    'prob_true_cal': prob_true_cal,
    'prob_pred_cal': prob_pred_cal
}

with open('07_calibration_insights.pkl', 'wb') as f:
    pickle.dump(calibration_insights, f)

print(f"✓ Saved: 07_calibration_insights.pkl")

print("\n" + "="*80)
print("✓ PHASE 7 COMPLETE")
print("="*80)
