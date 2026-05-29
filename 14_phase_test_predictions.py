"""
========================================================
PHASE 14: TEST PREDICTIONS - Generate Submission File
========================================================
Objective: Generate test predictions using 5-seed ensemble.

MODEL: 5-Seed XGBoost with optimized threshold
FEATURES: 147 engineered features
THRESHOLD: 0.015925

OUTPUT: submission.csv (CoilID, Y)
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("PHASE 14: TEST PREDICTIONS - STARTING")
print("="*80)

# Load data and insights
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

with open('01_eda_insights.pkl', 'rb') as f:
    eda_insights = pickle.load(f)
with open('02_correlation_insights.pkl', 'rb') as f:
    corr_insights = pickle.load(f)
with open('04_shap_insights.pkl', 'rb') as f:
    shap_insights = pickle.load(f)
with open('05_engineering_insights.pkl', 'rb') as f:
    eng = pickle.load(f)
with open('06_imbalance_insights.pkl', 'rb') as f:
    imb = pickle.load(f)
with open('13_final_insights.pkl', 'rb') as f:
    final = pickle.load(f)

X_engineered_train = eng['X_engineered']
y_train = train_df['Y']
selected_params = imb['selected_params']
final_threshold = final['final_threshold']
seeds = final['seeds_used']

print(f"\nLoaded training features: {X_engineered_train.shape}")
print(f"Final threshold: {final_threshold:.6f}")
print(f"Seeds: {seeds}")

# ═══════════════════════════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING FOR TEST SET
# ═══════════════════════════════════════════════════════════════════════════════════════
print("\nEngineering test features...")

X_features = [col for col in train_df.columns if col.startswith('X')]
X_train = train_df[X_features].fillna(train_df[X_features].median())
X_test = test_df[X_features].fillna(train_df[X_features].median())

feature_groups = corr_insights['feature_groups']
top_shap_features = shap_insights['top_shap_features'][:10]
top_interactions = shap_insights['top_interactions'].head(10)

# Initialize engineered features dataframe
engineered_features_test = pd.DataFrame(index=X_test.index)

# 1. GROUP AGGREGATIONS
print("  • Group aggregations...")
for group_id, features in feature_groups.items():
    if len(features) > 1:
        group_data = X_test[features]
        engineered_features_test[f'G{group_id}_mean'] = group_data.mean(axis=1)
        engineered_features_test[f'G{group_id}_std'] = group_data.std(axis=1)
        engineered_features_test[f'G{group_id}_max'] = group_data.max(axis=1)
        engineered_features_test[f'G{group_id}_min'] = group_data.min(axis=1)
        engineered_features_test[f'G{group_id}_range'] = group_data.max(axis=1) - group_data.min(axis=1)
        engineered_features_test[f'G{group_id}_cv'] = (group_data.std(axis=1) / (group_data.mean(axis=1).abs() + 1e-10))

# 2. INTERACTION FEATURES
print("  • Interaction features...")
for idx, row in top_interactions.iterrows():
    f1, f2 = row['Feature1'], row['Feature2']
    engineered_features_test[f'{f1}_{f2}_prod'] = X_test[f1] * X_test[f2]
    engineered_features_test[f'{f1}_{f2}_ratio'] = X_test[f1] / (X_test[f2] + 1e-10)
    engineered_features_test[f'{f1}_{f2}_diff'] = X_test[f1] - X_test[f2]

# 3. TOP SHAP FEATURE RATIOS
print("  • SHAP feature ratios...")
for i, feat in enumerate(top_shap_features):
    for j, other_feat in enumerate(top_shap_features):
        if i < j:
            engineered_features_test[f'{feat}_{other_feat}_ratio'] = X_test[feat] / (X_test[other_feat] + 1e-10)

# 4. PROCESS STATE INDICATORS
print("  • Process state indicators...")
high_pressure_features = ['X10', 'X13', 'X29', 'X30', 'X31', 'X32', 'X33']
high_pressure_features = [f for f in high_pressure_features if f in X_features]
engineered_features_test['high_pressure_regime'] = (X_test[high_pressure_features] > X_train[high_pressure_features].quantile(0.75)).sum(axis=1)

high_temp_features = ['X4', 'X5', 'X6', 'X7', 'X8', 'X9']
high_temp_features = [f for f in high_temp_features if f in X_features]
engineered_features_test['high_temp_regime'] = (X_test[high_temp_features] > X_train[high_temp_features].quantile(0.75)).sum(axis=1)

engineered_features_test['regime_instability'] = engineered_features_test['high_pressure_regime'] + engineered_features_test['high_temp_regime']

low_risk_features = ['X34', 'X35']
low_risk_features = [f for f in low_risk_features if f in X_features]
if low_risk_features:
    engineered_features_test['low_defect_likelihood'] = (X_test[low_risk_features] < X_train[low_risk_features].quantile(0.25)).sum(axis=1)

# 5. STATISTICAL FEATURES
print("  • Statistical features...")
engineered_features_test['row_mean'] = X_test.mean(axis=1)
engineered_features_test['row_std'] = X_test.std(axis=1)
engineered_features_test['row_max'] = X_test.max(axis=1)
engineered_features_test['row_min'] = X_test.min(axis=1)
engineered_features_test['row_range'] = X_test.max(axis=1) - X_test.min(axis=1)
engineered_features_test['row_skew'] = X_test.skew(axis=1)
engineered_features_test['row_cv'] = X_test.std(axis=1) / (X_test.mean(axis=1).abs() + 1e-10)

# Extreme values
engineered_features_test['n_extreme_high'] = (X_test > X_train.quantile(0.95)).sum(axis=1)
engineered_features_test['n_extreme_low'] = (X_test < X_train.quantile(0.05)).sum(axis=1)

# 6. CLEANUP
print("  • Cleaning features...")
engineered_features_test = engineered_features_test.replace([np.inf, -np.inf], np.nan)
engineered_features_test = engineered_features_test.fillna(engineered_features_test.median())
engineered_features_test = engineered_features_test.fillna(0)

# Combine with original features
X_test_engineered = pd.concat([X_test, engineered_features_test], axis=1)

# Reorder to match training feature order
X_test_engineered = X_test_engineered[X_engineered_train.columns]

print(f"  Test engineered features shape: {X_test_engineered.shape}")

# ═══════════════════════════════════════════════════════════════════════════════════════
# 5-SEED ENSEMBLE PREDICTIONS ON TEST SET
# ═══════════════════════════════════════════════════════════════════════════════════════
print(f"\n5-Seed ensemble predictions...")

all_test_preds = []

for seed_idx, seed in enumerate(seeds):
    print(f"  Seed {seed_idx+1}/5 ({seed})...", end=' ')

    params_seed = selected_params.copy()
    params_seed['random_state'] = seed

    model = xgb.XGBClassifier(**params_seed)
    model.fit(X_engineered_train, y_train, verbose=False)

    pred = model.predict_proba(X_test_engineered)[:, 1]
    all_test_preds.append(pred)

    print(f"pred mean: {pred.mean():.6f}")

# Average predictions
avg_test_pred = np.mean(all_test_preds, axis=0)
print(f"\nEnsemble mean prediction: {avg_test_pred.mean():.6f}")
print(f"Probability range: {avg_test_pred.min():.6f} - {avg_test_pred.max():.6f}")

# ═══════════════════════════════════════════════════════════════════════════════════════
# APPLY THRESHOLD AND CREATE SUBMISSION
# ═══════════════════════════════════════════════════════════════════════════════════════
print(f"\nApplying threshold: {final_threshold:.6f}")

y_pred_final = (avg_test_pred >= final_threshold).astype(int)

n_defects = (y_pred_final == 1).sum()
n_normal = (y_pred_final == 0).sum()

print(f"  Predicted defects: {n_defects} ({100*n_defects/len(y_pred_final):.2f}%)")
print(f"  Predicted normal:  {n_normal} ({100*n_normal/len(y_pred_final):.2f}%)")

# Create submission
submission = pd.DataFrame({
    'CoilID': test_df['CoilID'],
    'Y': y_pred_final
})

submission.to_csv('submission.csv', index=False)
print(f"\n✓ submission.csv ({len(submission)} rows)")

# Also save with probabilities
submission_prob = pd.DataFrame({
    'CoilID': test_df['CoilID'],
    'Y': y_pred_final,
    'Probability': avg_test_pred
})

submission_prob.to_csv('submission_with_probabilities.csv', index=False)
print(f"✓ submission_with_probabilities.csv")

# Display first few rows
print(f"\nFirst 10 predictions:")
print(submission.head(10).to_string(index=False))

print("\n" + "="*80)
print("✓ PHASE 14 COMPLETE - SUBMISSION READY")
print("="*80)
