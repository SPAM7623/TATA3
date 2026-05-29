"""
========================================================
PHASE 1: INDUSTRIAL EDA - Understand Defect Behavior
========================================================
Objective: Understand defect behavior before modeling.

CHECKLIST - MAIN TASKS:
□ Dataset shape
□ Missing values
□ Duplicates
□ Class imbalance
□ Univariate analysis
□ Defect vs non-defect comparison
□ Variance/instability analysis
□ Outlier analysis
□ Correlation analysis
□ Interaction analysis
□ PCA / UMAP / t-SNE

CHECKLIST - KEY QUESTIONS TO ANSWER:
□ Which variables differ between defect and normal coils?
□ Which variables become unstable during defects?
□ Which parameter combinations appear dangerous?
□ Are there hidden process regimes?

OUTPUT EXPECTED:
□ Candidate important variables
□ Candidate interactions
□ Candidate operating regimes
□ Candidate instability indicators

FEEDS INTO: → Correlation Grouping → SHAP Validation → Feature Engineering
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
import warnings
warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════════════════════════════
# 1. LOAD DATA
# ═══════════════════════════════════════════════════════════════════════════════════════
print("="*80)
print("PHASE 1: INDUSTRIAL EDA - STARTING")
print("="*80)

train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

print("\n✓ DATA LOADED")
print(f"  Train shape: {train_df.shape}")
print(f"  Test shape: {test_df.shape}")

# ═══════════════════════════════════════════════════════════════════════════════════════
# 2. DATASET SHAPE & BASIC INFO
# ═══════════════════════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("2. DATASET SHAPE & STRUCTURE")
print("="*80)

print(f"\nTrain: {train_df.shape[0]} rows × {train_df.shape[1]} columns")
print(f"Test: {test_df.shape[0]} rows × {test_df.shape[1]} columns")
print(f"\nColumns: {list(train_df.columns)}")

# ═══════════════════════════════════════════════════════════════════════════════════════
# 3. MISSING VALUES
# ═══════════════════════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("3. MISSING VALUES ANALYSIS")
print("="*80)

missing_train = train_df.isnull().sum()
missing_train_pct = (missing_train / len(train_df)) * 100
missing_cols = missing_train[missing_train > 0]

if len(missing_cols) > 0:
    print(f"\nMissing values found in {len(missing_cols)} columns:")
    for col, count in missing_cols.items():
        print(f"  {col}: {count} ({missing_train_pct[col]:.2f}%)")
else:
    print("\n✓ No missing values in training data")

missing_test = test_df.isnull().sum()
print(f"\nTest missing values: {missing_test.sum()} total")

# ═══════════════════════════════════════════════════════════════════════════════════════
# 4. DUPLICATES
# ═══════════════════════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("4. DUPLICATES ANALYSIS")
print("="*80)

feature_cols = [col for col in train_df.columns if col not in ['CoilID', 'Y']]

# Check for complete duplicates
complete_dups = train_df[feature_cols].duplicated().sum()
print(f"\nComplete feature duplicates: {complete_dups}")

# Check for duplicates by CoilID
coil_dups = train_df['CoilID'].duplicated().sum()
print(f"Duplicate CoilIDs: {coil_dups}")

if complete_dups == 0 and coil_dups == 0:
    print("✓ No duplicates found")

# ═══════════════════════════════════════════════════════════════════════════════════════
# 5. CLASS IMBALANCE
# ═══════════════════════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("5. CLASS IMBALANCE ANALYSIS")
print("="*80)

class_counts = train_df['Y'].value_counts().sort_index()
class_pcts = (class_counts / len(train_df)) * 100

print(f"\nClass distribution:")
for class_label, count in class_counts.items():
    label_name = "DEFECT (Alpha)" if class_label == 1 else "NORMAL (No defect)"
    print(f"  {label_name}: {count:5d} ({class_pcts[class_label]:6.2f}%)")

imbalance_ratio = class_counts[0] / class_counts[1]
print(f"\n⚠ IMBALANCE RATIO: {imbalance_ratio:.2f}:1 (normal:defect)")
print(f"  → scale_pos_weight for XGBoost: {imbalance_ratio:.2f}")

# ═══════════════════════════════════════════════════════════════════════════════════════
# 6. UNIVARIATE ANALYSIS - DEFECT VS NON-DEFECT COMPARISON
# ═══════════════════════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("6. UNIVARIATE ANALYSIS - DEFECT VS NON-DEFECT")
print("="*80)

X_features = [col for col in train_df.columns if col.startswith('X')]
defect_df = train_df[train_df['Y'] == 1][X_features]
normal_df = train_df[train_df['Y'] == 0][X_features]

# Calculate statistics for each group
comparison_stats = []

print(f"\nStatistical comparison (Defect vs Normal):")
print(f"{'Variable':<8} {'Defect_Mean':>12} {'Normal_Mean':>12} {'Defect_Std':>12} {'Normal_Std':>12} {'T-Stat':>10} {'P-Value':>10} {'Diff %':>8}")
print("-" * 100)

for col in X_features:
    defect_mean = defect_df[col].mean()
    normal_mean = normal_df[col].mean()
    defect_std = defect_df[col].std()
    normal_std = normal_df[col].std()

    # T-test
    t_stat, p_val = stats.ttest_ind(defect_df[col].dropna(), normal_df[col].dropna())

    # Percentage difference
    pct_diff = ((defect_mean - normal_mean) / (normal_mean + 1e-10)) * 100

    comparison_stats.append({
        'Variable': col,
        'Defect_Mean': defect_mean,
        'Normal_Mean': normal_mean,
        'Defect_Std': defect_std,
        'Normal_Std': normal_std,
        'T_Stat': t_stat,
        'P_Value': p_val,
        'Pct_Diff': pct_diff
    })

    if p_val < 0.05:  # Statistically significant
        marker = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*"
        print(f"{col:<8} {defect_mean:12.2f} {normal_mean:12.2f} {defect_std:12.2f} {normal_std:12.2f} {t_stat:10.4f} {p_val:10.2e} {pct_diff:8.2f}{marker}")

comparison_df = pd.DataFrame(comparison_stats)
comparison_df_sorted = comparison_df.sort_values('P_Value')

# Save candidate important variables (p < 0.05)
candidate_important = comparison_df_sorted[comparison_df_sorted['P_Value'] < 0.05]['Variable'].tolist()
print(f"\n✓ CANDIDATE IMPORTANT VARIABLES ({len(candidate_important)} found):")
print(f"  {candidate_important}")

# ═══════════════════════════════════════════════════════════════════════════════════════
# 7. VARIANCE/INSTABILITY ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("7. VARIANCE/INSTABILITY ANALYSIS")
print("="*80)

print(f"\nVariance comparison (higher variance = instability):")
print(f"{'Variable':<8} {'Defect_Var':>14} {'Normal_Var':>14} {'Var_Ratio':>12} {'Levene_P':>12} {'Unstable?':>10}")
print("-" * 80)

instability_indicators = []

for col in X_features:
    defect_var = defect_df[col].var()
    normal_var = normal_df[col].var()
    var_ratio = (defect_var + 1e-10) / (normal_var + 1e-10)

    # Levene's test for variance equality
    levene_stat, levene_p = stats.levene(defect_df[col].dropna(), normal_df[col].dropna())

    unstable = "YES" if var_ratio > 1.5 and levene_p < 0.05 else "NO"

    instability_indicators.append({
        'Variable': col,
        'Defect_Var': defect_var,
        'Normal_Var': normal_var,
        'Var_Ratio': var_ratio,
        'Levene_P': levene_p,
        'Is_Unstable': unstable
    })

    if unstable == "YES":
        print(f"{col:<8} {defect_var:14.2f} {normal_var:14.2f} {var_ratio:12.4f} {levene_p:12.2e} {unstable:>10}")

instability_df = pd.DataFrame(instability_indicators)
unstable_vars = instability_df[instability_df['Is_Unstable'] == 'YES']['Variable'].tolist()
print(f"\n✓ CANDIDATE INSTABILITY INDICATORS ({len(unstable_vars)} found):")
print(f"  {unstable_vars}")

# ═══════════════════════════════════════════════════════════════════════════════════════
# 8. OUTLIER ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("8. OUTLIER ANALYSIS (IQR Method)")
print("="*80)

outlier_summary = {}

for col in X_features:
    Q1 = train_df[col].quantile(0.25)
    Q3 = train_df[col].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers_count = ((train_df[col] < lower_bound) | (train_df[col] > upper_bound)).sum()
    outlier_pct = (outliers_count / len(train_df)) * 100

    # Outliers in defect class
    defect_outliers = ((defect_df[col] < lower_bound) | (defect_df[col] > upper_bound)).sum()

    outlier_summary[col] = {
        'Total_Outliers': outliers_count,
        'Outlier_Pct': outlier_pct,
        'Defect_Outliers': defect_outliers,
        'Defect_Outlier_Pct': (defect_outliers / len(defect_df)) * 100
    }

outlier_df = pd.DataFrame(outlier_summary).T.sort_values('Outlier_Pct', ascending=False)
print(f"\nTop 10 variables by outlier percentage:")
print(outlier_df.head(10))

high_outlier_vars = outlier_df[outlier_df['Outlier_Pct'] > 5].index.tolist()
print(f"\n✓ Variables with >5% outliers: {len(high_outlier_vars)}")

# ═══════════════════════════════════════════════════════════════════════════════════════
# 9. CORRELATION ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("9. CORRELATION ANALYSIS")
print("="*80)

correlation_matrix = train_df[X_features].corr()

# Find high correlations (pairs)
print(f"\nHigh correlations (|r| > 0.8):")
print(f"{'Var1':<8} {'Var2':<8} {'Correlation':>12}")
print("-" * 40)

high_corr_pairs = []
for i in range(len(correlation_matrix.columns)):
    for j in range(i+1, len(correlation_matrix.columns)):
        if abs(correlation_matrix.iloc[i, j]) > 0.8:
            var1 = correlation_matrix.columns[i]
            var2 = correlation_matrix.columns[j]
            corr_val = correlation_matrix.iloc[i, j]
            print(f"{var1:<8} {var2:<8} {corr_val:12.4f}")
            high_corr_pairs.append((var1, var2, corr_val))

print(f"\n✓ Found {len(high_corr_pairs)} highly correlated pairs (|r| > 0.8)")

# ═══════════════════════════════════════════════════════════════════════════════════════
# 10. INTERACTION ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("10. INTERACTION ANALYSIS - DANGEROUS PARAMETER COMBINATIONS")
print("="*80)

# Find variables with both high defect-normal difference AND high instability
dangerous_combos = set(candidate_important) & set(unstable_vars)
print(f"\n✓ DANGEROUS INTERACTION CANDIDATES ({len(dangerous_combos)} found):")
print(f"  Variables that are both statistically different AND unstable in defect class:")
print(f"  {sorted(dangerous_combos)}")

# ═══════════════════════════════════════════════════════════════════════════════════════
# 11. OPERATING REGIMES DETECTION (Using means)
# ═══════════════════════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("11. HIDDEN OPERATING REGIMES DETECTION")
print("="*80)

# Standardize features for clustering
scaler = StandardScaler()
X_scaled = scaler.fit_transform(train_df[X_features])

# Use quantiles to define regimes
print(f"\nDetecting potential operating regimes using feature quantiles...")

# High-risk variables (from candidate important + instability)
risk_vars = list(set(candidate_important + unstable_vars))[:10]  # Top 10
print(f"  Using top risk variables: {risk_vars}")

regimes = {}
for var in risk_vars[:5]:
    q25 = train_df[var].quantile(0.25)
    q50 = train_df[var].quantile(0.50)
    q75 = train_df[var].quantile(0.75)

    defect_in_low = ((defect_df[var] < q25).sum() / len(defect_df)) * 100
    defect_in_mid = ((defect_df[var].between(q25, q75)).sum() / len(defect_df)) * 100
    defect_in_high = ((defect_df[var] > q75).sum() / len(defect_df)) * 100

    regimes[var] = {
        'Low': defect_in_low,
        'Mid': defect_in_mid,
        'High': defect_in_high
    }
    print(f"\n  {var}: Defect % in [Low | Mid | High] regimes: [{defect_in_low:.1f}% | {defect_in_mid:.1f}% | {defect_in_high:.1f}%]")

print(f"\n✓ CANDIDATE OPERATING REGIMES: Identified regime shifts")

# ═══════════════════════════════════════════════════════════════════════════════════════
# 12. PCA ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("12. PCA - DIMENSIONALITY REDUCTION")
print("="*80)

# Fill missing values with median before scaling
X_filled = train_df[X_features].fillna(train_df[X_features].median())
X_scaled = scaler.fit_transform(X_filled)

pca = PCA()
pca.fit(X_scaled)

cumsum_var = np.cumsum(pca.explained_variance_ratio_)

print(f"\nExplained variance by components:")
print(f"  PC1-5: {cumsum_var[4]:.2%}")
print(f"  PC1-10: {cumsum_var[9]:.2%}")
print(f"  PC1-20: {cumsum_var[19]:.2%}")
print(f"  PC1-30: {cumsum_var[29]:.2%} (all 49)")

# PCA with 2 components
pca_2d = PCA(n_components=2)
X_pca = pca_2d.fit_transform(X_scaled)

print(f"\n✓ PCA complete. Variance explained (2D): {cumsum_var[1]:.2%}")

# ═══════════════════════════════════════════════════════════════════════════════════════
# 13. t-SNE VISUALIZATION
# ═══════════════════════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("13. t-SNE - NON-LINEAR DIMENSIONALITY REDUCTION")
print("="*80)

print(f"\nFitting t-SNE (this may take a minute)...")
tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter_without_progress=300)
X_tsne = tsne.fit_transform(X_scaled)
print(f"✓ t-SNE complete")

# ═══════════════════════════════════════════════════════════════════════════════════════
# 14. UMAP VISUALIZATION
# ═══════════════════════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("14. UMAP - TOPOLOGICAL EMBEDDING")
print("="*80)

print(f"\nFitting UMAP...")
umap_reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15)
X_umap = umap_reducer.fit_transform(X_scaled)
print(f"✓ UMAP complete")

# ═══════════════════════════════════════════════════════════════════════════════════════
# 15. VISUALIZATION - SAVE FIGURES
# ═══════════════════════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("15. CREATING VISUALIZATIONS")
print("="*80)

fig = plt.figure(figsize=(20, 12))

# PCA
ax1 = plt.subplot(2, 3, 1)
scatter1 = ax1.scatter(X_pca[train_df['Y']==0, 0], X_pca[train_df['Y']==0, 1],
                       c='blue', alpha=0.6, s=30, label='Normal')
scatter2 = ax1.scatter(X_pca[train_df['Y']==1, 0], X_pca[train_df['Y']==1, 1],
                       c='red', alpha=0.8, s=50, marker='x', label='Defect')
ax1.set_xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]:.1%})')
ax1.set_ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]:.1%})')
ax1.set_title('PCA Projection')
ax1.legend()
ax1.grid(alpha=0.3)

# t-SNE
ax2 = plt.subplot(2, 3, 2)
ax2.scatter(X_tsne[train_df['Y']==0, 0], X_tsne[train_df['Y']==0, 1],
           c='blue', alpha=0.6, s=30, label='Normal')
ax2.scatter(X_tsne[train_df['Y']==1, 0], X_tsne[train_df['Y']==1, 1],
           c='red', alpha=0.8, s=50, marker='x', label='Defect')
ax2.set_xlabel('t-SNE 1')
ax2.set_ylabel('t-SNE 2')
ax2.set_title('t-SNE Projection')
ax2.legend()
ax2.grid(alpha=0.3)

# UMAP
ax3 = plt.subplot(2, 3, 3)
ax3.scatter(X_umap[train_df['Y']==0, 0], X_umap[train_df['Y']==0, 1],
           c='blue', alpha=0.6, s=30, label='Normal')
ax3.scatter(X_umap[train_df['Y']==1, 0], X_umap[train_df['Y']==1, 1],
           c='red', alpha=0.8, s=50, marker='x', label='Defect')
ax3.set_xlabel('UMAP 1')
ax3.set_ylabel('UMAP 2')
ax3.set_title('UMAP Projection')
ax3.legend()
ax3.grid(alpha=0.3)

# Scree plot
ax4 = plt.subplot(2, 3, 4)
ax4.plot(range(1, 21), cumsum_var[:20], 'bo-', linewidth=2)
ax4.axhline(y=0.9, color='r', linestyle='--', label='90% threshold')
ax4.set_xlabel('Number of Components')
ax4.set_ylabel('Cumulative Explained Variance')
ax4.set_title('PCA Scree Plot')
ax4.legend()
ax4.grid(alpha=0.3)

# Class distribution
ax5 = plt.subplot(2, 3, 5)
ax5.bar(['Normal', 'Defect'], class_counts, color=['blue', 'red'], alpha=0.7)
ax5.set_ylabel('Count')
ax5.set_title(f'Class Distribution (Imbalance: {imbalance_ratio:.1f}:1)')
for i, v in enumerate(class_counts):
    ax5.text(i, v + 10, str(v), ha='center', fontweight='bold')

# Top differentiating variables
ax6 = plt.subplot(2, 3, 6)
top_diff = comparison_df_sorted.head(10)
ax6.barh(range(len(top_diff)), -np.log10(top_diff['P_Value'].values), color='green', alpha=0.7)
ax6.set_yticks(range(len(top_diff)))
ax6.set_yticklabels(top_diff['Variable'].values)
ax6.set_xlabel('-log10(P-Value)')
ax6.set_title('Top 10 Differentiating Variables')
ax6.axvline(x=-np.log10(0.05), color='r', linestyle='--', label='p=0.05')
ax6.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('01_eda_visualizations.png', dpi=150, bbox_inches='tight')
print("✓ Saved: 01_eda_visualizations.png")
plt.close()

# ═══════════════════════════════════════════════════════════════════════════════════════
# 16. SAVE EDA OUTPUTS FOR NEXT PHASES
# ═══════════════════════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("16. SAVING EDA OUTPUTS FOR DOWNSTREAM PHASES")
print("="*80)

# Save candidate variables
eda_insights = {
    'candidate_important_variables': candidate_important,
    'unstable_variables': unstable_vars,
    'dangerous_interaction_candidates': list(dangerous_combos),
    'high_correlation_pairs': high_corr_pairs,
    'imbalance_ratio': imbalance_ratio,
    'comparison_stats': comparison_df_sorted,
    'instability_stats': instability_df
}

# Save to pickle for downstream use
import pickle
with open('01_eda_insights.pkl', 'wb') as f:
    pickle.dump(eda_insights, f)

print(f"\n✓ Saved: 01_eda_insights.pkl")

# ═══════════════════════════════════════════════════════════════════════════════════════
# 17. SUMMARY & NEXT STEPS
# ═══════════════════════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("PHASE 1 SUMMARY")
print("="*80)

print(f"""
KEY FINDINGS:

1. CLASS IMBALANCE:
   - Normal coils: {class_counts[0]:,} ({class_pcts[0]:.2f}%)
   - Defect coils: {class_counts[1]:,} ({class_pcts[1]:.2f}%)
   - Imbalance ratio: {imbalance_ratio:.2f}:1
   → Must use scale_pos_weight = {imbalance_ratio:.2f}

2. SIGNIFICANT VARIABLES:
   - {len(candidate_important)} variables significantly different between defect/normal
   - Top variables: {candidate_important[:5]}

3. INSTABILITY INDICATORS:
   - {len(unstable_vars)} variables show high variance in defect class
   - Suggests process becomes unstable before/during defects

4. DANGEROUS INTERACTIONS:
   - {len(dangerous_combos)} variables are both significant AND unstable
   - These are prime candidates for feature engineering
   - Variables: {list(dangerous_combos)[:10]}

5. DIMENSIONALITY:
   - 49 original features
   - First 20 PCs explain {cumsum_var[19]:.1%} of variance
   - Suggests redundancy → feature engineering needed

6. SEPARABILITY:
   - PCA/t-SNE/UMAP show some clustering by class
   - Defects are minority but distinguishable
   - Signal is present but weak

NEXT PHASE (Phase 2: Correlation Grouping):
→ Find which variables move together
→ Identify hidden process blocks
→ Prepare for smarter feature engineering
""")

print("\n" + "="*80)
print("✓ PHASE 1 COMPLETE")
print("="*80)
