"""
========================================================
PHASE 2: CORRELATION GROUPING - Find Hidden Process Blocks
========================================================
Objective: Find hidden process blocks.

CHECKLIST - MAIN TASKS:
□ Correlation heatmap
□ Hierarchical clustering
□ Feature grouping

OUTPUT EXPECTED:
□ Correlated feature blocks
□ Redundant features
□ Candidate process groups

INFERENCE: If X7,X12,X18 move together: → likely same hidden process block

FEEDS INTO: → Feature Engineering → SHAP Interpretation → Interaction Analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist, squareform
import pickle
import warnings
warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════════════════════════════
# 1. LOAD DATA AND EDA INSIGHTS
# ═══════════════════════════════════════════════════════════════════════════════════════
print("="*80)
print("PHASE 2: CORRELATION GROUPING - STARTING")
print("="*80)

train_df = pd.read_csv('train.csv')

# Load EDA insights
with open('01_eda_insights.pkl', 'rb') as f:
    eda_insights = pickle.load(f)

candidate_important = eda_insights['candidate_important_variables']

X_features = [col for col in train_df.columns if col.startswith('X')]

print(f"\n✓ DATA LOADED")
print(f"  Total features: {len(X_features)}")
print(f"  Candidate important: {len(candidate_important)}")

# ═══════════════════════════════════════════════════════════════════════════════════════
# 2. COMPUTE CORRELATION MATRIX
# ═══════════════════════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("2. COMPUTING CORRELATION MATRIX")
print("="*80)

X_filled = train_df[X_features].fillna(train_df[X_features].median())
corr_matrix = X_filled.corr().abs()

print(f"\n✓ Correlation matrix computed")
print(f"  Shape: {corr_matrix.shape}")

# ═══════════════════════════════════════════════════════════════════════════════════════
# 3. HIERARCHICAL CLUSTERING ON CORRELATION MATRIX
# ═══════════════════════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("3. HIERARCHICAL CLUSTERING")
print("="*80)

# Convert correlation to distance (dissimilarity)
distance_matrix = 1 - corr_matrix
distance_condensed = squareform(distance_matrix)

# Perform hierarchical clustering
linkage_matrix = linkage(distance_condensed, method='ward')

print(f"\n✓ Hierarchical clustering completed")

# Get clusters at different cut thresholds
print(f"\nCluster formation at different distance thresholds:")
for threshold in [0.3, 0.4, 0.5, 0.6]:
    clusters = fcluster(linkage_matrix, threshold, criterion='distance')
    n_clusters = len(np.unique(clusters))
    print(f"  Distance {threshold}: {n_clusters} clusters")

# Use threshold that gives reasonable number of groups (around 8-12)
optimal_threshold = 0.5
clusters = fcluster(linkage_matrix, optimal_threshold, criterion='distance')
n_clusters = len(np.unique(clusters))

print(f"\n→ Using threshold: {optimal_threshold} → {n_clusters} clusters")

# ═══════════════════════════════════════════════════════════════════════════════════════
# 4. BUILD FEATURE GROUPS
# ═══════════════════════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("4. FEATURE GROUPING")
print("="*80)

feature_groups = {}
for i, feature in enumerate(X_features):
    cluster_id = clusters[i]
    if cluster_id not in feature_groups:
        feature_groups[cluster_id] = []
    feature_groups[cluster_id].append(feature)

# Sort by group size
feature_groups = dict(sorted(feature_groups.items(),
                             key=lambda x: len(x[1]), reverse=True))

print(f"\n✓ {n_clusters} Feature groups identified:")
print(f"\n{'Group':<8} {'Size':<8} {'Features':<60}")
print("-" * 80)

for group_id, features in feature_groups.items():
    features_str = ', '.join(features)
    if len(features_str) > 60:
        features_str = features_str[:57] + "..."
    print(f"G{group_id:<7} {len(features):<8} {features_str:<60}")

# ═══════════════════════════════════════════════════════════════════════════════════════
# 5. INTRA-GROUP CORRELATION ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("5. INTRA-GROUP CORRELATION ANALYSIS")
print("="*80)

print(f"\nAverage within-group correlations (cohesion):")
print(f"{'Group':<8} {'Size':<8} {'Avg_Corr':<12} {'Min_Corr':<12} {'Max_Corr':<12}")
print("-" * 60)

group_stats = {}
for group_id, features in feature_groups.items():
    if len(features) > 1:
        submatrix = corr_matrix.loc[features, features]

        # Get upper triangle (excluding diagonal)
        mask = np.triu(np.ones_like(submatrix, dtype=bool), k=1)
        correlations = submatrix.values[mask]

        avg_corr = correlations.mean()
        min_corr = correlations.min()
        max_corr = correlations.max()
    else:
        avg_corr = min_corr = max_corr = np.nan

    group_stats[group_id] = {
        'avg_corr': avg_corr,
        'min_corr': min_corr,
        'max_corr': max_corr
    }

    print(f"G{group_id:<7} {len(features):<8} {avg_corr:<12.4f} {min_corr:<12.4f} {max_corr:<12.4f}")

# ═══════════════════════════════════════════════════════════════════════════════════════
# 6. IDENTIFY REDUNDANT FEATURES
# ═══════════════════════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("6. REDUNDANT FEATURES ANALYSIS")
print("="*80)

print(f"\nHighly redundant features within each group (|r| > 0.9):")

redundant_features = []
for group_id, features in feature_groups.items():
    if len(features) > 1:
        for i, feat1 in enumerate(features):
            for feat2 in features[i+1:]:
                if corr_matrix.loc[feat1, feat2] > 0.9:
                    redundant_features.append((feat1, feat2, corr_matrix.loc[feat1, feat2]))
                    print(f"  G{group_id}: {feat1} <--> {feat2}: {corr_matrix.loc[feat1, feat2]:.4f}")

print(f"\n✓ Found {len(redundant_features)} highly redundant pairs")

# ═══════════════════════════════════════════════════════════════════════════════════════
# 7. INTERPRET PROCESS GROUPS
# ═══════════════════════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("7. PROCESS GROUP INTERPRETATION")
print("="*80)

print(f"\nInterpretation of each feature group:")
print(f"(✓ = candidate important variable from Phase 1)\n")

important_by_group = {}
for group_id, features in feature_groups.items():
    important_in_group = [f for f in features if f in candidate_important]
    important_by_group[group_id] = important_in_group

    print(f"Group {group_id} ({len(features)} features):")
    print(f"  Features: {features}")

    if important_in_group:
        print(f"  ✓ IMPORTANT: {important_in_group}")
        print(f"  → This group likely represents a key process block")
    else:
        print(f"  → Supportive process block")

    # Get average correlation with important variables
    if important_in_group:
        avg_corr_to_important = []
        for feat in features:
            if feat not in important_in_group:
                for imp in important_in_group:
                    avg_corr_to_important.append(corr_matrix.loc[feat, imp])

        if avg_corr_to_important:
            print(f"  → Avg correlation to important vars: {np.mean(avg_corr_to_important):.4f}")

    print()

# ═══════════════════════════════════════════════════════════════════════════════════════
# 8. VISUALIZATION - CORRELATION HEATMAP WITH CLUSTERING
# ═══════════════════════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("8. CREATING VISUALIZATIONS")
print("="*80)

fig, axes = plt.subplots(2, 2, figsize=(20, 16))

# Full correlation heatmap
ax1 = axes[0, 0]
sns.heatmap(corr_matrix, cmap='RdBu_r', vmin=0, vmax=1, ax=ax1,
            xticklabels=False, yticklabels=False, cbar_kws={'label': 'Correlation'})
ax1.set_title('Full Correlation Matrix (All 49 Features)')

# Clustered heatmap (reorder by cluster)
ax2 = axes[0, 1]
cluster_order = []
for group_id in sorted(feature_groups.keys()):
    cluster_order.extend(feature_groups[group_id])

corr_clustered = corr_matrix.loc[cluster_order, cluster_order]
sns.heatmap(corr_clustered, cmap='RdBu_r', vmin=0, vmax=1, ax=ax2,
            xticklabels=False, yticklabels=False, cbar_kws={'label': 'Correlation'})
ax2.set_title(f'Correlation Matrix (Clustered - {n_clusters} Groups)')

# Draw group boundaries
y_offset = 0
for group_id in sorted(feature_groups.keys()):
    group_size = len(feature_groups[group_id])
    ax2.axhline(y=y_offset, color='black', linewidth=2)
    ax2.axvline(x=y_offset, color='black', linewidth=2)
    y_offset += group_size

# Dendrogram
ax3 = axes[1, 0]
dendrogram(linkage_matrix, ax=ax3, labels=X_features, leaf_font_size=8)
ax3.set_title('Hierarchical Clustering Dendrogram')
ax3.axhline(y=optimal_threshold, color='r', linestyle='--', label=f'Cut threshold={optimal_threshold}')
ax3.legend()
ax3.set_ylabel('Distance')

# Group sizes
ax4 = axes[1, 1]
group_ids = list(feature_groups.keys())
group_sizes = [len(feature_groups[gid]) for gid in group_ids]
colors = ['green' if any(f in candidate_important for f in feature_groups[gid]) else 'blue'
          for gid in group_ids]
bars = ax4.bar(range(len(group_ids)), group_sizes, color=colors, alpha=0.7)
ax4.set_xticks(range(len(group_ids)))
ax4.set_xticklabels([f'G{gid}' for gid in group_ids])
ax4.set_ylabel('Number of Features')
ax4.set_title('Feature Group Sizes\n(Green = Has important variables)')
ax4.grid(axis='y', alpha=0.3)

# Add values on bars
for i, bar in enumerate(bars):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(height)}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('02_correlation_grouping.png', dpi=150, bbox_inches='tight')
print("\n✓ Saved: 02_correlation_grouping.png")
plt.close()

# ═══════════════════════════════════════════════════════════════════════════════════════
# 9. SAVE OUTPUTS FOR NEXT PHASES
# ═══════════════════════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("9. SAVING OUTPUTS FOR DOWNSTREAM PHASES")
print("="*80)

correlation_insights = {
    'feature_groups': feature_groups,
    'feature_group_stats': group_stats,
    'redundant_features': redundant_features,
    'important_by_group': important_by_group,
    'correlation_matrix': corr_matrix,
    'cluster_order': cluster_order,
    'n_clusters': n_clusters
}

with open('02_correlation_insights.pkl', 'wb') as f:
    pickle.dump(correlation_insights, f)

print(f"\n✓ Saved: 02_correlation_insights.pkl")

# ═══════════════════════════════════════════════════════════════════════════════════════
# 10. SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("PHASE 2 SUMMARY")
print("="*80)

print(f"""
KEY FINDINGS:

1. FEATURE GROUPING:
   - {n_clusters} distinct feature groups identified
   - Largest group: {max(len(f) for f in feature_groups.values())} features
   - Smallest group: {min(len(f) for f in feature_groups.values())} feature(s)

2. HIGHLY REDUNDANT PAIRS:
   - {len(redundant_features)} feature pairs with |r| > 0.9
   - These are candidates for removal/aggregation
   - Examples: {[f"{f1}-{f2}" for f1, f2, _ in redundant_features[:3]]}

3. PROCESS BLOCK MAPPING:
   - Important variables distributed across {len([g for g in important_by_group.values() if g])} groups
   - Groups with important vars represent key process stages
   - Other groups are supporting/secondary processes

4. FEATURE ENGINEERING STRATEGY:
   - Create group-level aggregations (mean, std, max, min)
   - Create ratios between key groups
   - Create variance indicators for unstable groups
   - This reduces dimensionality while preserving information

NEXT PHASE (Phase 3: Baseline Modeling):
→ Build baseline XGBoost model
→ Test LightGBM and Random Forest
→ Select strongest model family
→ Establish baseline performance metrics
""")

print("\n" + "="*80)
print("✓ PHASE 2 COMPLETE")
print("="*80)
