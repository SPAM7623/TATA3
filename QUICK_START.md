# Alpha Defect Prediction Workflow - Quick Start Guide

## ✓ CURRENT STATUS: Phases 1-3 Complete (23% of workflow)

---

## What's Been Done

### Phase 1: Industrial EDA ✓
```bash
python 01_phase_industrial_eda.py
# Outputs: 01_eda_insights.pkl, 01_eda_visualizations.png
```
**Key Finding**: 39 significant variables, 32 correlated pairs, weak but learnable signal

### Phase 2: Correlation Grouping ✓
```bash
python 02_phase_correlation_grouping.py
# Outputs: 02_correlation_insights.pkl, 02_correlation_grouping.png
```
**Key Finding**: 27 process blocks identified, 12 redundant pairs found

### Phase 3: Baseline Modeling ✓
```bash
python 03_phase_baseline_modeling.py
# Outputs: 03_modeling_insights.pkl, 03_baseline_modeling.png
```
**Key Finding**: XGBoost wins with 0.8574 ROC-AUC

---

## Key Metrics Summary

| Phase | Key Metric | Value | Status |
|-------|-----------|-------|--------|
| 1 | Important Variables | 39 | ✓ |
| 1 | Imbalance Ratio | 19.48:1 | ✓ |
| 2 | Feature Groups | 27 | ✓ |
| 2 | Redundant Pairs | 12 | ✓ |
| 3 | Best Model | XGBoost | ✓ |
| 3 | ROC-AUC | 0.8574 | ✓ |
| 3 | Recall | 0.2736 | ✓ |

---

## Files Overview

```
CODE SCRIPTS (Ready to Run)
├─ 01_phase_industrial_eda.py (26 KB)
├─ 02_phase_correlation_grouping.py (16 KB)
├─ 03_phase_baseline_modeling.py (19 KB)
└─ [Phases 4-13 to be created]

DATA OUTPUTS (Pickled)
├─ 01_eda_insights.pkl (7.8 KB)
├─ 02_correlation_insights.pkl (23 KB)
└─ 03_modeling_insights.pkl (45 KB)

VISUALIZATIONS
├─ 01_eda_visualizations.png (567 KB)
├─ 02_correlation_grouping.png (192 KB)
└─ 03_baseline_modeling.png (236 KB)

DOCUMENTATION
├─ WORKFLOW_STATUS.md (Current phase overview)
├─ PHASE_COMPLETION_REPORT.md (Detailed findings)
└─ QUICK_START.md (This file)

RAW DATA
├─ train.csv (1.1 MB, 1352 rows, 49 features)
└─ test.csv (264 KB, 339 rows, 49 features)
```

---

## Quick Command Reference

### Run All Completed Phases
```bash
# Run Phase 1: EDA (takes ~2 minutes)
python 01_phase_industrial_eda.py

# Run Phase 2: Correlation Grouping (takes ~30 seconds)
python 02_phase_correlation_grouping.py

# Run Phase 3: Baseline Modeling (takes ~5 minutes)
python 03_phase_baseline_modeling.py
```

### View Results
```bash
# View workflow status
cat WORKFLOW_STATUS.md

# View detailed report
cat PHASE_COMPLETION_REPORT.md

# View generated visualizations
# 01_eda_visualizations.png - PCA, t-SNE, UMAP, distributions
# 02_correlation_grouping.png - Correlation heatmap, dendrogram
# 03_baseline_modeling.png - ROC curves, PR curves
```

### Load Insights in Python
```python
import pickle

# Phase 1 EDA insights
with open('01_eda_insights.pkl', 'rb') as f:
    eda = pickle.load(f)
    print(eda['candidate_important_variables'])

# Phase 2 Correlation insights
with open('02_correlation_insights.pkl', 'rb') as f:
    corr = pickle.load(f)
    print(corr['feature_groups'])

# Phase 3 Modeling insights
with open('03_modeling_insights.pkl', 'rb') as f:
    model = pickle.load(f)
    print(f"XGBoost ROC-AUC: {np.mean(model['xgb_scores']['roc_auc']):.4f}")
```

---

## The 13-Phase Workflow

```
✓ Phase 1:  Industrial EDA
✓ Phase 2:  Correlation Grouping
✓ Phase 3:  Baseline Modeling
→ Phase 4:  SHAP Interpretation       [NEXT]
→ Phase 5:  Feature Engineering
→ Phase 6:  Imbalance Handling
→ Phase 7:  Calibration & Probability
→ Phase 8:  Threshold Optimization
→ Phase 9:  Stability Analysis
→ Phase 10: Overfitting Analysis
→ Phase 11: Ensemble Analysis
→ Phase 12: Multi-Seed XGB
→ Phase 13: Final Threshold Re-optimization
```

---

## Big Picture Insight Flow

Each phase outputs insights that feed into the next:

```
Phase 1: Find suspicious variables/interactions
    ↓
Phase 2: Find hidden process blocks
    ↓
Phase 3: Learn defect signal with XGBoost
    ↓
Phase 4: Validate EDA hypotheses via SHAP  ← NEXT
    ↓
Phase 5: Create process-state descriptors
    ↓
Phase 6: Protect rare defects (imbalance)
    ↓
Phase 7: Understand probability scale
    ↓
Phase 8: Convert ranking into predictions
    ↓
Phase 9: Verify robustness
    ↓
Phase 10: Measure generalization
    ↓
Phase 11: Test ensembles
    ↓
Phase 12: Reduce variance
    ↓
Phase 13: Maximize leaderboard score
```

---

## How to Extend This Workflow

### To Add Phase 4 (SHAP)
1. Load `03_modeling_insights.pkl` for XGBoost OOF predictions
2. Train final XGBoost model on full training set
3. Calculate SHAP values
4. Create summary plots
5. Save insights to `04_shap_insights.pkl`
6. Generate visualization `04_shap_analysis.png`

### To Add Phase 5 (Feature Engineering)
1. Load `02_correlation_insights.pkl` for feature groups
2. Load `04_shap_insights.pkl` for feature importance (from Phase 4)
3. Create group aggregations
4. Create interaction features
5. Train new model with engineered features
6. Compare to baseline
7. Save top performing features to `05_engineered_features.pkl`

---

## Key Findings to Remember

### Phase 1 Discoveries
- **Top Important**: X35 (p=5.79e-23), X13 (p=3.53e-21), X36, X34, X10
- **Imbalance**: 19.48:1 normal:defect ratio
- **Dimensionality**: 49 features → 20 PCs explain 89.7% variance
- **Separability**: Weak but exploitable signal

### Phase 2 Discoveries
- **Process Blocks**: 27 groups mapped from correlation structure
- **Key Block**: Group 1 has 8 features with 0.91 avg correlation (7 are important!)
- **Redundancy**: 12 pairs with |r| > 0.9 are candidates for aggregation
- **Strategy**: Create group-level features for engineering

### Phase 3 Discoveries
- **Winner**: XGBoost (0.8574 ROC-AUC)
- **Probability**: All predictions skewed toward zero
- **Threshold**: Default 0.5 is wrong → need custom threshold (Phase 8)
- **Signal**: Present and learnable, but weak

---

## Important Notes

### For Reproducibility
- All random_state = 42
- Stratified K-Fold maintains class ratio
- Missing values handled via median imputation
- No scaling for tree models (they don't need it)

### For Next Developer
1. The workflow is designed with insight cascading
2. Each phase's pickle output feeds the next phase
3. Keep the checklist comments in code for tracking
4. Don't skip any phase - they build on each other
5. The "mindset" in WORKFLOW_STATUS.md is important

### For Debugging
- If a phase fails, check the pickle from previous phase loaded correctly
- Verify missing value handling (median imputation)
- Check random_state consistency
- Ensure data shapes match expected (1352 train, 339 test)

---

## Performance Baselines

Keep these in mind for improvement targets:

| Metric | Phase 3 (Baseline) | Target | Phase |
|--------|-------------------|--------|-------|
| ROC-AUC | 0.8574 | 0.87+ | 5-8 |
| PR-AUC | 0.3710 | 0.45+ | 5-8 |
| Recall | 0.2736 | 0.35+ | 8 |
| Precision | 0.5133 | 0.60+ | 8 |

Expected improvements from:
- Phase 5 (Feature Engineering): +2-5% ROC-AUC
- Phase 6 (Imbalance Handling): +1-3% Recall
- Phase 8 (Threshold Optimization): +5-10% PR-AUC

---

## Git & Version Control

### Current Branch
```
Branch: claude/relaxed-dijkstra-dy7xg
Commits: 2
Status: Pushed to remote
```

### To Continue Development
```bash
git checkout claude/relaxed-dijkstra-dy7xg
git pull origin claude/relaxed-dijkstra-dy7xg

# After creating Phase 4
git add 04_phase*.py 04_*.pkl 04_*.png
git commit -m "Phase 4: SHAP Interpretation"
git push origin claude/relaxed-dijkstra-dy7xg
```

---

## Last Updated
2026-05-29 | Phases 1-3 Complete | 23% of Workflow
