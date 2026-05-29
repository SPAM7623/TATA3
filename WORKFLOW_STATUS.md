# Alpha Defect Prediction - End-to-End Workflow
**Status**: PHASES 1-3 COMPLETE ✓

---

## WORKFLOW OVERVIEW
This project implements a 13-phase systematic approach to predict Alpha defects in hot rolling mills through Industrial EDA, feature engineering, and threshold optimization.

**Big Picture Insight Flow**:
```
EDA → Find suspicious variables/interactions
   ↓
Correlation Grouping → Find hidden process blocks
   ↓
Baseline XGB → Learn defect signal
   ↓
SHAP → Validate EDA hypotheses
   ↓
Feature Engineering → Create process-state descriptors
   ↓
Imbalance Handling → Protect rare defects
   ↓
Calibration Analysis → Understand probability scale
   ↓
Threshold Optimization → Convert ranking into predictions
   ↓
Stability Analysis → Verify robustness
   ↓
Overfitting Analysis → Measure generalization
   ↓
Multi-Seed Averaging → Reduce variance
   ↓
Final Threshold Optimization → Maximize leaderboard score
```

---

## PHASE 1: INDUSTRIAL EDA ✓ COMPLETE
**File**: `01_phase_industrial_eda.py`  
**Output**: `01_eda_insights.pkl`, `01_eda_visualizations.png`

### Checklist Status:
- ✓ Dataset shape: 1352 × 51 (49 features + CoilID + Y)
- ✓ Missing values: 12 columns with missing values (X15: 11.83%, others <3%)
- ✓ Duplicates: None found
- ✓ Class imbalance: 95.12% normal / 4.88% defect (19.48:1 ratio)
- ✓ Univariate analysis: 39 variables significantly different (p<0.05)
- ✓ Defect vs normal comparison: COMPLETED
- ✓ Variance/instability analysis: COMPLETED
- ✓ Outlier analysis: 14 variables with >5% outliers
- ✓ Correlation analysis: 32 pairs with |r| > 0.8
- ✓ Interaction analysis: COMPLETED
- ✓ PCA/UMAP/t-SNE: Dimensionality reduction complete

### Key Findings:
1. **Class Imbalance**: 19.48:1 (normal:defect) → Must use scale_pos_weight = 19.48
2. **Significant Variables**: 39 variables differ significantly between classes
3. **Top Important Variables**: X35, X13, X36, X34, X10
4. **Dimensionality**: First 20 PCs explain 89.7% of variance (redundancy present)
5. **Separability**: Classes are partially separable but signal is weak

### Insights Generated:
- Candidate important variables: 39 features
- Candidate instability indicators: 0 (high variance in defect class)
- High-correlation variable pairs: 32 pairs (|r| > 0.8)
- Operating regimes detected through quantile analysis

---

## PHASE 2: CORRELATION GROUPING ✓ COMPLETE
**File**: `02_phase_correlation_grouping.py`  
**Output**: `02_correlation_insights.pkl`, `02_correlation_grouping.png`

### Checklist Status:
- ✓ Correlation heatmap: Full and clustered visualizations created
- ✓ Hierarchical clustering: Ward linkage with distance threshold 0.5
- ✓ Feature grouping: 27 feature groups identified

### Key Findings:
1. **Feature Groups**: 27 distinct process blocks identified
   - Largest group: 8 features
   - Smallest group: 1 feature
   
2. **Process Block Distribution**:
   - Important variables distributed across 19 groups
   - Groups with important vars represent key process stages
   - Supporting groups are secondary/auxiliary processes

3. **Redundancy**: 12 highly redundant pairs (|r| > 0.9)
   - Examples: X10-X13 (0.956), X30-X31 (0.968), X32-X33 (0.948)
   - Candidates for removal/aggregation

4. **Correlation Structure**:
   - Main blocks: Temperature parameters, mechanical properties, process indicators
   - Strong within-group coherence suggests real process blocks

### Insights Generated:
- Feature group assignments for 49 variables
- Group cohesion metrics (average within-group correlations)
- Redundant feature pairs
- Process block interpretation guide

---

## PHASE 3: BASELINE MODELING ✓ COMPLETE
**File**: `03_phase_baseline_modeling.py`  
**Output**: `03_modeling_insights.pkl`, `03_baseline_modeling.png`

### Checklist Status:
- ✓ XGBoost: ROC-AUC = 0.8574 ± 0.0535
- ✓ LightGBM: ROC-AUC = 0.8452 ± 0.0586
- ✓ Random Forest: ROC-AUC = 0.8493 ± 0.0662
- ✓ Cross-validation: 5-Fold Stratified
- ✓ Metrics: ROC-AUC, PR-AUC, Recall, Precision, F1, MCC

### Model Comparison:
```
Metric          XGBoost      LightGBM     RandomForest
─────────────────────────────────────────────────────
ROC-AUC         ★0.8574      0.8452       0.8493
PR-AUC          0.3710       ★0.4345      0.3827
Recall          ★0.2736      0.2132       0.1209
Precision       0.5133       ★0.8000      0.3905
F1              ★0.3546      0.3309       0.1800
MCC             0.3512       ★0.3946      0.1970
```

### Model Selection:
**WINNER: XGBoost** ✓
- Highest ROC-AUC (0.8574)
- Best recall (0.2736)
- Best F1 (0.3546)
- Optimal balance for defect detection

### Key Observations:
1. **Probability Distribution**: Heavily skewed toward 0 (defects are rare)
2. **Threshold Issue**: Default 0.5 threshold is inappropriate
3. **Signal Quality**: Clear learnable signal present but weak
4. **Variance**: Moderate variance across folds (expected with small defect set)

### Insights Generated:
- XGBoost OOF predictions (for SHAP and calibration)
- Model performance baselines
- Performance comparison data
- Probability distribution analysis

---

## CURRENT STATUS: 3 of 13 Phases Complete

### ✓ Completed Phases:
1. Industrial EDA - Data understanding and feature analysis
2. Correlation Grouping - Process block identification
3. Baseline Modeling - Model selection (XGBoost)

### 🔄 Upcoming Phases:
4. **SHAP Interpretation** - Validate EDA findings with model explanations
5. **Feature Engineering** - Create process-state descriptors
6. **Imbalance Handling** - Ensure robust handling of rare defects
7. **Calibration & Probability Analysis** - Understand prediction confidence
8. **Threshold Optimization** - Find optimal decision threshold
9. **Stability Analysis** - Verify robustness and generalization
10. **Overfitting Analysis** - Measure train vs test gap
11. **Ensemble Analysis** - Test model combinations
12. **Multi-Seed XGB** - Reduce prediction variance
13. **Final Threshold Re-optimization** - Maximize leaderboard score

---

## DATA SUMMARY
- **Train**: 1352 samples, 49 features, 66 defects (4.88%)
- **Test**: 339 samples, 49 features (no target)
- **Missing Values**: Present in 12 columns (handled via median imputation)
- **Feature Types**: All continuous numerical
- **Class Imbalance**: 19.48:1 (highly imbalanced)

---

## TECHNICAL SETUP
- **Python Version**: 3.11
- **Key Libraries**: pandas, numpy, scikit-learn, xgboost, lightgbm, matplotlib, seaborn, umap, scipy
- **Cross-Validation**: 5-Fold Stratified K-Fold
- **Random Seeds**: 42 (reproducible)

---

## FILES GENERATED
1. **Data Files**:
   - `train.csv` - Training data
   - `test.csv` - Test data

2. **Phase 1 Outputs**:
   - `01_eda_insights.pkl` - EDA findings (candidate variables, instability metrics)
   - `01_eda_visualizations.png` - PCA, t-SNE, UMAP, scree plot, class distribution

3. **Phase 2 Outputs**:
   - `02_correlation_insights.pkl` - Feature groups, redundant pairs, statistics
   - `02_correlation_grouping.png` - Correlation heatmap, dendrogram, group sizes

4. **Phase 3 Outputs**:
   - `03_modeling_insights.pkl` - OOF predictions, CV scores, model comparison
   - `03_baseline_modeling.png` - ROC curves, PR curves, probability distributions

---

## NEXT IMMEDIATE STEPS
1. **Phase 4 (SHAP)**: Explain XGBoost predictions to validate EDA hypotheses
2. **Phase 5 (Feature Engineering)**: Create interaction/aggregation features based on correlation groups
3. **Phase 6 (Imbalance Handling)**: Implement SMOTE or weighted training
4. **Phase 7 (Calibration)**: Analyze probability calibration and adjust thresholds
5. **Phase 8 (Threshold Optimization)**: Find optimal decision boundary for leaderboard

---

## KEY MINDSET REMINDER
**Do NOT think**: "Which X variable predicts Y?"  
**Think**: "Which hidden thermo-mechanical process states, instability patterns, and parameter interactions create metallurgical conditions that lead to Alpha defect formation?"

This workflow focuses on **process understanding** before optimization.

---

Generated: 2026-05-29  
Workflow Version: 1.0
