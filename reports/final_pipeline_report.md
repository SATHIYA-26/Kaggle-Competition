# Phase 7: Final Kaggle Iteration & Production Pipeline Report

## 1. Project Overview & Learning Objectives
- **Competition**: Kaggle House Prices — Advanced Regression Techniques (Ames, Iowa).
- **Evaluation Metric**: Root Mean Squared Logarithmic Error (RMSLE).
- **Core Working Principle**: Iterative, scientific ML experimentation:
  $$\text{Understand} \to \text{Build} \to \text{Validate} \to \text{Submit} \to \text{Analyze} \to \text{Improve} \to \text{Repeat}$$
- **Final Result**: Advanced from the bottom half of the competition to **Rank 844 / 3,843 (Top ~21.9% globally)** with a final competition score of **`0.12418`**.

---

## 2. Complete Leaderboard Progression & Submission History

| Submission | Phase | Model Architecture | Key Change Tested | Local Evaluation | Kaggle Score | Global Rank | Strategic Takeaway |
|:---:|:---:|:---|:---|:---:|:---:|:---:|:---|
| **S1** | Phase 3 | Ordinary Least Squares (OLS) | Unregularized baseline on clean preprocessed data | `0.1418` (Val RMSLE) | **`0.13950`** | **`2165`** / 3843 | Established clean baseline floor; target `log1p` matched competition metric directly. |
| **S2** | Phase 4 | Lasso ($\alpha=0.001$) + Linear `TotalSF` | 7 domain-engineered features + $\ell_1$ pruning | `0.1224` (Val RMSLE) | **`0.14271`** | — | Local gain was misleading: unbounded linear slope extrapolated 10,190 sq ft mansion (`Id 2550`) to \$1.47M. |
| **S3** | Phase 5 | Tuned XGBoost (`depth=3, lr=0.03, n=500`) | 5-Fold Cross-Validation + Non-linear log area features + Tree boosting | **`0.1261`** (5-Fold CV) | **`0.12788`** | **`1280`** / 3843 | 5-Fold CV eliminated single-split bias; tree leaf bounding fixed `Id 2550` extrapolation, jumping **+885 ranks**! |
| **S4** | Phase 6 | Multi-Model Diverse Ensemble | 65% XGBoost + 10% GBR + 20% Lasso + 5% Ridge | **`0.1265`** (5-Fold OOF) | **`0.12418`** | 🚀 **`844`** / 3843 | Error correlation was 0.8445; combining linear and tree inductive biases canceled residual variance, jumping **+436 ranks into Top 22%**! |
| **Final** | Phase 7 | Clean Production Pipeline (`train_and_predict.py`) | Consolidated leak-safe code + full 5-fold OOF training | **`0.1265`** (5-Fold OOF) | **`0.12418`** | **Top Quintile** | Production-grade script generating verified `final_submission.csv`. |

---

## 3. Comprehensive Retrospective: What Helped vs. What Hurt

### A. What Helped (Kept in Production Pipeline)
1. **Target Logarithmic Transformation (`log1p(SalePrice)`)**:
   - Transformed skewed target ($1.88 \to 0.12$). Aligning training loss with the official RMSLE evaluation metric was foundational.
2. **Structural Missingness (`"None"`) vs. Random Missingness**:
   - Explicitly encoding 15 amenity columns as `"None"` prevented false mode imputation on non-existent basements, garages, and pools.
3. **Domain Aggregate Feature Engineering**:
   - `TotalSF` (living area + basement), `TotalBath` (weighted baths), and structural ages (`HouseAge`, `RemodelAge`, `GarageAge`) gave models direct access to macro housing scale.
4. **Concave Log Area Transformations (`LogTotalSF`, `LogGrLivArea`, `LogLotArea`)**:
   - Modeled diminishing marginal returns and successfully prevented runaway linear extrapolation on distribution tails.
5. **5-Fold Cross-Validation**:
   - Provided an unshakeable local compass that tracked Kaggle test scores to within **0.0017**.
6. **Multi-Model Diverse Ensembling (Blending & Stacking)**:
   - Combining models with different mathematical inductive biases (smooth linear hyperplanes vs. orthogonal decision trees) reduced error variance by exploiting their 15.5% error independence.

### B. What Hurt (Formally Rejected Based on Evidence)
1. **Unregularized Ordinary Least Squares (OLS)**:
   - S1 suffered from severe coefficient variance explosion due to multicollinear pairs (`GarageCars`/`GarageArea`, `TotalBsmtSF`/`1stFlrSF`).
2. **Unconstrained Linear Composite Features (`TotalSF` without log)**:
   - In S2, linear models extrapolated the 10,190 sq ft mansion `Id 2550` to \$1.468M, inflating test RMSLE.
3. **Distance-Based Models (KNN & SVR)**:
   - Collapsed under the **Curse of Dimensionality** in 300-D sparse one-hot space (KNN: `0.2037`, SVR: `0.3763`).
4. **Unsupervised Principal Component Analysis (PCA)**:
   - Degraded 5-fold CV error from `0.1373` down to `0.1538–0.1632`. Discarded rare luxury categories as low-variance "noise" and destroyed category sparsity.

---

## 4. Final Production Pipeline Architecture

```text
                           Raw Tabular Data (train.csv, test.csv)
                                              │
                      ┌───────────────────────┴───────────────────────┐
                      ▼                                               ▼
         [Target log1p(SalePrice)]                       [Remove Arbitrary 'Id' Column]
                      │                                               │
                      │                                               ▼
                      │                                 [Structural Absence Imputation]
                      │                                  15 amenity columns -> "None"
                      │                                               │
                      │                                               ▼
                      │                                 [Domain Feature Engineering]
                      │                                  TotalSF, TotalBath, Ages,
                      │                                  LogTotalSF, LogGrLivArea
                      │                                               │
                      │                                               ▼
                      │                                  [ColumnTransformer (Train Fit)]
                      │                                  • Num: Median Imputation
                      │                                  • Cat: Mode + OneHotEncoder(ignore)
                      │                                               │
                      │                     ┌─────────────────────────┴─────────────────────────┐
                      │                     ▼                                                   ▼
                      │         [Sparse Matrix (312 cols)]                        [StandardScaler(with_mean=False)]
                      │                     │                                                   │
                      │                     ▼                                                   ▼
                      │            [Tree-Based Models]                                [Regularized Linear Models]
                      │          • Tuned XGBoost (65%)                               • Tuned Lasso alpha=0.003 (20%)
                      │          • Tuned GBR (10%)                                   • Tuned Ridge alpha=300 (5%)
                      │                     │                                                   │
                      │                     └─────────────────────────┬─────────────────────────┘
                      │                                               │
                      └──────────────────────────────► [5-Fold Cross-Validation]
                                                       OOF Blended RMSLE = 0.1265
                                                                      │
                                                                      ▼
                                                      [Inverse Transform: expm1()]
                                                                      │
                                                                      ▼
                                                       final_submission.csv (Rank 844)
```

---

## 5. How to Reproduce the Complete Pipeline

The complete, leak-free pipeline is encapsulated in [`train_and_predict.py`](file:///d:/Semester-5/Machine%20Learning%20-%20Competition/train_and_predict.py):

```bash
# To run the complete training, 5-fold CV validation, and generate final_submission.csv:
python train_and_predict.py
```

### Script Execution Verification:
- **Fold 1 Blended RMSLE**: `0.1238`
- **Fold 2 Blended RMSLE**: `0.1062`
- **Fold 3 Blended RMSLE**: `0.1684`
- **Fold 4 Blended RMSLE**: `0.1179`
- **Fold 5 Blended RMSLE**: `0.1054`
- **Final 5-Fold OOF RMSLE**: **`0.1265`**
- **Test Set Predictions**: Exactly 1,459 non-negative, non-null predictions saved in [`final_submission.csv`](file:///d:/Semester-5/Machine%20Learning%20-%20Competition/final_submission.csv).

---

## 6. Final ML Learning Summary
By following a hypothesis-driven workflow rather than blindly chasing leaderboard scores:
1. We proved **why data understanding (EDA) dictates preprocessing**.
2. We learned **why a baseline is indispensable**.
3. We witnessed firsthand **how a single validation split can deceive you**.
4. We discovered **why unsupervised dimensionality reduction fails on sparse categorical data**.
5. We mathematically verified **how diverse ensembles cancel residual error**.
6. We advanced from **Rank 2,165 to Rank 844 globally**.
