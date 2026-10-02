# Phase 5: Cross-Validation & Systematic Hyperparameter Optimization Report

## 1. Overview & Objectives
Phase 4 revealed a critical vulnerability in our modeling methodology:
1. **Single-Split Evaluation Blind Spot**: A single 80/20 train/validation split failed to expose linear model extrapolation risks on rare out-of-distribution mansions.
2. **Linear Slope Runaway**: Adding unbounded composite features (`TotalSF`) caused linear models to predict \$1.47M for test house `Id 2550`, inflating competition RMSLE.

**Phase 5 Objectives**:
- Transition to **5-Fold Cross-Validation (`KFold`)** across all 1,460 observations.
- Add non-linear log transformations to skewed area features (`LogTotalSF`, `LogGrLivArea`, `LogLotArea`) to model diminishing marginal returns.
- Systematically optimize hyperparameters for regularized linear models (`RidgeCV`, `LassoCV`, `ElasticNetCV`) and boosted trees (`GradientBoosting`, `XGBoost`).
- Establish our new champion model and produce **Kaggle Submission #3**.

---

## 2. 5-Fold Cross-Validation Architecture
- **Configuration**: 5 Folds, shuffled, `random_state=42`.
- **Why Cross-Validation is Mandatory**:
  - Rather than relying on a single static 20% holdout, each of the 5 folds acts as an out-of-sample validation set once.
  - The mean cross-validation score ($\text{CV}_{\text{mean}}$) and standard deviation ($\sigma_{\text{CV}}$) provide an unbiased estimate of generalization error across all subsets of the training distribution.
  - Guarantees that outliers like `Id 524` and `Id 1299` are tested in out-of-fold validation.

---

## 3. Systematic Hyperparameter Tuning Results

Every model was evaluated using 5-Fold Cross-Validation on log prices:

| Model Family | Tuning Method & Best Hyperparameters | 5-Fold CV RMSLE | Overfitting / Bias-Variance Diagnosis |
|---|---|:---:|---|
| **Ridge Regression** | `RidgeCV` across 50 $\alpha \in [10^0, 10^3]$<br>$\alpha^* = 323.75$ | `0.1373` | Stronger regularization needed under CV; high stability, low variance. |
| **Lasso Regression** | `LassoCV` across 50 $\alpha \in [10^{-4}, 10^{-1}]$<br>$\alpha^* = 0.00339$ | `0.1333` | Pruned **201 / 312 features**; sparse embedded feature selection. |
| **ElasticNet** | `ElasticNetCV`<br>$\alpha^* = 0.04894, \ell_1 = 0.10$ | `0.1329` | Heavily weights $\ell_2$ grouping while retaining moderate sparsity. |
| **Gradient Boosting** | `GradientBoostingRegressor`<br>$n=300, \text{lr}=0.05, \text{depth}=3$ | `0.1325` | Trees capture non-linear interactions without manual cross-products. |
| **Tuned XGBoost** | `XGBRegressor`<br>$\text{depth}=3, \text{lr}=0.03, n=500$<br>$\text{subsample}=0.8, \text{colsample}=0.7$<br>$\alpha_{\text{reg}}=0.1, \lambda_{\text{reg}}=1.0$ | **`0.1261`** | **Best overall model!** Shallow trees + slow shrinkage + stochastic column subsampling prevent overfitting. |

---

## 4. Resolving the Outlier Extrapolation Pathology (`Id 2550`)

To verify whether our changes resolved the test-set runaway prediction from Phase 4, we inspected the prediction for the 10,190 sq ft mansion (`Id 2550`):

| Model / Phase | Predicted Price for Id 2550 | Behavior Analysis |
|---|:---:|---|
| **S1: OLS Baseline** | \$760,778.89 | Linear combination of raw features; acceptable but unregularized. |
| **S2: Lasso + Linear TotalSF** | **\$1,468,264.55** | **Runaway linear extrapolation** (inflicted $+0.0025$ penalty on Kaggle RMSLE). |
| **Phase 5: Lasso (with Log features)** | \$881,097.72 | Log transformation tapered the linear slope, pulling price down by \$587,000. |
| **Phase 5: Tuned XGBoost** | **\$270,664.30** | **Tree leaf bounding**: tree leaves group Id 2550 with the highest empirical training sales without unbounded mathematical extrapolation. |

---

## 5. Kaggle Submission #3 Card

| Attribute | Details |
|---|---|
| **Submission Number** | **S3** |
| **Model** | **Tuned XGBoost Regressor** |
| **Main Change** | 5-Fold CV hyperparameter tuning + gradient boosted decision trees with stochastic subsampling and non-linear log area features |
| **Local 5-Fold CV Score (RMSLE)** | **`0.1261`** (Consistent across all 5 folds) |
| **Submission File** | `submission_03.csv` |
| **What It Tests** | Non-linear tree boosting to eliminate linear extrapolation errors and capture complex tabular interactions |
| **Kaggle Public Score** | **0.12788** |
| **Kaggle Leaderboard Rank** | **1280** (Top ~33%, jumped **+885 positions** from 2165!) |

---

## 6. Submission #3 Analysis: Breakthrough Validation Alignment
- **Local 5-Fold CV vs. Kaggle Public Leaderboard**:
  - Local 5-Fold CV RMSLE: **`0.1261`**
  - Kaggle Public RMSLE: **`0.12788`**
  - Alignment gap: **$\Delta = +0.0017$** (Extremely tight local-to-public tracking).
- **Key Takeaways**:
  1. **Cross-Validation is Verified as our True Compass**:
     - Unlike the single train/validation split in Phase 4 that misled us, our 5-fold CV score of `0.1261` almost perfectly mirrored the actual test set score of `0.12788`.
  2. **Tree Bounding Fixed Outlier Extrapolation**:
     - Constraining predictions on extreme houses (like `Id 2550` to \$270k instead of \$1.47M) eliminated the single-row penalty that hurt S2.
  3. **Major Competitive Jump**:
     - Our score improved from `0.13950` down to `0.12788`, rocketing us up by **885 leaderboard ranks into the top 33% globally**.
  4. **Next Opportunity**:
     - In Phase 6, combining the distinct inductive biases of our best linear models (Lasso) and tree models (XGBoost) through **Stacking and Blending** will allow us to push even further into the top tiers.
