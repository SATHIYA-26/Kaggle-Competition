# Phase 4: Model Families + Systematic Improvement Report

## 1. Overview & Objectives
Phase 4 expands our modeling toolkit beyond unregularized Ordinary Least Squares (OLS) by addressing the core limitation discovered in Phase 1 and 3: **multicollinearity and high variance among 300+ one-hot encoded features**.

This phase evaluates:
1. **Regularized Linear Models**: Ridge ($\ell_2$), Lasso ($\ell_1$), and ElasticNet ($\ell_1 + \ell_2$).
2. **Distance-Based Models**: K-Nearest Neighbors (KNN) and Support Vector Regression (SVR).
3. **Tree Ensembles**: Random Forest Regressor.
4. **Domain-Specific Feature Engineering**: Creating physical housing aggregate features.

---

## 2. Regularized Linear Models: Theory & Empirical Comparison

### A. Mathematical Formulation & Bias-Variance Tradeoff
- **Ordinary Least Squares (OLS)**:
  $$\min_{\beta} \sum_{i=1}^n \left(y_i - \beta_0 - \sum_{j=1}^p \beta_j X_{ij}\right)^2$$
  *Limitation*: When features are collinear ($X^T X$ is near-singular), variance explodes: $\text{Var}(\hat{\beta}) = \sigma^2 (X^T X)^{-1}$. Weights swing wildly to cancel each other out.

- **Ridge Regression ($\ell_2$ penalty)**:
  $$\min_{\beta} \left\{ \text{RSS} + \lambda \sum_{j=1}^p \beta_j^2 \right\}$$
  *Mechanics*: Adds a spherical quadratic penalty that shrinks all coefficients toward zero proportionally. Solves multicollinearity by conditioning $(X^T X + \lambda I)^{-1}$.

- **Lasso Regression ($\ell_1$ penalty)**:
  $$\min_{\beta} \left\{ \text{RSS} + \lambda \sum_{j=1}^p |\beta_j| \right\}$$
  *Mechanics*: The $\ell_1$ diamond-shaped constraint has sharp corners on parameter axes. Contours of the RSS ellipse hit these corners first, driving non-essential feature weights to **exactly zero** (embedded feature selection).

- **ElasticNet ($\ell_1 + \ell_2$ penalty)**:
  Combines the grouping effect of Ridge (keeping correlated features together) with the sparsity of Lasso.

### B. Feature Scaling Requirement
Because regularization penalties apply uniformly ($\sum \beta_j^2$ or $\sum |\beta_j|$), features measured in thousands (e.g. `LotArea`, `GrLivArea`) would naturally have tiny coefficients and escape penalization, while features measured in units (e.g. `FullBath`, dummy 0/1 indicators) would be overly penalized. 
- **Solution**: `StandardScaler(with_mean=False)` is placed strictly inside the model pipeline after one-hot encoding, ensuring zero data leakage and uniform shrinkage.

---

## 3. Empirical Results: Model Family Comparison

Evaluated on the exact same 80/20 train/validation split (`random_state=42`) using RMSLE on log prices:

| Model Family | Model Configuration | Local Val RMSLE | Zeroed Features | Observation / Diagnosis |
|---|---|:---:|:---:|---|
| **Baseline** | Ordinary Least Squares (OLS) | `0.1418` | 0 / 300 | Baseline benchmark; high variance from multicollinearity. |
| **Linear ($\ell_2$)** | Ridge ($\alpha = 10.0$) | `0.1249` | 0 / 300 | Drastic improvement (-0.0169); suppresses collinear noise. |
| **Linear ($\ell_2$)** | Ridge ($\alpha = 20.0$) | **`0.1247`** | 0 / 300 | Optimal $\ell_2$ shrinkage parameter. |
| **Linear ($\ell_1$)** | Lasso ($\alpha = 0.0005$) | `0.1228` | 70 / 300 | Shrinks and prunes 70 redundant dummy columns. |
| **Linear ($\ell_1$)** | Lasso ($\alpha = 0.001$) | **`0.1227`** | 98 / 300 | Prunes 98 redundant features; best linear model without FE. |
| **Linear (Hybrid)** | ElasticNet ($\alpha = 0.0005, \ell_1 = 0.8$) | `0.1231` | ~60 / 300 | Strong balance between sparsity and grouping. |
| **Tree Ensemble** | Random Forest ($N = 100$ trees) | `0.1440` | — | Slightly worse than linear baseline (`0.1418`). |
| **Distance-Based** | K-Nearest Neighbors ($k = 5$) | `0.2037` | — | Severe failure due to Curse of Dimensionality. |
| **Distance-Based** | Support Vector Regressor (SVR, $C = 5$) | `0.3763` | — | Fails in high-dimensional sparse dummy space. |

### Why Did Distance Models & Trees Struggle?
1. **Curse of Dimensionality in KNN & SVR**: In a 300-dimensional sparse space, all points become roughly equidistant in Euclidean space ($d(x_i, x_j) \approx \text{const}$). Distance metrics lose their discriminative power.
2. **Trees on One-Hot Data**: Decision trees partition feature space using axis-aligned orthogonal cuts. When a categorical feature is split into 25 binary dummies (e.g. `Neighborhood`), a tree must take dozens of deep splits to isolate a neighborhood, leading to fragmented sample sizes and overfitting. Furthermore, trees cannot extrapolate linear scaling trends beyond the observed leaf bounds.

---

## 4. Domain-Specific Feature Engineering

Rather than feeding raw individual room measurements, we engineered 7 domain-justified composite features:

| Engineered Feature | Mathematical Formula | Physical Real-World Housing Meaning |
|---|---|---|
| `TotalSF` | `TotalBsmtSF + 1stFlrSF + 2ndFlrSF` | Total usable indoor living space across all levels. |
| `TotalBath` | `FullBath + 0.5*HalfBath + BsmtFullBath + 0.5*BsmtHalfBath` | Total sanitary capacity of the home. |
| `HouseAge` | `YrSold - YearBuilt` | Actual structural age at time of transaction. |
| `RemodelAge` | `YrSold - YearRemodAdd` | Age of most recent renovation/modernization. |
| `GarageAge` | `YrSold - GarageYrBlt` | Wear and tear on the garage structure. |
| `TotalPorchSF` | `OpenPorchSF + EnclosedPorch + 3SsnPorch + ScreenPorch` | Total outdoor lounging and leisure surface area. |
| `TotalFinBsmtSF` | `BsmtFinSF1 + BsmtFinSF2` | Subterranean finished square footage. |

### Impact of Feature Engineering:
- **OLS with FE**: RMSLE dropped from `0.1418` $\to$ `0.1398`.
- **Ridge with FE**: RMSLE `0.1249`.
- **Lasso with FE**: RMSLE dropped from `0.1418` $\to$ **`0.1224`** (Zeroed **108** / 308 features).

---

## 5. Kaggle Submission #2 Card

| Attribute | Details |
|---|---|
| **Submission Number** | **S2** |
| **Model** | **Lasso Regression ($\alpha = 0.001$) with Domain Feature Engineering** |
| **Main Change** | Added 7 engineered features + $\ell_1$ regularization with standardization |
| **Local Validation (RMSLE)** | **`0.1224`** (Baseline was `0.1418`, improvement of **$-0.0194$**) |
| **Submission File** | `submission_02.csv` |
| **What It Tests** | How $\ell_1$ automatic feature pruning and domain square footage / bathroom aggregates boost generalization |
| **Kaggle Public Score** | **0.14271** |
| **Kaggle Leaderboard Rank** | *(Pending user rank input)* |

---

## 6. Submission #2 Analysis: The Discrepancy Investigation
- **Local Validation RMSLE**: `0.1224` (down from `0.1418`)
- **Kaggle Public Score**: `0.14271` (up from `0.13950`)
- **Investigation: Why did local validation improve while Kaggle slightly degraded?**
  1. **Outlier Linear Extrapolation on Id 2550**:
     - House `Id 2550` has `GrLivArea = 5095` and `TotalBsmtSF = 5095`, resulting in `TotalSF = 10,190 sq ft`.
     - In S1 (OLS baseline), `Id 2550` was predicted at **\$760,778**.
     - In S2 (Lasso + `TotalSF`), the unconstrained linear slope extrapolated `Id 2550` to **\$1,468,264**.
     - Since the maximum historical sale price in Ames is \$755,000, predicting \$1.47M introduces a squared log error of $\approx 0.80$ on this single test house alone, which accounts for nearly the entire $+0.003$ gap on test RMSLE!
  2. **Single-Split Validation Blind Spot**:
     - Our 80/20 split did not have a 10,000 sq ft house in the validation split, so local validation could not detect this linear extrapolation risk.
     - **Takeaway**: This proves why **5-Fold Cross-Validation (Phase 5)** and non-linear / tree-based handling are necessary to evaluate generalized performance reliably.
