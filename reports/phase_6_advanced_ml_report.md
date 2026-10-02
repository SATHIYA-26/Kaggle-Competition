# Phase 6: Ensembles, PCA & Advanced ML Report

## 1. Overview & Objectives
Phase 6 bridges the advanced theoretical concepts from Machine Learning to optimize our competitive Kaggle pipeline:
1. **PCA as a Controlled Hypothesis-Driven Experiment**: Evaluate whether dimensionality reduction improves or harms high-dimensional one-hot encoded housing data.
2. **Error Orthogonality & Diversity**: Measure error correlation across fundamentally different algorithm families (Linear Models vs. Gradient Boosted Trees).
3. **Out-of-Fold (OOF) Blending & Stacking**: Combine the distinct inductive biases of our best models into an ensemble that cancels individual model errors.
4. **Produce Kaggle Submission #4**.

---

## 2. The PCA Controlled Experiment: Theory vs. Evidence

### A. The Hypothesis
In classical ML, Principal Component Analysis (PCA) is often used to compress high-dimensional feature spaces ($p = 312$), eliminate multicollinearity, and reduce overfitting by projecting data onto orthogonal axes of maximum variance:
$$\max_{w} \text{Var}(X w) \quad \text{subject to } \|w\|_2 = 1$$

### B. Empirical Results (5-Fold CV on Log Prices)
Standardized features were evaluated with varying numbers of principal components versus the uncompressed feature space:

| Configuration | Dimensions ($k$) | Variance Explained | 5-Fold CV RMSLE | Outcome vs. No PCA |
|---|:---:|:---:|:---:|:---:|
| **Baseline (No PCA)** | **312 features** | **100.0%** | **`0.1373`** | **Optimal** |
| PCA | 150 components | 89.3% | `0.1611` | $+0.0238$ (Degraded) |
| PCA | 100 components | 75.4% | `0.1632` | $+0.0259$ (Degraded) |
| PCA | 50 components | 54.2% | `0.1573` | $+0.0200$ (Degraded) |
| PCA | 30 components | 42.2% | `0.1538` | $+0.0165$ (Degraded) |

### C. Why Did PCA Degrade Performance on One-Hot Encoded Data?
1. **Variance Does Not Equal Predictive Power**:
   - In sparse categorical data, rare luxury features (such as `Neighborhood_StoneBr` or `RoofMatl_WdShngl`) appear in only 1-2% of records.
   - Their sample variance is tiny: $\sigma^2 = p(1 - p) \approx 0.01 \times 0.99 \approx 0.0099$.
   - Because PCA is **unsupervised**, it treats these low-variance directions as "noise" and discards them in early components.
   - However, in housing economics, luxury neighborhoods and rare roof materials have **massive price signals**. Discarding low-variance principal components discards our highest-value features!
2. **Destruction of Category Semantics & Sparsity**:
   - One-hot binary columns are discrete indicator coordinates ($x \in \{0, 1\}$). Linear combinations of binary variables yield dense, uninterpretable real numbers that confuse downstream linear shrinkage.
3. **Engineering Decision**: **PCA is definitively rejected** from our production pipeline.

---

## 3. Multi-Model Ensembling & Error Orthogonality

### A. Why Diverse Ensembles Work
If two models make identical predictions, averaging them provides zero benefit. However, when models have **different inductive biases**, their errors are imperfectly correlated:
$$\text{Var}\left(w_1 e_1 + w_2 e_2\right) = w_1^2 \sigma_1^2 + w_2^2 \sigma_2^2 + 2 w_1 w_2 \text{Cov}(e_1, e_2) < \min(\sigma_1^2, \sigma_2^2)$$

- **Lasso / Ridge**: Fit a smooth global linear hyperplane. Excellent at capturing global linear scaling across hundreds of sparse one-hot dummy features, but prone to bias on complex non-linear feature interactions.
- **XGBoost / Gradient Boosting**: Partition feature space using axis-aligned orthogonal cuts. Excellent at capturing non-linear interactions (e.g. `OverallQual * TotalSF`), but prone to variance on smooth global trends.

### B. Empirical Error Correlation Check
We calculated the residual errors ($e = y_{\text{true}} - \hat{y}$) across all 1,460 out-of-fold predictions:
$$\text{Corr}(e_{\text{XGBoost}}, e_{\text{Lasso}}) = \mathbf{0.8445}$$
Because error correlation is **0.8445** (substantially below 1.0), **over 15.5% of their error variance is independent**, meaning their individual mistakes cancel out when blended.

---

## 4. Out-of-Fold (OOF) Blending Architecture & Results

Using 5-Fold Cross-Validation, models were trained on 4 folds and used to predict on the 5th holdout fold to generate true out-of-fold predictions:

| Model Family | Model Description | 5-Fold OOF RMSLE | Ensemble Role |
|---|---|:---:|---|
| **Tree 1** | Tuned XGBoost (`depth=3, lr=0.03, n=500, colsample=0.7`) | `0.1276` | Primary non-linear interaction engine |
| **Tree 2** | Tuned Gradient Boosting (`depth=3, lr=0.05, n=300`) | `0.1339` | Tree diversity & variance smoother |
| **Linear 1** | Tuned Lasso (`alpha=0.003, max_iter=10000`) | `0.1380` | Sparse global linear anchor (pruned 201 features) |
| **Linear 2** | Tuned Ridge (`alpha=300.0`) | `0.1415` | $\ell_2$ smooth regularizer |
| **Ensemble** | **Weighted Blend (65% XGB + 10% GBR + 20% Lasso + 5% Ridge)** | **`0.1265`** | **Optimal multi-model balance** |

### Test Outlier Check (`Id 2550`):
- **S1 (OLS)**: \$760,778
- **S2 (Lasso Linear)**: \$1,468,264 (Extrapolation blowup)
- **S3 (XGBoost Alone)**: \$270,664 (Aggressive tree clamp)
- **S4 (Ensemble Blend)**: **\$408,016.66** (Balanced, grounded luxury price)

---

## 5. Kaggle Submission #4 Card

| Attribute | Details |
|---|---|
| **Submission Number** | **S4** |
| **Model** | **Multi-Model Out-Of-Fold Diverse Ensemble (65% XGBoost + 10% GBR + 20% Lasso + 5% Ridge)** |
| **Main Change** | Multi-model ensembling combining non-linear boosted trees with sparse regularized linear models + empirical rejection of PCA |
| **Local 5-Fold OOF Score** | **`0.1265`** (OOF RMSLE across all 1,460 samples) |
| **Submission File** | `submission_04.csv` |
| **What It Tests** | Error cancellation between complementary linear and tree-based model families |
| **Kaggle Public Score** | *(Pending submission)* |
| **Kaggle Leaderboard Rank** | *(Pending submission)* |
