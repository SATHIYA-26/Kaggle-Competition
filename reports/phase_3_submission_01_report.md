# Phase 3: First Modeling Milestone & Kaggle Submission #1 Report

## 1. Milestone Objective
Establish our first end-to-end Machine Learning pipeline from raw data to a verified Kaggle submission file. This milestone tests the unregularized, un-engineered baseline signal using Ordinary Least Squares (OLS) Linear Regression.

---

## 2. Train / Validation Strategy
- **Split Configuration**: 80% Training ($N = 1,168$), 20% Validation ($N = 292$), fixed seed `random_state=42`.
- **Leakage Prevention**:
  - The `preprocessor` is fitted strictly on `X_train`.
  - `X_val` is transformed using the learned training parameters.
- **Evaluation Metric**: Root Mean Squared Logarithmic Error (RMSLE). Because our target variable is $y_{\text{log}} = \log(1 + y)$, calculating Root Mean Squared Error (RMSE) on log predictions directly yields the official competition metric:
  $$\text{RMSLE} = \sqrt{\frac{1}{n_{\text{val}}} \sum_{i=1}^{n_{\text{val}}} \left(\hat{y}_{\text{log}, i} - y_{\text{log}, i}\right)^2}$$

---

## 3. Baseline Model: Ordinary Least Squares Linear Regression
- **Algorithm**: Standard Ordinary Least Squares (`sklearn.linear_model.LinearRegression`).
- **Mathematical Form**:
  $$\hat{y}_{\text{log}} = \beta_0 + \sum_{j=1}^{p} \beta_j X_j$$
- **Local Validation Performance**:
  - **Training RMSLE**: `0.1101`
  - **Validation RMSLE**: `0.1418`
- **Analysis of Local Results**:
  - The gap between Training RMSLE ($0.1101$) and Validation RMSLE ($0.1418$) indicates mild overfitting, which is expected because $p = 302$ features are fitted on $N = 1,168$ training rows without coefficient shrinkage.
  - However, $0.1418$ is a remarkably respectable initial baseline for raw tabular housing data, proving that our preprocessing and target log-transformation provide a clean foundation.

---

## 4. End-to-End Prediction & Test Verification
To maximize test prediction quality, the complete pipeline was fitted on the full training set ($N = 1,460$) and evaluated on `test.csv` ($N = 1,459$):

```text
Raw test.csv (1459, 80)
   ↓
Structural None imputation
   ↓
preprocessor.transform() (1459, 302)
   ↓
LinearRegression.predict() [Log Space]
   ↓
np.expm1() [Inverse Transform]
   ↓
Predicted SalePrice in US Dollars
```

### Verification Checks:
- **File**: `submission_01.csv`
- **Total Rows**: Exactly 1,459 prediction rows + 1 header row (`1460` lines total).
- **Columns**: `Id`, `SalePrice` (matches `sample_submission.csv` exactly).
- **Missing Values**: `0` NaNs or null values.
- **Prediction Bounds**:
  - Minimum predicted price: **\$38,200.50** (strictly positive, no collapse).
  - Maximum predicted price: **\$760,778.89** (realistic upper bound).
  - Mean predicted price: **\$178,554.99** (close to empirical training mean of \$180,921).

---

## 5. Kaggle Submission #1 Card

| Attribute | Details |
|---|---|
| **Submission Number** | S1 |
| **Model** | Ordinary Least Squares Linear Regression |
| **Main Change** | Initial baseline end-to-end pipeline |
| **Local Validation (RMSLE)** | **0.1418** |
| **Submission File** | `submission_01.csv` |
| **What It Tests** | Baseline predictive capacity of preprocessed features without regularization or feature engineering |
| **Kaggle Public Score** | **0.13950** |
| **Kaggle Leaderboard Rank** | **2165 / 3843** (Top ~56%) |

---

## 6. Submission #1 Analysis & Alignment Check
- **Local Validation vs. Kaggle Leaderboard**:
  - Local Validation RMSLE: **0.1418**
  - Kaggle Public RMSLE: **0.13950**
  - $\Delta = -0.0023$ (Kaggle score is slightly stronger than local validation).
- **Why did this occur?**
  1. **Validation Split Integrity**: The local 80/20 split was realistic and conservative. It did not suffer from optimistic leakage.
  2. **Sample Size Advantage**: While local validation was trained on 80% of data (1,168 rows), the final submission pipeline was fitted on 100% of data (1,460 rows). Extra samples provided additional stability for the 302 parameters.
  3. **Strong Foundation**: Achieving top 56% globally on submission #1 with standard unregularized OLS proves that our structural imputation, one-hot encoding, and target logarithmic transformation provide an exceptionally sound baseline.
- **Next Frontier**:
  - Unregularized OLS suffers from variance inflation due to multicollinear pairs identified in EDA.
  - Adding regularization (Ridge/Lasso) and domain-specific feature engineering in Phase 4 is our next path to push toward the top tiers.
