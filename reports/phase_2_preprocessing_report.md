# Phase 2: Reliable Data Pipeline & Preprocessing Report

## 1. Overview
The goal of Phase 2 was to construct a robust, leakage-free, reproducible preprocessing pipeline that transforms raw tabular data into machine-learning-ready matrices without peeking at validation or test distributions.

---

## 2. Pipeline Design & Components

### A. Separation of Features and Identifiers
- `Id` was removed from both training features `X` and test features `test_X`. `Id` is an arbitrary tracking number; including it allows models to memorize index artifacts rather than genuine housing patterns.
- `SalePrice` was separated as the continuous regression target $y$.
- Target variable transformed into log-space: $y_{\text{log}} = \log(1 + y)$ (`np.log1p`), achieving near-normal symmetry (skewness = 0.1213).

### B. Structural Amenity Imputation
Before pipeline estimation, 15 categorical features where missingness represents structural absence (no garage, no pool, no fireplace, no basement) were imputed with the string `"None"`:
- `PoolQC`, `MiscFeature`, `Alley`, `Fence`, `MasVnrType`, `FireplaceQu`, `GarageType`, `GarageFinish`, `GarageQual`, `GarageCond`, `BsmtQual`, `BsmtCond`, `BsmtExposure`, `BsmtFinType1`, `BsmtFinType2`.

### C. Scikit-Learn `ColumnTransformer` Architecture
The preprocessing core is built using Scikit-Learn's `ColumnTransformer`, ensuring strict encapsulation:

```python
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median'))
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numeric_cols),
    ('cat', categorical_transformer, categorical_cols)
])
```

- **Numerical Sub-Pipeline**: 36 features. Uses `SimpleImputer(strategy='median')` because median is robust to outliers and skewed distributions.
- **Categorical Sub-Pipeline**: 43 features. Uses `SimpleImputer(strategy='most_frequent')` for unhandled missing values, followed by `OneHotEncoder(handle_unknown='ignore')`.
- **Unknown Category Protection**: `handle_unknown='ignore'` handles unseen categories in the test set by encoding them as all-zero rows across that category's one-hot dummy columns, preventing runtime crashes.

---

## 3. Important Decisions

1. **Feature Scaling Placement**:
   - `StandardScaler` was deliberately **omitted** from the base preprocessor.
   - *Rationale*: Tree-based models (Random Forest, Gradient Boosting, XGBoost) split on rank order and are scaling-invariant. Densifying sparse one-hot matrices with standard scaling slows down tree models. Scaling is attached to model-specific pipelines (e.g., Ridge/Lasso) where required.
2. **Outlier Preservation**:
   - Extreme points (e.g., large houses with atypical pricing) were retained for the initial baseline. Filtering outliers before establishing a clean baseline obscures whether future gains come from model architecture or sample selection.
3. **Data Leakage Guarantee**:
   - `preprocessor.fit()` is called **only on the training split**. The fitted transformers are then used to `.transform()` the validation and test splits.

---

## 4. Output Dimensions & Verification
- **Training Matrix (`X_processed`)**: `(1460, 302)` | Residual missing values: `0`
- **Test Matrix (`test_X_processed`)**: `(1459, 302)` | Residual missing values: `0`
- **Target Vector (`y_log`)**: `(1460,)` | Skewness: `0.1213`
