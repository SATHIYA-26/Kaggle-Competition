# Phase 1: Dataset Understanding & Exploratory Data Analysis (EDA) Report

## 1. Problem Definition & Objectives
- **Competition**: Kaggle House Prices — Advanced Regression Techniques.
- **Goal**: Predict the final sale price (`SalePrice`) of residential homes in Ames, Iowa.
- **Evaluation Metric**: Root Mean Squared Logarithmic Error (RMSLE) between predicted values and actual sale prices:
  $$\text{RMSLE} = \sqrt{\frac{1}{n} \sum_{i=1}^n \left(\log(1 + y_i) - \log(1 + \hat{y}_i)\right)^2}$$
- **Data Shapes**:
  - Training dataset (`train.csv`): 1,460 observations, 81 columns (80 features + `SalePrice`).
  - Test dataset (`test.csv`): 1,459 observations, 80 columns (features only, `SalePrice` omitted).

---

## 2. Target Variable Analysis (`SalePrice`)
- **Distribution**:
  - Original `SalePrice` exhibits severe positive right-skewness: **Skewness = 1.8829**.
  - Minimum price: \$34,900; Median: \$163,000; Mean: \$180,921; Maximum: \$755,000.
  - The long right tail violates the normality and homoscedasticity assumptions of linear models and over-penalizes prediction errors on expensive properties.
- **Logarithmic Transformation**:
  - Applying $z = \log(1 + \text{SalePrice})$ (`np.log1p`) normalizes the distribution: **Skewness = 0.1213** (nearly symmetric Gaussian).
  - **Critical Alignment**: Minimizing standard Root Mean Squared Error (RMSE) on $z$ mathematically minimizes Kaggle's evaluation metric (**RMSLE**) directly.

---

## 3. Missing Value Investigation & Structural Absence
Missing values in this dataset fall into two fundamentally distinct categories:

### A. Structural Absence (Amenity Does Not Exist)
In 15 categorical features, `NaN` does NOT indicate missing/corrupted data; according to `data_description.txt`, it explicitly indicates the absence of that amenity:
- `PoolQC` (99.5% NaN): No swimming pool.
- `MiscFeature` (96.3% NaN): No miscellaneous feature (shed, elevator, etc.).
- `Alley` (93.8% NaN): No alley access.
- `Fence` (80.8% NaN): No fence.
- `FireplaceQu` (47.3% NaN): No fireplace.
- `GarageType`, `GarageFinish`, `GarageQual`, `GarageCond` (~5.5% NaN): No garage.
- `BsmtQual`, `BsmtCond`, `BsmtExposure`, `BsmtFinType1`, `BsmtFinType2` (~2.5% NaN): No basement.
- `MasVnrType` (59.7% NaN): No masonry veneer.
- **Resolution**: Imputed explicitly as `"None"` so the model treats absence as a distinct categorical state rather than mode imputation.

### B. Random Missingness (True Missing Values)
- `LotFrontage` (17.7% missing in train, 15.6% in test): Right-skewed numerical dimension.
- `MasVnrArea` (0.5% in train, 1.0% in test): Numerical area.
- `GarageYrBlt` (5.5% in train, 5.3% in test): Year built for garage.
- Rare test-set missing entries: `MSZoning` (4), `Utilities` (2), `BsmtFullBath` (2), `BsmtHalfBath` (2), `Functional` (2), `Exterior1st` (1), `Exterior2nd` (1), `TotalBsmtSF` (1), `BsmtUnfSF` (1), `BsmtFinSF1` (1), `BsmtFinSF2` (1), `KitchenQual` (1), `GarageCars` (1), `GarageArea` (1), `SaleType` (1).
- **Resolution**: Handled via robust median (numerical) and mode (categorical) imputation inside the pipeline.

---

## 4. Key Relationships & Multicollinearity

### Top Linear Correlations with `SalePrice`:
1. `OverallQual` ($r = 0.791$): Overall material and finish of the house.
2. `GrLivArea` ($r = 0.709$): Above-ground living area in sq ft.
3. `GarageCars` ($r = 0.640$) / `GarageArea` ($r = 0.623$): Garage capacity and footprint.
4. `TotalBsmtSF` ($r = 0.614$) / `1stFlrSF` ($r = 0.606$): Foundation and floor footprint.
5. `FullBath` ($r = 0.561$): Full bathrooms above grade.
6. `TotRmsAbvGrd` ($r = 0.534$): Total rooms above grade.
7. `YearBuilt` ($r = 0.523$) & `YearRemodAdd` ($r = 0.507$): House age and modernization.

### Multicollinear Feature Pairs ($r \ge 0.80$):
- `GarageCars` & `GarageArea` ($r = 0.882$): Redundant representation of garage size.
- `YearBuilt` & `GarageYrBlt` ($r = 0.826$): Garages are typically built when the house is constructed.
- `GrLivArea` & `TotRmsAbvGrd` ($r = 0.825$): Living area directly drives the room count.
- `TotalBsmtSF` & `1stFlrSF` ($r = 0.820$): In standard architecture, the basement ceiling matches the first floor perimeter.

---

## 5. EDA Conclusions & Next Steps
- Target variable must be modeled in logarithmic space (`y_log`).
- Pipeline must cleanly separate structural missingness (`"None"`) from random numerical/categorical missingness.
- High multicollinearity indicates that regularized linear models (Ridge/Lasso) and tree ensembles will be critical in future phases to prevent variance inflation.
