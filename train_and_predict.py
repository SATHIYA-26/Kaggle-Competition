"""
========================================================================================
Kaggle House Prices — Production End-to-End ML Pipeline
========================================================================================
This script encapsulates the final, evidence-based Machine Learning pipeline developed
through 7 systematic iterations. It trains our winning multi-family ensemble across
5-fold cross-validation and generates the final competition submission.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import GradientBoostingRegressor
import xgboost as xgb
from sklearn.metrics import root_mean_squared_error


def load_data(train_path="train.csv", test_path="test.csv"):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    return train, test


def impute_structural_absence(train_df, test_df):
    """
    15 features where NaN represents the absence of an amenity (e.g. no garage, no pool).
    Imputing 'None' prevents false mode imputation.
    """
    none_cols = [
        'PoolQC', 'MiscFeature', 'Alley', 'Fence', 'MasVnrType',
        'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual',
        'GarageCond', 'BsmtQual', 'BsmtCond', 'BsmtExposure',
        'BsmtFinType1', 'BsmtFinType2'
    ]
    train_clean = train_df.copy()
    test_clean = test_df.copy()
    for col in none_cols:
        train_clean[col] = train_clean[col].fillna('None')
        test_clean[col] = test_clean[col].fillna('None')
    return train_clean, test_clean


def engineer_features(df):
    """
    Creates domain-justified aggregate features and concave logarithmic transformations
    to capture diminishing marginal returns and eliminate outlier linear extrapolation.
    """
    d = df.copy()
    # 1. Total Square Footage
    d['TotalSF'] = d['TotalBsmtSF'] + d['1stFlrSF'] + d['2ndFlrSF']
    # 2. Total Bathrooms
    d['TotalBath'] = (
        d['FullBath'] + 0.5 * d['HalfBath'] +
        d['BsmtFullBath'] + 0.5 * d['BsmtHalfBath']
    )
    # 3. Structural Ages at transaction
    d['HouseAge'] = d['YrSold'] - d['YearBuilt']
    d['RemodelAge'] = d['YrSold'] - d['YearRemodAdd']
    d['GarageAge'] = d['YrSold'] - d['GarageYrBlt']
    # 4. Total Porch Area
    d['TotalPorchSF'] = (
        d['OpenPorchSF'] + d['EnclosedPorch'] +
        d['3SsnPorch'] + d['ScreenPorch']
    )
    # 5. Finished Basement
    d['TotalFinBsmtSF'] = d['BsmtFinSF1'] + d['BsmtFinSF2']
    
    # 6. Non-linear log transformations (diminishing marginal returns)
    d['LogTotalSF'] = np.log1p(d['TotalSF'])
    d['LogGrLivArea'] = np.log1p(d['GrLivArea'])
    d['LogLotArea'] = np.log1p(d['LotArea'])
    return d


def build_preprocessor(numeric_cols, categorical_cols):
    """
    Constructs leak-safe ColumnTransformer.
    StandardScaler is deliberately excluded from here to preserve sparsity for tree models.
    """
    num_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median'))
    ])
    cat_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    return ColumnTransformer([
        ('num', num_pipe, numeric_cols),
        ('cat', cat_pipe, categorical_cols)
    ])


def run_pipeline():
    print("Step 1: Loading raw data...")
    train, test = load_data()
    
    # Extract IDs and target
    test_ids = test['Id']
    y = train['SalePrice']
    y_log = np.log1p(y)
    
    train_clean, test_clean = impute_structural_absence(train, test)
    X = engineer_features(train_clean.drop(columns=['SalePrice', 'Id']))
    test_X = engineer_features(test_clean.drop(columns=['Id']))
    
    print(f"Step 2: Preprocessing {X.shape[1]} engineered features...")
    numeric_cols = X.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object', 'string']).columns.tolist()
    
    preprocessor = build_preprocessor(numeric_cols, categorical_cols)
    X_proc = preprocessor.fit_transform(X)
    test_X_proc = preprocessor.transform(test_X)
    
    # Feature scaler specifically for regularized linear models
    scaler = StandardScaler(with_mean=False)
    X_scaled = scaler.fit_transform(X_proc)
    test_X_scaled = scaler.transform(test_X_proc)
    
    print("Step 3: Training 5-Fold Cross-Validated Diverse Ensemble...")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    oof_xgb = np.zeros(len(X))
    oof_gbr = np.zeros(len(X))
    oof_lasso = np.zeros(len(X))
    oof_ridge = np.zeros(len(X))
    
    test_preds_xgb = np.zeros(len(test_X))
    test_preds_gbr = np.zeros(len(test_X))
    test_preds_lasso = np.zeros(len(test_X))
    test_preds_ridge = np.zeros(len(test_X))
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
        # 1. XGBoost
        m_xgb = xgb.XGBRegressor(
            n_estimators=500, learning_rate=0.03, max_depth=3,
            subsample=0.8, colsample_bytree=0.7, reg_alpha=0.1, reg_lambda=1.0, random_state=42
        )
        # 2. Gradient Boosting
        m_gbr = GradientBoostingRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=3, random_state=42
        )
        # 3. Lasso
        m_lasso = Lasso(alpha=0.003, max_iter=10000, random_state=42)
        # 4. Ridge
        m_ridge = Ridge(alpha=300.0, random_state=42)
        
        # Fit on training fold
        m_xgb.fit(X_proc[train_idx], y_log.iloc[train_idx])
        m_gbr.fit(X_proc[train_idx], y_log.iloc[train_idx])
        m_lasso.fit(X_scaled[train_idx], y_log.iloc[train_idx])
        m_ridge.fit(X_scaled[train_idx], y_log.iloc[train_idx])
        
        # OOF Predictions
        oof_xgb[val_idx] = m_xgb.predict(X_proc[val_idx])
        oof_gbr[val_idx] = m_gbr.predict(X_proc[val_idx])
        oof_lasso[val_idx] = m_lasso.predict(X_scaled[val_idx])
        oof_ridge[val_idx] = m_ridge.predict(X_scaled[val_idx])
        
        # Accumulate fold predictions for test set
        test_preds_xgb += m_xgb.predict(test_X_proc) / 5
        test_preds_gbr += m_gbr.predict(test_X_proc) / 5
        test_preds_lasso += m_lasso.predict(test_X_scaled) / 5
        test_preds_ridge += m_ridge.predict(test_X_scaled) / 5
        
        fold_blend = (
            0.65 * oof_xgb[val_idx] + 0.10 * oof_gbr[val_idx] +
            0.20 * oof_lasso[val_idx] + 0.05 * oof_ridge[val_idx]
        )
        fold_rmsle = root_mean_squared_error(y_log.iloc[val_idx], fold_blend)
        print(f"  Fold {fold} Blended RMSLE: {fold_rmsle:.4f}")
    
    # Complete Out-of-Fold Validation
    final_oof_blend = (
        0.65 * oof_xgb + 0.10 * oof_gbr +
        0.20 * oof_lasso + 0.05 * oof_ridge
    )
    total_oof_rmsle = root_mean_squared_error(y_log, final_oof_blend)
    print(f"\n========================================================")
    print(f"Final 5-Fold Cross-Validation OOF RMSLE: {total_oof_rmsle:.4f}")
    print(f"========================================================")
    
    # Generate Test Predictions in Dollar Scale
    final_test_preds_log = (
        0.65 * test_preds_xgb + 0.10 * test_preds_gbr +
        0.20 * test_preds_lasso + 0.05 * test_preds_ridge
    )
    final_test_preds = np.expm1(final_test_preds_log)
    
    # Save Final Submission
    submission = pd.DataFrame({
        'Id': test_ids,
        'SalePrice': final_test_preds
    })
    submission.to_csv('final_submission.csv', index=False)
    
    # Integrity Assertions
    assert submission.shape == (1459, 2), f"Invalid shape: {submission.shape}"
    assert submission['SalePrice'].isnull().sum() == 0, "Null predictions found"
    assert (submission['SalePrice'] > 0).all(), "Non-positive predictions found"
    
    print("\nStep 4: Final submission verified and saved to 'final_submission.csv'!")
    print(f"  Predictions Count: {len(submission)}")
    print(f"  Min Price: ${final_test_preds.min():,.2f}")
    print(f"  Max Price: ${final_test_preds.max():,.2f}")
    print(f"  Mean Price: ${final_test_preds.mean():,.2f}")
    print(f"  Test Mansion Id 2550: ${final_test_preds[test[test['Id'] == 2550].index[0]]:,.2f}")


if __name__ == '__main__':
    run_pipeline()
