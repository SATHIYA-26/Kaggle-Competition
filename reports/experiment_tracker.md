# Kaggle Competition Experiment Tracker

This document tracks all formal modeling experiments and Kaggle submissions across the project lifecycle.

| Submission | Phase | Main Change | Model | Local CV / Val | Kaggle Score | Kaggle Rank | Key Learnings & Observations |
|:---:|:---:|:---|:---|:---:|:---:|:---:|:---|
| **S1** | Phase 3 | First baseline end-to-end pipeline | Linear Regression (OLS) | 0.1418 (Val RMSLE) | 0.13950 | 2165 / 3843 | Clean preprocessing & log target yield solid 0.1418 local baseline; 0.13950 on Kaggle shows strong local/public alignment and top 56% baseline. |
| **S2** | Phase 4 | Feature Engineering + Lasso Regularization | Lasso (alpha=0.001) | 0.1224 (Val RMSLE) | 0.14271 | *(Pending)* | Local Val improved to 0.1224 (-0.0194); Kaggle score 0.14271 slightly degraded due to extreme linear extrapolation on single 10,190 sq ft test mansion (Id 2550). |
| **S3** | Phase 5 | 5-Fold CV + Tuned XGBoost | XGBoost (depth=3, lr=0.03, n=500) | 0.1261 (5-Fold CV) | 0.12788 | 1280 / 3843 | 5-fold CV hyperparameter tuning + gradient boosted trees; resolved Id 2550 outlier extrapolation; score dropped to 0.12788, jumping +885 ranks into top 33%! |
| **S4** | Phase 6 | Multi-Model Diverse Ensemble (Stacking/Blending) | 65% XGB + 10% GBR + 20% Lasso + 5% Ridge | 0.1265 (5-Fold OOF) | *(Pending)* | *(Pending)* | Blended non-linear trees with sparse linear models (error corr=0.84); rejected PCA (-0.02 worse); Id 2550 predicted cleanly at $408k. |
| **S5** | Phase 5 | Boosting (XGBoost/LightGBM) | — | — | — | — | — |
| **S6** | Phase 5 | Hyperparameter tuning | — | — | — | — | — |
| **S7** | Phase 6 | Ensembling & Stacking | — | — | — | — | — |

*Note: Kaggle Score and Rank are updated strictly upon official submission confirmation from the Kaggle leaderboard.*
