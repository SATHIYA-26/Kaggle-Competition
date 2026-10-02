# Kaggle Competition Experiment Tracker

This document tracks all formal modeling experiments and Kaggle submissions across the project lifecycle.

| Submission | Phase | Main Change | Model | Local CV / Val | Kaggle Score | Kaggle Rank | Key Learnings & Observations |
|:---:|:---:|:---|:---|:---:|:---:|:---:|:---|
| **S1** | Phase 3 | First baseline end-to-end pipeline | Linear Regression (OLS) | 0.1418 (Val RMSLE) | 0.13950 | 2165 / 3843 | Clean preprocessing & log target yield solid 0.1418 local baseline; 0.13950 on Kaggle shows strong local/public alignment and top 56% baseline. |
| **S2** | Phase 4 | Feature Engineering + Lasso Regularization | Lasso (alpha=0.001) | 0.1224 (Val RMSLE) | *(Pending)* | *(Pending)* | Added TotalSF, TotalBath, Ages; L1 penalty pruned 108 redundant dummy features, dropping RMSLE from 0.1418 to 0.1224 (-0.0194). |
| **S3** | Phase 4 | Regularization (Ridge/Lasso) | — | — | — | — | — |
| **S4** | Phase 4 | Tree ensemble (Random Forest) | — | — | — | — | — |
| **S5** | Phase 5 | Boosting (XGBoost/LightGBM) | — | — | — | — | — |
| **S6** | Phase 5 | Hyperparameter tuning | — | — | — | — | — |
| **S7** | Phase 6 | Ensembling & Stacking | — | — | — | — | — |

*Note: Kaggle Score and Rank are updated strictly upon official submission confirmation from the Kaggle leaderboard.*
