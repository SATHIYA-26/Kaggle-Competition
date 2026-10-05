# 🏠 House Prices — End-to-End Machine Learning Project

> **Kaggle Competition:** House Prices - Advanced Regression Techniques

## 🎯 Overview

This project uses the **House Prices Kaggle competition** as an end-to-end Machine Learning playground.

The objective is to go beyond building a single model and systematically explore the complete ML workflow — from **EDA and preprocessing to advanced models, ensembles, and Kaggle submissions**.

The focus is on:

**Learn → Experiment → Compare → Validate → Submit → Analyze → Improve**

---

## 🧠 Our Approach

We will explore different Machine Learning techniques step-by-step and compare how they perform on the same problem.

```text
Dataset
  ↓
EDA
  ↓
Preprocessing
  ↓
Feature Engineering
  ↓
Baseline
  ↓
Linear Models
  ↓
Regularization
  ↓
Tree Models
  ↓
Distance / Kernel Models
  ↓
Ensemble Learning
  ↓
Boosting
  ↓
Feature Selection
  ↓
PCA
  ↓
Cross Validation
  ↓
Hyperparameter Tuning
  ↓
Stacking
  ↓
Final Ensemble
```

Each technique is treated as an **experiment**, not just something to add to the project.

---

## 🔬 ML Techniques We Explore

### Regression
- Linear Regression
- Ridge
- Lasso
- ElasticNet

### Tree-Based Models
- Decision Tree
- Random Forest

### Distance & Kernel Methods
- KNN
- SVR

### Ensemble & Boosting
- AdaBoost
- Gradient Boosting
- XGBoost
- Stacking

### Optimization & Representation
- Feature Engineering
- Feature Selection
- PCA
- Cross Validation
- Grid Search
- Randomized Search
- Hyperparameter Tuning

---

## 🗺️ Project Phases

### Phase 1 — Data Understanding
`EDA → Distribution → Correlation → Missing Values → Outliers`

### Phase 2 — Data Preparation
`Cleaning → Imputation → Encoding → Pipelines → Leakage Prevention`

### Phase 3 — Baseline
`Linear Regression → First Benchmark → Kaggle Submission`

### Phase 4 — Model Exploration
`Ridge → Lasso → ElasticNet → Decision Tree → Random Forest → KNN → SVR`

### Phase 5 — Feature Engineering
`House-Specific Features → Feature Selection → Representation Experiments`

### Phase 6 — Advanced Models
`AdaBoost → Gradient Boosting → XGBoost`

### Phase 7 — Optimization
`Cross Validation → Grid Search → Random Search → Hyperparameter Tuning`

### Phase 8 — Dimensionality Reduction
`Scaling → PCA → Comparison`

### Phase 9 — Ensemble
`Model Blending → Stacking → Final Ensemble`

### Phase 10 — Final Pipeline
`Best Experiments → Clean Pipeline → Final Kaggle Submission`

---

## 🏆 Kaggle Submission Strategy

We will make **multiple meaningful submissions** throughout the project.

```text
Submission 1
    ↓
Baseline

Submission 2+
    ↓
Model / Feature / Technique Improvement

    ↓
Compare with previous results

    ↓
Submit only meaningful experiments

    ↓
Analyze Kaggle result

    ↓
Next experiment
```

The number of submissions is **not fixed**.

We submit when an experiment provides either:

- A potential performance improvement
- A useful ML learning
- A meaningful comparison

---

## 📊 Experiment Tracking

Each important experiment will track:

```text
Model
Features
Preprocessing
Validation
Hyperparameters
CV Score
Kaggle Score
Observation
Decision
```

This allows us to understand **which changes actually matter** instead of randomly changing the pipeline.

---

## 📈 Evaluation

The competition uses **RMSLE**.

We therefore experiment with logarithmic target transformation:

```python
y_log = np.log1p(y)
```

and convert predictions back using:

```python
pred = np.expm1(pred_log)
```

**Lower Kaggle score = better performance.**

---

## 📁 Project Structure

```text
house-prices-ml/
│
├── data/
│   ├── train.csv
│   └── test.csv
│
├── notebooks/
│   └── prediction.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── features.py
│   ├── models.py
│   ├── evaluation.py
│   └── submission.py
│
├── submissions/
│   ├── submission_01.csv
│   ├── submission_02.csv
│   └── ...
│
├── results/
│   └── experiment_log.csv
│
└── README.md
```

---

## 🎓 Final Goal

By the end of the project, we aim to have:

```text
Complete ML Pipeline
        +
Multiple ML Models
        +
Feature Engineering
        +
Cross Validation
        +
Hyperparameter Tuning
        +
Ensemble Learning
        +
Multiple Kaggle Submissions
        +
Experiment History
        +
Practical ML Understanding
```

---

## 🔥 Philosophy

> **Don't just chase the leaderboard.**
>
> **Build → Experiment → Understand → Validate → Submit → Analyze → Improve.**
