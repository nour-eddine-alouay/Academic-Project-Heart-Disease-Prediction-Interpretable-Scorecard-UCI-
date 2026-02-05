# Heart Disease Prediction & Interpretable Scorecard (UCI) — SPLEX Project

This project applies machine learning to the **UCI Heart Disease** dataset to predict the presence of heart disease (**binary classification**) and proposes an **interpretable** approach in the form of a **medical scorecard**.

> ✅ Author: **Nour Eddine ALOUAY**  
> 📊 Dataset: UCI Heart Disease (via KaggleHub)

---

## Table of Contents
- [Overview](#overview)
- [Objectives](#objectives)
- [Dataset](#dataset)
- [Methodology](#methodology)
  - [1) Data loading & cleaning](#1-data-loading--cleaning)
  - [2) Preprocessing & leakage prevention](#2-preprocessing--leakage-prevention)
  - [3) Baseline models (cross-validation)](#3-baseline-models-cross-validation)
  - [4) Hyperparameter tuning (GridSearchCV)](#4-hyperparameter-tuning-gridsearchcv)
  - [5) Feature selection](#5-feature-selection)
  - [6) Unsupervised exploration (PCA + KMeans)](#6-unsupervised-exploration-pca--kmeans)
  - [7) Bayesian Network (pgmpy)](#7-bayesian-network-pgmpy)
  - [8) Interpretable scorecard + integer score](#8-interpretable-scorecard--integer-score)
- [Results](#results)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Running the Notebook](#running-the-notebook)
- [Technical Notes](#technical-notes)
- [Reproducibility](#reproducibility)
- [Limitations & Future Work](#limitations--future-work)
- [Credits](#credits)
- [License](#license)

---

## Overview
The **Heart Disease** dataset contains clinical variables (age, blood pressure, cholesterol, ECG results, chest pain type, etc.).  
The goal is to predict the target variable:

- `target = 1`: heart disease present  
- `target = 0`: heart disease absent  

---

## Objectives
1. Build a rigorous ML pipeline for clinical/tabular data
2. Compare several supervised baseline models
3. Tune hyperparameters using **GridSearchCV**
4. Evaluate **feature selection** approaches
5. Explore data structure with **PCA + KMeans**
6. Experiment with a **Bayesian Network** (discretization + structure learning)
7. Propose a final **interpretable** approach: **scorecard + integer score**

---

## Dataset
The dataset is fetched via **KaggleHub**:

- Kaggle source: `thisishusseinali/uci-heart-disease-data`

⚠️ The notebook downloads the dataset automatically; no data files are stored in this repository.

### Important cleaning step: duplicates
The raw dataset contains many duplicates:
- before cleaning: ~606 rows  
- after dropping duplicates: ~302 unique samples  

✅ Dropping duplicates helps avoid **data leakage** (train/test contamination).

---

## Methodology

### 1) Data loading & cleaning
- Download using `kagglehub.dataset_download`
- Inspect schema: dtypes, target distribution
- Check missing values
- Remove duplicates (`df.drop_duplicates()`)

### 2) Preprocessing & leakage prevention
Preprocessing is handled with `ColumnTransformer` + `Pipeline`:
- Numeric: median imputation + scaling (`StandardScaler`)
- Categorical: mode imputation + one-hot encoding (`OneHotEncoder`)

✅ The preprocessing is included inside CV/tuning pipelines (`cross_validate`, `GridSearchCV`) to prevent leakage.

### 3) Baseline models (cross-validation)
Models evaluated with **5-fold Stratified CV**:
- Logistic Regression
- SVM (RBF)
- k-NN
- Gaussian Naive Bayes
- Random Forest

Metrics:
- Accuracy
- F1-score
- ROC-AUC

### 4) Hyperparameter tuning (GridSearchCV)
Tuning performed with **ROC-AUC** as the main objective:
- Logistic Regression: `C`
- SVM: `C`, `gamma`
- kNN: `n_neighbors`, `weights`, `p`
- Random Forest: `n_estimators`, `max_depth`, `min_samples_split`, etc.

### 5) Feature selection
Compared three approaches:
- **Filter**: `SelectFdr` + ANOVA F-test
- **Embedded**: Logistic Regression with **L1** regularization
- **Embedded**: Elastic Net via `SGDClassifier`

### 6) Unsupervised exploration (PCA + KMeans)
- Apply PCA (2D) after preprocessing
- Run KMeans clustering with `k=2`
- Compare clusters to the true target using:
  - Explained variance (2 PCs)
  - ARI (Adjusted Rand Index)
  - Silhouette score

### 7) Bayesian Network (pgmpy)
Because Bayesian Networks in `pgmpy` require discrete variables:
- Discretize numeric features into quantile bins (`pd.qcut`)
- Learn structure (Hill Climb + BIC-like score)
- Fit CPDs via maximum likelihood
- Predict `P(target=1)` using Variable Elimination inference
- Evaluate ROC-AUC (CV + test)

### 8) Interpretable scorecard + integer score
Final interpretable approach:
- Discretize numeric features into quantile bins (`KBinsDiscretizer`)
- One-hot encode categorical features
- Train sparse model: **Logistic Regression L1**
- Convert coefficients to **integer points**
- Choose the threshold using the **Youden index** (ROC-based)

Output: a simple additive score using ~**36 rules** (after rounding).

---

## Results
Exact values may vary slightly depending on environment, but overall:

- Best tuned discriminative model: **LogReg_tuned**, test ROC-AUC ≈ **0.93**
- Final scorecard (integer score): test ROC-AUC ≈ **0.903**
- Bayesian Network: test ROC-AUC ≈ **0.87**

✅ Final trade-off: a small performance drop for a large gain in interpretability.

---

## Repository Structure
Recommended structure:
