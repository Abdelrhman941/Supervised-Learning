# Supervised Learning - Complete Reference Guide

A comprehensive resource for understanding, selecting, and applying supervised learning algorithms across regression and classification tasks.

## 📋 Table of Contents

1. [Overview](#overview)
2. [Model Selection Guide](#model-selection-guide)
3. [Linear Models](#linear-models)
4. [Tree-Based Models](#tree-based-models)
5. [Ensemble Methods](#ensemble-methods)
6. [Instance-Based Learning](#instance-based-learning)
7. [Support Vector Machines](#support-vector-machines)
8. [Probabilistic Models](#probabilistic-models)
9. [Performance Comparison](#performance-comparison)
10. [Decision Framework](#decision-framework)
11. [Further Resources](#further-resources)

<div style="width: 100%; height: 10px; background: linear-gradient(to right, gray, white, orange,white, gray); border-radius: 5px; margin: 20px 0;"></div>

## Overview

Supervised learning algorithms learn from labeled training data to make predictions on new, unseen data. This repository covers the most important algorithms across different paradigms, organized by their underlying mathematical principles and use cases.

**Key Considerations:**
- **Problem Type**: Regression vs Classification
- **Data Size**: Small $(< 1K)$, Medium $(1K-100K)$, Large $(> 100K)$
- **Feature Count**: Low $(< 100)$, High $(> 100)$
- **Interpretability**: High vs Low requirement
- **Training Time**: Fast vs Slow acceptable
- **Prediction Speed**: Real-time vs Batch processing

<div style="width: 100%; height: 10px; background: linear-gradient(to right, gray, white, orange,white, gray); border-radius: 5px; margin: 20px 0;"></div>

## Model Selection Guide

### 🎯 Quick Decision Tree

```
Problem Type?
├── Regression
│   ├── Linear relationship? → Linear Regression
│   ├── Non-linear + interpretable? → Decision Trees/Random Forest
│   └── High accuracy needed? → XGBoost/LightGBM
└── Classification
    ├── Linear separable? → Logistic Regression/SVM
    ├── Small dataset? → Naive Bayes/KNN
    ├── Need interpretability? → Decision Trees
    └── High accuracy? → Random Forest/Boosting
```

### 📊 Data Characteristics Matrix

| Data Condition | Recommended Models | Avoid |
|---|---|---|
| **Small dataset (< 1K)** | Naive Bayes, KNN, Linear models | Deep ensembles, Neural networks |
| **Large dataset (> 100K)** | XGBoost, LightGBM, Linear models | KNN (unless optimized) |
| **High-dimensional** | Linear models, SVM, Random Forest | KNN, Naive Bayes |
| **Categorical features** | Tree-based, CatBoost | Linear models (without encoding) |
| **Missing values** | Tree-based, XGBoost, CatBoost | KNN, SVM, Naive Bayes |
| **Need interpretability** | Linear, Decision Trees, Naive Bayes | Ensemble methods, SVM |
| **Real-time prediction** | Linear, Naive Bayes, simple trees | Large ensembles, KNN |

<div style="width: 100%; height: 10px; background: linear-gradient(to right, gray, white, orange,white, gray); border-radius: 5px; margin: 20px 0;"></div>

## Linear Models

Linear models assume a linear relationship between features and target variable. They're fast, interpretable, and work well with high-dimensional data.

<div style="background: linear-gradient(135deg,rgb(139, 162, 80),rgb(50, 165, 136)); 
            color: #ffffff; 
            width: 100%; 
            height: 40px; 
            text-align: center; 
            font-weight: bold; 
            line-height: 40px; 
            margin: 15px 0; 
            font-size: 24px; 
            border-radius: 10px; 
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);">
    1. Linear Regression
</div>

**Core Equation:**
$$
ŷ = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ
$$

**Cost Function (MSE):**
$$
J(\beta) = \frac{1}{2m} \sum_{i=1}^{m} \left( h_{\beta}(x^{(i)}) - y^{(i)} \right)^2
$$

**When to Use:**
- Linear relationship between features and target
- Need interpretable coefficients
- Fast training/prediction required
- High-dimensional data with regularization

**When NOT to Use:**
- Non-linear relationships
- Multicollinearity without regularization
- Outliers present (use robust regression instead)

**Variants:**
- **Ridge (L2)**: Handles multicollinearity, shrinks coefficients
- **Lasso (L1)**: Feature selection, sparse solutions
- **Elastic Net**: Combines L1 + L2 penalties

<div style="background: linear-gradient(135deg,rgb(139, 162, 80),rgb(50, 165, 136)); 
            color: #ffffff; 
            width: 100%; 
            height: 40px; 
            text-align: center; 
            font-weight: bold; 
            line-height: 40px; 
            margin: 15px 0; 
            font-size: 24px; 
            border-radius: 10px; 
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);">
    2. Logistic Regression
</div>

**Core Equation:**
$$
P(y=1) = \frac{1}{1 + e^{-(\beta_{0} + \beta_{1}x_{1} + ... + \beta_{n}x_{n})}}
$$

**Cost Function (Log-likelihood):**
$$
J(\beta) = -\frac{1}{m} \sum_{i=1}^{m} \left[ y^{(i)} \log(h_{\beta}(x^{(i)})) + (1-y^{(i)}) \log(1-h_{\beta}(x^{(i)})) \right]
$$

**When to Use:**
- Binary or multiclass classification
- Need probability estimates
- Linear decision boundary acceptable
- Interpretable feature impacts required

**When NOT to Use:**
- Complex non-linear patterns
- Perfect separation (causes convergence issues)
- Very small datasets

<div style="width: 100%; height: 10px; background: linear-gradient(to right, gray, white, orange,white, gray); border-radius: 5px; margin: 20px 0;"></div>

## Tree-Based Models

Tree-based models make decisions by splitting data based on feature values. They handle non-linear relationships naturally and require minimal preprocessing.

<div style="background: linear-gradient(135deg,rgb(139, 162, 80),rgb(50, 165, 136)); 
            color: #ffffff; 
            width: 100%; 
            height: 40px; 
            text-align: center; 
            font-weight: bold; 
            line-height: 40px; 
            margin: 15px 0; 
            font-size: 24px; 
            border-radius: 10px; 
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);">
    Decision Trees
</div>

**Splitting Criteria:**
- **Gini Impurity**: $Gini = 1 - \sum p²ᵢ$
- **Entropy**: $Entropy = -\sum pᵢ \log₂(pᵢ)$
- **MSE** (regression): $MSE = \frac{1}{n} \sum (yᵢ - ȳ)²$

**Information Gain:**
$$
IG = Entropy(parent) - \sum \left( \frac{nᵢ}{n} \right) \times Entropy(childᵢ)
$$

**When to Use:**
- Non-linear relationships
- Mixed data types
- Need interpretability
- Feature interactions important

**When NOT to Use:**
- Linear relationships (overcomplicates)
- Small datasets (prone to overfitting)
- Need smooth decision boundaries

<div style="width: 100%; height: 10px; background: linear-gradient(to right, gray, white, orange,white, gray); border-radius: 5px; margin: 20px 0;"></div>

## Ensemble Methods

Ensemble methods combine multiple models to improve performance and reduce overfitting.

> ### Bagging

Trains multiple models on different subsets of data, then averages predictions.

<div style="background: linear-gradient(135deg,rgb(139, 162, 80),rgb(50, 165, 136)); 
            color: #ffffff; 
            width: 100%; 
            height: 40px; 
            text-align: center; 
            font-weight: bold; 
            line-height: 40px; 
            margin: 15px 0; 
            font-size: 24px; 
            border-radius: 10px; 
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);">
    Random Forest
</div>

**Key Concepts:**
- Bootstrap sampling of data
- Random feature selection at each split
- Majority voting (classification) or averaging (regression)

**When to Use:**
- General-purpose algorithm (works well on most datasets)
- Need feature importance estimates
- Robust to overfitting
- Handle large datasets

**When NOT to Use:**
- Need high interpretability
- Very small datasets
- Linear relationships

<div style="background: linear-gradient(135deg,rgb(139, 162, 80),rgb(50, 165, 136)); 
            color: #ffffff; 
            width: 100%; 
            height: 40px; 
            text-align: center; 
            font-weight: bold; 
            line-height: 40px; 
            margin: 15px 0; 
            font-size: 24px; 
            border-radius: 10px; 
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);">
    Extra Trees (Extremely Randomized Trees)
</div>

- More randomness than Random Forest
- Faster training
- Better for very large datasets

> ### Boosting

Sequential learning where each model corrects previous model's errors.

<div style="background: linear-gradient(135deg,rgb(139, 162, 80),rgb(50, 165, 136)); 
            color: #ffffff; 
            width: 100%; 
            height: 40px; 
            text-align: center; 
            font-weight: bold; 
            line-height: 40px; 
            margin: 15px 0; 
            font-size: 24px; 
            border-radius: 10px; 
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);">
    AdaBoost
</div>

**Weight Update:**
$$
wᵢ⁽ᵗ⁺¹⁾ = wᵢ⁽ᵗ⁾ × \exp(αₜ × I(yᵢ ≠ hₜ(xᵢ)))
$$

**Model Weight:**
$$
αₜ = ½ × \ln\left(\frac{1 - εₜ}{εₜ}\right)
$$

<div style="background: linear-gradient(135deg,rgb(139, 162, 80),rgb(50, 165, 136)); 
            color: #ffffff; 
            width: 100%; 
            height: 40px; 
            text-align: center; 
            font-weight: bold; 
            line-height: 40px; 
            margin: 15px 0; 
            font-size: 24px; 
            border-radius: 10px; 
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);">
    Gradient Boosting
</div>

**Residual Update:**
$$
rᵢₘ = yᵢ - Fₘ₋₁(xᵢ)
$$

**Model Update:**
$$
Fₘ(x) = Fₘ₋₁(x) + γₘhₘ(x)
$$

<div style="background: linear-gradient(135deg,rgb(139, 162, 80),rgb(50, 165, 136)); 
            color: #ffffff; 
            width: 100%; 
            height: 40px; 
            text-align: center; 
            font-weight: bold; 
            line-height: 40px; 
            margin: 15px 0; 
            font-size: 24px; 
            border-radius: 10px; 
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);">
    XGBoost
</div>

**Objective Function:**
$$
Obj = \sum L(yᵢ, ŷᵢ) + \sum Ω(fₖ)
$$

**Advantages:**
- Built-in regularization
- Handles missing values
- Parallel processing
- Cross-validation built-in

<div style="background: linear-gradient(135deg,rgb(139, 162, 80),rgb(50, 165, 136)); 
            color: #ffffff; 
            width: 100%; 
            height: 40px; 
            text-align: center; 
            font-weight: bold; 
            line-height: 40px; 
            margin: 15px 0; 
            font-size: 24px; 
            border-radius: 10px; 
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);">
    LightGBM
</div>

- Leaf-wise tree growth
- Faster than XGBoost
- Lower memory usage
- Built-in categorical feature support

<div style="background: linear-gradient(135deg,rgb(139, 162, 80),rgb(50, 165, 136)); 
            color: #ffffff; 
            width: 100%; 
            height: 40px; 
            text-align: center; 
            font-weight: bold; 
            line-height: 40px; 
            margin: 15px 0; 
            font-size: 24px; 
            border-radius: 10px; 
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);">
    CatBoost
</div>

- Automatic categorical encoding
- Robust to hyperparameters
- Built-in overfitting protection
- No need for extensive preprocessing

> ### Hybrid Methods

<div style="background: linear-gradient(135deg,rgb(139, 162, 80),rgb(50, 165, 136)); 
            color: #ffffff; 
            width: 100%; 
            height: 40px; 
            text-align: center; 
            font-weight: bold; 
            line-height: 40px; 
            margin: 15px 0; 
            font-size: 24px; 
            border-radius: 10px; 
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);">
    Voting
</div>

**Hard Voting:**
$$
ŷ = mode(h₁(x), h₂(x), ..., hₖ(x))
$$

**Soft Voting:**
$$
ŷ = argmax\left(\frac{1}{k} \sum P(class|x)\right)
$$

<div style="background: linear-gradient(135deg,rgb(139, 162, 80),rgb(50, 165, 136)); 
            color: #ffffff; 
            width: 100%; 
            height: 40px; 
            text-align: center; 
            font-weight: bold; 
            line-height: 40px; 
            margin: 15px 0; 
            font-size: 24px; 
            border-radius: 10px; 
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);">
    Stacking
</div>

Uses a meta-learner to combine base model predictions:
```
Level 0: Base models (h₁, h₂, ..., hₖ)
Level 1: Meta-model learns optimal combination
```

<div style="width: 100%; height: 10px; background: linear-gradient(to right, gray, white, orange,white, gray); border-radius: 5px; margin: 20px 0;"></div>

## Instance-Based Learning

<div style="background: linear-gradient(135deg,rgb(139, 162, 80),rgb(50, 165, 136)); 
            color: #ffffff; 
            width: 100%; 
            height: 40px; 
            text-align: center; 
            font-weight: bold; 
            line-height: 40px; 
            margin: 15px 0; 
            font-size: 24px; 
            border-radius: 10px; 
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);">
    K-Nearest Neighbors (KNN)
</div>

**Distance Metrics:**
- **Euclidean**: $d = \sqrt{\sum (xᵢ - yᵢ)²}$
- **Manhattan**: $d = \sum |xᵢ - yᵢ|$
- **Minkowski**: $d = \left( \sum |xᵢ - yᵢ|^{p} \right)^{1/p}$

**Prediction:**
- **Classification**: Majority vote among k neighbors
- **Regression**: Average of k neighbor values

**When to Use:**
- Small to medium datasets
- Irregular decision boundaries
- Local patterns important
- Simple baseline needed

**When NOT to Use:**
- High-dimensional data (curse of dimensionality)
- Large datasets (slow prediction)
- Noisy data
- All features equally scaled not possible

<div style="width: 100%; height: 10px; background: linear-gradient(to right, gray, white, orange,white, gray); border-radius: 5px; margin: 20px 0;"></div>

> ## Support Vector Machines
<div style="background: linear-gradient(135deg,rgb(139, 162, 80),rgb(50, 165, 136)); 
            color: #ffffff; 
            width: 100%; 
            height: 40px; 
            text-align: center; 
            font-weight: bold; 
            line-height: 40px; 
            margin: 15px 0; 
            font-size: 24px; 
            border-radius: 10px; 
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);">
    SVR , SVC 
</div>

**Optimization Objective:**
$$
\min \frac{1}{2}||w||² + C\sum\xiᵢ
\text{subject to: } yᵢ(w·xᵢ + b) ≥ 1 - ξᵢ
$$

**Kernel Functions:**
- **Linear**: $K(xᵢ, xⱼ) = xᵢ·xⱼ$
- **RBF**: $K(xᵢ, xⱼ) = \exp(-γ||xᵢ - xⱼ||²)$
- **Polynomial**: $K(xᵢ, xⱼ) = (γxᵢ·xⱼ + r)ᵈ$

**When to Use:**
- High-dimensional data
- Clear margin between classes
- Kernel trick needed for non-linear patterns
- Robust to outliers

**When NOT to Use:**
- Very large datasets (slow training)
- Noisy data with overlapping classes
- Need probability estimates
- Multiple classes (requires one-vs-rest)

<div style="width: 100%; height: 10px; background: linear-gradient(to right, gray, white, orange,white, gray); border-radius: 5px; margin: 20px 0;"></div>

## Probabilistic Models

<div style="background: linear-gradient(135deg,rgb(139, 162, 80),rgb(50, 165, 136)); 
            color: #ffffff; 
            width: 100%; 
            height: 40px; 
            text-align: center; 
            font-weight: bold; 
            line-height: 40px; 
            margin: 15px 0; 
            font-size: 24px; 
            border-radius: 10px; 
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);">
    Naive Bayes
</div>

**Bayes' Theorem:**
$$
P(y|x) = \frac{P(x|y)P(y)}{P(x)}
$$

**Gaussian Naive Bayes:**
$$
P(xᵢ|y) = \frac{1}{\sqrt{2\pi\sigma²y}} \times \exp\left(-\frac{(xᵢ - μy)²}{2σ²y}\right)
$$

**When to Use:**
- Text classification
- Small datasets
- Features are conditionally independent
- Need fast training/prediction
- Baseline model

**When NOT to Use:**
- Strong feature correlations
- Numerical features with complex distributions
- Need high accuracy (usually)

<div style="width: 100%; height: 10px; background: linear-gradient(to right, gray, white, orange,white, gray); border-radius: 5px; margin: 20px 0;"></div>

## Performance Comparison

| **📂 Category**         | **🧠 Algorithm**           | ⚡ **Training Speed** | 🚀 **Prediction Speed** | 🎯 **Accuracy** | 🔍 **Interpretability** | 🔥 **Overfitting Risk** |
| ----------------------- | -------------------------- | -------------------- | ----------------------- | --------------- | ----------------------- | ----------------------- |
| **Linear Models**       | **Linear Regression**      | 🟢 Very Fast         | 🟢 Very Fast            | 🟡 Low–Medium   | 🟢 Very High            | 🟢 Low                  |
|                         | **Logistic Regression**    | 🟢 Fast              | 🟢 Very Fast            | 🟡 Medium       | 🟢 High                 | 🟢 Low                  |
| **Tree-Based**          | **Decision Trees**         | 🟢 Fast              | 🟢 Fast                 | 🟡 Medium       | 🟢 Very High            | 🔴 High                 |
| **Ensemble → Bagging**  | **Random Forest**          | 🟡 Medium            | 🟢 Fast                 | 🟢 High         | 🟡 Medium               | 🟢 Low                  |
|                         | **Extra Trees**            | 🟡 Medium            | 🟢 Fast                 | 🟢 High         | 🟡 Medium               | 🟡 Medium               |
| **Ensemble → Boosting** | **AdaBoost**               | 🟡 Medium            | 🟢 Fast                 | 🟢 High         | 🔴 Low                  | 🟡 Medium               |
|                         | **Gradient Boosting**      | 🔴 Medium–Slow       | 🟢 Fast                 | 🟢 Very High    | 🔴 Low                  | 🟡 Medium               |
|                         | **XGBoost**                | 🔴 Medium–Slow       | 🟢 Fast                 | 🟢 Very High    | 🔴 Low                  | 🟡 Medium               |
|                         | **LightGBM**               | 🟢 Fast              | 🟢 Very Fast            | 🟢 Very High    | 🔴 Low                  | 🟡 Medium               |
|                         | **CatBoost**               | 🟡 Medium            | 🟢 Fast                 | 🟢 Very High    | 🔴 Low                  | 🟢 Low                  |
| **Ensemble → Hybrid**   | **Stacking**               | 🔴 Slow (depends)    | 🟡 Medium               | 🟢 Very High    | 🔴 Low                  | 🟡 Medium–High          |
|                         | **Voting Classifier**      | 🟢 Fast              | 🟡 Medium               | 🟢 High         | 🟡 Medium               | 🟢 Low                  |
| **Lazy Learners**       | **K-Nearest Neighbors**    | 🟢 Very Fast (train) | 🔴 Slow                 | 🟢 Medium–High  | 🟡 Medium               | 🟡 Medium               |
| **Kernel Methods**      | **Support Vector Machine** | 🔴 Slow              | 🟢 Fast                 | 🟢 High         | 🔴 Low                  | 🟡 Medium               |
| **Probabilistic**       | **Naive Bayes**            | 🟢 Very Fast         | 🟢 Very Fast            | 🟡 Medium       | 🟢 High                 | 🟢 Low                  |

📌 Legend for Symbol Colors →  
```             
🟢 → Excellent / Fast / Low risk       
🟡 → Moderate / Depends on data or conditions     
🔴 → Weak / Slow / High risk  
```

<div style="width: 100%; height: 10px; background: linear-gradient(to right, gray, white, orange,white, gray); border-radius: 5px; margin: 20px 0;"></div>

## Decision Framework

### 1. Start with Problem Definition
- **Regression** or **Classification**?
- **Sample size**: $< 1K, 1K-100K, > 100K?$
- **Feature count**: $< 100, 100-1000, > 1000?$
- **Target distribution**: Balanced or imbalanced?

### 2. Consider Constraints
- **Interpretability**: Required or not?
- **Training time**: Limited or flexible?
- **Prediction speed**: Real-time or batch?
- **Memory**: Limited or abundant?

### 3. Algorithm Selection Priority

**For Most Cases (Start Here):**
1. **Random Forest** - Robust, good performance
2. **XGBoost/LightGBM** - If accuracy is critical
3. **Logistic/Linear Regression** - If interpretability needed

**For Specific Scenarios:**
- **Small dataset**: Naive Bayes, KNN, Linear models
- **High-dimensional**: Linear models, SVM
- **Categorical heavy**: CatBoost, Tree-based
- **Text data**: Naive Bayes, Linear models
- **Real-time**: Linear models, Naive Bayes

### 4. Evaluation Strategy
```python
# Typical evaluation pipeline
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix

# 1. Cross-validation for model selection
cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')

# 2. Hyperparameter tuning
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')

# 3. Final evaluation on test set
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

<div style="width: 100%; height: 10px; background: linear-gradient(to right, gray, white, orange,white, gray); border-radius: 5px; margin: 20px 0;"></div>

## Further Resources

### 📚 Essential References
- **Scikit-learn Documentation**: [sklearn.org](https://scikit-learn.org/)
- **Elements of Statistical Learning**: Free PDF by Hastie, Tibshirani, Friedman
- **Pattern Recognition and Machine Learning**: Christopher Bishop

### 🔧 Implementation Guides
- **XGBoost Documentation**: [xgboost.readthedocs.io](https://xgboost.readthedocs.io/)
- **LightGBM Guide**: [lightgbm.readthedocs.io](https://lightgbm.readthedocs.io/)
- **CatBoost Tutorial**: [catboost.ai](https://catboost.ai/)

### 📊 Visualization Tools
- **Yellowbrick**: ML visualization library
- **SHAP**: Model interpretability
- **ELI5**: Explain ML models

### 🧠 Interactive Learning
- **Coursera ML Course**: Andrew Ng
- **Fast.ai**: Practical machine learning
- **Kaggle Learn**: Free micro-courses

<div style="width: 100%; height: 10px; background: linear-gradient(to right, gray, white, orange,white, gray); border-radius: 5px; margin: 20px 0;"></div>

## 🚀 Quick Start Templates

### Model Comparison Template
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

models = {
    'Random Forest': RandomForestClassifier(),
    'Logistic Regression': LogisticRegression(),
    'SVM': SVC()
}

for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=5)
    print(f"{name}: {scores.mean():.3f} (+/- {scores.std() * 2:.3f})")
```

### Hyperparameter Tuning Template
```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7, None],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(
    RandomForestClassifier(),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

grid_search.fit(X_train, y_train)
print(f"Best parameters : {grid_search.best_params_}")
print(f"Best score      : {grid_search.best_score_:.3f}")
```
