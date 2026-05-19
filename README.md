# CMPE-257-Diabetes-Project
# Diabetes Multi-Class Detection (CMPE 257)

## Project Overview
This repository contains a complete machine learning pipeline for detecting and classifying the stage of diabetes in patients. The project specifically focuses on predicting three distinct classes: **No Diabetes**, **Pre-Diabetes**, and **Type 2 Diabetes** (excluding Type 1 and Gestational cases). 

Through extensive data processing, feature engineering, and hyperparameter optimization, we evaluate four distinct classification models to determine the most effective algorithm for medical diagnosis based on patient metrics.

## Database Used
[Diabetes Health Indicators Database](https://www.kaggle.com/datasets/mohankrishnathalla/diabetes-health-indicators-dataset)

## Team Members & Roles
* **Daniel Cai:** Data processing and cleaning
* **Yuxin Zhang:** Diagrams and feature engineering
* **Nhi Nguyen:** Modeling algorithm and evaluation
* **Katriel Chiu:** Model visualization and hyperparameter optimization

---

## Machine Learning Pipeline

### 1. Data Processing and Cleaning
* **Target Pruning:** Filtered out `Type 1` and `Gestational` diabetes stages to focus purely on Type 2 progression.
* **Data Formatting:** Converted specific binary/categorical columns (`family_history_diabetes`, `hypertension_history`, etc.) into proper categorical types.
* **Validation:** Verified data integrity (no missing values, checked for duplicate rows, evaluated descriptive statistics).

### 2. Feature Engineering & Selection
* **Custom Features:** Created domain-specific composite features to better capture patient health:
    * `glucose_diff`: Difference between postprandial and fasting glucose.
    * `cholesterol_ratio`: Ratio of LDL to HDL cholesterol.
    * `lifestyle_score`: A composite metric combining diet, physical activity, sleep, and screen time.
* **Multicollinearity Handling:** Generated a Pearson Correlation Heatmap and programmatically dropped one feature from highly correlated pairs (|r| > 0.8), prioritizing the feature with the highest correlation to the target label.
* **Feature Selection:** Utilized a 5-fold Stratified Cross-Validation loop with a Logistic Regression model to iteratively test and identify the optimal number of features to keep.

### 3. Modeling & Evaluation
We trained and evaluated four primary models using standard scaling and an 80/20 train-test split:
1.  **Logistic Regression**
2.  **Decision Tree**
3.  **Random Forest**
4.  **XGBoost**

**Evaluation Metrics Included:**
* Accuracy (Test and 5-Fold Cross-Validation)
* Macro F1-Score
* Confusion Matrices
* ROC Curves & AUC
* Feature Importance visualization (derived from Random Forest)

### 4. Hyperparameter Optimization
To maximize model performance, we utilized both **GridSearchCV** and **RandomizedSearchCV** to fine-tune the models. 
* Scoring was evaluated on both `accuracy` and `f1_macro`.
* Extensive comparative visualizations were generated to analyze the performance differences between baseline models, Grid Search tuned models, and Randomized Search tuned models.
