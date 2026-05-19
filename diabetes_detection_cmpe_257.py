# -*- coding: utf-8 -*-
"""Diabetes Detection - CMPE 257.ipynb


Original file is located at
    https://colab.research.google.com/drive/1vsB4qpOdp3img9U2SzjSXjjR5Pds5eTg

---
# DIABETES MULTI-CLASS DETECTION

## GROUP MEMBERS: Katriel Chiu, Nhi Nguyen, Yuxin Zhang, Daniel Cai

---

---
# TASKS
## Daniel: Data processing and cleaning
## Yuxin: Diagrams and feature engineering
## Nhi: Modeling algorithm and evaluation
## Katriel: Model visualization and hyperparameter optimization
---
"""

'LIBRARIES'
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler  # for scaling
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

"""## Data processing and cleaning"""

df = pd.read_csv('diabetes_dataset.csv')
df_filtered = df[
    ~df['diabetes_stage'].isin(['Type 1', 'Gestational']) #prune type 1 and gestational entries
]
print(df_filtered['diabetes_stage'].value_counts())
df_filtered.to_csv("diabetes_filtered.csv", index=False) #saves results to new csv after prune

df = pd.read_csv("diabetes_filtered.csv")

encoded_cats = [
    'family_history_diabetes',
    'hypertension_history',
    'cardiovascular_history',
    'diagnosed_diabetes'
]

for col in encoded_cats:
    df[col] = df[col].astype('category') #manually set encoded categorical columns to not be numerical

categorical_cols = df.select_dtypes(include=['object', 'category']).columns
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns

print("Categorical Columns:")
print(categorical_cols)

print("\nNumerical Columns:")
print(numerical_cols)

df.info() #general overview + shows no missing values as total number of entries = number of non null

print("Duplicate rows:", df.duplicated().sum())

df.describe(include=['int64', 'float64']).T #descriptive stats of numerical data

"""## Diagrams and feature engineering"""

# keep only 3 classes
df = df[df["diabetes_stage"].isin(["No Diabetes", "Pre-Diabetes", "Type 2"])].copy()


# map labels to numbers
label_map = {
    "No Diabetes": 0,
    "Pre-Diabetes": 1,
    "Type 2": 2
}
df["label"] = df["diabetes_stage"].map(label_map)

# drop columns
df = df.drop(["diabetes_stage", "diagnosed_diabetes", "diabetes_risk_score"], axis=1, errors="ignore")

###### Feature Engineering ######

# glucose difference
df["glucose_diff"] = df["glucose_postprandial"] - df["glucose_fasting"]

# cholesterol ratio
df["cholesterol_ratio"] = df["ldl_cholesterol"] / df["hdl_cholesterol"]

# lifestyle score
df["lifestyle_score"] = (
    df["diet_score"]
    + df["physical_activity_minutes_per_week"] / 100
    + df["sleep_hours_per_day"]
    - df["screen_time_hours_per_day"]
)

# encode categorical columns into numeric
df_encoded = df.copy()

cat_cols = [
    "gender",
    "ethnicity",
    "employment_status",
    "smoking_status",
    "income_level",
    "education_level"
]

for col in cat_cols:
    freq_map = df_encoded[col].value_counts(normalize=True)
    df_encoded[col] = df_encoded[col].map(freq_map)


###### Pearson Correlation for Multicollinearity ######

# correlation matrix before dropping features
corr_matrix = df_encoded.corr(method="pearson")

# heatmap before dropping features
corr = df_encoded.drop(columns=["label"]).corr(method="pearson")
plt.figure(figsize=(30, 20))
sns.heatmap(corr, cmap="coolwarm", annot=False)
plt.title("Feature Correlation Heatmap")
plt.show()

# find highly correlated feature pairs
threshold = 0.8
high_corr_pairs = []

for i in range(len(corr_matrix.columns)):
    for j in range(i):
        corr_value = corr_matrix.iloc[i, j]
        if abs(corr_value) > threshold:
            high_corr_pairs.append((
                corr_matrix.columns[i],
                corr_matrix.columns[j],
                corr_value
            ))

print("Highly correlated feature pairs (|r| > 0.8):")
for f1, f2, r in high_corr_pairs:
    print(f"{f1} vs {f2} collinearity: {r:.3f}")

# drop one feature from each highly correlated pair
# keep the one more correlated with the label
to_drop = set()

for f1, f2, r in high_corr_pairs:
    corr_f1 = abs(corr_matrix.loc[f1, "label"])
    corr_f2 = abs(corr_matrix.loc[f2, "label"])

    if corr_f1 < corr_f2:
        to_drop.add(f1)
    else:
        to_drop.add(f2)

# never drop label
to_drop.discard("label")

print("\nDropped Features due to multicollinearity:")
print(list(to_drop))
df_encoded = df_encoded.drop(columns=list(to_drop), errors="ignore")


###### Feature Selection ######

# compute feature correlation with label
corr_with_label = df_encoded.corr()["label"].abs().sort_values(ascending=False)

# remove label itself
feature_ranking = corr_with_label.drop("label")

# test different numbers of features
results = []
best_k = 0
best_score = 0

# try all features
max_k = len(feature_ranking)

X = df_encoded[feature_ranking.index]
y = df_encoded["label"]

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scaler = StandardScaler()  # initialize scaler

for k in range(5, max_k+1):  # start from 5 features
    top_features = feature_ranking.head(k).index
    acc_scores = []

    for train_index, test_index in kf.split(X, y):
        X_train, X_test = X.iloc[train_index][top_features], X.iloc[test_index][top_features]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # scale within each fold
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = LogisticRegression(max_iter=2000)
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        acc_scores.append(accuracy_score(y_test, y_pred))

    mean_acc = np.mean(acc_scores)
    results.append((k, mean_acc))

    if mean_acc > best_score:
        best_score = mean_acc
        best_k = k

print("\nAccuracy for different numbers of features (5-fold CV):")
for k, acc in results:
    print(f"{k} features: {acc:.4f}")

print("\nBest number of features:", best_k)
print("Best CV accuracy:", round(best_score, 4))

# plot performance vs number of features
ks = [r[0] for r in results]
scores = [r[1] for r in results]

plt.figure(figsize=(8, 5))
plt.plot(ks, scores, marker="o")
plt.title("Average CV Accuracy vs Number of Features")
plt.xlabel("Number of Features")
plt.ylabel("Average CV Accuracy")
plt.grid(True)
plt.show()

# select best features
selected_features = feature_ranking.head(best_k).index.tolist()

# only keep these features + label
df_selected = df_encoded[selected_features + ["label"]]

print("\nSelected Features:")
print(selected_features)

###### Class Distribution Plot ######

counts = df_selected["label"].value_counts().sort_index()

plt.figure(figsize=(6, 4))
plt.bar(["No Diabetes", "Prediabetes", "Type 2"], counts)
plt.title("Class Distribution")
plt.xlabel("Class")
plt.ylabel("Count")
plt.show()

"""##Modeling algorithm and evaluation"""

CLASS_NAMES = ["No Diabetes", "Pre-Diabetes", "Type 2"]
X = df_encoded[selected_features].values
y = df_encoded["label"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

print(f"Train: {len(X_train)} samples  |  Test: {len(X_test)} samples\n")

# train 4 models
models = {
    "Logistic Regression": LogisticRegression(max_iter=2000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(max_depth=8, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    "XGBoost": XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric='mlogloss',
        random_state=42,
        n_jobs=-1
    )
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy", n_jobs=-1)
    results[name] = {
        "model":    model,
        "y_pred":   y_pred,
        "acc":      accuracy_score(y_test, y_pred),
        "f1_macro": f1_score(y_test, y_pred, average="macro"),
        "cv":       cv_scores.mean(),
        "cv_std":   cv_scores.std(),
        "cm":       confusion_matrix(y_test, y_pred),
        "report":   classification_report(y_test, y_pred, target_names=CLASS_NAMES, output_dict=True),
    }
model_names = list(results.keys())

print(f"{'Model':<22} {'Accuracy':>10} {'Macro F1':>10} {'CV':>8}")
for name, r in results.items():
    print(f"{name:<22} {r['acc']:>10.4f} {r['f1_macro']:>10.4f} {r['cv']:>8.4f}")
best = max(results, key=lambda n: results[n]["f1_macro"])
print(f"\nBest model: {best}  (Macro F1 = {results[best]['f1_macro']:.4f})")

# model comparison (Test vs CV Accuracy)
fig, ax = plt.subplots(figsize=(10, 5))
x, w = np.arange(len(model_names)), 0.35
b1 = ax.bar(x - w/2, [results[m]["acc"] for m in model_names], w,
            label="Test Accuracy", color="steelblue")
b2 = ax.bar(x + w/2, [results[m]["cv"]  for m in model_names], w,
            label="CV Accuracy (5-fold)", color="darkorange",
            yerr=[results[m]["cv_std"] for m in model_names], capsize=4)
ax.set_xticks(x); ax.set_xticklabels(model_names)
ax.set_ylim(0, 1.1); ax.set_ylabel("Accuracy")
ax.set_title("Model Comparison: Test vs Cross-Validation Accuracy")
ax.legend(); ax.bar_label(b1, fmt="%.3f", padding=3); ax.bar_label(b2, fmt="%.3f", padding=3)
plt.tight_layout()
plt.show()

# confusion matrices
fig, axes = plt.subplots(1, 4, figsize=(20, 4))
for ax, name in zip(axes, model_names):
    cm = results[name]["cm"]
    sns.heatmap(cm.astype(float) / cm.sum(axis=1, keepdims=True),
                annot=cm, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                ax=ax, cbar=False)
    ax.set_title(f"{name}\nAccuracy = {results[name]['acc']:.3f}")
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
plt.suptitle("Confusion Matrices", fontsize=13)
plt.tight_layout()
plt.show()

# F1 score for each class
colors = ["steelblue", "darkorange", "green"]
fig, ax = plt.subplots(figsize=(10, 5))
x, w = np.arange(len(model_names)), 0.25
for i, cls in enumerate(CLASS_NAMES):
    ax.bar(x + i*w, [results[m]["report"][cls]["f1-score"] for m in model_names],
           w, label=cls, color=colors[i])
ax.set_xticks(x + w); ax.set_xticklabels(model_names)
ax.set_ylim(0, 1.1); ax.set_ylabel("F1 Score")
ax.set_title("F1 Score by Model for each class"); ax.legend()
plt.tight_layout()
plt.show()

# ROC curves
y_bin = label_binarize(y_test, classes=[0, 1, 2])
fig, axes = plt.subplots(1, 4, figsize=(20, 4))
for ax, name in zip(axes, model_names):
    y_prob = results[name]["model"].predict_proba(X_test)
    for i, cls in enumerate(CLASS_NAMES):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
        ax.plot(fpr, tpr, color=colors[i], lw=2,
                label=f"{cls} (AUC={auc(fpr,tpr):.2f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_title(name); ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.legend(fontsize=8); ax.set_xlim([0, 1]); ax.set_ylim([0, 1.05])
plt.suptitle("ROC Curves", fontsize=13)
plt.tight_layout()
plt.show()

# Feature Importance — Random Forest
rf = results["Random Forest"]["model"]
imp = pd.Series(rf.feature_importances_, index=selected_features).sort_values()
fig, ax = plt.subplots(figsize=(10, 7))
imp.plot.barh(ax=ax, color="steelblue")

# Add percentage labels to the end of each bar
total = imp.sum()
for bar, value in zip(ax.patches, imp.values):
    pct = value / total
    ax.text(
        bar.get_width() + 0.001,
        bar.get_y() + bar.get_height() / 2,
        f"{pct:.4f}",
        va="center", fontsize=9
    )

ax.set_xlim(right=ax.get_xlim()[1] * 1.1)
ax.set_title("Important Features — Random Forest", fontsize=13)
ax.set_xlabel("Importance")
plt.tight_layout()
plt.show()

# summary
print(f"{'Model':<22} {'Test Accuracy':>9} {'CV Accuracy':>8} {'Macro F1':>9}")
for name in model_names:
    r = results[name]["report"]
    print(f"{name:<22} {results[name]['acc']:>13.4f} {results[name]['cv']:>11.4f} {r['macro avg']['f1-score']:>9.4f}")

best = max(results, key=lambda m: results[m]["acc"])
print(f"\nBest model: {best}  (Accuracy = {results[best]['acc']:.4f})")
print("\n" + classification_report(y_test, results[best]["y_pred"], target_names=CLASS_NAMES))

"""## Model visualization and hyperparameter optimization"""

# Create a pipeline for scaling
# Currently not using this in the script
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("logreg", LogisticRegression(random_state = 42))
])

# Logistic Regression hyperparameter tuning

# Define hyperparameters (will be used for all search algorithms)
lg_hyperparameters = {
    'penalty': [None, 'l2'],
    'tol': [0.005, 0.01, 0.025],
    'C': [0.01, 0.1, 1, 10, 100],
    'solver': ['sag', 'saga'],
    'max_iter': [25, 50, 100, 250, 500]
}

# Define our model
logistic_regression = LogisticRegression(random_state = 42);

# Grid Search

# Perform grid search on our training data
# Accuracy
lg_grid_search_acc = GridSearchCV(logistic_regression, lg_hyperparameters, n_jobs = -1, cv = 5, refit = 'accuracy',
                              scoring = {'f1_macro': 'f1_macro', 'accuracy': 'accuracy'}, verbose = 2);
lg_grid_search_acc.fit(X_train, y_train);

# Macro F1
lg_grid_search_f1 = GridSearchCV(logistic_regression, lg_hyperparameters, n_jobs = -1, cv = 5, refit = 'f1_macro',
                              scoring = {'f1_macro': 'f1_macro', 'accuracy': 'accuracy'}, verbose = 2);
lg_grid_search_f1.fit(X_train, y_train);

# Print out results
print("Logistic Regression Hyperparameter Tuning Results (Grid Search)\n")

print("Accuracy")
print("Top Hyperparameters:", lg_grid_search_acc.best_params_)
print(f"Top CV Scores: {lg_grid_search_acc.best_score_:.4f}")
print(f"Test Accuracy: {lg_grid_search_acc.score(X_test, y_test):.4f}")

print("\nMacro F1")
print("Top Hyperparameters:", lg_grid_search_f1.best_params_)
print(f"Top CV Scores: {lg_grid_search_f1.best_score_:.4f}")
print(f"Test Accuracy: {lg_grid_search_f1.score(X_test, y_test):.4f}")

# Randomized Search

# Perform randomized search on our training data
# Accuracy
lg_random_search_acc = RandomizedSearchCV(logistic_regression, lg_hyperparameters, n_jobs = -1, cv = 5, refit = 'accuracy',
                                      scoring = {'f1_macro': 'f1_macro', 'accuracy': 'accuracy'}, verbose = 2);
lg_random_search_acc.fit(X_train, y_train);

# Macro F1
lg_random_search_f1 = RandomizedSearchCV(logistic_regression, lg_hyperparameters, n_jobs = -1, cv = 5, refit = 'f1_macro',
                                      scoring = {'f1_macro': 'f1_macro', 'accuracy': 'accuracy'}, verbose = 2);
lg_random_search_f1.fit(X_train, y_train);

# Print out results
print("Logistic Regression Hyperparameter Tuning Results (Randomized Search)\n")

print("Accuracy")
print("Top Hyperparameters:", lg_random_search_acc.best_params_)
print(f"Top CV Scores: {lg_random_search_acc.best_score_:.4f}")
print(f"Test Accuracy: {lg_random_search_acc.score(X_test, y_test):.4f}")

print("\nMacro F1")
print("Top Hyperparameters:", lg_random_search_f1.best_params_)
print(f"Top CV Scores: {lg_random_search_f1.best_score_:.4f}")
print(f"Test Accuracy: {lg_random_search_f1.score(X_test, y_test):.4f}")

# Decision Tree hyperparameter tuning

# Define hyperparameters (will be used for all search algorithms)
dt_hyperparameters = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [3, 5, 7],
    'min_samples_split': [2, 3, 5],
    'min_samples_leaf': [1, 2, 3],
    'max_features': [None, 'sqrt', 'log2']
}

# Define our model
decision_tree = DecisionTreeClassifier(random_state = 42)

# Grid Search

# Perform grid search on our training data
# Accuracy
dt_grid_search_acc = GridSearchCV(decision_tree, dt_hyperparameters, n_jobs = -1, cv = 5, refit = 'accuracy',
                           scoring = {'f1_macro': 'f1_macro', 'accuracy': 'accuracy'}, verbose = 2);
dt_grid_search_acc.fit(X_train, y_train);

# Macro F1
dt_grid_search_f1 = GridSearchCV(decision_tree, dt_hyperparameters, n_jobs = -1, cv = 5, refit = 'f1_macro',
                           scoring = {'f1_macro': 'f1_macro', 'accuracy': 'accuracy'}, verbose = 2);
dt_grid_search_f1 .fit(X_train, y_train);

# Print out results
print("Decision Tree Hyperparameter Tuning Results (Grid Search)")

print("Accuracy")
print("Top Hyperparameters:", dt_grid_search_acc.best_params_)
print(f"Top CV Scores: {dt_grid_search_acc.best_score_:.4f}")
print(f"Test Accuracy: {dt_grid_search_acc.score(X_test, y_test):.4f}")

print("\nMacro F1")
print("Top Hyperparameters:", dt_grid_search_f1.best_params_)
print(f"Top CV Scores: {dt_grid_search_f1.best_score_:.4f}")
print(f"Test Accuracy: {dt_grid_search_f1.score(X_test, y_test):.4f}")

# Other possible tuning options
# Bayesian Optimization

# Randomized Search

# Perform randomized search on our training data
# Accuracy
dt_random_search_acc = RandomizedSearchCV(decision_tree, dt_hyperparameters, n_jobs = -1, cv = 5, refit = 'accuracy',
                           scoring = {'f1_macro': 'f1_macro', 'accuracy': 'accuracy'}, verbose = 2);
dt_random_search_acc.fit(X_train, y_train);

# Macro F1
dt_random_search_f1 = RandomizedSearchCV(decision_tree, dt_hyperparameters, n_jobs = -1, cv = 5, refit = 'f1_macro',
                           scoring = {'f1_macro': 'f1_macro', 'accuracy': 'accuracy'}, verbose = 2);
dt_random_search_f1.fit(X_train, y_train);

# Print out results
print("Decision Tree Hyperparameter Tuning Results (Randomized Search)\n")

print("Accuracy")
print("Top Hyperparameters:", dt_random_search_acc.best_params_)
print(f"Top CV Scores: {dt_random_search_acc.best_score_:.4f}")
print(f"Test Accuracy: {dt_random_search_acc.score(X_test, y_test):.4f}")

print("\nMacro F1")
print("Top Hyperparameters:", dt_random_search_f1.best_params_)
print(f"Top CV Scores: {dt_random_search_f1.best_score_:.4f}")
print(f"Test Accuracy: {dt_random_search_f1.score(X_test, y_test):.4f}")

# Random Forest hyperparameter tuning

# Define hyperparameters (will be used for all search algorithms)
rf_hyperparameters = {
    'n_estimators': [100, 150, 200],
    'criterion': ['gini', 'entropy'],
    'max_depth': [10, 15, 20],
    'min_samples_split': [2, 3, 5],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True]
}

# Define our model
random_forest = RandomForestClassifier(random_state = 42)

# Grid Search

# Perform randomized search on our training data
# Accuracy
rf_grid_search_acc = GridSearchCV(random_forest, rf_hyperparameters, n_jobs = -1, cv = 5, refit = 'accuracy',
                           scoring = {'f1_macro': 'f1_macro', 'accuracy': 'accuracy'}, verbose = 2);
rf_grid_search_acc.fit(X_train, y_train);

# Macro F1
rf_grid_search_f1 = GridSearchCV(random_forest, rf_hyperparameters, n_jobs = -1, cv = 5, refit = 'f1_macro',
                           scoring = {'f1_macro': 'f1_macro', 'accuracy': 'accuracy'}, verbose = 2);
rf_grid_search_f1.fit(X_train, y_train);

# Print out results
print("Random Forest Hyperparameter Tuning Results (Grid Search)\n")

print("Accuracy")
print("Top Hyperparameters:", rf_grid_search_acc.best_params_)
print(f"Top CV Scores: {rf_grid_search_acc.best_score_:.4f}")
print(f"Test Accuracy: {rf_grid_search_acc.score(X_test, y_test):.4f}")

print("\nMacro F1")
print("Top Hyperparameters:", rf_grid_search_f1.best_params_)
print(f"Top CV Scores: {rf_grid_search_f1.best_score_:.4f}")
print(f"Test Accuracy: {rf_grid_search_f1.score(X_test, y_test):.4f}")

# Randomized Search

# Perform randomized search on our training data
# Accuracy
rf_random_search_acc = RandomizedSearchCV(random_forest, rf_hyperparameters, n_jobs = -1, cv = 5, refit = 'f1_macro',
                           scoring = {'f1_macro': 'f1_macro', 'accuracy': 'accuracy'}, verbose = 2);
rf_random_search_acc.fit(X_train, y_train);

# Macro F1
rf_random_search_f1 = RandomizedSearchCV(random_forest, rf_hyperparameters, n_jobs = -1, cv = 5, refit = 'f1_macro',
                           scoring = {'f1_macro': 'f1_macro', 'accuracy': 'accuracy'}, verbose = 2);
rf_random_search_f1.fit(X_train, y_train);

# Print out results
print("Random Forest Hyperparameter Tuning Results (Randomized Search)\n")

print("Accuracy")
print("Top Hyperparameters:", rf_random_search_acc.best_params_)
print(f"Top CV Scores: {rf_random_search_acc.best_score_:.4f}")
print(f"Test Accuracy: {rf_random_search_acc.score(X_test, y_test):.4f}")

print("\nMacro F1")
print("Top Hyperparameters:", rf_random_search_f1.best_params_)
print(f"Top CV Scores: {rf_random_search_f1.best_score_:.4f}")
print(f"Test Accuracy: {rf_random_search_f1.score(X_test, y_test):.4f}")

# XGBoost hyperparameter tuning

# Define hyperparameters (will be used for all search algorithms)
xgb_hyperparameters = {
    'n_estimators': [100, 150, 200],
    'max_depth': [4, 5, 6],
    'learning_rate': [0.05, 0.1],
    'subsample': [0.6, 0.8],
    'colsample_bytree': [0.6, 0.8],
    'min_child_weight': [1, 3, 5],
    'eval_metric': ['mlogloss', 'auc']
}

# Define our model
xgb = XGBClassifier(random_state = 42)

# Grid Search

# Perform randomized search on our training data
# Accuracy
xgb_grid_search_acc = GridSearchCV(xgb, xgb_hyperparameters, n_jobs = -1, cv = 5, refit = 'accuracy',
                           scoring = {'f1_macro': 'f1_macro', 'accuracy': 'accuracy'}, verbose = 2);
xgb_grid_search_acc.fit(X_train, y_train);

# Macro F1
xgb_grid_search_f1 = GridSearchCV(xgb, xgb_hyperparameters, n_jobs = -1, cv = 5, refit = 'f1_macro',
                           scoring = {'f1_macro': 'f1_macro', 'accuracy': 'accuracy'}, verbose = 2);
xgb_grid_search_f1.fit(X_train, y_train);

# Print out results
print("XGBoost Hyperparameter Tuning Results (Grid Search)\n")

print("Accuracy")
print("Top Hyperparameters:", xgb_grid_search_acc.best_params_)
print(f"Top CV Scores: {xgb_grid_search_acc.best_score_:.4f}")
print(f"Test Accuracy: {xgb_grid_search_acc.score(X_test, y_test):.4f}")

print("\nMacro F1")
print("Top Hyperparameters:", xgb_grid_search_f1.best_params_)
print(f"Top CV Scores: {xgb_grid_search_f1.best_score_:.4f}")
print(f"Test Accuracy: {xgb_grid_search_f1.score(X_test, y_test):.4f}")

# Randomized Search

# Perform randomized search on our training data
# Accuracy
xgb_random_search_acc = RandomizedSearchCV(xgb, xgb_hyperparameters, n_jobs = -1, cv = 5, refit = 'f1_macro',
                           scoring = {'f1_macro': 'f1_macro', 'accuracy': 'accuracy'}, verbose = 2);
xgb_random_search_acc.fit(X_train, y_train);

# Macro F1
xgb_random_search_f1 = RandomizedSearchCV(xgb, xgb_hyperparameters, n_jobs = -1, cv = 5, refit = 'f1_macro',
                           scoring = {'f1_macro': 'f1_macro', 'accuracy': 'accuracy'}, verbose = 2);
xgb_random_search_f1.fit(X_train, y_train);

# Print out results
print("Random Forest Hyperparameter Tuning Results (Randomized Search)\n")

print("Accuracy")
print("Top Hyperparameters:", xgb_random_search_acc.best_params_)
print(f"Top CV Scores: {xgb_random_search_acc.best_score_:.4f}")
print(f"Test Accuracy: {xgb_random_search_acc.score(X_test, y_test):.4f}")

print("\nMacro F1")
print("Top Hyperparameters:", xgb_random_search_f1.best_params_)
print(f"Top CV Scores: {xgb_random_search_f1.best_score_:.4f}")
print(f"Test Accuracy: {xgb_random_search_f1.score(X_test, y_test):.4f}")

# Consolidate results
# Grid Search CV Accuracy & Test
results_grid_acc = {'Logistic Regression': lg_grid_search_acc.best_score_, 'Decision Tree': dt_grid_search_acc.best_score_,
                    'Random Forest': rf_grid_search_acc.best_score_, 'XGBoost': xgb_grid_search_acc.best_score_}
results_grid_acc_test = {'Logistic Regression': lg_grid_search_acc.score(X_test, y_test), 'Decision Tree': dt_grid_search_acc.score(X_test, y_test),
                         'Random Forest': rf_grid_search_acc.score(X_test, y_test), 'XGBoost': xgb_grid_search_acc.score(X_test, y_test)}

# Grid Search CV Macro F1 & Test
results_grid_f1 = {'Logistic Regression': lg_grid_search_f1.best_score_, 'Decision Tree': dt_grid_search_f1.best_score_,
                   'Random Forest': rf_grid_search_f1.best_score_, 'XGBoost': xgb_grid_search_f1.best_score_}
results_grid_f1_test = {'Logistic Regression': lg_grid_search_f1.score(X_test, y_test), 'Decision Tree': dt_grid_search_f1.score(X_test, y_test),
                        'Random Forest': rf_grid_search_f1.score(X_test, y_test), 'XGBoost': xgb_grid_search_f1.score(X_test, y_test)}

# Random Search CV Accuracy & Test
results_random_acc = {'Logistic Regression': lg_random_search_acc.best_score_, 'Decision Tree': dt_random_search_acc.best_score_,
                      'Random Forest': rf_random_search_acc.best_score_, 'XGBoost': xgb_random_search_acc.best_score_}
results_random_acc_test = {'Logistic Regression': lg_random_search_acc.score(X_test, y_test), 'Decision Tree': dt_random_search_acc.score(X_test, y_test),
                           'Random Forest': rf_random_search_acc.score(X_test, y_test), 'XGBoost': xgb_random_search_acc.score(X_test, y_test)}

# Random Search CV Macro F1 & Test
results_random_f1 = {'Logistic Regression': lg_random_search_f1.best_score_, 'Decision Tree': dt_random_search_f1.best_score_,
                     'Random Forest': rf_random_search_f1.best_score_, 'XGBoost': xgb_random_search_f1.best_score_}
results_random_f1_test = {'Logistic Regression': lg_random_search_f1.score(X_test, y_test), 'Decision Tree': dt_random_search_f1.score(X_test, y_test),
                          'Random Forest': rf_random_search_f1.score(X_test, y_test), 'XGBoost': xgb_random_search_f1.score(X_test, y_test)}

# Grid Search results comparison (CV Accuracy vs Test Accuracy)
fig, ax = plt.subplots(figsize=(10, 5))
x, w = np.arange(len(model_names)), 0.35
b1 = ax.bar(x - w/2, [results_grid_acc[m] for m in model_names], w,
            label="CV Accuracy (5-fold)", color="steelblue")
b2 = ax.bar(x + w/2, [results_grid_acc_test[m] for m in model_names], w,
            label="Test Accuracy", color="darkorange")
ax.set_xticks(x); ax.set_xticklabels(model_names)
ax.set_ylim(0, 1.1); ax.set_ylabel("Accuracy")
ax.set_title("Grid Search Comparison: Cross-Validation vs Test Accuracy")
ax.legend(); ax.bar_label(b1, fmt="%.3f", padding=3); ax.bar_label(b2, fmt="%.3f", padding=3)
plt.tight_layout()
plt.show()

# Grid Search results comparison (CV Macro F1 vs Test Macro F1)
fig, ax = plt.subplots(figsize=(10, 5))
x, w = np.arange(len(model_names)), 0.35
b1 = ax.bar(x - w/2, [results_grid_f1[m] for m in model_names], w,
            label="CV Macro F1 (5-fold)", color="steelblue")
b2 = ax.bar(x + w/2, [results_grid_f1_test[m] for m in model_names], w,
            label="Test Macro F1", color="darkorange")
ax.set_xticks(x); ax.set_xticklabels(model_names)
ax.set_ylim(0, 1.1); ax.set_ylabel("Macro F1")
ax.set_title("Grid Search Comparison: Cross-Validation vs Test Macro F1")
ax.legend(); ax.bar_label(b1, fmt="%.3f", padding=3); ax.bar_label(b2, fmt="%.3f", padding=3)
plt.tight_layout()
plt.show()

# Random Search results comparison (CV Accuracy vs Test Accuracy)
fig, ax = plt.subplots(figsize=(10, 5))
x, w = np.arange(len(model_names)), 0.35
b1 = ax.bar(x - w/2, [results_random_acc[m] for m in model_names], w,
            label="CV Accuracy (5-fold)", color="steelblue")
b2 = ax.bar(x + w/2, [results_random_acc_test[m] for m in model_names], w,
            label="Test Accuracy", color="darkorange")
ax.set_xticks(x); ax.set_xticklabels(model_names)
ax.set_ylim(0, 1.1); ax.set_ylabel("Accuracy")
ax.set_title("Random Search Comparison: Cross-Validation vs Test Accuracy")
ax.legend(); ax.bar_label(b1, fmt="%.3f", padding=3); ax.bar_label(b2, fmt="%.3f", padding=3)
plt.tight_layout()
plt.show()

# Random Search results comparison (CV Macro F1 vs Test Macro F1)
fig, ax = plt.subplots(figsize=(10, 5))
x, w = np.arange(len(model_names)), 0.35
b1 = ax.bar(x - w/2, [results_random_f1[m] for m in model_names], w,
            label="CV Macro F1 (5-fold)", color="steelblue")
b2 = ax.bar(x + w/2, [results_random_f1_test[m] for m in model_names], w,
            label="Test Macro F1", color="darkorange")
ax.set_xticks(x); ax.set_xticklabels(model_names)
ax.set_ylim(0, 1.1); ax.set_ylabel("Macro F1")
ax.set_title("Random Search Comparison: Cross-Validation vs Test Macro F1")
ax.legend(); ax.bar_label(b1, fmt="%.3f", padding=3); ax.bar_label(b2, fmt="%.3f", padding=3)
plt.tight_layout()
plt.show()

# Grid Search vs Random Search results comparison (CV Accuracy)
fig, ax = plt.subplots(figsize=(10, 5))
x, w = np.arange(len(model_names)), 0.35
b1 = ax.bar(x - w/2, [results_grid_acc[m] for m in model_names], w,
            label="Grid Search", color="steelblue")
b2 = ax.bar(x + w/2, [results_random_acc[m] for m in model_names], w,
            label="Random Search", color="darkorange")
ax.set_xticks(x); ax.set_xticklabels(model_names)
ax.set_ylim(0, 1.1); ax.set_ylabel("Accuracy")
ax.set_title("Grid Search vs Random Search: Cross-Validation (5-fold) Accuracy Results")
ax.legend(); ax.bar_label(b1, fmt="%.3f", padding=3); ax.bar_label(b2, fmt="%.3f", padding=3)
plt.tight_layout()
plt.show()

# Grid Search vs Random Search results comparison (Test Accuracy)
fig, ax = plt.subplots(figsize=(10, 5))
x, w = np.arange(len(model_names)), 0.35
b1 = ax.bar(x - w/2, [results_grid_acc_test[m] for m in model_names], w,
            label="Grid Search", color="steelblue")
b2 = ax.bar(x + w/2, [results_random_acc_test[m] for m in model_names], w,
            label="Random Search", color="darkorange")
ax.set_xticks(x); ax.set_xticklabels(model_names)
ax.set_ylim(0, 1.1); ax.set_ylabel("Accuracy")
ax.set_title("Grid Search vs Random Search: Test (5-fold) Accuracy Results")
ax.legend(); ax.bar_label(b1, fmt="%.3f", padding=3); ax.bar_label(b2, fmt="%.3f", padding=3)
plt.tight_layout()
plt.show()

# Grid Search vs Random Search results comparison (CV Macro F1)
fig, ax = plt.subplots(figsize=(10, 5))
x, w = np.arange(len(model_names)), 0.35
b1 = ax.bar(x - w/2, [results_grid_f1[m] for m in model_names], w,
            label="Grid Search", color="steelblue")
b2 = ax.bar(x + w/2, [results_random_f1[m] for m in model_names], w,
            label="Random Search", color="darkorange")
ax.set_xticks(x); ax.set_xticklabels(model_names)
ax.set_ylim(0, 1.1); ax.set_ylabel("Macro F1")
ax.set_title("Grid Search vs Random Search: Cross-Validation (5-fold) Macro F1 Results")
ax.legend(); ax.bar_label(b1, fmt="%.3f", padding=3); ax.bar_label(b2, fmt="%.3f", padding=3)
plt.tight_layout()
plt.show()

# Grid Search vs Random Search results comparison (Test Macro F1)
fig, ax = plt.subplots(figsize=(10, 5))
x, w = np.arange(len(model_names)), 0.35
b1 = ax.bar(x - w/2, [results_grid_f1_test[m] for m in model_names], w,
            label="Grid Search", color="steelblue")
b2 = ax.bar(x + w/2, [results_random_f1_test[m] for m in model_names], w,
            label="Random Search", color="darkorange")
ax.set_xticks(x); ax.set_xticklabels(model_names)
ax.set_ylim(0, 1.1); ax.set_ylabel("Macro F1")
ax.set_title("Grid Search vs Random Search: Test (5-fold) Macro F1 Results")
ax.legend(); ax.bar_label(b1, fmt="%.3f", padding=3); ax.bar_label(b2, fmt="%.3f", padding=3)
plt.tight_layout()
plt.show()

# Consolidate best hyperparameters from tuning
best_params_grid_acc = [lg_grid_search_acc.best_params_, dt_grid_search_acc.best_params_, rf_grid_search_acc.best_params_, xgb_grid_search_acc.best_params_]
best_params_random_acc = [lg_random_search_acc.best_params_, dt_random_search_acc.best_params_, rf_random_search_acc.best_params_, xgb_random_search_acc.best_params_]
best_params_grid_f1 = [lg_grid_search_f1.best_params_, dt_grid_search_f1.best_params_, rf_grid_search_f1.best_params_, xgb_grid_search_f1.best_params_]
best_params_random_f1 = [lg_random_search_f1.best_params_, dt_random_search_f1.best_params_, rf_random_search_f1.best_params_, xgb_random_search_f1.best_params_]

# Summarize the best hyperparameters for each model
print("Best Hyperparameters (CV 5-fold)\n")
print("Grid Search Hyperparameters (Accuracy)")
for name, params in zip(model_names, best_params_grid_acc):
  print(f"{name:<22}-> {params}")

print("\nRandom Search Hyperparameters (Accuracy)")
for name, params in zip(model_names, best_params_random_acc):
  print(f"{name:<22}-> {params}")

print("\nGrid Search Hyperparameters (Macro F1)")
for name, params in zip(model_names, best_params_grid_f1):
  print(f"{name:<22}-> {params}")

print("\nRandom Search Hyperparameters (Macro F1)")
for name, params in zip(model_names, best_params_random_f1):
  print(f"{name:<22}-> {params}")

# Summarize results after hyperparameter tuning
print("Hyperparameter Tuning Summary (CV 5-fold)\n")

print("Grid Search Results")
print(f"{'Model':<22} {'Accuracy':>10} {'Macro F1':>10} {'Test Accuracy':>15} {'Test Macro F1':>15} ")
for name, acc, f1, acc_test, f1_test in zip(model_names, results_grid_acc, results_grid_f1, results_grid_acc_test, results_grid_f1_test):
    print(f"{name:<22} {results_grid_acc[acc]:>9.5f} {results_grid_f1[f1]:>10.5f} {results_grid_acc_test[acc_test]:>10.5f} {results_grid_f1_test[f1_test]:>15.5f}")

print("\nRandom Search Results")
print(f"{'Model':<22} {'Accuracy':>10} {'Macro F1':>10} {'Test Accuracy':>15} {'Test Macro F1':>15} ")
for name, acc, f1, acc_test, f1_test in zip(model_names, results_random_acc, results_random_f1, results_random_acc_test, results_random_f1_test):
    print(f"{name:<22} {results_random_acc[acc]:>9.5f} {results_random_f1[f1]:>10.5f} {results_random_acc_test[acc_test]:>10.5f} {results_random_f1_test[f1_test]:>15.5f}")

print("\nPrevious Best Model Reults")

best = max(results, key=lambda n: results[n]["f1_macro"])
print(f"Best Model: {best}  (Macro F1 = {results[best]['f1_macro']:.5f})")

best = max(results, key=lambda m: results[m]["acc"])
print(f"Best Model: {best}  (Test Accuracy = {results[best]['acc']:.5f})")

print("\nNew Best Model Results")

best_acc = max(results_grid_acc, key = lambda m: results_grid_acc[m])
best_f1 = max(results_grid_f1, key = lambda m: results_grid_f1[m])
best_acc_test = max(results_grid_acc_test, key = lambda m: results_grid_acc_test[m])
best_f1_test = max(results_grid_f1_test, key = lambda m: results_grid_f1_test[m])

print(f"Best Model: {best_acc}  (Accuracy = {results_grid_acc[best_acc]:.5f})")
print(f"Best Model: {best_f1}  (Macro F1 = {results_grid_f1[best_f1]:.5f})")
print(f"Best Model: {best_acc_test}  (Test Accuracy = {results_grid_acc_test[best_acc_test]:.5f})")
print(f"Best Model: {best_f1_test}  (Test Macro F1 = {results_grid_f1_test[best_f1_test]:.5f})")