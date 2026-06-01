# ============================================================
# HEART FAILURE PREDICTION — LEAF-WISE GRADIENT BOOSTING MODEL
# Algorithm  : LightGBM (Light Gradient Boosting Machine)
# Tuning     : RandomizedSearchCV (5-fold cross-validation)
# Dataset    : heart_failure_clinical_records.csv (5000 records)
# Output     : lgbm_model.pkl  (model + scaler saved for Streamlit)
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report,
                             roc_auc_score, roc_curve)
import lightgbm as lgb

pd.set_option('display.max_rows', 10)
pd.set_option('display.max_columns', 10)

# ─────────────────────────────────────────────
# 1. LOAD DATASET
# ─────────────────────────────────────────────
print("=" * 65)
print("  LIGHTGBM — HEART FAILURE PREDICTION")
print("=" * 65)

df = pd.read_csv("heart_failure_clinical_records.csv")

print("\n[1] ORIGINAL DATASET (before exclusion)")
print(f"    Shape : {df.shape[0]} records x {df.shape[1]} features")
print(df)

# ─────────────────────────────────────────────
# 2. VARIABLE EXCLUSION
# ─────────────────────────────────────────────
EXCLUDED = ["diabetes", "smoking", "time"]
df_clean = df.drop(columns=EXCLUDED)

print(f"\n[2] DATASET AFTER VARIABLE EXCLUSION")
print(f"    Excluded : {EXCLUDED}")
print(f"    Shape    : {df_clean.shape[0]} records x {df_clean.shape[1]} features")
print(df_clean)

# ─────────────────────────────────────────────
# 3. FEATURES & TARGET SPLIT
# ─────────────────────────────────────────────
X = df_clean.drop(columns=["DEATH_EVENT"])
y = df_clean["DEATH_EVENT"]

print(f"\n[3] FEATURES & TARGET")
print(f"    Features : {X.columns.tolist()}")
print(f"    Target   : DEATH_EVENT  (0 = survived, 1 = death)")
print(f"    Class distribution:")
print(f"      Survived (0) : {(y == 0).sum()} ({(y == 0).mean()*100:.1f}%)")
print(f"      Death    (1) : {(y == 1).sum()} ({(y == 1).mean()*100:.1f}%)")

# ─────────────────────────────────────────────
# 4. TRAIN / TEST SPLIT (80:20)
# ─────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n[4] TRAIN / TEST SPLIT  (80% : 20%)")
print(f"    Training set : {X_train.shape[0]} records")
print(f"    Test set     : {X_test.shape[0]} records")

# ─────────────────────────────────────────────
# 5. FEATURE SCALING (StandardScaler)
# ─────────────────────────────────────────────
scaler         = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

print(f"\n[5] FEATURE SCALING")
print(f"    Method : StandardScaler (mean=0, std=1)")
print(f"    Applied to all {X.shape[1]} input features")

# ─────────────────────────────────────────────
# 6. PARAMETER TUNING — RandomizedSearchCV
# ─────────────────────────────────────────────
print(f"\n[6] PARAMETER TUNING — RandomizedSearchCV")
print(f"    Searching best hyperparameters over 5-fold cross-validation ...")
print(f"    (This may take a few minutes — please wait)\n")

param_dist = {
    "n_estimators"     : list(range(200, 1201, 50)),
    "learning_rate"    : [0.01, 0.03, 0.05, 0.07, 0.09],
    "num_leaves"       : list(range(20, 150, 1)),
    "max_depth"        : list(range(5, 20)),
    "colsample_bytree" : [0.6, 0.7, 0.8, 0.9, 1.0],
    "subsample"        : [0.6, 0.7, 0.8, 0.9, 1.0],
    "min_child_samples": [10, 20, 30, 40, 50],
    "reg_alpha"        : [0, 0.01, 0.1, 0.5],
    "reg_lambda"       : [0, 0.01, 0.1, 0.5],
}

base_lgb = lgb.LGBMClassifier(
    boosting_type = "gbdt",
    objective     = "binary",
    random_state  = 42,
    verbose       = -1,
    n_jobs        = -1
)

cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

search = RandomizedSearchCV(
    estimator           = base_lgb,
    param_distributions = param_dist,
    n_iter              = 50,
    scoring             = "accuracy",
    cv                  = cv_strategy,
    random_state        = 42,
    n_jobs              = -1,
    verbose             = 1
)

search.fit(X_train_scaled, y_train)

best_params   = search.best_params_
best_cv_score = search.best_score_

param_order = ["colsample_bytree", "learning_rate", "max_depth",
               "min_child_samples", "n_estimators", "num_leaves",
               "reg_alpha", "reg_lambda", "subsample"]

print(f"\n    Parameter tuning complete!")
print(f"\n{'─'*65}")
print(f"    BEST PARAMETERS FOUND")
print(f"{'─'*65}")
for param in param_order:
    if param in best_params:
        print(f"    {param:<22} : {best_params[param]}")
print(f"{'─'*65}")
print(f"    Cross-validation score  : {best_cv_score*100:.2f}%")
print(f"{'─'*65}")

# ─────────────────────────────────────────────
# 7. TRAIN FINAL MODEL WITH BEST PARAMETERS
# ─────────────────────────────────────────────
print(f"\n[7] TRAINING FINAL MODEL WITH BEST PARAMETERS ...")

best_model = lgb.LGBMClassifier(
    boosting_type = "gbdt",
    objective     = "binary",
    random_state  = 42,
    verbose       = -1,
    n_jobs        = -1,
    **best_params
)

best_model.fit(X_train_scaled, y_train)
print(f"    Training complete.")

# ─────────────────────────────────────────────
# 8. EVALUATION ON TEST SET
# ─────────────────────────────────────────────
y_pred      = best_model.predict(X_test_scaled)
y_pred_prob = best_model.predict_proba(X_test_scaled)[:, 1]

accuracy  = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall    = recall_score(y_test, y_pred)
f1        = f1_score(y_test, y_pred)
auc_roc   = roc_auc_score(y_test, y_pred_prob)

print(f"\n[8] MODEL EVALUATION — TEST SET RESULTS")
print(f"{'─'*65}")
print(f"    Accuracy  : {accuracy*100:.2f}%")
print(f"    Precision : {precision*100:.2f}%")
print(f"    Recall    : {recall*100:.2f}%")
print(f"    F1 Score  : {f1*100:.2f}%")
print(f"    AUC-ROC   : {auc_roc*100:.2f}%")
print(f"{'─'*65}")

print(f"\n    Classification Report:")
print(classification_report(y_test, y_pred,
                             target_names=["Survived (0)", "Death (1)"]))

# ─────────────────────────────────────────────
# 9. OVERFITTING CHECK
# ─────────────────────────────────────────────
print(f"\n[9] OVERFITTING CHECK")
print(f"{'─'*65}")

train_pred     = best_model.predict(X_train_scaled)
train_accuracy = accuracy_score(y_train, train_pred)
test_accuracy  = accuracy_score(y_test, y_pred)
gap            = abs(train_accuracy - test_accuracy) * 100

print(f"    Train Accuracy : {train_accuracy*100:.2f}%")
print(f"    Test Accuracy  : {test_accuracy*100:.2f}%")
print(f"    Gap            : {gap:.2f}%")
print(f"{'─'*65}")

if gap <= 3:
    print(f"    Result  : NO overfitting detected — model generalises well")
    print(f"    Verdict : Results are trustworthy for new patient predictions")
elif gap <= 7:
    print(f"    Result  : MILD overfitting detected (gap = {gap:.2f}%)")
    print(f"    Verdict : Acceptable, consider increasing reg_alpha / reg_lambda")
else:
    print(f"    Result  : SIGNIFICANT overfitting detected (gap = {gap:.2f}%)")
    print(f"    Verdict : Model memorised training data — increase regularisation")

print(f"{'─'*65}")

# ─────────────────────────────────────────────
# 10. VISUALISATIONS
# ─────────────────────────────────────────────
print(f"\n[10] GENERATING VISUALISATIONS ...")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("LightGBM — Heart Failure Prediction Results",
             fontsize=14, fontweight="bold")

# ── 10a. Confusion Matrix ────────────────────
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", ax=axes[0],
            xticklabels=["Survived", "Death"],
            yticklabels=["Survived", "Death"])
axes[0].set_title("Confusion Matrix", fontweight="bold")
axes[0].set_xlabel("Predicted Label")
axes[0].set_ylabel("True Label")

# ── 10b. ROC Curve ───────────────────────────
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
axes[1].plot(fpr, tpr, color="#1D9E75", lw=2,
             label=f"LightGBM (AUC = {auc_roc:.3f})")
axes[1].plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1)
axes[1].set_xlabel("False Positive Rate")
axes[1].set_ylabel("True Positive Rate")
axes[1].set_title("ROC Curve", fontweight="bold")
axes[1].legend(loc="lower right")
axes[1].grid(alpha=0.3)

# ── 10c. Evaluation Metrics Bar Chart ────────
metric_labels = ["Accuracy", "Precision", "Recall", "F1 Score", "AUC-ROC"]
metric_values = [accuracy, precision, recall, f1, auc_roc]
bar_colors    = ["#1D9E75", "#185FA5", "#7F77DD", "#D85A30", "#BA7517"]
bars = axes[2].bar(metric_labels, [v * 100 for v in metric_values],
                   color=bar_colors, width=0.5)
axes[2].set_ylim(0, 115)
axes[2].set_ylabel("Score (%)")
axes[2].set_title("Evaluation Metrics", fontweight="bold")
axes[2].grid(axis="y", alpha=0.3)
for bar, val in zip(bars, metric_values):
    axes[2].text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 1.5,
                 f"{val*100:.1f}%",
                 ha="center", va="bottom", fontsize=9, fontweight="bold")

plt.tight_layout()
plt.savefig("lightgbm_results.png", dpi=150, bbox_inches="tight")
plt.show()
print(f"    Saved -> lightgbm_results.png")

# ── 10d. Feature Importance ──────────────────
importance = best_model.feature_importances_
feat_df    = pd.DataFrame({"Feature"   : X.columns.tolist(),
                            "Importance": importance})
feat_df    = feat_df.sort_values("Importance", ascending=True)

fig2, ax2 = plt.subplots(figsize=(8, 5))
feat_colors = ["#1D9E75" if i == len(feat_df) - 1 else "#9FE1CB"
               for i in range(len(feat_df))]
ax2.barh(feat_df["Feature"], feat_df["Importance"], color=feat_colors)
ax2.set_xlabel("Feature Importance Score")
ax2.set_title("LightGBM — Feature Importance", fontweight="bold")
ax2.grid(axis="x", alpha=0.3)
for i, imp in enumerate(feat_df["Importance"]):
    ax2.text(imp + 0.5, i, str(imp), va="center", fontsize=9)
plt.tight_layout()
plt.savefig("lightgbm_feature_importance.png", dpi=150, bbox_inches="tight")
plt.show()
print(f"    Saved -> lightgbm_feature_importance.png")

# ─────────────────────────────────────────────
# 11. SUMMARY TABLE
# ─────────────────────────────────────────────
print(f"\n{'=' * 65}")
print(f"  LIGHTGBM MODEL — FINAL SUMMARY")
print(f"{'=' * 65}")
print(f"  Algorithm      : Leaf-Wise Gradient Boosting (LightGBM)")
print(f"  Dataset        : Heart Failure Clinical Records")
print(f"  Records used   : {len(df_clean)} (after exclusion)")
print(f"  Features used  : {X.shape[1]}")
print(f"  Train/Test     : 80% / 20%")
print(f"  Tuning method  : RandomizedSearchCV (50 iterations, 5-fold CV)")
print(f"{'─' * 65}")
print(f"  BEST HYPERPARAMETERS")
for param in param_order:
    if param in best_params:
        print(f"    {param:<22} : {best_params[param]}")
print(f"{'─' * 65}")
print(f"  PERFORMANCE METRICS")
print(f"    Cross-val accuracy : {best_cv_score*100:.2f}%")
print(f"    Test Accuracy      : {accuracy*100:.2f}%")
print(f"    Test Precision     : {precision*100:.2f}%")
print(f"    Test Recall        : {recall*100:.2f}%")
print(f"    Test F1 Score      : {f1*100:.2f}%")
print(f"    Test AUC-ROC       : {auc_roc*100:.2f}%")
print(f"{'─' * 65}")
print(f"  OVERFITTING CHECK")
print(f"    Train Accuracy     : {train_accuracy*100:.2f}%")
print(f"    Test Accuracy      : {test_accuracy*100:.2f}%")
print(f"    Gap                : {gap:.2f}%")
print(f"{'=' * 65}")

# ─────────────────────────────────────────────
# 12. SAVE MODEL & SCALER (.pkl)
# ─────────────────────────────────────────────
print(f"\n[12] SAVING MODEL & SCALER ...")

model_data = {
    "model"        : best_model,
    "scaler"       : scaler,
    "feature_names": X.columns.tolist(),
    "best_params"  : best_params,
    "metrics"      : {
        "accuracy" : accuracy,
        "precision": precision,
        "recall"   : recall,
        "f1_score" : f1,
        "roc_auc"  : auc_roc,
        "cv_score" : best_cv_score,
        "train_acc": train_accuracy,
        "gap"      : gap,
    }
}

joblib.dump(model_data, "lgbm_model.pkl")

print(f"    Saved -> lgbm_model.pkl")
print(f"\n{'=' * 65}")
print(f"  Output files:")
print(f"    lgbm_model.pkl                  — model + scaler for Streamlit")
print(f"    lightgbm_results.png            — confusion matrix, ROC, metrics")
print(f"    lightgbm_feature_importance.png — feature importance chart")
print(f"{'=' * 65}")