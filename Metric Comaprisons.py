# ============================================================
# HEART FAILURE PREDICTION — MODEL COMPARISON (FINAL VERSION)
# Separate bar charts matching your screenshot style + Train/Test for Accuracy
# ============================================================

import matplotlib.pyplot as plt
import numpy as np

print("=" * 65)
print("  MODEL COMPARISON — HEART FAILURE PREDICTION")
print("=" * 65)

# ─────────────────────────────────────────────
# 1. EXACT METRICS FROM YOUR RESULTS
# ─────────────────────────────────────────────
results = {
    "LightGBM": {
        "accuracy": 0.9940, "precision": 0.9904, "recall": 0.9904,
        "f1_score": 0.9904, "roc_auc": 0.9998
    },
    "ANN": {
        "accuracy": 0.9780, "precision": 0.9740, "recall": 0.9554,
        "f1_score": 0.9646, "roc_auc": 0.9830
    },
    "SVM": {
        "accuracy": 0.9660, "precision": 0.9348, "recall": 0.9586,
        "f1_score": 0.9465, "roc_auc": 0.9720
    }
}

# Training accuracies for the Accuracy chart (from your screenshots)
train_accuracies = {
    "LightGBM": 0.9995,
    "ANN": 0.9945,
    "SVM": 0.9895
}

model_names = ["LightGBM", "ANN", "SVM"]
metric_keys = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
metric_labels = ["Accuracy", "Precision", "Recall", "F1 Score", "AUC-ROC"]
model_colors = ["#1D9E75", "#185FA5", "#7F77DD"]  # Teal=LightGBM, Blue=ANN, Purple=SVM

print("  All metrics loaded successfully\n")

# ─────────────────────────────────────────────
# 2. PRINT COMPARISON TABLE
# ─────────────────────────────────────────────
print(f"  COMPARISON TABLE")
print(f"{'─' * 65}")
print(f"  {'Metric':<14} {'LightGBM':>12} {'ANN':>12} {'SVM':>12} {'Best':>12}")
print(f"  {'─' * 61}")

for key, label in zip(metric_keys, metric_labels):
    vals = [results[n][key] for n in model_names]
    best = model_names[int(np.argmax(vals))]
    strs = [f"{v * 100:.2f}%" for v in vals]
    print(f"  {label:<14} {strs[0]:>12} {strs[1]:>12} {strs[2]:>12} {best:>12}")

print(f"{'─' * 65}")

# ─────────────────────────────────────────────
# 3. GROUPED BAR CHART — ALL 5 METRICS (kept for convenience)
# ─────────────────────────────────────────────
x = np.arange(len(metric_labels))
width = 0.25

fig, ax = plt.subplots(figsize=(13, 6))
fig.suptitle("Model Comparison — Heart Failure Prediction",
             fontsize=15, fontweight="bold", y=1.01)

for i, (name, color) in enumerate(zip(model_names, model_colors)):
    vals = [results[name][k] * 100 for k in metric_keys]
    bars = ax.bar(x + i * width, vals, width,
                  label=name, color=color, alpha=0.88,
                  edgecolor="white", linewidth=0.6)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.4,
                f"{val:.1f}",
                ha="center", va="bottom",
                fontsize=8, fontweight="bold",
                color="#333333")

ax.set_xticks(x + width)
ax.set_xticklabels(metric_labels, fontsize=11)
ax.set_ylim(0, 115)
ax.set_ylabel("Score (%)", fontsize=11)
ax.set_xlabel("Evaluation Metric", fontsize=11)
ax.legend(fontsize=10, loc="lower right", framealpha=0.9, edgecolor="#CCCCCC")
ax.grid(axis="y", alpha=0.3, linestyle="--")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig("comparison_all_metrics.png", dpi=150, bbox_inches="tight")
plt.show()
print(f"\n  Saved -> comparison_all_metrics.png")

# ─────────────────────────────────────────────
# 4. SEPARATE BAR CHARTS (exactly like your screenshots)
# ─────────────────────────────────────────────

# 4.1 Accuracy Comparison (Train + Test)
fig_acc, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(model_names))
width = 0.35

# Training bars (light gray)
train_vals = [train_accuracies[name] * 100 for name in model_names]
ax.bar(x - width / 2, train_vals, width, label='Training Accuracy',
       color='lightgray', alpha=0.9, edgecolor="white")

# Testing bars (model colors)
test_vals = [results[name]["accuracy"] * 100 for name in model_names]
for i, color in enumerate(model_colors):
    ax.bar(x[i] + width / 2, test_vals[i], width, label='Testing Accuracy' if i == 0 else "",
           color=color, alpha=0.88, edgecolor="white")

# Add percentage labels
for i, v in enumerate(train_vals):
    ax.text(x[i] - width / 2, v + 0.5, f"{v:.1f}%", ha='center', va='bottom', fontsize=9, fontweight='bold')
for i, v in enumerate(test_vals):
    ax.text(x[i] + width / 2, v + 0.5, f"{v:.1f}%", ha='center', va='bottom', fontsize=9, fontweight='bold')

ax.set_title("Accuracy Comparison (Generalization Analysis)", fontsize=13, fontweight="bold")
ax.set_ylabel("Accuracy (%)", fontsize=11)
ax.set_xticks(x)
ax.set_xticklabels(model_names, fontsize=11)
ax.set_ylim(0, 105)
ax.legend(fontsize=10)
ax.grid(axis="y", alpha=0.3, linestyle="--")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig("accuracy_comparison.png", dpi=150, bbox_inches="tight")
plt.show()
print("  Saved -> accuracy_comparison.png")

# 4.2 Precision, Recall, F1-Score, AUC-ROC (single bars per model)
other_metrics = [
    ("precision", "Precision Comparison", "Precision (%)", "precision_comparison.png"),
    ("recall", "Recall Comparison", "Recall (%)", "recall_comparison.png"),
    ("f1_score", "F1-Score Comparison", "F1-Score (%)", "f1_score_comparison.png"),
    ("roc_auc", "AUC Comparison", "AUC (%)", "auc_comparison.png")
]

for key, chart_title, y_label, filename in other_metrics:
    fig, ax = plt.subplots(figsize=(10, 6))
    vals = [results[name][key] * 100 for name in model_names]

    bars = ax.bar(model_names, vals, color=model_colors, alpha=0.88,
                  edgecolor="white", linewidth=0.6)

    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.5,
                f"{val:.1f}%", ha="center", va="bottom",
                fontsize=10, fontweight="bold")

    ax.set_title(f"Model Performance: {chart_title}", fontsize=13, fontweight="bold")
    ax.set_ylabel(y_label, fontsize=11)
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Saved -> {filename}")

# ─────────────────────────────────────────────
# 5. FINAL SUMMARY
# ─────────────────────────────────────────────
print(f"\n{'=' * 65}")
print(f"  FINAL SUMMARY")
print(f"{'=' * 65}")

wins = {name: 0 for name in model_names}
for key in metric_keys:
    vals = [results[n][key] for n in model_names]
    best_idx = int(np.argmax(vals))
    wins[model_names[best_idx]] += 1

for name in model_names:
    print(f"  {name:<12} : {wins[name]} / {len(metric_keys)} metrics won")

best_overall = max(wins, key=wins.get)
print(f"\n  Best overall model : {best_overall}")
print(f"  (Based on most metric categories won)")
print(f"{'=' * 65}")
print(f"\n  Output files generated:")
print(f"    comparison_all_metrics.png     — grouped chart")
print(f"    accuracy_comparison.png        — Train vs Test Accuracy")
print(f"    precision_comparison.png       — Precision only")
print(f"    recall_comparison.png          — Recall only")
print(f"    f1_score_comparison.png        — F1-Score only")
print(f"    auc_comparison.png             — AUC-ROC only")
print(f"{'=' * 65}")