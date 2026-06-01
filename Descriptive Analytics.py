# ============================================================
# HEART FAILURE MORTALITY PREDICTION - Descriptive Analytics & Visualisation
#   Step 1 : Missing values check
#   Step 2 : Statistical significance tests (variable exclusion)
#   Step 3 : Descriptive statistics per variable
#   Step 4 : All visualisations (histograms, bar charts,
#             box-plots, correlation heatmap)
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from scipy.stats import pointbiserialr, chi2_contingency

pd.set_option('display.max_rows', 10)
pd.set_option('display.max_columns', 10)

# ─────────────────────────────────────────────
# LOAD DATASET
# ─────────────────────────────────────────────
df       = pd.read_csv("heart_failure_clinical_records.csv")
EXCLUDED = ['diabetes', 'smoking', 'time']
df_clean = df.drop(columns=EXCLUDED)

CONTINUOUS = ['age', 'creatinine_phosphokinase', 'ejection_fraction',
              'platelets', 'serum_creatinine', 'serum_sodium']
BINARY     = ['anaemia', 'diabetes', 'high_blood_pressure', 'sex', 'smoking']

print("=" * 70)
print("  HEART FAILURE — DESCRIPTIVE ANALYTICS & VISUALISATION")
print("=" * 70)
print(f"\n  Original dataset : {df.shape[0]} records x {df.shape[1]} features")
print(f"  Cleaned dataset  : {df_clean.shape[0]} records x {df_clean.shape[1]} features")
print(f"  Excluded         : {EXCLUDED}")


# ═══════════════════════════════════════════════════════════════
# STEP 1 — MISSING VALUES CHECK
# ═══════════════════════════════════════════════════════════════
print("\n\n" + "=" * 70)
print("  STEP 1 — MISSING VALUES CHECK")
print("=" * 70)
missing = df.isnull().sum()
if missing.sum() == 0:
    print("\n  No missing values detected. Dataset is clean.")
    print(f"  All {df.shape[1]} variables have complete data across {df.shape[0]} records.")
else:
    print("\n  Missing values found:")
    print(missing[missing > 0])


# ═══════════════════════════════════════════════════════════════
# STEP 2 — STATISTICAL SIGNIFICANCE & VARIABLE EXCLUSION
# ═══════════════════════════════════════════════════════════════
print("\n\n" + "=" * 70)
print("  STEP 2 — STATISTICAL SIGNIFICANCE TEST vs DEATH_EVENT")
print("=" * 70)
print(f"\n  {'Variable':<28} {'Test':<20} {'Statistic':<14} {'p-value':<12} {'Sig?'}")
print(f"  {'-'*68}")

sig_results = {}

for col in CONTINUOUS + ['time']:
    r, p = pointbiserialr(df[col], df['DEATH_EVENT'])
    sig  = "YES" if p < 0.05 else "NO"
    sig_results[col] = {'stat': f"r={r:.3f}", 'p': p, 'sig': sig}
    print(f"  {col:<28} {'Point-biserial':<20} r={r:>6.3f}       p={p:<10.4f} {sig}")

for col in BINARY:
    ct            = pd.crosstab(df[col], df['DEATH_EVENT'])
    chi2, p, _, _ = chi2_contingency(ct)
    sig           = "YES" if p < 0.05 else "NO"
    sig_results[col] = {'stat': f"chi2={chi2:.3f}", 'p': p, 'sig': sig}
    print(f"  {col:<28} {'Chi-squared':<20} chi2={chi2:>5.3f}       p={p:<10.4f} {sig}")

print(f"\n  Variable Exclusion Rationale:")
print(f"  {'-'*68}")
excl_reasons = {
    'diabetes': f"Not statistically significant (p={sig_results['diabetes']['p']:.4f}, "
                f"chi2=0.587). No meaningful association with DEATH_EVENT.",
    'smoking' : f"Not statistically significant (p={sig_results['smoking']['p']:.4f}, "
                f"chi2=0.399). Weakest predictor in the dataset.",
    'time'    :  "Follow-up duration recorded AFTER clinical event — data leakage risk. "
                 "Excluded despite strong correlation (r=-0.527).",
}
for var, reason in excl_reasons.items():
    print(f"\n  EXCLUDED : {var}")
    print(f"  Reason   : {reason}")

print(f"\n  Retained features : {df_clean.columns.tolist()}")

# ═══════════════════════════════════════════════════════════════
# STEP 3 — DESCRIPTIVE STATISTICS & VISUALISATION
# ═══════════════════════════════════════════════════════════════
print("\n\n" + "=" * 70)
print("  STEP 3 — DESCRIPTIVE STATISTICS & VISUALISATION")
print("=" * 70)

# ── 3.1 Overall Statistical Summary ──────────────────────────
print(f"\n  3.1 Overall Statistical Summary (Continuous Variables)")
print(f"{'─'*70}")
summary = df_clean[CONTINUOUS].describe().round(2)
print(summary.to_string())

# ── 3.2 Per-variable descriptive stats ───────────────────────
print(f"\n  3.2 Per-Variable Descriptive Summary")
print(f"{'─'*70}")
stats_rows = []
for col in CONTINUOUS:
    s = df_clean[col].describe()
    stats_rows.append({
        'Variable': col,
        'Count'   : int(s['count']),
        'Mean'    : round(s['mean'], 2),
        'Std'     : round(s['std'], 2),
        'Min'     : round(s['min'], 2),
        '25%'     : round(s['25%'], 2),
        '50%'     : round(s['50%'], 2),
        '75%'     : round(s['75%'], 2),
        'Max'     : round(s['max'], 2),
    })
stats_df = pd.DataFrame(stats_rows).set_index('Variable')
print(stats_df.to_string())

# ── 3.3 Binary Variables Summary ─────────────────────────────
print(f"\n  3.3 Binary Variable Distribution")
print(f"{'─'*70}")
retained_binary = [b for b in BINARY if b not in EXCLUDED]
print(f"\n  {'Variable':<25} {'Yes (1)':<12} {'No (0)':<12} {'Yes %':<10} {'No %'}")
print(f"  {'─'*65}")
for col in retained_binary + ['DEATH_EVENT']:
    yes = (df_clean[col] == 1).sum()
    no  = (df_clean[col] == 0).sum()
    print(f"  {col:<25} {yes:<12} {no:<12} {yes/len(df_clean)*100:<10.1f}% {no/len(df_clean)*100:.1f}%")

# ── 3.4 Class Balance ─────────────────────────────────────────
print(f"\n  3.4 Target Class Distribution (DEATH_EVENT)")
print(f"{'─'*70}")
survived = (df_clean['DEATH_EVENT'] == 0).sum()
died     = (df_clean['DEATH_EVENT'] == 1).sum()
print(f"  Survived (0) : {survived} records ({survived/len(df_clean)*100:.1f}%)")
print(f"  Death    (1) : {died}  records ({died/len(df_clean)*100:.1f}%)")
print(f"  Class imbalance ratio : {survived/died:.2f}:1")


# ═══════════════════════════════════════════════════════════════
# VISUALISATIONS
# ═══════════════════════════════════════════════════════════════

TEAL   = '#1D9E75'
BLUE   = '#185FA5'
CORAL  = '#D85A30'
PURPLE = '#7F77DD'
AMBER  = '#BA7517'
COLORS = [TEAL, BLUE, CORAL, PURPLE, AMBER, '#639922']

# ────────────────────────────────────────────────────────────────
# Figure 1 — Statistical Summary Tables (one per continuous var)
# ────────────────────────────────────────────────────────────────
for idx, col in enumerate(CONTINUOUS):
    s   = df_clean[col].describe().round(2)
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    ax.axis('off')

    col_labels = [col.replace('_', ' ').title()]
    row_labels = ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']
    cell_vals  = [[f"{s[r]:.2f}"] for r in row_labels]

    tbl = ax.table(
        cellText   = cell_vals,
        rowLabels  = row_labels,
        colLabels  = col_labels,
        cellLoc    = 'center',
        loc        = 'center',
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1.3, 1.6)

    # Style header row
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor('#CCCCCC')
        if r == 0:
            cell.set_facecolor(COLORS[idx])
            cell.set_text_props(color='white', fontweight='bold')
        elif r % 2 == 0:
            cell.set_facecolor('#F5F5F5')
        else:
            cell.set_facecolor('white')

    ax.set_title(f"Statistical Summary of {col.replace('_',' ').title()}",
                 fontweight='bold', fontsize=11, pad=12)
    plt.tight_layout()
    fname = f"stats_summary_{col}.png"
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"  Saved -> {fname}")


# ────────────────────────────────────────────────────────────────
# Figure 2 — Histograms with binning for each continuous variable
# ────────────────────────────────────────────────────────────────
bin_configs = {
    'age'                     : (7, 'Age Group (years)'),
    'creatinine_phosphokinase': (8, 'CPK Level (mcg/L)'),
    'ejection_fraction'       : (7, 'Ejection Fraction (%)'),
    'platelets'               : (7, 'Platelet Count (kiloplatelets/mL)'),
    'serum_creatinine'        : (7, 'Serum Creatinine (mg/dL)'),
    'serum_sodium'            : (7, 'Serum Sodium (mEq/L)'),
}

for idx, (col, (bins, xlabel)) in enumerate(bin_configs.items()):
    fig, ax = plt.subplots(figsize=(8, 5))

    data    = df_clean[col]
    counts, edges, patches = ax.hist(data, bins=bins,
                                      color=COLORS[idx], edgecolor='white',
                                      linewidth=0.8)

    # Add frequency labels on top of each bar
    for count, patch in zip(counts, patches):
        if count > 0:
            ax.text(patch.get_x() + patch.get_width()/2,
                    patch.get_height() + max(counts)*0.01,
                    f'{int(count)}',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Add bin range labels on x-axis
    bin_labels = []
    for i in range(len(edges)-1):
        bin_labels.append(f"{edges[i]:.0f} – {edges[i+1]:.0f}")
    ax.set_xticks([(edges[i] + edges[i+1])/2 for i in range(len(edges)-1)])
    ax.set_xticklabels(bin_labels, rotation=20, ha='right', fontsize=9)

    title = col.replace('_', ' ').title()
    ax.set_title(f"Distribution of {title} by Group",
                 fontsize=13, fontweight='bold')
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel("Frequency", fontsize=11)
    ax.set_ylim(0, max(counts) * 1.15)
    ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    fname = f"histogram_{col}.png"
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"  Saved -> {fname}")


# ────────────────────────────────────────────────────────────────
# Figure 3 — Binary Variables Bar Charts
# ────────────────────────────────────────────────────────────────
binary_labels = {
    'anaemia'            : ('No Anaemia', 'Anaemia'),
    'high_blood_pressure': ('Normal BP', 'High BP'),
    'sex'                : ('Female', 'Male'),
    'DEATH_EVENT'        : ('Survived', 'Death'),
}

for idx, (col, (label0, label1)) in enumerate(binary_labels.items()):
    no_count  = (df_clean[col] == 0).sum()
    yes_count = (df_clean[col] == 1).sum()
    total     = no_count + yes_count

    fig, ax = plt.subplots(figsize=(6, 4.5))
    bars = ax.bar([label0, label1], [no_count, yes_count],
                  color=[TEAL, CORAL], width=0.45, edgecolor='white')

    for bar, cnt in zip(bars, [no_count, yes_count]):
        pct = cnt / total * 100
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + total*0.01,
                f"{cnt}\n({pct:.1f}%)",
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    title = col.replace('_', ' ').title()
    ax.set_title(f"Distribution of {title}", fontsize=13, fontweight='bold')
    ax.set_ylabel("Number of Patients", fontsize=11)
    ax.set_ylim(0, max(no_count, yes_count) * 1.2)
    ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    fname = f"barchart_{col}.png"
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"  Saved -> {fname}")


# ────────────────────────────────────────────────────────────────
# Figure 4 — Boxplots: continuous vars by DEATH_EVENT
# ────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
fig.suptitle("Distribution of Clinical Features by Outcome (Survived vs Death)",
             fontsize=13, fontweight='bold')

for ax, col, color in zip(axes.flat, CONTINUOUS, COLORS):
    survived_data = df_clean[df_clean['DEATH_EVENT'] == 0][col]
    death_data    = df_clean[df_clean['DEATH_EVENT'] == 1][col]

    bp = ax.boxplot([survived_data, death_data],
                    labels=['Survived', 'Death'],
                    patch_artist=True,
                    medianprops=dict(color='black', linewidth=2))

    bp['boxes'][0].set_facecolor(TEAL + '99')
    bp['boxes'][1].set_facecolor(CORAL + '99')

    title = col.replace('_', ' ').title()
    ax.set_title(title, fontweight='bold', fontsize=10)
    ax.set_ylabel("Value", fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig("boxplots_by_outcome.png", dpi=150, bbox_inches='tight')
plt.show()
print(f"  Saved -> boxplots_by_outcome.png")


# ────────────────────────────────────────────────────────────────
# Figure 5 — Correlation Heatmap (cleaned dataset)
# ────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 7))
corr = df_clean.corr().round(2)
mask = np.triu(np.ones_like(corr, dtype=bool))

sns.heatmap(corr, mask=mask, annot=True, fmt='.2f',
            cmap='RdYlGn', center=0, ax=ax,
            linewidths=0.5, linecolor='white',
            annot_kws={"size": 9})

ax.set_title("Feature Correlation Heatmap (After Variable Exclusion)",
             fontsize=13, fontweight='bold', pad=12)
plt.tight_layout()
plt.savefig("correlation_heatmap.png", dpi=150, bbox_inches='tight')
plt.show()
print(f"  Saved -> correlation_heatmap.png")


# ─────────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────────
print(f"\n{'=' * 70}")
print(f"  ALL FILES SAVED SUCCESSFULLY")
print(f"{'=' * 70}")
print(f"\n  Statistical Summary Tables (one per variable):")
for col in CONTINUOUS:
    print(f"    stats_summary_{col}.png")
print(f"\n  Histograms with Binning:")
for col in CONTINUOUS:
    print(f"    histogram_{col}.png")
print(f"\n  Binary Variable Bar Charts:")
for col in binary_labels:
    print(f"    barchart_{col}.png")
print(f"\n  Additional Plots:")
print(f"    boxplots_by_outcome.png      — boxplots split by survived vs death")
print(f"    correlation_heatmap.png      — feature correlation heatmap")
print(f"{'=' * 70}")