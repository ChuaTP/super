# ============================================================
# HEART FAILURE MORTALITY PREDICTION - DATASET LOADING
# ============================================================

import pandas as pd

pd.set_option('display.max_rows', 10)
pd.set_option('display.max_columns', 10)

# ─────────────────────────────────────────────
# 1. LOAD FULL DATASET (5000 records)
# ─────────────────────────────────────────────
df = pd.read_csv("heart_failure_clinical_records.csv")

print("=" * 60)
print("1. DATASET LOADED SUCCESSFULLY")
print("=" * 60)
print(f"   Total Records : {df.shape[0]}")
print(f"   Total Features: {df.shape[1]}")
print("\n   Dataset Before Variable Exclusion:")
print(df)

# ─────────────────────────────────────────────
# 2. VARIABLE INFORMATION TABLE
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("2. DATASET OVERVIEW - ALL VARIABLES & DATA TYPES")
print("=" * 60)

variable_info = {
    "age"                      : ("float64", "Continuous", "Age of patient (years)"),
    "anaemia"                  : ("int64",   "Binary",     "Decrease of red blood cells (0=No, 1=Yes)"),
    "creatinine_phosphokinase" : ("int64",   "Continuous", "Level of CPK enzyme in blood (mcg/L)"),
    "diabetes"                 : ("int64",   "Binary",     "Patient has diabetes (0=No, 1=Yes)"),
    "ejection_fraction"        : ("int64",   "Continuous", "% of blood leaving heart per contraction"),
    "high_blood_pressure"      : ("int64",   "Binary",     "Patient has hypertension (0=No, 1=Yes)"),
    "platelets"                : ("float64", "Continuous", "Platelets in blood (kiloplatelets/mL)"),
    "serum_creatinine"         : ("float64", "Continuous", "Level of creatinine in blood (mg/dL)"),
    "serum_sodium"             : ("int64",   "Continuous", "Level of sodium in blood (mEq/L)"),
    "sex"                      : ("int64",   "Binary",     "Gender of patient (0=Female, 1=Male)"),
    "smoking"                  : ("int64",   "Binary",     "Patient smokes (0=No, 1=Yes)"),
    "time"                     : ("int64",   "Continuous", "Follow-up period (days)"),
    "DEATH_EVENT"              : ("int64",   "Binary",     "TARGET: Patient died (0=No, 1=Yes)"),
}

print(f"\n{'Variable':<30} {'Dtype':<10} {'Type':<12} {'Description'}")
print("-" * 90)
for var, (dtype, vtype, desc) in variable_info.items():
    tag = " <- TARGET" if var == "DEATH_EVENT" else ""
    print(f"{var:<30} {dtype:<10} {vtype:<12} {desc}{tag}")

# ─────────────────────────────────────────────
# 3. LOAD CLEANED DATASET (Variables Excluded)
# ─────────────────────────────────────────────
EXCLUDED = ["diabetes", "smoking", "time"]

df_clean = df.drop(columns=EXCLUDED)

print("\n" + "=" * 60)
print("3. CLEANED DATASET (AFTER VARIABLE EXCLUSION)")
print("=" * 60)
print(f"\n   Original : {df.shape[0]} records x {df.shape[1]} features")
print(f"   Cleaned  : {df_clean.shape[0]} records x {df_clean.shape[1]} features")
print(f"   Excluded : {EXCLUDED}")
print(f"\n   Retained Variables:")

for col in df_clean.columns:
    dtype = str(df_clean[col].dtype)
    tag = " <- TARGET" if col == "DEATH_EVENT" else ""
    print(f"   +  {col:<30} ({dtype}){tag}")

print(f"\n   Dataset After Variable Exclusion of diabetes, smoking, and time:")
print(df_clean)

# ─────────────────────────────────────────────
# 4. SAVE CLEANED DATASET
# ─────────────────────────────────────────────
df_clean.to_csv("heart_failure_cleaned.csv", index=False)
print(f"\n{'=' * 60}")
print(f"   Cleaned dataset saved -> heart_failure_cleaned.csv")
print(f"{'=' * 60}")
print(f"\n   Next step : Run descriptive_analytics.py")
print(f"               for missing value checks, significance")
print(f"               tests, and full visualisations.")
print(f"{'=' * 60}")