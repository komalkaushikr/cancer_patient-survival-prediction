"""
01_explore.py  --  STAGE 1: UNDERSTAND THE DATA.

Read-only. We change NOTHING here. The goal is to know the data well enough to
make decisions: what to drop, what the target is, where leakage hides.
"""

import pandas as pd
import config

df = pd.read_csv(config.RAW_CSV)


print("SHAPE:", df.shape, "  (rows, columns)")
# WHY: tells you if you even have enough data to model. 100k rows = plenty.

print("\nDTYPES:")
print(df.dtypes)
# WHY: catches numbers stored as text and dates stored as 'object'.
# Diagnosis_Date is object here -> it needs parsing, not treating as a category.

print("\nMISSING per column:")
print(df.isnull().sum())
# WHY: this is your cleaning to-do list. (This dataset shows zero -- suspicious,
# real medical data is never this clean. Likely synthetic. Say that out loud.)

print("\nDUPLICATE ROWS:", df.duplicated().sum())
# WHY: duplicate rows can land in BOTH train and test and inflate your score.
# An interviewer will ask about this.

print("\nCARDINALITY (unique values per text column):")
for col in df.select_dtypes("object").columns:
    print(f"  {col:16s} {df[col].nunique():>6}")
# WHY: decides encode-vs-drop. 2 uniques = clean binary. 1400+ = too granular.

print(f"\nTARGET = {config.TARGET}")
print(df[config.TARGET].value_counts())
print(df[config.TARGET].value_counts(normalize=True).round(3))
# WHY: class balance decides whether you need class_weight later.
# ~64% Deceased / ~36% Alive = mild imbalance, manageable.

print("\nNUMERIC SUMMARY:")
print(df.describe())
# WHY: reveals impossible values (age 200, negative cost) before they poison
# the model.

# ---- THE LEAKAGE CHECK (the most important block in this file) ----------
print("\nLEAKAGE CHECK -- Survival_Months by Status:")
print(df.groupby(config.TARGET)["Survival_Months"].describe()[["mean", "min", "max"]])
print(
    "\n-> If the two groups differ a lot, Survival_Months is measured FROM the\n"
    "   outcome and is NOT known at diagnosis. That is target leakage. We drop\n"
    "   it as a feature (config.DROP_COLS) while keeping it to UNDERSTAND data."
)
