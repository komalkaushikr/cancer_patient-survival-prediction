"""
02_clean_split.py  --  STAGE 2: SAFE CLEANING, then SPLIT.

ORDER IS THE LESSON HERE. We only do operations that do NOT learn anything from
the data (dropping columns, parsing a date, fixing impossible values). Then we
SPLIT. Anything that learns a parameter (scaling, encoding, imputation) happens
AFTER the split, fit on TRAIN only -- that lives in 03_features.py.

This script SAVES processed/train.csv and processed/test.csv so later stages
just load them. Saving artifacts between stages is how real pipelines work.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
import config

df = pd.read_csv(config.RAW_CSV)
print("Loaded:", df.shape)

# 1) Drop ID + leakage columns.
#    SAFE before split: removing a column learns nothing from the data.
df = df.drop(columns=config.DROP_COLS)
print("After dropping", config.DROP_COLS, "->", df.shape)

# 2) Parse the date and derive simple features.
#    SAFE before split: each row is transformed on its own, no learned parameter.
df[config.DATE_COL] = pd.to_datetime(df[config.DATE_COL], errors="coerce")
df["diag_year"] = df[config.DATE_COL].dt.year
df["diag_month"] = df[config.DATE_COL].dt.month
df = df.drop(columns=[config.DATE_COL])   # raw date no longer needed

# 3) Fix impossible values with a domain rule.
#    SAFE before split: a fixed rule (0 <= age <= 120), not a learned threshold.
before = len(df)
df = df[df["Age"].between(0, 120)]
print(f"Dropped {before - len(df)} rows with impossible Age")

# 4) Drop high-cardinality text for the FIRST model. Revisit later.
df = df.drop(columns=[c for c in config.HIGH_CARDINALITY if c in df.columns])
print("Modeling columns:", list(df.columns))

# 5) ===== THE FIREWALL: SPLIT =====
#    stratify=target keeps the same Alive/Deceased ratio in BOTH train and test,
#    so your test set is a fair mirror of reality.
train, test = train_test_split(
    df,
    test_size=config.TEST_SIZE,
    random_state=config.RANDOM_STATE,
    stratify=df[config.TARGET],
)
print("Train:", train.shape, " Test:", test.shape)
print("Train balance:\n", train[config.TARGET].value_counts(normalize=True).round(3))
print("Test balance:\n", test[config.TARGET].value_counts(normalize=True).round(3))

train.to_csv(config.TRAIN_CSV, index=False)
test.to_csv(config.TEST_CSV, index=False)
print("\nSaved ->", config.TRAIN_CSV)
print("Saved ->", config.TEST_CSV)
