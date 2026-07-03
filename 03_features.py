"""
03_features.py  --  STAGE 3: ENCODE + SCALE (the firewall in action).

This is where "fit on train only" stops being a slogan and becomes code.
We build a ColumnTransformer that applies the RIGHT transform to each column
type, fit it on TRAIN, and apply it to BOTH sets. Then we show the WRONG way
(fit on everything) so you can see the leak with your own eyes.

Encoding decisions (say these out loud in an interview):
  - Age, diag_year, diag_month : numeric      -> StandardScaler
  - Gender, Cancer_Type, Treatment_Type : nominal (no order) -> OneHotEncoder
  - Stage : ORDINAL (I < II < III < IV, real severity order) -> OrdinalEncoder

Run:  python 03_features.py
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
import config

train = pd.read_csv(config.TRAIN_CSV)
test = pd.read_csv(config.TEST_CSV)

# ---- separate features (X) from target (y) -----------------------------
# Positive class = Deceased (=1). WHY: "death" is the event of interest, so a
# false negative = "we predicted Alive but they died" = the costly medical error.
y_train = (train[config.TARGET] == "Deceased").astype(int)
y_test = (test[config.TARGET] == "Deceased").astype(int)
X_train = train.drop(columns=[config.TARGET])
X_test = test.drop(columns=[config.TARGET])

# ---- declare which columns get which transform -------------------------
numeric = ["Age", "diag_year", "diag_month"]
nominal = ["Gender", "Cancer_Type", "Treatment_Type"]
ordinal = ["Stage"]

# Stage must be listed LEAST -> MOST severe for OrdinalEncoder.
# Print the actual values and CONFIRM this order matches clinical severity:
stage_order = sorted(X_train["Stage"].unique())
print("Stage categories (verify this is least->most severe):", stage_order)

# ---- THE RIGHT WAY: one transformer, fit on TRAIN only -----------------
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric),
        ("nom", OneHotEncoder(handle_unknown="ignore", sparse_output=False), nominal),
        ("ord", OrdinalEncoder(categories=[stage_order]), ordinal),
    ]
)

X_train_enc = preprocessor.fit_transform(X_train)   # LEARNS means/std/categories from TRAIN
X_test_enc = preprocessor.transform(X_test)         # APPLIES the same numbers to test

feature_names = preprocessor.get_feature_names_out()
print("\nEncoded shape  train:", X_train_enc.shape, " test:", X_test_enc.shape)
print("Feature names:", list(feature_names))

# ---- PROOF that it worked: train's scaled Age has mean 0, std 1 ---------
age_idx = list(feature_names).index("num__Age")
print(f"\nRIGHT way -> train Age scaled mean={X_train_enc[:, age_idx].mean():.6f} "
      f"(should be ~0), std={X_train_enc[:, age_idx].std():.4f} (should be ~1)")

# ---- THE WRONG WAY: fit the scaler on ALL data (train + test) ----------
# This lets the test set influence the numbers used on the train set = LEAK.
all_age = pd.concat([X_train["Age"], X_test["Age"]])
leaky_mean = all_age.mean()
clean_mean = X_train["Age"].mean()
print(f"\nWRONG way -> scaler mean uses ALL data: {leaky_mean:.4f}")
print(f"RIGHT way -> scaler mean uses TRAIN only: {clean_mean:.4f}")
print("They differ because the wrong way peeked at the test set. The gap is\n"
      "small here (train is 80% of data), but with small data, time series, or\n"
      "target encoding it becomes large -- and it is ALWAYS a bug. Never fit on\n"
      "the test set. This is the single most common leakage mistake.")

# Save the encoded arrays + fitted preprocessor for the modeling stage.
np.save(config.PROCESSED_DIR / "X_train_enc.npy", X_train_enc)
np.save(config.PROCESSED_DIR / "X_test_enc.npy", X_test_enc)
np.save(config.PROCESSED_DIR / "y_train.npy", y_train.to_numpy())
np.save(config.PROCESSED_DIR / "y_test.npy", y_test.to_numpy())
print("\nSaved encoded arrays to", config.PROCESSED_DIR)
