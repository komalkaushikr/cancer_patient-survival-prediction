"""
config.py  --  single source of truth.

Every other script imports from here, so there is ONE place to change a path
or a column decision. This is a real-project habit: never hardcode the same
path in five files.
"""

from pathlib import Path

# ---- paths -------------------------------------------------------------
# Point this at YOUR csv. The r"..." prefix makes Windows backslashes literal.
RAW_CSV = r"E:\Desktop\cancer\archive\india_cancer_patients_2022_2025.csv"

PROCESSED_DIR = Path(__file__).parent / "processed"
PROCESSED_DIR.mkdir(exist_ok=True)          # created automatically if missing

TRAIN_CSV = PROCESSED_DIR / "train.csv"
TEST_CSV = PROCESSED_DIR / "test.csv"

# ---- column roles  (the WHY for each is in WORKFLOW.md) ----------------
TARGET = "Status"                            # what we predict: Alive vs Deceased

DROP_COLS = [
    "Patient_ID",       # identifier: unique per row, zero predictive signal
    "Survival_Months",  # LEAKAGE: measured from the outcome, unknown at diagnosis
]

# High-cardinality text. We leave these OUT of the first model to keep it simple,
# and revisit later with target/frequency encoding once a baseline exists.
HIGH_CARDINALITY = ["City", "Hospital_Name", "State"]

DATE_COL = "Diagnosis_Date"

# ---- reproducibility ---------------------------------------------------
RANDOM_STATE = 42      # fixed seed -> same split every run (interviewers love this)
TEST_SIZE = 0.20       # 80% train / 20% test
