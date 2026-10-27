"""Load and clean the OMNI fixed-width text file.

Columns are unnamed in the raw file. The first three fields are year,
day-of-year, hour. Everything else is anonymous and must be named by you
in COLUMN_NAMES below before any modeling.
"""
import numpy as np
import pandas as pd

RAW_PATH = "data/raw/omni_01_av.txt"

# TODO: rename these. At minimum keep the first three.
# The file has ~54 fields per row; count them and name them all.
COLUMN_NAMES = [f"col_{i}" for i in range(54)]
COLUMN_NAMES[0] = "year"
COLUMN_NAMES[1] = "day_of_year"
COLUMN_NAMES[2] = "hour"

# Per-column plausible physical range (lo, hi). Values outside are set NaN.
# TODO: tighten these once you know what each column is.
DEFAULT_RANGE = (-1e6, 1e6)

def load_omni(path=RAW_PATH, plausible_range=None):
    rows = []
    dropped = 0
    with open(path, "r") as f:
        for line in f:
            parts = line.split()
            if len(parts) < len(COLUMN_NAMES):
                dropped += 1
                continue
            rows.append(parts[: len(COLUMN_NAMES)])
    df = pd.DataFrame(rows, columns=COLUMN_NAMES).astype(float)
    df = df.replace([999.9, 999.99, 9999999.0, 999999.99, 99999.99, 9999.0,
                     99.0, 99.99, 99999.0], np.nan)
    if plausible_range is not None:
        for col, (lo, hi) in plausible_range.items():
            if col in df.columns:
                df.loc[(df[col] < lo) | (df[col] > hi), col] = np.nan
    else:
        lo, hi = DEFAULT_RANGE
        df[(df < lo) | (df > hi)] = np.nan
    print(f"Loaded {len(df)} rows, dropped {dropped} malformed rows.")
    return df