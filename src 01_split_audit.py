"""Temporal split + leakage audit. All numbers printed here go into
results/split_audit.md (which you write yourself)."""
import os
import numpy as np
import pandas as pd
from omni_loader import load_omni

TARGET = None  # TODO: set to a real column name, e.g. "col_25"
TRAIN_END_YEAR = 1973  # TODO: adjust to your chosen temporal cutoff

assert TARGET is not None, "Pick a target column before running."

df = load_omni()
df = df.dropna(subset=[TARGET]).copy()

train = df[df["year"] <= TRAIN_END_YEAR]
test = df[df["year"] > TRAIN_END_YEAR]

print(f"Train rows: {len(train)}, Test rows: {len(test)}")
print(f"Train target mean/std: {train[TARGET].mean():.4f} / {train[TARGET].std():.4f}")
print(f"Test  target mean/std: {test[TARGET].mean():.4f} / {test[TARGET].std():.4f}")
print("Missing target — train: {:.1%}, test: {:.1%}".format(
    df.loc[df['year'] <= TRAIN_END_YEAR, TARGET].isna().mean(),
    df.loc[df['year'] >  TRAIN_END_YEAR, TARGET].isna().mean()))

dups = df.duplicated(subset=["year", "day_of_year"]).sum()
print(f"Duplicate (year, day_of_year) rows: {dups}")

# Correlation leakage check on numeric features
num = df.select_dtypes(include=[np.number]).drop(columns=[TARGET])
corrs = num.apply(lambda s: s.corr(df[TARGET]))
print("Features with |corr| > 0.95 (possible leakage):")
print(corrs[corrs.abs() > 0.95])

os.makedirs("results", exist_ok=True)
np.save("data/processed/train_idx.npy", train.index.to_numpy())
np.save("data/processed/test_idx.npy", test.index.to_numpy())