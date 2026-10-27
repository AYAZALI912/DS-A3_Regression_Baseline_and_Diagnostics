"""Learning curve (TimeSeriesSplit) and error slices: by year, by target
magnitude, and by day-of-year quartile. These slices are meaningful for
this dataset in ways a random-feature median split is not."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import learning_curve, TimeSeriesSplit
from sklearn.metrics import mean_squared_error

from omni_loader import load_omni

TARGET = None  # TODO
assert TARGET is not None

df = load_omni().dropna(subset=[TARGET]).reset_index(drop=True)
train_idx = np.load("data/processed/train_idx.npy")
test_idx = np.load("data/processed/test_idx.npy")
FEATURES = [c for c in df.columns if c not in (TARGET, "year", "day_of_year", "hour")]

X_train, X_test = df.loc[train_idx, FEATURES], df.loc[test_idx, FEATURES]
y_train, y_test = df.loc[train_idx, TARGET], df.loc[test_idx, TARGET]

pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("model", Ridge(alpha=1.0)),
]).fit(X_train, y_train)

sizes, tr, va = learning_curve(pipe, X_train, y_train,
    train_sizes=np.linspace(0.1, 1.0, 5),
    cv=TimeSeriesSplit(n_splits=5),
    scoring="neg_mean_squared_error", n_jobs=-1)
plt.plot(sizes, -tr.mean(1), label="train"); plt.plot(sizes, -va.mean(1), label="val")
plt.xlabel("Train size"); plt.ylabel("RMSE"); plt.legend(); plt.title("Learning curve")
plt.savefig("results/figures/learning_curve.png", dpi=120); plt.close()

pred = pipe.predict(X_test)
test = df.loc[test_idx, ["year", "day_of_year", TARGET]].copy()
test["pred"] = pred

def rmse(a, b): return float(np.sqrt(mean_squared_error(a, b)))
print("Overall test RMSE:", rmse(test[TARGET], test["pred"]))

print("\nBy year:")
for yr, g in test.groupby("year"):
    print(f"  {int(yr)}: n={len(g)}, RMSE={rmse(g[TARGET], g['pred']):.4f}")

test["target_quartile"] = pd.qcut(test[TARGET], 4, labels=False, duplicates="drop")
print("\nBy target quartile:")
for q, g in test.groupby("target_quartile"):
    print(f"  Q{int(q)}: n={len(g)}, RMSE={rmse(g[TARGET], g['pred']):.4f}")

test["doy_quartile"] = pd.qcut(test["day_of_year"], 4, labels=False, duplicates="drop")
print("\nBy day-of-year quartile:")
for q, g in test.groupby("doy_quartile"):
    print(f"  Q{int(q)}: n={len(g)}, RMSE={rmse(g[TARGET], g['pred']):.4f}")