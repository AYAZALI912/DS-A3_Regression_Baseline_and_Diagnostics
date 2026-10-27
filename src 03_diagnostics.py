"""Residual, heteroscedasticity, and influence diagnostics on the TRAIN set
of the best linear/Ridge model."""
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.linear_model import Ridge
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from omni_loader import load_omni

TARGET = None  # TODO
assert TARGET is not None

df = load_omni().dropna(subset=[TARGET]).reset_index(drop=True)
train_idx = np.load("data/processed/train_idx.npy")
FEATURES = [c for c in df.columns if c not in (TARGET, "year", "day_of_year", "hour")]

X = df.loc[train_idx, FEATURES]
y = df.loc[train_idx, TARGET]

pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("model", Ridge(alpha=1.0)),
]).fit(X, y)

X_scaled = pipe.named_steps["scaler"].transform(pipe.named_steps["imputer"].transform(X))
X_sm = sm.add_constant(X_scaled)
ols = sm.OLS(y.values, X_sm).fit()
resid, fitted = ols.resid, ols.fittedvalues

import os; os.makedirs("results/figures", exist_ok=True)

plt.figure(); plt.scatter(fitted, resid, s=6, alpha=0.4)
plt.axhline(0, color="red"); plt.xlabel("Fitted"); plt.ylabel("Residuals")
plt.title("Residuals vs Fitted")
plt.savefig("results/figures/resid_vs_fitted.png", dpi=120); plt.close()

fig = sm.qqplot(resid, line="45", fit=True)
plt.savefig("results/figures/qq.png", dpi=120); plt.close(fig)

bp = sm.stats.diagnostic.het_breuschpagan(resid, X_sm)
infl = ols.get_influence()
cooks = infl.cooks_distance[0]
threshold = 4 / len(X)
summary = {
    "breusch_pagan_lm": float(bp[0]),
    "breusch_pagan_p": float(bp[1]),
    "max_cooks_d": float(cooks.max()),
    "cooks_threshold": float(threshold),
    "n_points_over_threshold": int((cooks > threshold).sum()),
}
print(json.dumps(summary, indent=2))
# TODO: paste this JSON into results/diagnostics_summary.json yourself.