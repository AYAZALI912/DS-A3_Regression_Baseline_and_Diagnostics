"""Naive baseline + linear/ridge/lasso with TimeSeriesSplit CV tuning.
Tuning happens on TRAIN only. Test is touched once, at the end."""
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score

from omni_loader import load_omni

TARGET = None  # TODO: same as 01
assert TARGET is not None

df = load_omni().dropna(subset=[TARGET]).reset_index(drop=True)
train_idx = np.load("data/processed/train_idx.npy")
test_idx = np.load("data/processed/test_idx.npy")

FEATURES = [c for c in df.columns if c not in (TARGET, "year", "day_of_year", "hour")]
X_train, X_test = df.loc[train_idx, FEATURES], df.loc[test_idx, FEATURES]
y_train, y_test = df.loc[train_idx, TARGET], df.loc[test_idx, TARGET]

def pipe(model):
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", model),
    ])

tscv = TimeSeriesSplit(n_splits=5)

def evaluate(name, fitted):
    pred = fitted.predict(X_test)
    return {"model": name,
            "test_rmse": float(np.sqrt(mean_squared_error(y_test, pred))),
            "test_r2": float(r2_score(y_test, pred))}

results = []

dummy = DummyRegressor(strategy="mean").fit(X_train, y_train)
results.append(evaluate("dummy_mean", dummy))

results.append(evaluate("linear", pipe(LinearRegression()).fit(X_train, y_train)))

ridge = GridSearchCV(pipe(Ridge()), {"model__alpha": [0.01, 0.1, 1, 10, 100]},
                     cv=tscv, scoring="neg_mean_squared_error").fit(X_train, y_train)
results.append(evaluate(f"ridge_a{ridge.best_params_['model__alpha']}", ridge))

lasso = GridSearchCV(pipe(Lasso(max_iter=20000)),
                     {"model__alpha": [0.001, 0.01, 0.1, 1]},
                     cv=tscv, scoring="neg_mean_squared_error").fit(X_train, y_train)
nz = int(np.sum(lasso.best_estimator_.named_steps["model"].coef_ != 0))
res = evaluate(f"lasso_a{lasso.best_params_['model__alpha']}", lasso)
res["n_nonzero"] = nz
results.append(res)

pd.DataFrame(results).to_csv("results/model_comparison.csv", index=False)
print(pd.DataFrame(results))