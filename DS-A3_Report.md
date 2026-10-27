# DS-A3 Regression Baseline and Diagnostics

## Aim
Build a transparent regression baseline for daily solar-wind speed using a frozen, time-ordered split and inspect where prediction errors may be unreliable.

## Data and method
The supplied OMNI daily text file has 23,376 rows and 55 columns. The target is solar-wind speed (km/s); predictors are proton density, IMF magnitude, GSM Bz, and flow pressure. Sentinel/fill values are treated as missing. Rows with missing target or selected predictors are removed, dates are ordered, and duplicate dates are dropped. The resulting analysis has 19,019 usable rows: 13,313 training, 2,853 validation, and 2,853 chronological test rows.

The naive baseline predicts the training median. Candidate models are ordinary linear regression, ridge (alpha=1), and lasso (alpha=0.01), using median imputation and standardization. The model is selected by validation RMSE; the test split is reserved for final evaluation.

## Results
The validation-selected model was **Linear regression**. On the held-out chronological test set, it achieved MAE=65.453 km/s, RMSE=78.812 km/s, and R²=0.244 (n=2,853). The naive median baseline on the same test set achieved MAE=70.736 km/s, RMSE=90.670 km/s, and R²=-0.001. Full comparison metrics are in `model_comparison.csv`; test predictions and residual slices are also supplied.

## Diagnostics and interpretation
Residual-versus-prediction and residual-distribution plots are included. The prediction-bin table summarizes mean residual, MAE, and residual spread across prediction ranges. Review these slices for systematic under/over-prediction and changing error variance.

## Claim boundary
This is an interpretable predictive baseline, not a causal study. Linear associations may reflect shared solar-wind conditions, temporal dependence, measurement quality, or unobserved variables. The chronological test set is a stronger check than a random split but does not guarantee performance under future distribution shifts. Sentinel rules and positional column mapping should be independently checked against the dataset documentation before extending the claims.

## Reproduction
Install packages in `requirements.txt`, place the frozen file at `data/omni_01_av.txt` (or beside the notebook), and run the notebook top-to-bottom. Scripts and raw outputs should be committed together with a dataset hash and the exact repository commit.
