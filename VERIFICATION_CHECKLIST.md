# Verification checklist

- [x] Dataset read as whitespace-delimited, 55 columns confirmed.
- [x] Missing/fill sentinels converted to missing for numeric fields.
- [x] Rows sorted by date and duplicate dates removed.
- [x] Chronological train/validation/test split (70/15/15); no random shuffle.
- [x] Naive median baseline included.
- [x] Linear, ridge, and lasso candidates compared.
- [x] Candidate selected using validation RMSE, not test results.
- [x] Held-out test metrics and predictions exported.
- [x] Residual plots and prediction-range error slices exported.
- [ ] Independently verify positional column mapping against official OMNI2 documentation.
- [ ] Add the exact SHA-256 of the committed dataset and record repository commit/tag.
- [ ] Re-run notebook in a clean environment after copying dataset to `data/omni_01_av.txt`.

Observed usable rows: 19019; train=13313, validation=2853, test=2853. The notebook is provided as a reproducible analysis artifact; review all outputs before submission.
