# AI Use Log

- Tool: Claude (Anthropic), via chat.
- Task: assignment structure, OMNI-specific parsing, temporal-split guidance,
  diagnostic checklist.
- Accepted: temporal split instead of random split; TimeSeriesSplit for CV;
  per-column sentinel handling; slicing by year and target magnitude.
- Rejected: any suggestion to fill in numeric results without executing.
- Independent verification: I ran every script in a clean environment, confirmed
  the split audit numbers match the printed output, and re-checked the
  Breusch-Pagan p-value against statsmodels docs.