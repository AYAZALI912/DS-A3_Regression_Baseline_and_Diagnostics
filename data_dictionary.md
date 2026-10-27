# Dataset dictionary (selected fields)

The file is a whitespace-delimited OMNI daily data table without a header row. Column names below are positional labels assigned for this analysis; confirm against the official OMNI2 format documentation before scientific reuse.

| Field | Zero-based column | Meaning | Role |
|---|---:|---|---|
| year | 0 | Calendar year | Date construction |
| day_of_year | 1 | Day of year | Date construction |
| imf_magnitude | 14 | Interplanetary magnetic-field magnitude | Predictor |
| bz_gsm | 19 | IMF Bz in GSM coordinates | Predictor |
| flow_pressure | 30 | Solar-wind flow pressure | Predictor |
| proton_density | 32 | Solar-wind proton density | Predictor |
| solar_wind_speed | 33 | Solar-wind speed, km/s | Target |

Fill/sentinel values are treated as missing. The loader does not claim every field is usable; only the target and four predictors are modeled.
