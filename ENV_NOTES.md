# Environment notes

- **Time:** 2026-10-18 01:30
- **Machine:** x86_64
- **Byte order:** little
- **OS:** Linux-6.6.87.2-microsoft-standard-WSL2-x86_64-with-glibc2.29
- **Python:** 3.8.10
- **CPU cores:** 12
- **Memory (GB):** 9.5

## Imports

| Package | Status | Version or error |
| --- | --- | --- |
| numpy | ok | 1.24.4 |
| pandas | ok | 2.0.3 |
| scipy | ok | 1.10.1 |
| sklearn | ok | 1.3.2 |
| pyarrow | ok | 17.0.0 |
| matplotlib | ok | 3.7.5 |
| joblib | ok | 1.4.2 |
| xgboost | ok | 2.1.4 |
| shap | ok | 0.44.1 |
| snapml | MISSING | ModuleNotFoundError: No module named 'snapml' |

## Smoke tests

| Test | Result |
| --- | --- |
| scikit-learn RandomForest fit | ok (0.25s) |
| xgboost fit + predict | ok (0.30s) |
| shap TreeExplainer | ok on xgboost (0.08s, shape (200, 10)) |
| parquet write/read | ok (1.05s) |

## Verdict

GREEN: use XGBoost + TreeSHAP as planned.
