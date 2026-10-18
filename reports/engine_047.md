# Turbofan Engine Maintenance Assessment Report: Engine 47

## Status Overview
- **Engine ID**: 47
- **Fleet Urgency Rank**: Rank 100 of 100
- **Operational Cycles Observed**: 73
- **Risk Classification**: healthy
- **Predicted Remaining Useful Life (RUL)**: 127.2 cycles
- **Fleet Average Predicted RUL**: 86.6 cycles

## Risk Classification Criteria
- **Critical Risk Threshold**: Predicted RUL below 30.0 cycles
- **Warning Risk Threshold**: Predicted RUL below 60.0 cycles
- **Current Fleet Status**: Ranked 100 of 100 engines (healthy status)

## Primary Telemetry & Degradation Drivers
The primary sensor drivers identified by TreeSHAP attribution and their deviation from baseline (measured over initial 20 cycles):

| Sensor | Symbol | Description | Impact on RUL (cycles) | Direction | Current Value | Healthy Baseline | Baseline Deviation | Recent Slope per Cycle |
|---|---|---|---|---|---|---|---|---|
| `s11` | **Ps30** | Static pressure at HPC outlet | `6.4` | lengthens predicted life | `47.44` | `47.597` | `-0.157` | `-0.06` |
| `s12` | **phi** | Ratio of fuel flow to Ps30 | `5.7` | lengthens predicted life | `521.29` | `521.25` | `0.04` | `0.11` |
| `s14` | **NRc** | Corrected core speed | `4.1` | lengthens predicted life | `8125.57` | `8121.823` | `3.747` | `1.4525` |
| `s9` | **Nc** | Physical core speed | `4.1` | lengthens predicted life | `9038.32` | `9043.952` | `-5.632` | `-1.7675` |
| `s4` | **T50** | Total temperature at LPT outlet | `4.0` | lengthens predicted life | `1402.79` | `1410.296` | `-7.506` | `-3.425` |

Engine age effect on predicted RUL: -2.3 cycles.

## Diagnostic Assessment
Engine telemetry exhibits exemplary operational health, ranking among the least urgent units across the fleet. Core parameters remain closely aligned with baseline readings established during the initial burn-in window. The primary positive life contributors include `s11` (Ps30 - Static pressure at HPC outlet) and `s12` (phi - Ratio of fuel flow to Ps30), both of which lengthen predicted life. Negative rate-of-change metrics indicate stable thermo-fluid operating dynamics.

## Operational Action Protocol
- **CONTINUE UNRESTRICTED FLIGHT OPERATIONS**: Retain standard flight dispatch availability without operational constraints.
- **STANDARD MONITORING**: Maintain routine scheduled inspection intervals.

## Model Reliability & Technical Context
- **Model Architecture**: XGBoost
- **Model Test Root Mean Square Error (RMSE)**: 11.824 cycles
- **Model Test Mean Absolute Error (MAE)**: 8.648 cycles
- **Target Training RUL Cap**: 125 cycles
- **Operational Caveat**: Predictions near the cap mean 'at least roughly this many cycles', because the model was trained on RUL capped at that value.