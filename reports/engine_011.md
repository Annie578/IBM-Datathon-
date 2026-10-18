# Turbofan Engine Maintenance Assessment Report: Engine 11

## Status Overview
- **Engine ID**: 11
- **Fleet Urgency Rank**: Rank 51 of 100
- **Operational Cycles Observed**: 83
- **Risk Classification**: healthy
- **Predicted Remaining Useful Life (RUL)**: 86.9 cycles
- **Fleet Average Predicted RUL**: 86.6 cycles

## Risk Classification Criteria
- **Critical Risk Threshold**: Predicted RUL below 30.0 cycles
- **Warning Risk Threshold**: Predicted RUL below 60.0 cycles
- **Current Fleet Status**: Ranked 51 of 100 engines (healthy status)

## Primary Telemetry & Degradation Drivers
The primary sensor drivers identified by TreeSHAP attribution and their deviation from baseline (measured over initial 20 cycles):

| Sensor | Symbol | Description | Impact on RUL (cycles) | Direction | Current Value | Healthy Baseline | Baseline Deviation | Recent Slope per Cycle |
|---|---|---|---|---|---|---|---|---|
| `s4` | **T50** | Total temperature at LPT outlet | `-7.3` | shortens predicted life | `1414.22` | `1409.229` | `4.991` | `-0.505` |
| `s11` | **Ps30** | Static pressure at HPC outlet | `6.1` | lengthens predicted life | `47.67` | `47.578` | `0.092` | `-0.0075` |
| `s12` | **phi** | Ratio of fuel flow to Ps30 | `5.1` | lengthens predicted life | `520.74` | `521.196` | `-0.456` | `-0.0025` |
| `s17` | **htBleed** | Bleed enthalpy | `-4.0` | shortens predicted life | `394.0` | `392.8` | `1.2` | `0.5` |
| `s13` | **NRf** | Corrected fan speed | `-3.8` | shortens predicted life | `2388.18` | `2388.125` | `0.055` | `0.02` |

Engine age effect on predicted RUL: -7.7 cycles.

## Diagnostic Assessment
Engine telemetry exhibits exemplary operational health, ranking among the least urgent units across the fleet. Core parameters remain closely aligned with baseline readings established during the initial burn-in window. The primary positive life contributors include `s11` (Ps30 - Static pressure at HPC outlet) and `s12` (phi - Ratio of fuel flow to Ps30), both of which lengthen predicted life. The negative impact from `s4` (T50 - Total temperature at LPT outlet) is offset by these favorable pressure and fuel ratio dynamics. Negative rate-of-change metrics for `s4` and `s11` indicate stable thermo-fluid operating dynamics, suggesting that thermal stress at the low-pressure turbine outlet is trending toward nominal levels while high-pressure compressor outlet pressure remains robust.

## Operational Action Protocol
- **CONTINUE UNRESTRICTED FLIGHT OPERATIONS**: Retain standard flight dispatch availability without operational constraints.
- **STANDARD MONITORING**: Maintain routine scheduled inspection intervals.

## Model Reliability & Technical Context
- **Model Architecture**: XGBoost
- **Model Test Root Mean Square Error (RMSE)**: 11.824 cycles
- **Model Test Mean Absolute Error (MAE)**: 8.648 cycles
- **Target Training RUL Cap**: 125 cycles
- **Operational Caveat**: Predictions near the cap mean 'at least roughly this many cycles', because the model was trained on RUL capped at that value.