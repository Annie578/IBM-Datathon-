# Turbofan Engine Maintenance Assessment Report: Engine 81

## Status Overview
- **Engine ID**: 81
- **Fleet Urgency Rank**: Rank 1 of 100
- **Operational Cycles Observed**: 213
- **Risk Classification**: critical
- **Predicted Remaining Useful Life (RUL)**: 6.3 cycles
- **Fleet Average Predicted RUL**: 86.6 cycles

## Risk Classification Criteria
- **Critical Risk Threshold**: Predicted RUL below 30.0 cycles
- **Warning Risk Threshold**: Predicted RUL below 60.0 cycles
- **Current Fleet Status**: Ranked 1 of 100 engines (critical status)

## Primary Telemetry & Degradation Drivers
The primary sensor drivers identified by TreeSHAP attribution and their deviation from baseline (measured over initial 20 cycles):

| Sensor | Symbol | Description | Impact on RUL (cycles) | Direction | Current Value | Healthy Baseline | Baseline Deviation | Recent Slope per Cycle |
|---|---|---|---|---|---|---|---|---|
| `s4` | **T50** | Total temperature at LPT outlet | `-16.4` | shortens predicted life | `1427.83` | `1401.087` | `26.743` | `1.0475` |
| `s11` | **Ps30** | Static pressure at HPC outlet | `-14.6` | shortens predicted life | `48.11` | `47.267` | `0.843` | `0.0175` |
| `s3` | **T30** | Total temperature at HPC outlet | `-13.5` | shortens predicted life | `1598.35` | `1586.814` | `11.536` | `0.2125` |
| `s7` | **P30** | Total pressure at HPC outlet | `-7.1` | shortens predicted life | `551.53` | `554.13` | `-2.601` | `0.0675` |
| `s12` | **phi** | Ratio of fuel flow to Ps30 | `-6.9` | shortens predicted life | `519.67` | `522.018` | `-2.348` | `-0.0125` |

Engine age effect on predicted RUL: 6.9 cycles.

## Diagnostic Assessment
Engine exhibits severe multi-subsystem degradation placing it at the very top of the fleet urgency ranking. The dominant driver of reduced life expectancy is `s4` (T50 - Total temperature at LPT outlet), which drastically shortens predicted life. Thermal stress is compounded by `s11` (Ps30 - Static pressure at HPC outlet). Elevated outlet temperatures and static pressure variations indicate advanced stage turbine and compressor wear, with positive parameter slopes demonstrating accelerating degradation rather than stable cruise performance.

## Operational Action Protocol
- **IMMEDIATE GROUNDING & SHOP INSPECTION**: Remove engine from flight rotation prior to next operational sortie.
- **HOT SECTION BORESCOPE**: Conduct detailed borescope inspection of Low Pressure Turbine (LPT) and High Pressure Compressor (HPC) stages.
- **SEAL & BLADE CLEARANCE VERIFICATION**: Inspect static pressure taps and turbine blade clearance for thermal erosion.

## Model Reliability & Technical Context
- **Model Architecture**: XGBoost
- **Model Test Root Mean Square Error (RMSE)**: 11.824 cycles
- **Model Test Mean Absolute Error (MAE)**: 8.648 cycles
- **Target Training RUL Cap**: 125 cycles
- **Operational Caveat**: Predictions near the cap mean 'at least roughly this many cycles', because the model was trained on RUL capped at that value.