"""Step 3: explain the predictions and write the files the rest of the team uses.

Run from the repo root (after 01 and 02):
    python scripts/03_explain.py

Reads   models/rul_model.json (or .joblib), data/processed/*, outputs/metrics.json
Writes  outputs/fleet.csv                       every test engine ranked by predicted RUL (rank 1 = most urgent)
        outputs/explanations/engine_NNN.json    one file per engine: THIS is the only input the LLM gets
        outputs/global_importance.json          which sensors matter most across the fleet
        outputs/demo_engines.json               one critical, one middle, one healthy engine for the demo

Design rules for the engine JSON files:
  * The true RUL is NOT included. The LLM must explain the model, not see the answer key.
  * Every number the report may quote is in the file, already rounded.
  * Sensor names and descriptions come from the fixed table below (standard C-MAPSS documentation),
    so the LLM never has to invent what a sensor measures.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

from common import load_table

# Standard C-MAPSS sensor documentation (Saxena et al., 2008). Operating settings are listed too.
SENSOR_INFO = {
    "op1": ("altitude", "Operating setting 1 (altitude)"),
    "op2": ("Mach", "Operating setting 2 (Mach number)"),
    "op3": ("TRA", "Operating setting 3 (throttle resolver angle)"),
    "s1": ("T2", "Total temperature at fan inlet"),
    "s2": ("T24", "Total temperature at LPC outlet"),
    "s3": ("T30", "Total temperature at HPC outlet"),
    "s4": ("T50", "Total temperature at LPT outlet"),
    "s5": ("P2", "Pressure at fan inlet"),
    "s6": ("P15", "Total pressure in bypass duct"),
    "s7": ("P30", "Total pressure at HPC outlet"),
    "s8": ("Nf", "Physical fan speed"),
    "s9": ("Nc", "Physical core speed"),
    "s10": ("epr", "Engine pressure ratio"),
    "s11": ("Ps30", "Static pressure at HPC outlet"),
    "s12": ("phi", "Ratio of fuel flow to Ps30"),
    "s13": ("NRf", "Corrected fan speed"),
    "s14": ("NRc", "Corrected core speed"),
    "s15": ("BPR", "Bypass ratio"),
    "s16": ("farB", "Burner fuel-air ratio"),
    "s17": ("htBleed", "Bleed enthalpy"),
    "s18": ("Nf_dmd", "Demanded fan speed"),
    "s19": ("PCNfR_dmd", "Demanded corrected fan speed"),
    "s20": ("W31", "HPT coolant bleed"),
    "s21": ("W32", "LPT coolant bleed"),
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["xgb", "rf"], default=None, help="default: whatever outputs/metrics.json says")
    p.add_argument("--data-dir", default="data/processed")
    p.add_argument("--model-dir", default="models")
    p.add_argument("--out-dir", default="outputs")
    p.add_argument("--critical", type=float, default=30.0, help="predicted RUL (cycles) below this = critical")
    p.add_argument("--warning", type=float, default=60.0, help="predicted RUL (cycles) below this = warning")
    p.add_argument("--top-k", type=int, default=5, help="sensors listed per engine")
    p.add_argument("--baseline-cycles", type=int, default=20, help="must match the value used in step 1")
    return p.parse_args()


def group_name(col):
    """'s11_mean5' -> 's11', 'op1_dev' -> 'op1', 'cycle' -> 'cycle'."""
    return col.split("_")[0]


def load_model(kind, model_dir):
    if kind == "xgb":
        import xgboost as xgb

        m = xgb.XGBRegressor()
        m.load_model(os.path.join(model_dir, "rul_model.json"))
        return m
    import joblib

    return joblib.load(os.path.join(model_dir, "rul_model.joblib"))


def tree_shap(model, kind, X):
    """Exact TreeSHAP. Returns (values [n, features], base_values [n]). Values are in RUL cycles."""
    if kind == "xgb":
        import xgboost as xgb

        # XGBoost computes TreeSHAP itself, so this works even if the separate shap package is missing.
        contrib = model.get_booster().predict(xgb.DMatrix(X), pred_contribs=True)
        return contrib[:, :-1], contrib[:, -1]
    import shap

    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(X)
    base = float(np.ravel(explainer.expected_value)[0])
    return values, np.full(len(X), base)


def r(x, nd=2):
    return round(float(x), nd)


def risk_level(rul, critical, warning):
    if rul < critical:
        return "critical"
    if rul < warning:
        return "warning"
    return "healthy"


def main():
    args = parse_args()
    out_dir = args.out_dir
    os.makedirs(os.path.join(out_dir, "explanations"), exist_ok=True)

    with open(os.path.join(out_dir, "metrics.json")) as f:
        metrics = json.load(f)
    kind = args.model or metrics["model"]
    with open(os.path.join(args.data_dir, "feature_cols.json")) as f:
        feature_cols = json.load(f)
    with open(os.path.join(args.data_dir, "data_summary.json")) as f:
        summary = json.load(f)
    test_last = load_table(os.path.join(args.data_dir, "test_last")).sort_values("unit").reset_index(drop=True)

    model = load_model(kind, args.model_dir)
    X = test_last[feature_cols]
    raw_pred = model.predict(X)
    values, base = tree_shap(model, kind, X)

    # SHAP must add up to the prediction. If it does not, something is wrong, so stop loudly.
    gap = np.abs(values.sum(axis=1) + base - raw_pred).max()
    print(f"SHAP check: max |sum(SHAP) + base - prediction| = {gap:.4f} cycles")
    if gap > 0.5:
        raise RuntimeError("SHAP values do not add up to the model prediction; refusing to write explanations")

    # Collapse the ~100 engineered features into one effect per sensor.
    groups = {}
    for j, col in enumerate(feature_cols):
        groups.setdefault(group_name(col), []).append(j)
    effect = pd.DataFrame({g: values[:, idx].sum(axis=1) for g, idx in groups.items()})
    sensor_groups = [g for g in effect.columns if g != "cycle"]

    pred = np.clip(raw_pred, 0, None)
    order = np.argsort(pred, kind="stable")
    rank_of = np.empty(len(pred), dtype=int)
    rank_of[order] = np.arange(1, len(pred) + 1)

    model_context = {
        "model": "XGBoost" if kind == "xgb" else "RandomForest",
        "test_rmse_cycles": metrics["test_vs_true_rul"]["rmse"],
        "test_mae_cycles": metrics["test_vs_true_rul"]["mae"],
        "rul_cap_cycles": metrics["rul_cap"],
        "caveat": "Predictions near the cap mean 'at least roughly this many cycles', because the model was trained on RUL capped at that value.",
    }
    thresholds = {"critical_below_cycles": args.critical, "warning_below_cycles": args.warning}

    fleet_rows = []
    drivers_by_unit = {}
    for i, row in test_last.iterrows():
        unit = int(row["unit"])
        shap_row = effect.iloc[i]
        top = shap_row[sensor_groups].abs().sort_values(ascending=False).head(args.top_k).index.tolist()
        drivers = []
        for g in top:
            if g not in SENSOR_INFO or g not in row.index:
                continue
            value = float(row[g])
            dev = float(row[f"{g}_dev"])
            symbol, desc = SENSOR_INFO[g]
            s = float(shap_row[g])
            drivers.append(
                {
                    "sensor": g,
                    "symbol": symbol,
                    "description": desc,
                    "effect_on_rul_cycles": r(s, 1),
                    "direction": "shortens predicted life" if s < 0 else "lengthens predicted life",
                    "current_value": r(value, 3),
                    "healthy_baseline_value": r(value - dev, 3),
                    "deviation_from_baseline": r(dev, 3),
                    "recent_slope_per_cycle": r(row[f"{g}_slope5"], 4),
                }
            )
        drivers_by_unit[unit] = drivers

        p = float(pred[i])
        level = risk_level(p, args.critical, args.warning)
        cycles = int(row["cycle_id"])
        doc = {
            "schema_version": 1,
            "engine_id": unit,
            "subset": summary["subset"],
            "cycles_observed": cycles,
            "predicted_rul_cycles": r(p, 1),
            "risk_level": level,
            "risk_thresholds": thresholds,
            "fleet_rank_most_urgent_first": int(rank_of[i]),
            "fleet_size": int(len(test_last)),
            "fleet_average_prediction_cycles": r(base[i], 1),
            "engine_age_effect_on_rul_cycles": r(shap_row["cycle"], 1) if "cycle" in effect.columns else None,
            "baseline_cycles_used": int(min(cycles, args.baseline_cycles)),
            "top_drivers": drivers,
            "model_context": model_context,
        }
        with open(os.path.join(out_dir, "explanations", f"engine_{unit:03d}.json"), "w") as f:
            json.dump(doc, f, indent=2)

        fleet_rows.append(
            {
                "rank": int(rank_of[i]),
                "unit": unit,
                "cycles_observed": cycles,
                "predicted_rul": r(p, 1),
                "risk_level": level,
                "top_driver": drivers[0]["sensor"] if drivers else "",
                "top_driver_symbol": drivers[0]["symbol"] if drivers else "",
                "true_rul": r(row["true_rul"], 1),
                "error": r(p - float(row["true_rul"]), 1),
            }
        )

    fleet = pd.DataFrame(fleet_rows).sort_values("rank")
    fleet.to_csv(os.path.join(out_dir, "fleet.csv"), index=False)

    importance = effect[sensor_groups].abs().mean().sort_values(ascending=False)
    gi = [
        {
            "sensor": g,
            "symbol": SENSOR_INFO.get(g, ("", ""))[0],
            "description": SENSOR_INFO.get(g, ("", ""))[1],
            "mean_abs_effect_cycles": r(v, 2),
        }
        for g, v in importance.items()
    ]
    with open(os.path.join(out_dir, "global_importance.json"), "w") as f:
        json.dump(gi, f, indent=2)

    n = len(fleet)
    demo = {
        "critical": int(fleet.iloc[0]["unit"]),
        "middle": int(fleet.iloc[n // 2]["unit"]),
        "healthy": int(fleet.iloc[-1]["unit"]),
        "note": "Chosen only by predicted RUL (lowest, median, highest), not by how accurate the prediction was.",
    }
    with open(os.path.join(out_dir, "demo_engines.json"), "w") as f:
        json.dump(demo, f, indent=2)

    counts = fleet["risk_level"].value_counts().to_dict()
    print(f"wrote {n} engine files to {out_dir}/explanations/, fleet.csv, global_importance.json, demo_engines.json")
    print("risk levels:", counts)
    print("top sensors overall:", ", ".join(f"{d['sensor']} ({d['symbol']})" for d in gi[:5]))
    print("demo engines:", demo)
    print(fleet.head(5).to_string(index=False))
    print("next: send outputs/explanations/engine_*.json to the LLM lead, fleet.csv to Member 4")


if __name__ == "__main__":
    main()
