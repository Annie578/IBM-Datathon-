"""Step 1: load C-MAPSS, cap RUL, build per-engine rolling features, split by engine.

Run from the repo root:
    python scripts/01_load_and_features.py                 # FD001 by default
    python scripts/01_load_and_features.py --subset FD003

Reads   data/raw/train_FD001.txt, test_FD001.txt, RUL_FD001.txt
Writes  data/processed/train_features(.parquet|.csv)      all training rows, with a train/val split column
        data/processed/test_features(.parquet|.csv)       every test row (used for sensor trend plots)
        data/processed/test_last(.parquet|.csv)           last cycle of each test engine (used for scoring)
        data/processed/feature_cols.json                  the feature list every later script uses
        data/processed/data_summary.json                  counts, dropped sensors, validation engines
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

from common import OP_COLS, SENSOR_COLS, read_cmapss, read_rul, save_table


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--subset", default="FD001", help="C-MAPSS subset name, e.g. FD001")
    p.add_argument("--data-dir", default="data/raw")
    p.add_argument("--out-dir", default="data/processed")
    p.add_argument("--cap", type=int, default=125, help="cap on the training RUL target (cycles)")
    p.add_argument("--val-frac", type=float, default=0.2, help="fraction of TRAIN engines held out for validation")
    p.add_argument("--baseline-cycles", type=int, default=20, help="early cycles that define each engine's healthy baseline")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def find_constant_columns(train, cols, tol=1e-6):
    """Columns that never change in the training data carry no information."""
    return [c for c in cols if train[c].std() < tol]


def add_features(df, base_cols, baseline_cycles):
    """Per-engine features. Only past and current cycles are used, never future ones."""
    df = df.sort_values(["unit", "cycle"]).reset_index(drop=True)
    grouped = df.groupby("unit")
    pieces = {"cycle": df["cycle"]}

    for c in base_cols:
        pieces[c] = df[c]

    for c in base_cols:
        g = grouped[c]
        pieces[f"{c}_mean5"] = g.transform(lambda s: s.rolling(5, min_periods=1).mean())
        pieces[f"{c}_mean15"] = g.transform(lambda s: s.rolling(15, min_periods=1).mean())
        pieces[f"{c}_std5"] = g.transform(lambda s: s.rolling(5, min_periods=2).std()).fillna(0.0)
        # Deviation from this engine's own healthy baseline (first N cycles). The LLM report uses this too.
        baseline = g.transform(lambda s: s.iloc[:baseline_cycles].mean())
        pieces[f"{c}_dev"] = df[c] - baseline
        # Slope over the last 5 cycles: how fast the sensor is moving right now.
        pieces[f"{c}_slope5"] = g.transform(lambda s: (s - s.shift(4)) / 4.0).fillna(0.0)

    feats = pd.DataFrame(pieces)
    return pd.concat([df[["unit", "cycle"]].rename(columns={"cycle": "cycle_id"}), feats], axis=1)


def main():
    args = parse_args()
    sub = args.subset
    train_raw = read_cmapss(os.path.join(args.data_dir, f"train_{sub}.txt"))
    test_raw = read_cmapss(os.path.join(args.data_dir, f"test_{sub}.txt"))
    rul = read_rul(os.path.join(args.data_dir, f"RUL_{sub}.txt"))

    n_train_units = train_raw["unit"].nunique()
    n_test_units = test_raw["unit"].nunique()
    if len(rul) != n_test_units:
        raise ValueError(f"RUL file has {len(rul)} rows but test has {n_test_units} engines")

    candidates = OP_COLS + SENSOR_COLS
    dropped = find_constant_columns(train_raw, candidates)
    kept_sensors = [c for c in SENSOR_COLS if c not in dropped]
    kept_ops = [c for c in OP_COLS if c not in dropped]
    base_cols = kept_ops + kept_sensors
    print(f"{sub}: {n_train_units} train engines, {n_test_units} test engines")
    print(f"dropped constant columns ({len(dropped)}): {dropped}")
    print(f"keeping {len(base_cols)} base columns: {base_cols}")

    # ---- training: RUL target, capped ----
    train = add_features(train_raw, base_cols, args.baseline_cycles)
    max_cycle = train.groupby("unit")["cycle_id"].transform("max")
    train["rul"] = max_cycle - train["cycle_id"]
    train["rul_capped"] = train["rul"].clip(upper=args.cap)

    # ---- split by ENGINE, never by row ----
    rng = np.random.default_rng(args.seed)
    units = np.array(sorted(train["unit"].unique()))
    n_val = max(1, int(round(len(units) * args.val_frac)))
    val_units = sorted(rng.choice(units, size=n_val, replace=False).tolist())
    train["split"] = np.where(train["unit"].isin(val_units), "val", "train")

    # ---- test: features for every row, plus the last row of each engine for scoring ----
    test = add_features(test_raw, base_cols, args.baseline_cycles)
    test_last = test.sort_values(["unit", "cycle_id"]).groupby("unit").tail(1).reset_index(drop=True)
    test_last = test_last.merge(rul, on="unit", how="left")
    test_last["true_rul_capped"] = test_last["true_rul"].clip(upper=args.cap)

    feature_cols = [c for c in train.columns if c not in ("unit", "cycle_id", "rul", "rul_capped", "split")]

    paths = {
        "train_features": save_table(train, os.path.join(args.out_dir, "train_features")),
        "test_features": save_table(test, os.path.join(args.out_dir, "test_features")),
        "test_last": save_table(test_last, os.path.join(args.out_dir, "test_last")),
    }
    with open(os.path.join(args.out_dir, "feature_cols.json"), "w") as f:
        json.dump(feature_cols, f, indent=2)

    summary = {
        "subset": sub,
        "rul_cap": args.cap,
        "train_engines": int(n_train_units),
        "test_engines": int(n_test_units),
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "dropped_constant_columns": dropped,
        "base_columns": base_cols,
        "n_features": len(feature_cols),
        "val_engines": val_units,
        "files": paths,
    }
    with open(os.path.join(args.out_dir, "data_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # ---- sanity checks that fail loudly if something is wrong ----
    assert train.groupby("unit")["rul"].min().eq(0).all(), "every train engine must end at RUL 0"
    assert not train[feature_cols].isna().any().any(), "NaNs in train features"
    assert not test_last[feature_cols].isna().any().any(), "NaNs in test features"
    overlap = set(train.loc[train.split == "train", "unit"]) & set(val_units)
    assert not overlap, "an engine appears in both train and val"

    print(f"{len(feature_cols)} features, {len(train):,} train rows ({len(val_units)} engines held out for validation)")
    print("saved:", *paths.values(), sep="\n  ")
    print("next: python scripts/02_train.py")


if __name__ == "__main__":
    main()
