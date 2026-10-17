"""Step 2: train the RUL model and report RMSE, MAE and the NASA score.

Run from the repo root (after 01_load_and_features.py):
    python scripts/02_train.py                # XGBoost (default)
    python scripts/02_train.py --model rf     # scikit-learn Random Forest fallback

Reads   data/processed/train_features, test_last, feature_cols.json
Writes  models/rul_model.json (XGBoost) or models/rul_model.joblib (Random Forest)
        outputs/metrics.json
        outputs/test_predictions.csv     one row per test engine: true and predicted RUL
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd

from common import load_table


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["xgb", "rf"], default="xgb")
    p.add_argument("--data-dir", default="data/processed")
    p.add_argument("--model-dir", default="models")
    p.add_argument("--out-dir", default="outputs")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(p) - np.asarray(y)) ** 2)))


def mae(y, p):
    return float(np.mean(np.abs(np.asarray(p) - np.asarray(y))))


def nasa_score(y, p):
    """NASA asymmetric score. Lower is better. Late predictions (pred > true) are punished harder."""
    d = np.asarray(p, dtype=float) - np.asarray(y, dtype=float)
    return float(np.sum(np.where(d < 0, np.exp(-d / 13.0) - 1.0, np.exp(d / 10.0) - 1.0)))


def all_metrics(y, p):
    return {"rmse": round(rmse(y, p), 3), "mae": round(mae(y, p), 3), "nasa_score": round(nasa_score(y, p), 1)}


def build_model(kind, seed):
    if kind == "xgb":
        import xgboost as xgb

        return xgb.XGBRegressor(
            n_estimators=1500,
            learning_rate=0.03,
            max_depth=5,
            min_child_weight=5,
            subsample=0.8,
            colsample_bytree=0.7,
            reg_lambda=2.0,
            tree_method="hist",
            early_stopping_rounds=50,
            n_jobs=-1,
            random_state=seed,
        )
    from sklearn.ensemble import RandomForestRegressor

    return RandomForestRegressor(n_estimators=300, min_samples_leaf=5, max_features=0.5, n_jobs=-1, random_state=seed)


def main():
    args = parse_args()
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)

    with open(os.path.join(args.data_dir, "feature_cols.json")) as f:
        feature_cols = json.load(f)
    train_all = load_table(os.path.join(args.data_dir, "train_features"))
    test_last = load_table(os.path.join(args.data_dir, "test_last"))

    tr = train_all[train_all["split"] == "train"]
    va = train_all[train_all["split"] == "val"]
    if tr["unit"].isin(va["unit"]).any():
        raise RuntimeError("an engine is in both train and val: re-run 01_load_and_features.py")
    print(f"train: {len(tr):,} rows / {tr['unit'].nunique()} engines   val: {len(va):,} rows / {va['unit'].nunique()} engines")

    X_tr, y_tr = tr[feature_cols], tr["rul_capped"]
    X_va, y_va = va[feature_cols], va["rul_capped"]
    X_te = test_last[feature_cols]

    model = build_model(args.model, args.seed)
    t0 = time.time()
    best_iter = None
    if args.model == "xgb":
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        best_iter = int(model.best_iteration) + 1
    else:
        model.fit(X_tr, y_tr)
    fit_seconds = round(time.time() - t0, 2)
    print(f"{args.model} trained in {fit_seconds}s" + (f", best iteration {best_iter}" if best_iter else ""))

    def predict(X):
        return np.clip(model.predict(X), 0, None)

    # Validation (held-out engines, capped target, every cycle).
    val_metrics = all_metrics(y_va, predict(X_va))

    # Test: one prediction per engine at its last observed cycle.
    pred_te = predict(X_te)
    test_raw = all_metrics(test_last["true_rul"], pred_te)                # against the RUL values in RUL_FDxxx.txt
    test_capped = all_metrics(test_last["true_rul_capped"], pred_te)      # against RUL capped like the training target

    # Context: what does a model that ignores the sensors score?
    naive = np.full(len(test_last), float(tr["rul_capped"].mean()))
    naive_metrics = all_metrics(test_last["true_rul"], naive)

    metrics = {
        "model": args.model,
        "rul_cap": int(train_all["rul_capped"].max()),
        "n_features": len(feature_cols),
        "fit_seconds": fit_seconds,
        "best_iteration": best_iter,
        "validation_capped": val_metrics,
        "test_vs_true_rul": test_raw,
        "test_vs_capped_rul": test_capped,
        "baseline_predict_mean_vs_true_rul": naive_metrics,
        "test_engines": int(len(test_last)),
        "note": "test_vs_true_rul is the headline number. NASA score is summed over engines, so it grows with engine count.",
    }
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    preds = pd.DataFrame(
        {
            "unit": test_last["unit"],
            "cycle_id": test_last["cycle_id"],
            "true_rul": test_last["true_rul"],
            "pred_rul": np.round(pred_te, 1),
        }
    )
    preds["error"] = (preds["pred_rul"] - preds["true_rul"]).round(1)
    preds.to_csv(os.path.join(args.out_dir, "test_predictions.csv"), index=False)

    if args.model == "xgb":
        model_path = os.path.join(args.model_dir, "rul_model.json")  # JSON is byte-order independent, safe on s390x
        model.save_model(model_path)
    else:
        import joblib

        model_path = os.path.join(args.model_dir, "rul_model.joblib")
        joblib.dump(model, model_path)

    print()
    print("TEST (last cycle of each engine, vs true RUL):", test_raw)
    print("TEST (vs capped RUL):                         ", test_capped)
    print("VALIDATION (held-out train engines, capped):  ", val_metrics)
    print("BASELINE (always predict the mean):           ", naive_metrics)
    print(f"saved: {model_path}, {args.out_dir}/metrics.json, {args.out_dir}/test_predictions.csv")
    print("next: python scripts/03_explain.py")


if __name__ == "__main__":
    main()
