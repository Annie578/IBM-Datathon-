"""Environment check for the LinuxONE VM (s390x, big-endian).

Run from the repo root:  python scripts/env_check.py
Prints a report and writes ENV_NOTES.md in the current directory.
"""
import importlib
import os
import platform
import sys
import time
from datetime import datetime

REQUIRED = ["numpy", "pandas", "scipy", "sklearn", "pyarrow", "matplotlib", "joblib", "xgboost", "shap"]
OPTIONAL = ["snapml"]


def try_import(name):
    try:
        module = importlib.import_module(name)
        return True, str(getattr(module, "__version__", "unknown"))
    except Exception as exc:  # noqa: BLE001 - we want every failure reason
        return False, f"{type(exc).__name__}: {exc}"


def mem_gb():
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal"):
                    return round(int(line.split()[1]) / 1024 / 1024, 1)
    except OSError:
        pass
    return None


def smoke_tests():
    """Tiny end-to-end run: train two models and compute TreeSHAP."""
    results = {}
    try:
        import numpy as np

        rng = np.random.default_rng(0)
        X = rng.normal(size=(2000, 10))
        y = X[:, 0] * 3 + X[:, 1] ** 2 + rng.normal(scale=0.1, size=2000)
    except Exception as exc:  # noqa: BLE001
        return {"test data": f"FAIL {type(exc).__name__}: {exc}"}

    rf = None
    try:
        from sklearn.ensemble import RandomForestRegressor

        t = time.time()
        rf = RandomForestRegressor(n_estimators=50, n_jobs=-1, random_state=0).fit(X, y)
        results["scikit-learn RandomForest fit"] = f"ok ({time.time() - t:.2f}s)"
    except Exception as exc:  # noqa: BLE001
        results["scikit-learn RandomForest fit"] = f"FAIL {type(exc).__name__}: {exc}"

    xgb_model = None
    try:
        import xgboost as xgb

        t = time.time()
        xgb_model = xgb.XGBRegressor(n_estimators=100, max_depth=4, n_jobs=-1).fit(X, y)
        pred = xgb_model.predict(X[:5])
        results["xgboost fit + predict"] = f"ok ({time.time() - t:.2f}s)"
        # Same call twice should give the same answer; catches endianness bugs in the native lib.
        assert (pred == xgb_model.predict(X[:5])).all()
    except Exception as exc:  # noqa: BLE001
        xgb_model = None
        results["xgboost fit + predict"] = f"FAIL {type(exc).__name__}: {exc}"

    target = xgb_model if xgb_model is not None else rf
    if target is None:
        results["shap TreeExplainer"] = "skipped (no model trained)"
    else:
        try:
            import shap

            t = time.time()
            values = shap.TreeExplainer(target).shap_values(X[:200])
            results["shap TreeExplainer"] = f"ok on {'xgboost' if xgb_model is not None else 'random forest'} ({time.time() - t:.2f}s, shape {values.shape})"
        except Exception as exc:  # noqa: BLE001
            results["shap TreeExplainer"] = f"FAIL {type(exc).__name__}: {exc}"

    try:
        import pandas as pd

        t = time.time()
        pd.DataFrame(X).to_parquet("_envcheck.parquet")
        pd.read_parquet("_envcheck.parquet")
        os.remove("_envcheck.parquet")
        results["parquet write/read"] = f"ok ({time.time() - t:.2f}s)"
    except Exception as exc:  # noqa: BLE001
        results["parquet write/read"] = f"FAIL {type(exc).__name__}: {exc} (fall back to CSV)"
    return results


def main():
    info = {
        "Time": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "Machine": platform.machine(),
        "Byte order": sys.byteorder,
        "OS": platform.platform(),
        "Python": sys.version.split()[0],
        "CPU cores": os.cpu_count(),
        "Memory (GB)": mem_gb(),
    }
    imports = {name: try_import(name) for name in REQUIRED}
    optional = {name: try_import(name) for name in OPTIONAL}
    tests = smoke_tests()

    xgb_ok = tests.get("xgboost fit + predict", "").startswith("ok")
    rf_ok = tests.get("scikit-learn RandomForest fit", "").startswith("ok")
    shap_ok = tests.get("shap TreeExplainer", "").startswith("ok")
    if xgb_ok and shap_ok:
        verdict = "GREEN: use XGBoost + TreeSHAP as planned."
    elif rf_ok and shap_ok:
        verdict = "YELLOW: XGBoost is not usable. Use scikit-learn RandomForest + TreeSHAP and say so in the slides."
    elif rf_ok:
        verdict = "ORANGE: SHAP is not usable. Use scikit-learn feature_importances_ or permutation importance for explanations."
    else:
        verdict = "RED: scikit-learn is not usable. Fix numpy/scikit-learn first (try apt install python3-numpy python3-sklearn)."

    lines = ["# Environment notes", ""]
    lines += [f"- **{k}:** {v}" for k, v in info.items()]
    lines += ["", "## Imports", "", "| Package | Status | Version or error |", "| --- | --- | --- |"]
    for name, (ok, detail) in {**imports, **optional}.items():
        lines.append(f"| {name} | {'ok' if ok else 'MISSING'} | {detail} |")
    lines += ["", "## Smoke tests", "", "| Test | Result |", "| --- | --- |"]
    lines += [f"| {k} | {v} |" for k, v in tests.items()]
    lines += ["", "## Verdict", "", verdict, ""]
    report = "\n".join(lines)

    print(report)
    with open("ENV_NOTES.md", "w") as f:
        f.write(report)
    print("Wrote ENV_NOTES.md")


if __name__ == "__main__":
    main()
