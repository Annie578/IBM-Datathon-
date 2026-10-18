"""Shared helpers for the pipeline scripts (loading raw C-MAPSS files, saving tables)."""
import os

import pandas as pd

OP_COLS = ["op1", "op2", "op3"]
SENSOR_COLS = [f"s{i}" for i in range(1, 22)]
RAW_COLS = ["unit", "cycle"] + OP_COLS + SENSOR_COLS


def read_cmapss(path):
    """Read one whitespace-separated C-MAPSS file (26 columns, no header)."""
    df = pd.read_csv(path, sep=r"\s+", header=None)
    df = df.dropna(axis=1, how="all")  # some copies have a trailing empty column
    if df.shape[1] != len(RAW_COLS):
        raise ValueError(f"{path}: expected {len(RAW_COLS)} columns, found {df.shape[1]}")
    df.columns = RAW_COLS
    df["unit"] = df["unit"].astype(int)
    df["cycle"] = df["cycle"].astype(int)
    return df


def read_rul(path):
    """Read RUL_FDxxx.txt: one true RUL per test engine, in engine order 1..N."""
    values = pd.read_csv(path, sep=r"\s+", header=None).iloc[:, 0].astype(float).to_numpy()
    return pd.DataFrame({"unit": range(1, len(values) + 1), "true_rul": values})


def save_table(df, base_path):
    """Save as parquet if pyarrow works (s390x may lack it), otherwise CSV. Returns the path used."""
    os.makedirs(os.path.dirname(base_path) or ".", exist_ok=True)
    try:
        df.to_parquet(base_path + ".parquet", index=False)
        stale = base_path + ".csv"
        if os.path.exists(stale):
            os.remove(stale)
        return base_path + ".parquet"
    except Exception:  # noqa: BLE001 - any parquet problem means fall back to CSV
        df.to_csv(base_path + ".csv", index=False)
        return base_path + ".csv"


def load_table(base_path):
    """Load a table saved by save_table, whichever format it ended up in."""
    if os.path.exists(base_path + ".parquet"):
        return pd.read_parquet(base_path + ".parquet")
    if os.path.exists(base_path + ".csv"):
        return pd.read_csv(base_path + ".csv")
    raise FileNotFoundError(f"no {base_path}.parquet or .csv found")
