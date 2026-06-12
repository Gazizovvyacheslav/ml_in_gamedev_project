"""Aggregate all exp*.csv into the protocol-standard participant2_results.csv.

Output columns follow team_modeling_protocol.txt section 16.
"""

from __future__ import annotations

import glob
import json

import numpy as np
import pandas as pd

import p2_common as C

METRIC_COLS = [
    "mae",
    "product_mae",
    "engagement_risk_mae",
    "medae",
    "p70_abs_error",
    "p90_abs_error",
    "r2",
    "small_mae",
    "normal_mae",
    "long_mae",
]

SCHEMA = (
    [
        "target",
        "model_family",
        "model_name",
        "objective_tag",
        "target_mode",
        "loss_function",
        "feature_set",
        "params",
        "fit_sec",
        "status",
    ]
    + [f"val_{m}" for m in METRIC_COLS]
    + [f"test_{m}" for m in METRIC_COLS]
    + ["backtest_mae_mean", "backtest_mae_std"]
)

PARAM_KEYS = [
    "depth",
    "learning_rate",
    "l2_leaf_reg",
    "iterations",
    "od_wait",
    "bootstrap_type",
    "subsample",
    "bagging_temperature",
    "one_hot_max_size",
    "max_ctr_complexity",
    "ctr_target_border_count",
    "grow_policy",
    "min_data_in_leaf",
    "max_leaves",
    "border_count",
    "feature_border_type",
    "rsm",
]


def main():
    files = sorted(glob.glob(str(C.OUTPUT_DIR / "exp*.csv")))
    frames = []
    for f in files:
        d = pd.read_csv(f)
        d["__source"] = f.split("/")[-1]
        frames.append(d)
    if not frames:
        print("no exp*.csv found")
        return
    df = pd.concat(frames, ignore_index=True, sort=False)

    def col(name, default=np.nan):
        """Return df[name] as a Series, or a constant Series if the column is absent."""
        if name in df.columns:
            return df[name]
        return pd.Series([default] * len(df), index=df.index)

    out = pd.DataFrame()
    out["target"] = col("target", C.NEXT_TARGET).fillna(C.NEXT_TARGET)
    out["model_family"] = "catboost"
    out["model_name"] = df["model_name"]
    out["objective_tag"] = df["experiment"]
    out["target_mode"] = col("target_mode", "p995").fillna("p995")
    out["loss_function"] = col("loss_function", "MAE").fillna("MAE")
    nfeat = col("n_features")
    out["feature_set"] = np.where(
        nfeat.notna(),
        "n_feat=" + nfeat.astype("Float64").astype("Int64").astype(str),
        "all",
    )

    def _params(row):
        d = {k: row[k] for k in PARAM_KEYS if k in row and pd.notna(row[k])}
        return json.dumps(d, default=float)

    out["params"] = df.apply(_params, axis=1)
    out["fit_sec"] = df.get("fit_sec")
    out["status"] = "ok"
    for m in METRIC_COLS:
        out[f"val_{m}"] = df.get(f"val_{m}")
        out[f"test_{m}"] = df.get(f"test_{m}")
    out["backtest_mae_mean"] = np.nan
    out["backtest_mae_std"] = df.get("val_mae_std")

    out = out[SCHEMA]
    path = C.OUTPUT_DIR / "participant2_results.csv"
    out.to_csv(path, index=False)
    print(f"[saved] {path}  ({len(out)} rows, {len(out.columns)} cols)")

    lb = (
        out.sort_values("val_mae")
        .groupby("objective_tag", as_index=False)
        .first()[["objective_tag", "model_name", "val_mae", "test_mae", "fit_sec"]]
    )
    print("\nBest per experiment (by val_mae):\n", lb.to_string(index=False))


if __name__ == "__main__":
    main()
