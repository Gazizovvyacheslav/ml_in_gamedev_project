"""exp12 - Ridge regression baseline (дополнение: Участник 2).

A transparent L2-regularised linear baseline on both targets, against DummyRegressor
(mean / median). Numeric features are standardised, categoricals one-hot encoded (rare
categories grouped). Shows how much the non-linear CatBoost improves over a plain linear fit.
"""

from __future__ import annotations

import time

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import p2_common as C
from preprocessing.preprocessing import TargetTransform


def build_model(num_cols, cat_cols, estimator):
    pre = ColumnTransformer(
        [
            ("num", StandardScaler(), num_cols),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", min_frequency=0.01),
                cat_cols,
            ),
        ]
    )
    return Pipeline([("pre", pre), ("est", estimator)])


def run_target(pack, target):
    rows = []
    for mode in ("raw", "p995", "log1p_p995"):
        tfm = TargetTransform(mode).fit(pack.y_train)
        ytr = tfm.transform(pack.y_train)
        for alpha in (0.1, 1.0, 10.0, 100.0):
            t0 = time.time()
            model = build_model(pack.num_cols, pack.cat_cols, Ridge(alpha=alpha))
            model.fit(pack.x_train, ytr)
            fit_sec = time.time() - t0
            r = dict(
                experiment="ridge",
                target=target,
                model_name=f"ridge_a{alpha}_{mode}",
                alpha=alpha,
                target_mode=mode,
                fit_sec=fit_sec,
            )
            for split, X, y in [
                ("val", pack.x_val, pack.y_val),
                ("test", pack.x_test, pack.y_test),
            ]:
                pred = tfm.inverse(model.predict(X))
                for k, v in C.metric_pack(y, pred).items():
                    r[f"{split}_{k}"] = v
            rows.append(r)

    for strat in ("mean", "median"):
        dm = DummyRegressor(strategy=strat).fit(pack.x_train, pack.y_train)
        r = dict(
            experiment="ridge",
            target=target,
            model_name=f"dummy_{strat}",
            alpha=np.nan,
            target_mode="raw",
            fit_sec=0.0,
        )
        for split, X, y in [
            ("val", pack.x_val, pack.y_val),
            ("test", pack.x_test, pack.y_test),
        ]:
            pred = np.maximum(dm.predict(X), 0.0)
            for k, v in C.metric_pack(y, pred).items():
                r[f"{split}_{k}"] = v
        rows.append(r)
    return rows


def main():
    rows = []
    for target in (C.NEXT_TARGET, C.CRM_TARGET):
        try:
            pack = C.get_pack(target_cols=(target,))[target]
        except Exception as e:
            print(f"[{target}] skipped: {type(e).__name__}: {e}")
            continue
        tr = run_target(pack, target)
        rows += tr
        best = min(tr, key=lambda x: x["val_mae"])
        print(
            f"[{target}] best: {best['model_name']} "
            f"val_mae={best['val_mae']:.2f} test_mae={best['test_mae']:.2f}"
        )

    res = C.save_results(rows, "exp12_ridge")
    for target in res.target.unique():
        sub = res[res.target == target].sort_values("val_mae").head(5)
        print(f"\n== {target} (top-5 by val_mae) ==")
        print(
            sub[
                ["model_name", "val_mae", "test_mae", "val_r2", "val_small_mae"]
            ].to_string(index=False)
        )


if __name__ == "__main__":
    main()
