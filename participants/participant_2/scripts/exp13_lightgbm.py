"""exp13 - LightGBM as an alternative boosting framework (Эксперимент 1).

Compact random search (objective regression_l1 / quantile, n_estimators, lr, num_leaves,
max_depth, feature/bagging fraction) on the next-session target, compared with the CatBoost
reference. The point is to justify the CatBoost choice by comparison, not by assumption.
Categorical features are passed natively via pandas 'category' dtype.
"""

from __future__ import annotations

import json
import random
import time


import p2_common as C
from preprocessing.preprocessing import TargetTransform


def to_cat(df, cat_cols):
    df = df.copy()
    for c in cat_cols:
        df[c] = df[c].astype("category")
    return df


def main():
    try:
        import lightgbm as lgb
    except OSError as e:
        print(f"LightGBM unavailable (likely missing libomp): {e}")
        C.save_results(
            [dict(experiment="lightgbm", model_name="unavailable", status=str(e))],
            "exp13_lightgbm",
        )
        return

    packs = C.get_pack(target_cols=(C.NEXT_TARGET,))
    p = packs[C.NEXT_TARGET]
    cat = p.cat_cols
    xtr = to_cat(p.x_train, cat)
    xva = to_cat(p.x_val, cat)
    xte = to_cat(p.x_test, cat)

    grid = dict(
        objective=["regression_l1", "quantile"],
        alpha=[0.40, 0.50],
        n_estimators=[600, 1000, 1400],
        learning_rate=[0.02, 0.03, 0.05],
        num_leaves=[31, 63],
        max_depth=[-1, 6, 8],
        feature_fraction=[0.8, 1.0],
        bagging_fraction=[0.8, 1.0],
        target_mode=["p995", "log1p_p995"],
    )
    rng = random.Random(42)
    configs, seen = [], set()
    while len(configs) < 14:
        cfg = {k: rng.choice(v) for k, v in grid.items()}
        if cfg["objective"] != "quantile":
            cfg["alpha"] = 0.5
        key = tuple(sorted(cfg.items()))
        if key in seen:
            continue
        seen.add(key)
        configs.append(cfg)

    rows = []
    for i, cfg in enumerate(configs):
        tfm = TargetTransform(cfg["target_mode"]).fit(p.y_train)
        params = dict(
            objective=cfg["objective"],
            n_estimators=cfg["n_estimators"],
            learning_rate=cfg["learning_rate"],
            num_leaves=cfg["num_leaves"],
            max_depth=cfg["max_depth"],
            feature_fraction=cfg["feature_fraction"],
            bagging_fraction=cfg["bagging_fraction"],
            bagging_freq=1,
            random_state=42,
            n_jobs=-1,
            verbosity=-1,
        )
        if cfg["objective"] == "quantile":
            params["alpha"] = cfg["alpha"]
        model = lgb.LGBMRegressor(**params)
        t0 = time.time()
        model.fit(
            xtr,
            tfm.transform(p.y_train),
            eval_set=[(xva, tfm.transform(p.y_val))],
            callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(0)],
        )
        fit_sec = time.time() - t0
        r = dict(
            experiment="lightgbm",
            model_name=f"lgb{i:02d}",
            model_family="lightgbm",
            **cfg,
            fit_sec=fit_sec,
        )
        for split, X, y in [("val", xva, p.y_val), ("test", xte, p.y_test)]:
            pred = tfm.inverse(model.predict(X))
            for k, v in C.metric_pack(y, pred).items():
                r[f"{split}_{k}"] = v
        rows.append(r)
        print(
            f"[{i + 1}/{len(configs)}] {cfg['objective']} val_mae={r['val_mae']:.2f} "
            f"test_mae={r['test_mae']:.2f} fit={fit_sec:.1f}s"
        )

    res = C.save_results(rows, "exp13_lightgbm")
    best = res.sort_values("val_mae").iloc[0]
    print(
        "\nbest LightGBM by val_mae:",
        best["model_name"],
        f"val_mae={best['val_mae']:.2f} test_mae={best['test_mae']:.2f} fit={best['fit_sec']:.1f}s",
    )

    ref = C.OUTPUT_DIR / "best_main_model.json"
    if ref.exists():
        cb = json.loads(ref.read_text())
        print(f"CatBoost reference (exp8): test_mae={cb['test_mae']:.2f}")


if __name__ == "__main__":
    main()
