"""exp8 - main CatBoost sweep by MAE (Эксперимент 2, distribution_of_responsoblities).

Wider random search on the next-session target over the doc's grid (iterations, depth, lr,
l2, min_data_in_leaf, random_strength, bootstrap, target_transform, clip_mode), selection by
validation MAE, then a short local-tuning note around the best config. This is the technical
best-model search (best_by_mae) that complements the architecture experiments.
"""

from __future__ import annotations

import json
import random


import p2_common as C


def clip_features(pack, mode):
    """clip_mode = 'none' | 'p005_p995' : clip numeric features to train percentiles."""
    if mode == "none":
        return pack
    lo = pack.x_train[pack.num_cols].quantile(0.005)
    hi = pack.x_train[pack.num_cols].quantile(0.995)
    q = C.subset_pack(pack, list(pack.feature_cols))
    for x in (q.x_train, q.x_val, q.x_test):
        x[pack.num_cols] = x[pack.num_cols].clip(lower=lo, upper=hi, axis=1)
    return q


def main():
    packs = C.get_pack(target_cols=(C.NEXT_TARGET,))
    p0 = packs[C.NEXT_TARGET]

    grid = dict(
        iterations=[1200, 1500, 1800],
        depth=[5, 6, 7, 8],
        learning_rate=[0.02, 0.03, 0.05],
        l2_leaf_reg=[3.0, 5.0, 7.0, 10.0],
        min_data_in_leaf=[20, 50],
        random_strength=[1.0, 1.5],
        bootstrap=["Bernoulli", "Bayesian"],
        target_mode=["p995", "log1p_p995"],
        clip_mode=["none", "p005_p995"],
    )
    rng = random.Random(42)
    n_samples = 40
    seen = set()
    configs = []
    while len(configs) < n_samples:
        cfg = {k: rng.choice(v) for k, v in grid.items()}
        key = tuple(sorted(cfg.items()))
        if key in seen:
            continue
        seen.add(key)
        configs.append(cfg)

    rows = []
    for i, cfg in enumerate(configs):
        hp = dict(
            depth=cfg["depth"],
            learning_rate=cfg["learning_rate"],
            l2_leaf_reg=cfg["l2_leaf_reg"],
            iterations=cfg["iterations"],
            min_data_in_leaf=cfg["min_data_in_leaf"],
            random_strength=cfg["random_strength"],
            od_wait=80,
        )
        extra = dict(bootstrap_type=cfg["bootstrap"])
        if cfg["bootstrap"] == "Bernoulli":
            extra["subsample"] = 0.85
        p = clip_features(p0, cfg["clip_mode"])
        model, fit_sec, tfm = C.fit_regressor(
            p, hp, loss_function="MAE", target_mode=cfg["target_mode"], extra=extra
        )
        r = dict(
            experiment="main_sweep",
            model_name=f"cfg{i:02d}",
            target=C.NEXT_TARGET,
            loss_function="MAE",
            **cfg,
            fit_sec=fit_sec,
        )
        r.update(C.eval_split(model, p, tfm))
        rows.append(r)
        if i % 8 == 0 or i == len(configs) - 1:
            print(
                f"[{i + 1}/{len(configs)}] best so far "
                f"val_mae={min(x['val_mae'] for x in rows):.2f}"
            )

    res = C.save_results(rows, "exp8_main_catboost_sweep")
    res = res.sort_values("val_mae").reset_index(drop=True)
    best = res.iloc[0]
    print("\nTop-5 by val_mae:")
    print(
        res[
            [
                "model_name",
                "depth",
                "learning_rate",
                "l2_leaf_reg",
                "iterations",
                "bootstrap",
                "target_mode",
                "clip_mode",
                "val_mae",
                "test_mae",
            ]
        ]
        .head(5)
        .to_string(index=False)
    )

    best_cfg = {
        k: (best[k].item() if hasattr(best[k], "item") else best[k]) for k in grid
    }
    with open(C.OUTPUT_DIR / "best_main_model.json", "w") as fh:
        json.dump(
            {
                "config": best_cfg,
                "val_mae": float(best["val_mae"]),
                "test_mae": float(best["test_mae"]),
            },
            fh,
            indent=2,
            default=str,
        )
    print(
        f"\nbest_by_mae: {best_cfg}  val_mae={best['val_mae']:.2f} "
        f"test_mae={best['test_mae']:.2f}"
    )


if __name__ == "__main__":
    main()
