"""exp2 - Bootstrap: MVS vs Bernoulli / Bayesian.

Adds Minimal Variance Sampling to the bootstrap comparison and checks subsample
0.7/0.8/0.9 for MVS. Stability is measured across seeds 42/52/62 (mean / std of val MAE),
and we also report long_mae to see the effect on the long tail.
"""

from __future__ import annotations

import numpy as np

import p2_common as C


def main():
    packs = C.get_pack(target_cols=(C.NEXT_TARGET,))
    p = packs[C.NEXT_TARGET]
    hp = C.load_best_hp()
    seeds = (42, 52, 62)

    configs = [
        dict(tag="bernoulli_ss0.8", bootstrap_type="Bernoulli", subsample=0.8),
        dict(tag="bayesian", bootstrap_type="Bayesian", bagging_temperature=1.0),
        dict(tag="mvs_ss0.7", bootstrap_type="MVS", subsample=0.7),
        dict(tag="mvs_ss0.8", bootstrap_type="MVS", subsample=0.8),
        dict(tag="mvs_ss0.9", bootstrap_type="MVS", subsample=0.9),
    ]

    rows = []
    for cfg in configs:
        tag = cfg.pop("tag")
        per_seed_val, per_seed_test, per_seed_long, fits = [], [], [], []
        last = None
        for sd in seeds:
            extra = dict(cfg)
            extra["random_seed"] = sd
            model, fit_sec, tfm = C.fit_regressor(
                p, hp, loss_function="MAE", target_mode="p995", extra=extra
            )
            m = C.eval_split(model, p, tfm)
            per_seed_val.append(m["val_mae"])
            per_seed_test.append(m["test_mae"])
            per_seed_long.append(m["val_long_mae"])
            fits.append(fit_sec)
            last = m
        r = dict(experiment="bootstrap_mvs", model_name=tag, **cfg)
        r["val_mae"] = float(np.mean(per_seed_val))
        r["val_mae_std"] = float(np.std(per_seed_val))
        r["test_mae"] = float(np.mean(per_seed_test))
        r["val_long_mae"] = float(np.mean(per_seed_long))
        r["fit_sec"] = float(np.mean(fits))

        for k, v in last.items():
            r.setdefault(k, v)
        rows.append(r)
        print(
            f"{tag}: val_mae={r['val_mae']:.2f}±{r['val_mae_std']:.2f} "
            f"test_mae={r['test_mae']:.2f} long_mae={r['val_long_mae']:.1f} "
            f"fit={r['fit_sec']:.1f}s"
        )

    res = C.save_results(rows, "exp2_bootstrap_mvs")
    print(
        "\n",
        res[
            [
                "model_name",
                "val_mae",
                "val_mae_std",
                "test_mae",
                "val_long_mae",
                "fit_sec",
            ]
        ]
        .sort_values("val_mae")
        .to_string(index=False),
    )


if __name__ == "__main__":
    main()
