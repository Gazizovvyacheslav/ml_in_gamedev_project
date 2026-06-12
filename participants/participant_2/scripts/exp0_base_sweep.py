"""exp0 - compact base CatBoost sweep on validation MAE.

The architecture experiments (SHAP selection, MVS, CTR, grow_policy, quantization, rsm)
are meant to be run locally around the 1-2 best base configs rather than as a full
cartesian product. This script finds that base config on the next-session target and
stores it for the other experiments to reuse.
"""

from __future__ import annotations

import json

import p2_common as C


def main():
    packs = C.get_pack(target_cols=(C.NEXT_TARGET,))
    p = packs[C.NEXT_TARGET]

    grid = []
    for depth in (5, 6, 7):
        for lr in (0.03, 0.05):
            for l2 in (3.0, 5.0, 8.0):
                grid.append(
                    dict(
                        depth=depth,
                        learning_rate=lr,
                        l2_leaf_reg=l2,
                        iterations=800,
                        od_wait=80,
                    )
                )

    rows = []
    for hp in grid:
        model, fit_sec, tfm = C.fit_regressor(
            p, hp, loss_function="MAE", target_mode="p995"
        )
        r = dict(hp)
        r["fit_sec"] = fit_sec
        r["best_iteration"] = model.get_best_iteration()
        r.update(C.eval_split(model, p, tfm))
        rows.append(r)
        print(
            f"depth={hp['depth']} lr={hp['learning_rate']} l2={hp['l2_leaf_reg']} "
            f"-> val_mae={r['val_mae']:.2f} test_mae={r['test_mae']:.2f} fit={fit_sec:.1f}s"
        )

    res = C.save_results(rows, "exp0_base_sweep")
    res = res.sort_values("val_mae").reset_index(drop=True)
    best = res.iloc[0]
    best_hp = dict(
        depth=int(best["depth"]),
        learning_rate=float(best["learning_rate"]),
        l2_leaf_reg=float(best["l2_leaf_reg"]),
        iterations=int(best["iterations"]),
        od_wait=int(best["od_wait"]),
    )

    with open(C.BEST_CONFIG_PATH, "w") as fh:
        json.dump(
            {
                "target_mode": "p995",
                "hp": best_hp,
                "val_mae": float(best["val_mae"]),
                "test_mae": float(best["test_mae"]),
            },
            fh,
            indent=2,
        )
    print("\nBEST base config (by val_mae):", best_hp, f"val_mae={best['val_mae']:.2f}")


if __name__ == "__main__":
    main()
