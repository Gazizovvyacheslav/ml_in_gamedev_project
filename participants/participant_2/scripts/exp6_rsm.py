"""exp6 (spare) - rsm: random subspace method.

Checks random feature sampling at each split (rsm = 0.6 / 0.8 / 1.0) as an extra
regularisation that may reduce overfitting and increase model diversity for ensembling.
"""

from __future__ import annotations

import p2_common as C


def main():
    packs = C.get_pack(target_cols=(C.NEXT_TARGET,))
    p = packs[C.NEXT_TARGET]
    hp = C.load_best_hp()

    rows = []
    for rsm in (0.6, 0.8, 1.0):
        model, fit_sec, tfm = C.fit_regressor(
            p, hp, loss_function="MAE", target_mode="p995", extra=dict(rsm=rsm)
        )
        r = dict(
            experiment="rsm",
            model_name=f"rsm{rsm}",
            rsm=rsm,
            fit_sec=fit_sec,
            model_size_kb=C.model_size_bytes(model) // 1024,
        )
        r.update(C.eval_split(model, p, tfm))
        rows.append(r)
        gap = r["val_mae"] - r["test_mae"]
        print(
            f"rsm={rsm}: val_mae={r['val_mae']:.2f} test_mae={r['test_mae']:.2f} "
            f"gap={gap:+.1f} fit={fit_sec:.1f}s"
        )

    res = C.save_results(rows, "exp6_rsm")
    print(
        "\n",
        res[["model_name", "val_mae", "test_mae", "fit_sec"]]
        .sort_values("val_mae")
        .to_string(index=False),
    )


if __name__ == "__main__":
    main()
