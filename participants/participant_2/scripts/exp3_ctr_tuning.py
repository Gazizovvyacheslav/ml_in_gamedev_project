"""exp3 - CTR tuning of categorical features.

Sweeps one_hot_max_size, max_ctr_complexity and ctr_target_border_count to see how much
the handling of categorical features (one-hot threshold, feature combinations and target
quantization for CTR) affects MAE / fit time / model size.
"""

from __future__ import annotations

import itertools

import p2_common as C


def main():
    packs = C.get_pack(target_cols=(C.NEXT_TARGET,))
    p = packs[C.NEXT_TARGET]
    hp = C.load_best_hp()

    one_hot = (2, 5, 10)
    ctr_complexity = (1, 2)
    ctr_border = (1, 3, 5)

    rows = []
    for ohms, mcc, ctbc in itertools.product(one_hot, ctr_complexity, ctr_border):
        extra = dict(
            one_hot_max_size=ohms, max_ctr_complexity=mcc, ctr_target_border_count=ctbc
        )
        model, fit_sec, tfm = C.fit_regressor(
            p, hp, loss_function="MAE", target_mode="p995", extra=extra
        )
        r = dict(
            experiment="ctr_tuning",
            model_name=f"ohms{ohms}_mcc{mcc}_ctbc{ctbc}",
            one_hot_max_size=ohms,
            max_ctr_complexity=mcc,
            ctr_target_border_count=ctbc,
            fit_sec=fit_sec,
            model_size_kb=C.model_size_bytes(model) // 1024,
        )
        r.update(C.eval_split(model, p, tfm))
        rows.append(r)
        print(
            f"ohms={ohms} mcc={mcc} ctbc={ctbc}: val_mae={r['val_mae']:.2f} "
            f"fit={fit_sec:.1f}s size={r['model_size_kb']}KB"
        )

    res = C.save_results(rows, "exp3_ctr_tuning")
    print(
        "\n",
        res[["model_name", "val_mae", "test_mae", "fit_sec", "model_size_kb"]]
        .sort_values("val_mae")
        .head(8)
        .to_string(index=False),
    )


if __name__ == "__main__":
    main()
