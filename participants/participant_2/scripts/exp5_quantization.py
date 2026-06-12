"""exp5 - Numeric feature quantization tuning.

Sweeps border_count and feature_border_type, and additionally tests a
per_float_feature_quantization with finer borders on a few of the most important numeric
features. We watch overall MAE and especially long_mae to see whether finer quantization
helps on the long tail.
"""

from __future__ import annotations

import itertools

import numpy as np

import p2_common as C


def main():
    packs = C.get_pack(target_cols=(C.NEXT_TARGET,))
    p = packs[C.NEXT_TARGET]
    hp = C.load_best_hp()

    border_counts = (64, 128, 254, 512)
    border_types = ("GreedyLogSum", "Median", "UniformAndQuantiles")

    rows = []
    for bc, bt in itertools.product(border_counts, border_types):
        extra = dict(border_count=bc, feature_border_type=bt)
        model, fit_sec, tfm = C.fit_regressor(
            p, hp, loss_function="MAE", target_mode="p995", extra=extra
        )
        r = dict(
            experiment="quantization",
            model_name=f"bc{bc}_{bt}",
            border_count=bc,
            feature_border_type=bt,
            fit_sec=fit_sec,
            model_size_kb=C.model_size_bytes(model) // 1024,
        )
        r.update(C.eval_split(model, p, tfm))
        rows.append(r)
        print(
            f"bc={bc} {bt}: val_mae={r['val_mae']:.2f} long={r['val_long_mae']:.1f} "
            f"fit={fit_sec:.1f}s size={r['model_size_kb']}KB"
        )

    base_model, _, tfm0 = C.fit_regressor(
        p, hp, loss_function="MAE", target_mode="p995"
    )
    imp = base_model.get_feature_importance()
    cols = list(p.feature_cols)
    num_set = set(p.num_cols)
    ranked_num = [cols[i] for i in np.argsort(imp)[::-1] if cols[i] in num_set]
    top_num = ranked_num[:3]
    feat_idx = {c: cols.index(c) for c in top_num}

    pffq = [f"{feat_idx[c]}:border_count=1024" for c in top_num]
    extra = dict(
        border_count=128,
        feature_border_type="GreedyLogSum",
        per_float_feature_quantization=pffq,
    )
    model, fit_sec, tfm = C.fit_regressor(
        p, hp, loss_function="MAE", target_mode="p995", extra=extra
    )
    r = dict(
        experiment="quantization",
        model_name="per_float_top3_1024",
        border_count=128,
        feature_border_type="GreedyLogSum",
        per_float_features="|".join(top_num),
        fit_sec=fit_sec,
        model_size_kb=C.model_size_bytes(model) // 1024,
    )
    r.update(C.eval_split(model, p, tfm))
    rows.append(r)
    print(
        f"per_float_top3_1024 ({','.join(top_num)}): val_mae={r['val_mae']:.2f} "
        f"long={r['val_long_mae']:.1f} fit={fit_sec:.1f}s"
    )

    res = C.save_results(rows, "exp5_quantization")
    print(
        "\n",
        res[
            [
                "model_name",
                "val_mae",
                "test_mae",
                "val_long_mae",
                "fit_sec",
                "model_size_kb",
            ]
        ]
        .sort_values("val_mae")
        .head(10)
        .to_string(index=False),
    )


if __name__ == "__main__":
    main()
