"""exp4 - Tree structure: SymmetricTree vs Depthwise vs Lossguide.

Compares the default symmetric (oblivious) trees with the more flexible Depthwise and
Lossguide growth policies, including their leaf-size / leaf-count controls. We look at
overall MAE plus the segment errors (small/normal/long) to see whether flexible trees
help on the heterogeneous segments.
"""

from __future__ import annotations

import p2_common as C


def main():
    packs = C.get_pack(target_cols=(C.NEXT_TARGET,))
    p = packs[C.NEXT_TARGET]
    hp = C.load_best_hp()

    configs = []

    configs.append(("symmetric", dict(hp), dict(grow_policy="SymmetricTree")))

    for mdl in (20, 50, 100):
        configs.append(
            (
                f"depthwise_mdl{mdl}",
                dict(hp),
                dict(grow_policy="Depthwise", min_data_in_leaf=mdl),
            )
        )

    for ml in (31, 63, 127):
        for mdl in (20, 50):
            h = dict(hp)
            h.pop("depth", None)
            configs.append(
                (
                    f"lossguide_ml{ml}_mdl{mdl}",
                    h,
                    dict(grow_policy="Lossguide", max_leaves=ml, min_data_in_leaf=mdl),
                )
            )

    rows = []
    for tag, h, extra in configs:
        model, fit_sec, tfm = C.fit_regressor(
            p, h, loss_function="MAE", target_mode="p995", extra=extra
        )
        r = dict(
            experiment="tree_structure",
            model_name=tag,
            grow_policy=extra.get("grow_policy"),
            min_data_in_leaf=extra.get("min_data_in_leaf"),
            max_leaves=extra.get("max_leaves"),
            fit_sec=fit_sec,
            model_size_kb=C.model_size_bytes(model) // 1024,
        )
        r.update(C.eval_split(model, p, tfm))
        rows.append(r)
        print(
            f"{tag}: val_mae={r['val_mae']:.2f} small={r['val_small_mae']:.1f} "
            f"normal={r['val_normal_mae']:.1f} long={r['val_long_mae']:.1f} "
            f"fit={fit_sec:.1f}s size={r['model_size_kb']}KB"
        )

    res = C.save_results(rows, "exp4_tree_structure")
    print(
        "\n",
        res[
            [
                "model_name",
                "val_mae",
                "test_mae",
                "val_small_mae",
                "val_normal_mae",
                "val_long_mae",
                "fit_sec",
                "model_size_kb",
            ]
        ]
        .sort_values("val_mae")
        .to_string(index=False),
    )


if __name__ == "__main__":
    main()
