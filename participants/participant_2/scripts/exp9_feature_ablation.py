"""exp9 - feature group ablation (Эксперимент 3, distribution_of_responsoblities).

Feature sets:
    session_only            - current/past session features
    session_install         - + install / acquisition attributes
    session_install_events  - full set (+ event aggregates)
    top_k_40 / 60 / 73      - k most important features (by full-model importance)

Goal: contribution of each group, whether event features are needed, model complexity /
inference speed trade-off.
"""

from __future__ import annotations

import numpy as np

import p2_common as C


def main():
    packs = C.get_pack(target_cols=(C.NEXT_TARGET,))
    p = packs[C.NEXT_TARGET]
    hp = C.load_best_hp()
    all_cols = list(p.feature_cols)
    groups = C.feature_groups(all_cols)
    print({k: len(v) for k, v in groups.items()})

    full_model, _, _ = C.fit_regressor(p, hp, loss_function="MAE", target_mode="p995")
    imp = full_model.get_feature_importance()
    order = [all_cols[i] for i in np.argsort(imp)[::-1]]

    feature_sets = {
        "session_only": groups["session"],
        "session_install": groups["session"] + groups["install"],
        "session_install_events": all_cols,
    }
    for k in (40, 60, len(all_cols)):
        feature_sets[f"top_k_{k}"] = order[:k]

    rows = []
    for name, cols in feature_sets.items():
        qp = C.subset_pack(p, cols)
        model, fit_sec, tfm = C.fit_regressor(
            qp, hp, loss_function="MAE", target_mode="p995"
        )
        r = dict(
            experiment="feature_ablation",
            model_name=name,
            feature_set=name,
            n_features=len(qp.feature_cols),
            fit_sec=fit_sec,
            model_size_kb=C.model_size_bytes(model) // 1024,
        )
        r.update(C.eval_split(model, qp, tfm))
        rows.append(r)
        print(
            f"{name} ({len(qp.feature_cols)} feats): val_mae={r['val_mae']:.2f} "
            f"test_mae={r['test_mae']:.2f} size={r['model_size_kb']}KB"
        )

    res = C.save_results(rows, "exp9_feature_ablation")
    print(
        "\n",
        res[
            [
                "model_name",
                "n_features",
                "val_mae",
                "test_mae",
                "fit_sec",
                "model_size_kb",
            ]
        ]
        .sort_values("val_mae")
        .to_string(index=False),
    )


if __name__ == "__main__":
    main()
