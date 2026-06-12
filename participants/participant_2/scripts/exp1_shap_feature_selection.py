"""exp1 - SHAP recursive feature selection vs manual top-k pruning.

Compares:
  * full model (all features)
  * CatBoost built-in select_features(algorithm="RecursiveByShapValues") for k=40/60/80
  * manual top-k pruning by PredictionValuesChange importance for k=40/60/80

Reports MAE, fit time, model size and the excluded-feature list, to decide whether the
model can be shrunk / inference sped up without losing accuracy.
"""

from __future__ import annotations

import json

import numpy as np
from catboost import CatBoostRegressor, Pool

import p2_common as C


def _reduced_pack(pack, keep_cols):
    """A lightweight view of a PreparedData pack restricted to keep_cols."""

    class _P:
        pass

    q = _P()
    q.x_train = pack.x_train[keep_cols]
    q.x_val = pack.x_val[keep_cols]
    q.x_test = pack.x_test[keep_cols]
    q.y_train, q.y_val, q.y_test = pack.y_train, pack.y_val, pack.y_test
    q.cat_cols = [c for c in pack.cat_cols if c in keep_cols]
    q.feature_cols = list(keep_cols)
    return q


def _row(tag, kind, k, model, fit_sec, pack, tfm, kept, dropped):
    r = dict(
        experiment="shap_feature_selection",
        model_name=tag,
        selection=kind,
        n_features=len(kept),
        fit_sec=fit_sec,
        model_size_kb=C.model_size_bytes(model) // 1024,
    )
    r.update(C.eval_split(model, pack, tfm))
    r["dropped_features"] = "|".join(dropped)
    return r


def main():
    packs = C.get_pack(target_cols=(C.NEXT_TARGET,))
    p = packs[C.NEXT_TARGET]
    hp = C.load_best_hp()
    all_cols = list(p.feature_cols)
    rows = []

    full_model, fit_sec, tfm = C.fit_regressor(
        p, hp, loss_function="MAE", target_mode="p995"
    )
    rows.append(
        _row("full", "none", len(all_cols), full_model, fit_sec, p, tfm, all_cols, [])
    )
    print(
        f"full ({len(all_cols)} feats): val_mae={rows[-1]['val_mae']:.2f} "
        f"test_mae={rows[-1]['test_mae']:.2f}"
    )

    imp = full_model.get_feature_importance()
    order = [all_cols[i] for i in np.argsort(imp)[::-1]]

    sel_tfm = tfm
    ytr = sel_tfm.transform(p.y_train)
    yva = sel_tfm.transform(p.y_val)
    train_pool = Pool(p.x_train, ytr, cat_features=p.cat_cols)
    val_pool = Pool(p.x_val, yva, cat_features=p.cat_cols)

    k_values = [k for k in (40, 60, 80) if k < len(all_cols)]
    if not k_values:
        k_values = [max(10, len(all_cols) // 2)]
    for k in k_values:
        selector = CatBoostRegressor(loss_function="MAE", **C.BASE_PARAMS, **hp)
        summary = selector.select_features(
            train_pool,
            eval_set=val_pool,
            features_for_select=list(range(len(all_cols))),
            num_features_to_select=k,
            algorithm="RecursiveByShapValues",
            steps=3,
            train_final_model=False,
            logging_level="Silent",
        )
        sel_idx = summary["selected_features"]
        sel_cols = [all_cols[i] for i in sel_idx]
        dropped = [c for c in all_cols if c not in set(sel_cols)]
        qp = _reduced_pack(p, sel_cols)
        m, fs, tf = C.fit_regressor(qp, hp, loss_function="MAE", target_mode="p995")
        rows.append(
            _row(f"shap_k{k}", "shap_recursive", k, m, fs, qp, tf, sel_cols, dropped)
        )
        print(
            f"shap_k{k}: val_mae={rows[-1]['val_mae']:.2f} test_mae={rows[-1]['test_mae']:.2f} "
            f"size={rows[-1]['model_size_kb']}KB"
        )

        top_cols = order[:k]
        dropped_m = [c for c in all_cols if c not in set(top_cols)]
        qp2 = _reduced_pack(p, top_cols)
        m2, fs2, tf2 = C.fit_regressor(qp2, hp, loss_function="MAE", target_mode="p995")
        rows.append(
            _row(f"topk_{k}", "manual_topk", k, m2, fs2, qp2, tf2, top_cols, dropped_m)
        )
        print(
            f"topk_{k}: val_mae={rows[-1]['val_mae']:.2f} test_mae={rows[-1]['test_mae']:.2f} "
            f"size={rows[-1]['model_size_kb']}KB"
        )

    res = C.save_results(rows, "exp1_shap_feature_selection")

    view = res[
        [
            "model_name",
            "selection",
            "n_features",
            "val_mae",
            "test_mae",
            "fit_sec",
            "model_size_kb",
        ]
    ].sort_values("val_mae")
    print("\n", view.to_string(index=False))

    shap40 = res[res.model_name == "shap_k40"].iloc[0]
    kept40 = [
        c for c in all_cols if c not in set(shap40["dropped_features"].split("|"))
    ]
    with open(C.OUTPUT_DIR / "feature_set_shap_40.json", "w") as fh:
        json.dump({"features": kept40}, fh, indent=2)


if __name__ == "__main__":
    main()
