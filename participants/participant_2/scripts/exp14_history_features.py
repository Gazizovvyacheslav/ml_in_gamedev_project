"""exp14 - time-aware history features (block 1), both targets, full metric set.

Compares, per target:
    baseline                 - original features (no hist_*)
    baseline + history       - + all hist_* features
    baseline + best history  - + only the hist_* features that rank high by importance
Selection is by validation MAE; test is reported once for reference. Writes a decision file
consumed by the final assembly (exp17).
"""

from __future__ import annotations

import json


import p2_common as C

TARGETS = [C.NEXT_TARGET, C.CRM_TARGET]
HP = dict(C.load_best_hp())


def evaluate(pack, name, target, feature_set):
    model, fit_sec, tfm = C.fit_regressor(
        pack, HP, loss_function="MAE", target_mode="p995"
    )
    r = dict(
        experiment="history_features",
        target=target,
        model_name=name,
        feature_set=feature_set,
        n_features=len(pack.feature_cols),
        fit_sec=fit_sec,
    )
    r.update(C.eval_split(model, pack, tfm))
    return r, model


def main():
    rows = []
    for target in TARGETS:
        pack_base = C.get_aug_pack(target, base_only=True)
        pack_hist = C.get_aug_pack(target, base_only=False)
        base_feats = list(pack_base.feature_cols)

        r_base, _ = evaluate(pack_base, "baseline", target, "baseline")
        r_hist, m_hist = evaluate(
            pack_hist, "baseline_plus_history", target, "all_features"
        )
        rows += [r_base, r_hist]

        imp = m_hist.get_feature_importance()
        cols = list(pack_hist.feature_cols)
        hist_rank = sorted(
            [
                (imp[i], cols[i])
                for i in range(len(cols))
                if cols[i].startswith("hist_")
            ],
            reverse=True,
        )
        best_hist = [c for _, c in hist_rank[:6]]
        keep = base_feats + best_hist
        pack_best = C.get_aug_pack(target, keep_features=keep)
        r_best, _ = evaluate(
            pack_best, "baseline_plus_best_history", target, "baseline+top6_hist"
        )
        r_best["best_hist_features"] = "|".join(best_hist)
        rows.append(r_best)

        use_history = min(r_hist["val_mae"], r_best["val_mae"]) < r_base["val_mae"]
        chosen = (
            (
                best_hist
                if r_best["val_mae"] <= r_hist["val_mae"]
                else [c for c in cols if c.startswith("hist_")]
            )
            if use_history
            else []
        )
        with open(C.OUTPUT_DIR / f"decision_history_{target}.json", "w") as fh:
            json.dump(
                {
                    "target": target,
                    "use_history": bool(use_history),
                    "best_hist_features": chosen,
                    "baseline_val_mae": r_base["val_mae"],
                    "history_val_mae": r_hist["val_mae"],
                    "best_history_val_mae": r_best["val_mae"],
                },
                fh,
                indent=2,
            )

        print(f"\n== {target} ==")
        for r in (r_base, r_hist, r_best):
            print(
                f"  {r['model_name']:<28} val_mae={r['val_mae']:.2f} "
                f"r2={r['val_r2']:.3f} product={r['val_product_mae']:.1f} "
                f"small={r['val_small_mae']:.1f} n={r['n_features']}"
            )
        print(f"  -> use_history={use_history}, top hist: {best_hist[:4]}")

    C.save_results(rows, "exp14_history_features")


if __name__ == "__main__":
    main()
