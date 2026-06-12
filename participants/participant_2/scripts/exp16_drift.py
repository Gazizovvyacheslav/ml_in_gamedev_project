"""exp16 - feature drift detection + removing unstable features (block 3), both targets.

For each feature, drift between train and test is measured:
    numeric      - PSI (population stability index, 10 quantile bins) + mean shift
    categorical  - share of test rows whose category is new/rare vs train
    all          - missing-rate change between train and test
Top drift-heavy features are found, then models are compared:
    all features / drop top-5 drift / drop top-10 drift / drop strong drift (PSI>0.25).
Writes a decision file (which features to drop) for the final assembly (exp17).
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

import p2_common as C

TARGETS = [C.NEXT_TARGET, C.CRM_TARGET]
HP = dict(C.load_best_hp())


def psi(train, test, bins=10):
    train = np.asarray(train, float)
    test = np.asarray(test, float)
    edges = np.unique(np.quantile(train, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    tr, _ = np.histogram(train, bins=edges)
    te, _ = np.histogram(test, bins=edges)
    tr = tr / max(tr.sum(), 1) + 1e-6
    te = te / max(te.sum(), 1) + 1e-6
    return float(np.sum((te - tr) * np.log(te / tr)))


def drift_table(pack):
    rows = []
    xtr, xte = pack.x_train, pack.x_test
    for c in pack.num_cols:
        rows.append(
            dict(
                feature=c,
                kind="num",
                psi=psi(xtr[c], xte[c]),
                mean_shift=abs(
                    float(xte[c].mean() - xtr[c].mean()) / (abs(xtr[c].mean()) + 1e-9)
                ),
                missing_shift=abs(float(xte[c].isna().mean() - xtr[c].isna().mean())),
            )
        )
    for c in pack.cat_cols:
        seen = set(xtr[c].astype(str).unique())
        new_share = float((~xte[c].astype(str).isin(seen)).mean())
        rows.append(
            dict(
                feature=c,
                kind="cat",
                psi=new_share * 1.0,
                mean_shift=np.nan,
                missing_shift=np.nan,
                new_cat_share=new_share,
            )
        )
    d = pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)
    return d


def fit_eval(target, drop, name):
    pack = C.get_aug_pack(target, base_only=False, drop_features=drop)
    model, fit_sec, tfm = C.fit_regressor(
        pack, HP, loss_function="MAE", target_mode="p995"
    )
    r = dict(
        experiment="drift",
        target=target,
        model_name=name,
        n_features=len(pack.feature_cols),
        n_dropped=len(drop),
        fit_sec=fit_sec,
    )
    r.update(C.eval_split(model, pack, tfm))
    return r


def main():
    rows = []
    for target in TARGETS:
        base = C.get_aug_pack(target, base_only=False)
        d = drift_table(base)
        d.to_csv(C.OUTPUT_DIR / f"drift_scores_{target}.csv", index=False)
        top10 = d.feature.head(10).tolist()
        top5 = top10[:5]
        strong = d[d.psi > 0.25].feature.tolist()
        print(f"\n== {target} == top drift: {top5}")
        print(
            d.head(8)[["feature", "kind", "psi", "mean_shift"]].to_string(index=False)
        )

        variants = [
            ([], "all_features"),
            (top5, "drop_top5_drift"),
            (top10, "drop_top10_drift"),
            (strong, f"drop_strong_psi>0.25_n{len(strong)}"),
        ]
        target_rows = []
        for drop, name in variants:
            r = fit_eval(target, drop, name)
            target_rows.append(r)
            gap = r["val_mae"] - r["test_mae"]
            print(
                f"  {name:<26} val_mae={r['val_mae']:.2f} test_mae={r['test_mae']:.2f} "
                f"gap={gap:+.1f} n={r['n_features']}"
            )
        rows += target_rows

        base_val = target_rows[0]["val_mae"]
        improving = [r for r in target_rows[1:] if r["val_mae"] <= base_val + 1e-6]
        if improving:
            pick = min(improving, key=lambda r: r["val_mae"])
            drop_map = {"drop_top5_drift": top5, "drop_top10_drift": top10}
            drop_list = drop_map.get(pick["model_name"], strong)
        else:
            pick, drop_list = target_rows[0], []
        with open(C.OUTPUT_DIR / f"decision_drift_{target}.json", "w") as fh:
            json.dump(
                {
                    "target": target,
                    "drop_features": drop_list,
                    "chosen_variant": pick["model_name"],
                    "all_val_mae": base_val,
                    "chosen_val_mae": pick["val_mae"],
                },
                fh,
                indent=2,
            )
        print(f"  -> drift drop: {pick['model_name']} ({len(drop_list)} features)")

    C.save_results(rows, "exp16_drift")


if __name__ == "__main__":
    main()
