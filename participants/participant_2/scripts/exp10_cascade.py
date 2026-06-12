"""exp10 - cascade model "classification -> regression" (Эксперимент 4).

A CatBoostClassifier predicts the engagement segment (small/normal/long), then a separate
CatBoostRegressor per segment produces the duration. Three routing variants:
    hard   - use the regressor of the arg-max class
    soft   - probability-weighted sum of the three segment regressors
    hybrid - confident rows -> segment regressor; unsure rows -> general model
Compared against the single general regressor (baseline). Reports MAE / ProductMAE /
EngagementRiskMAE / segment MAEs and the routing shares.
"""

from __future__ import annotations

import time

import numpy as np
from catboost import CatBoostClassifier

import p2_common as C
from preprocessing.preprocessing import TargetTransform

SEG_EDGES = (300.0, 1200.0)


def seg_label(y):
    y = np.asarray(y, dtype=float)
    return np.where(y <= SEG_EDGES[0], 0, np.where(y <= SEG_EDGES[1], 1, 2)).astype(int)


def main():
    packs = C.get_pack(target_cols=(C.NEXT_TARGET,))
    p = packs[C.NEXT_TARGET]
    hp = C.load_best_hp()
    cat = p.cat_cols

    ytr_seg = seg_label(p.y_train)

    t0 = time.time()
    clf = CatBoostClassifier(
        loss_function="MultiClass",
        eval_metric="MultiClass",
        class_names=[0, 1, 2],
        depth=hp["depth"],
        learning_rate=hp["learning_rate"],
        l2_leaf_reg=hp["l2_leaf_reg"],
        iterations=hp["iterations"],
        od_type="Iter",
        od_wait=80,
        random_seed=42,
        verbose=False,
        thread_count=-1,
    )
    clf.fit(
        p.x_train,
        ytr_seg,
        cat_features=cat,
        eval_set=(p.x_val, seg_label(p.y_val)),
        use_best_model=True,
    )
    clf_sec = time.time() - t0
    proba_val = clf.predict_proba(p.x_val)
    proba_test = clf.predict_proba(p.x_test)
    pred_cls_val = proba_val.argmax(1)
    acc = float((pred_cls_val == seg_label(p.y_val)).mean())
    print(f"classifier val accuracy={acc:.3f} fit={clf_sec:.1f}s")

    seg_models, seg_tfm = {}, {}
    for s in (0, 1, 2):
        mask = ytr_seg == s
        qp = C.subset_pack(p, list(p.feature_cols))
        sub_x = qp.x_train[mask]
        sub_y = p.y_train[mask]
        tfm = TargetTransform(mode="p995").fit(sub_y)
        from catboost import CatBoostRegressor

        m = CatBoostRegressor(
            loss_function="MAE",
            depth=hp["depth"],
            learning_rate=hp["learning_rate"],
            l2_leaf_reg=hp["l2_leaf_reg"],
            iterations=hp["iterations"],
            od_type="Iter",
            random_seed=42,
            verbose=False,
            thread_count=-1,
        )
        m.fit(sub_x, tfm.transform(sub_y), cat_features=cat)
        seg_models[s] = m
        seg_tfm[s] = tfm
        print(f"  segment {s}: {mask.sum()} train rows")

    general, _, gen_tfm = C.fit_regressor(
        p, hp, loss_function="MAE", target_mode="p995"
    )

    def seg_preds(X):
        return {s: seg_tfm[s].inverse(seg_models[s].predict(X)) for s in (0, 1, 2)}

    sp_val = seg_preds(p.x_val)
    sp_test = seg_preds(p.x_test)
    gen_val = gen_tfm.inverse(general.predict(p.x_val))
    gen_test = gen_tfm.inverse(general.predict(p.x_test))

    def route(kind, proba, sp, gen_pred, conf_thr=0.6):
        cls = proba.argmax(1)
        n = len(cls)
        if kind == "hard":
            out = np.array([sp[cls[i]][i] for i in range(n)])
            share = {f"share_{s}": float((cls == s).mean()) for s in (0, 1, 2)}
        elif kind == "soft":
            out = sum(proba[:, s] * sp[s] for s in (0, 1, 2))
            share = {"share_general": 0.0}
        else:
            conf = proba.max(1) >= conf_thr
            seg = np.array([sp[cls[i]][i] for i in range(n)])
            out = np.where(conf, seg, gen_pred)
            share = {"share_general": float((~conf).mean())}
        return np.maximum(out, 0.0), share

    rows = []

    for tag, pred_val, pred_test, share in [
        ("baseline_general", gen_val, gen_test, {}),
    ]:
        r = dict(experiment="cascade", model_name=tag, fit_sec=np.nan, **share)
        for split, y, pr in [("val", p.y_val, pred_val), ("test", p.y_test, pred_test)]:
            for k, v in C.metric_pack(y, pr).items():
                r[f"{split}_{k}"] = v
        rows.append(r)

    for kind in ("hard", "soft", "hybrid"):
        pv, sh_v = route(kind, proba_val, sp_val, gen_val)
        pt, _ = route(kind, proba_test, sp_test, gen_test)
        r = dict(
            experiment="cascade",
            model_name=f"cascade_{kind}",
            classifier_val_acc=acc,
            **sh_v,
        )
        for split, y, pr in [("val", p.y_val, pv), ("test", p.y_test, pt)]:
            for k, v in C.metric_pack(y, pr).items():
                r[f"{split}_{k}"] = v
        rows.append(r)
        print(
            f"{kind}: val_mae={r['val_mae']:.2f} product={r['val_product_mae']:.2f} "
            f"eng={r['val_engagement_risk_mae']:.2f} small={r['val_small_mae']:.1f} "
            f"long={r['val_long_mae']:.1f}"
        )

    res = C.save_results(rows, "exp10_cascade")
    print(
        "\n",
        res[
            [
                "model_name",
                "val_mae",
                "val_product_mae",
                "val_engagement_risk_mae",
                "val_small_mae",
                "val_normal_mae",
                "val_long_mae",
                "test_mae",
            ]
        ].to_string(index=False),
    )


if __name__ == "__main__":
    main()
