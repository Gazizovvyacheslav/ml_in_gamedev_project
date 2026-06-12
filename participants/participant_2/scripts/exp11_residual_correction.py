"""exp11 - residual correction of the base model (Эксперимент 5).

1. Split train chronologically into A (first part) and B (last part).
2. base_A trained on A; predict B -> residuals_B = y_B - pred_B (honest, no leakage).
3. A compact corrector is trained on B features -> residuals_B.
4. base_full trained on the whole train; final = base_full + corrector, evaluated on val/test.

Checks whether systematic error patterns remain (e.g. overprediction of short sessions).
"""

from __future__ import annotations

import time

import numpy as np
from catboost import CatBoostRegressor

import p2_common as C
from preprocessing.preprocessing import TargetTransform


def main():
    packs = C.get_pack(target_cols=(C.NEXT_TARGET,))
    p = packs[C.NEXT_TARGET]
    hp = C.load_best_hp()
    cat = p.cat_cols

    n = len(p.x_train)
    cut = int(n * 0.7)
    xa, ya = p.x_train.iloc[:cut], p.y_train[:cut]
    xb, yb = p.x_train.iloc[cut:], p.y_train[cut:]

    tfm_a = TargetTransform("p995").fit(ya)
    base_a = CatBoostRegressor(
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
    base_a.fit(xa, tfm_a.transform(ya), cat_features=cat)
    pred_b = tfm_a.inverse(base_a.predict(xb))
    resid_b = yb - pred_b

    t0 = time.time()
    corr = CatBoostRegressor(
        loss_function="MAE",
        depth=4,
        learning_rate=0.03,
        l2_leaf_reg=5.0,
        iterations=400,
        od_type="Iter",
        random_seed=42,
        verbose=False,
        thread_count=-1,
    )
    corr.fit(xb, resid_b, cat_features=cat)
    corr_sec = time.time() - t0

    base_full, base_sec, tfm = C.fit_regressor(
        p, hp, loss_function="MAE", target_mode="p995"
    )

    rows = []
    for split, X, y in [("val", p.x_val, p.y_val), ("test", p.x_test, p.y_test)]:
        base_pred = tfm.inverse(base_full.predict(X))
        corr_pred = corr.predict(X)
        final_pred = np.maximum(base_pred + corr_pred, 0.0)
        if split == "val":
            base_row = dict(
                experiment="residual_correction",
                model_name="base_only",
                fit_sec=base_sec,
            )
            final_row = dict(
                experiment="residual_correction",
                model_name="base_plus_correction",
                fit_sec=base_sec + corr_sec,
                mean_correction=float(np.mean(corr_pred)),
                mean_abs_correction=float(np.mean(np.abs(corr_pred))),
            )
        for k, v in C.metric_pack(y, base_pred).items():
            base_row[f"{split}_{k}"] = v
        for k, v in C.metric_pack(y, final_pred).items():
            final_row[f"{split}_{k}"] = v

    rows = [base_row, final_row]
    res = C.save_results(rows, "exp11_residual_correction")
    print(
        res[
            [
                "model_name",
                "val_mae",
                "test_mae",
                "val_small_mae",
                "val_normal_mae",
                "val_long_mae",
                "val_product_mae",
            ]
        ].to_string(index=False)
    )
    print(
        f"\nmean correction on val features: {final_row.get('mean_correction'):.2f} sec "
        f"(|mean| {final_row.get('mean_abs_correction'):.2f})"
    )
    delta = base_row["test_mae"] - final_row["test_mae"]
    print(
        f"test MAE: base={base_row['test_mae']:.2f} -> corrected={final_row['test_mae']:.2f} "
        f"(Δ={delta:+.2f})"
    )


if __name__ == "__main__":
    main()
