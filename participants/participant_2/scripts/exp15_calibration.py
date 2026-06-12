"""exp15 - regression calibration (block 2), both targets, full metric set.

The base CatBoost may systematically over/under-predict some segments. We fit a post-hoc
calibrator on an internal calibration split of validation (cal_fit), evaluate honestly on the
held-out part of validation (cal_eval) and only finally on test.

Methods: bin calibration (quantile bins -> mean actual), isotonic regression, and
segment calibration (isotonic per predicted small/normal/long segment).
Writes a decision file for the final assembly (exp17).
"""

from __future__ import annotations

import json

import numpy as np
from sklearn.isotonic import IsotonicRegression

import p2_common as C

TARGETS = [C.NEXT_TARGET, C.CRM_TARGET]
HP = dict(C.load_best_hp())
SEG = (300.0, 1200.0)


def seg_of(pred):
    pred = np.asarray(pred, float)
    return np.where(pred <= SEG[0], 0, np.where(pred <= SEG[1], 1, 2))


def fit_bin(raw, y, n_bins=20):
    edges = np.unique(np.quantile(raw, np.linspace(0, 1, n_bins + 1)))
    idx = np.clip(np.digitize(raw, edges[1:-1]), 0, len(edges) - 2)
    means = np.array(
        [
            y[idx == b].mean() if (idx == b).any() else np.nan
            for b in range(len(edges) - 1)
        ]
    )

    good = ~np.isnan(means)
    if good.any():
        means = np.interp(np.arange(len(means)), np.where(good)[0], means[good])
    return edges, means


def apply_bin(cal, raw):
    edges, means = cal
    idx = np.clip(np.digitize(raw, edges[1:-1]), 0, len(means) - 1)
    return means[idx]


def main():
    rows = []
    for target in TARGETS:
        pack = C.get_aug_pack(target, base_only=False)
        model, _, tfm = C.fit_regressor(
            pack, HP, loss_function="MAE", target_mode="p995"
        )
        raw_val = tfm.inverse(model.predict(pack.x_val))
        raw_test = tfm.inverse(model.predict(pack.x_test))

        n = len(raw_val)
        cut = int(n * 0.5)
        rf, yf = raw_val[:cut], pack.y_val[:cut]
        re, ye = raw_val[cut:], pack.y_val[cut:]

        bin_cal = fit_bin(rf, yf)
        iso = IsotonicRegression(out_of_bounds="clip").fit(rf, yf)
        seg_iso = {}
        sf = seg_of(rf)
        for s in (0, 1, 2):
            m = sf == s
            if m.sum() >= 50:
                seg_iso[s] = IsotonicRegression(out_of_bounds="clip").fit(rf[m], yf[m])

        def calibrate(method, raw):
            if method == "raw":
                return raw
            if method == "bin":
                return apply_bin(bin_cal, raw)
            if method == "isotonic":
                return iso.predict(raw)
            if method == "segment":
                out = raw.copy().astype(float)
                sg = seg_of(raw)
                for s in (0, 1, 2):
                    m = sg == s
                    if s in seg_iso and m.any():
                        out[m] = seg_iso[s].predict(raw[m])
                return out
            raise ValueError(method)

        methods = ["raw", "bin", "isotonic", "segment"]
        target_rows = []
        for method in methods:
            r = dict(
                experiment="calibration",
                target=target,
                model_name=method,
                calibration=method,
            )
            for split, raw, y in [("val", re, ye), ("test", raw_test, pack.y_test)]:
                pred = np.maximum(calibrate(method, raw), 0.0)
                for k, v in C.metric_pack(y, pred).items():
                    r[f"{split}_{k}"] = v
            target_rows.append(r)
            print(
                f"[{target}] {method:<9} val_mae={r['val_mae']:.2f} "
                f"r2={r['val_r2']:.3f} product={r['val_product_mae']:.1f} "
                f"small={r['val_small_mae']:.1f} long={r['val_long_mae']:.1f}"
            )
        rows += target_rows

        best = min(target_rows, key=lambda r: (r["val_mae"], r["model_name"] != "raw"))
        with open(C.OUTPUT_DIR / f"decision_calibration_{target}.json", "w") as fh:
            json.dump(
                {
                    "target": target,
                    "method": best["calibration"],
                    "raw_val_mae": target_rows[0]["val_mae"],
                    "best_val_mae": best["val_mae"],
                },
                fh,
                indent=2,
            )
        print(f"  -> calibration choice: {best['calibration']}")

    C.save_results(rows, "exp15_calibration")


if __name__ == "__main__":
    main()
