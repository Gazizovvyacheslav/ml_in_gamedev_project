"""exp17 - final technical CatBoost model (block 4), both targets.

Assembles the best decisions from the earlier blocks:
    - best CatBoost hyper-parameters (exp8 best_main_model for next-session)
    - feature set: baseline + chosen history features (exp14) - drift-heavy (exp16)
    - calibration method (exp15)
and trains the final model for each target, saving artifacts:
    final_model_<target>.cbm, final_features_<target>.json, final_config_<target>.json
plus a metrics table final_model_metrics.csv and a README_final.json.

Reported headline metrics are on test (final evaluation). A plain baseline (original
features, no history / drift / calibration) is included for comparison.
"""

from __future__ import annotations

import json
import time

import numpy as np
from sklearn.isotonic import IsotonicRegression

import p2_common as C
from exp15_calibration import fit_bin, apply_bin, seg_of

TARGETS = [C.NEXT_TARGET, C.CRM_TARGET]
FINAL_DIR = C.OUTPUT_DIR / "final_models"
FINAL_DIR.mkdir(exist_ok=True)

FULL_METRICS = [
    "mae",
    "medae",
    "p70_abs_error",
    "p90_abs_error",
    "r2",
    "small_mae",
    "normal_mae",
    "long_mae",
    "product_mae",
    "engagement_risk_mae",
    "wmape",
]


def load_decision(name, target, default):
    p = C.OUTPUT_DIR / f"decision_{name}_{target}.json"
    if p.exists():
        return json.loads(p.read_text())
    return default


def hp_for(target):
    """Use exp8's best technical config for next-session if available, else base HP."""
    if target == C.NEXT_TARGET and (C.OUTPUT_DIR / "best_main_model.json").exists():
        cfg = json.loads((C.OUTPUT_DIR / "best_main_model.json").read_text())["config"]
        hp = dict(
            depth=int(cfg["depth"]),
            learning_rate=float(cfg["learning_rate"]),
            l2_leaf_reg=float(cfg["l2_leaf_reg"]),
            iterations=int(cfg["iterations"]),
            min_data_in_leaf=int(cfg["min_data_in_leaf"]),
            random_strength=float(cfg["random_strength"]),
            od_wait=80,
        )
        extra = dict(bootstrap_type=cfg["bootstrap"])
        if cfg["bootstrap"] == "Bernoulli":
            extra["subsample"] = 0.85
        return hp, cfg["target_mode"], extra
    return dict(C.load_best_hp()), "p995", {}


def calibrate_test(method, raw_val, y_val, raw_test):
    """Fit calibrator on full validation, apply to test (final evaluation)."""
    if method == "raw":
        return raw_test
    if method == "bin":
        return apply_bin(fit_bin(raw_val, y_val), raw_test)
    if method == "isotonic":
        return (
            IsotonicRegression(out_of_bounds="clip")
            .fit(raw_val, y_val)
            .predict(raw_test)
        )
    if method == "segment":
        out = raw_test.astype(float).copy()
        sv, st = seg_of(raw_val), seg_of(raw_test)
        for s in (0, 1, 2):
            mv, mt = sv == s, st == s
            if mv.sum() >= 50 and mt.any():
                iso = IsotonicRegression(out_of_bounds="clip").fit(
                    raw_val[mv], y_val[mv]
                )
                out[mt] = iso.predict(raw_test[mt])
        return out
    raise ValueError(method)


def metrics_row(
    target,
    name,
    feature_set,
    calibration,
    drift_filter,
    y,
    pred,
    fit_sec,
    model_size,
    infer_us,
):
    m = C.metric_pack(y, pred)
    r = dict(
        target=target,
        model_name=name,
        feature_set=feature_set,
        calibration=calibration,
        drift_filter=drift_filter,
        fit_sec=round(fit_sec, 2),
        model_size_kb=model_size,
        inference_us_per_row=round(infer_us, 2),
    )
    for k in FULL_METRICS:
        r[k] = round(m[k], 3)
    return r


def main():
    rows = []
    readme = {}
    for target in TARGETS:
        dh = load_decision(
            "history", target, {"use_history": False, "best_hist_features": []}
        )
        dd = load_decision("drift", target, {"drop_features": []})
        dc = load_decision("calibration", target, {"method": "raw"})
        hp, tmode, extra = hp_for(target)

        pbase = C.get_aug_pack(target, base_only=True)
        mb, fsb, tb = C.fit_regressor(
            pbase, hp, loss_function="MAE", target_mode=tmode, extra=extra
        )
        pred_b = tb.inverse(mb.predict(pbase.x_test))
        rows.append(
            metrics_row(
                target,
                "baseline_reference",
                "baseline",
                "raw",
                "none",
                pbase.y_test,
                pred_b,
                fsb,
                C.model_size_bytes(mb) // 1024,
                0.0,
            )
        )

        base_feats = list(pbase.feature_cols)
        feats = base_feats + (dh["best_hist_features"] if dh["use_history"] else [])
        feats = [c for c in feats if c not in set(dd["drop_features"])]
        pack = C.get_aug_pack(target, keep_features=feats)

        model, fit_sec, tfm = C.fit_regressor(
            pack, hp, loss_function="MAE", target_mode=tmode, extra=extra
        )
        raw_val = tfm.inverse(model.predict(pack.x_val))
        raw_test = tfm.inverse(model.predict(pack.x_test))
        final_test = np.maximum(
            calibrate_test(dc["method"], raw_val, pack.y_val, raw_test), 0.0
        )

        t0 = time.time()
        for _ in range(3):
            model.predict(pack.x_test)
        infer_us = (time.time() - t0) / 3 / len(pack.x_test) * 1e6

        size_kb = C.model_size_bytes(model) // 1024
        fset = "baseline+history" if dh["use_history"] else "baseline"
        dfilter = dd.get("chosen_variant", "none")
        rows.append(
            metrics_row(
                target,
                "final",
                fset,
                dc["method"],
                dfilter,
                pack.y_test,
                final_test,
                fit_sec,
                size_kb,
                infer_us,
            )
        )

        tag = "next" if target == C.NEXT_TARGET else "crm"
        model.save_model(str(FINAL_DIR / f"final_model_{tag}.cbm"))
        (FINAL_DIR / f"final_features_{tag}.json").write_text(
            json.dumps(
                {"features": pack.feature_cols, "cat_features": pack.cat_cols}, indent=2
            )
        )
        (FINAL_DIR / f"final_config_{tag}.json").write_text(
            json.dumps(
                {
                    "target": target,
                    "hp": hp,
                    "target_mode": tmode,
                    "extra": extra,
                    "calibration": dc["method"],
                    "drift_filter": dfilter,
                    "use_history": dh["use_history"],
                },
                indent=2,
                default=str,
            )
        )

        base_mae = rows[-2]["mae"]
        fin_mae = rows[-1]["mae"]
        readme[target] = dict(
            n_features=len(pack.feature_cols),
            use_history=dh["use_history"],
            history_features=dh["best_hist_features"] if dh["use_history"] else [],
            drift_filter=dfilter,
            dropped_features=dd["drop_features"],
            calibration=dc["method"],
            target_mode=tmode,
            hp=hp,
            baseline_test_mae=base_mae,
            final_test_mae=fin_mae,
            improvement_sec=round(base_mae - fin_mae, 2),
            model_size_kb=size_kb,
            inference_us_per_row=round(infer_us, 2),
        )
        print(f"\n== {target} ==")
        print(
            f"  features={len(pack.feature_cols)} history={dh['use_history']} "
            f"drift={dfilter} calib={dc['method']}"
        )
        print(
            f"  baseline test MAE={base_mae:.2f} -> final test MAE={fin_mae:.2f} "
            f"(Δ={base_mae - fin_mae:+.2f}), size={size_kb}KB, "
            f"infer={infer_us:.1f}us/row"
        )

    C.save_results(rows, "exp17_final_model")

    import pandas as pd

    cols = (
        ["target", "model_name", "feature_set", "calibration", "drift_filter"]
        + FULL_METRICS
        + ["fit_sec", "model_size_kb", "inference_us_per_row"]
    )
    pd.DataFrame(rows)[cols].to_csv(
        C.OUTPUT_DIR / "final_model_metrics.csv", index=False
    )
    (C.OUTPUT_DIR / "README_final.json").write_text(
        json.dumps(readme, indent=2, default=str)
    )
    print("\n[saved] final_model_metrics.csv, README_final.json, final_models/*.cbm")


if __name__ == "__main__":
    main()
