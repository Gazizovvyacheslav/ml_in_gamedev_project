"""Train the final technical model on the FULL dataset (all rows, time split 70/15/15).

Same assembled decisions as exp17 (HP / feature set / history / drift / calibration), but with
max_rows=0 so the whole augmented dataset is used. Artifacts are written to
outputs/final_models_full/ so the 30k models are preserved. Metrics are saved incrementally
per target so partial progress survives a crash.
"""

from __future__ import annotations

import json
import time

import numpy as np
from sklearn.isotonic import IsotonicRegression

import p2_common as C
from exp15_calibration import fit_bin, apply_bin, seg_of

TARGETS = [C.NEXT_TARGET, C.CRM_TARGET]
FULL_DIR = C.OUTPUT_DIR / "final_models_full"
FULL_DIR.mkdir(exist_ok=True)

FULL = ["mae", "medae", "p70_abs_error", "p90_abs_error", "r2", "small_mae", "normal_mae",
        "long_mae", "product_mae", "engagement_risk_mae", "wmape"]


def load_decision(name, target, default):
    p = C.OUTPUT_DIR / f"decision_{name}_{target}.json"
    return json.loads(p.read_text()) if p.exists() else default


def hp_for(target):
    p = C.OUTPUT_DIR / "best_main_model.json"
    if target == C.NEXT_TARGET and p.exists():
        cfg = json.loads(p.read_text())["config"]
        hp = dict(depth=int(cfg["depth"]), learning_rate=float(cfg["learning_rate"]),
                  l2_leaf_reg=float(cfg["l2_leaf_reg"]), iterations=int(cfg["iterations"]),
                  min_data_in_leaf=int(cfg["min_data_in_leaf"]),
                  random_strength=float(cfg["random_strength"]), od_wait=80)
        extra = dict(bootstrap_type=cfg["bootstrap"])
        if cfg["bootstrap"] == "Bernoulli":
            extra["subsample"] = 0.85
        return hp, cfg["target_mode"], extra
    return dict(C.load_best_hp()), "p995", {}


def calibrate(method, raw_val, y_val, raw_test):
    if method == "bin":
        return apply_bin(fit_bin(raw_val, y_val), raw_test)
    if method == "isotonic":
        return IsotonicRegression(out_of_bounds="clip").fit(raw_val, y_val).predict(raw_test)
    if method == "segment":
        out = raw_test.astype(float).copy()
        sv, st = seg_of(raw_val), seg_of(raw_test)
        for s in (0, 1, 2):
            mv, mt = sv == s, st == s
            if mv.sum() >= 50 and mt.any():
                iso = IsotonicRegression(out_of_bounds="clip").fit(raw_val[mv], y_val[mv])
                out[mt] = iso.predict(raw_test[mt])
        return out
    return raw_test


def main():
    rows, readme = [], {}
    for target in TARGETS:
        t_all = time.time()
        dh = load_decision("history", target, {"use_history": False, "best_hist_features": []})
        dd = load_decision("drift", target, {"drop_features": []})
        dc = load_decision("calibration", target, {"method": "raw"})
        hp, tmode, extra = hp_for(target)

        pb = C.get_aug_pack(target, base_only=True, max_rows=0)
        feats = list(pb.feature_cols) + (dh["best_hist_features"] if dh["use_history"] else [])
        feats = [c for c in feats if c not in set(dd["drop_features"])]
        pack = C.get_aug_pack(target, keep_features=feats, max_rows=0)
        print(f"[{target}] rows train/val/test = {len(pack.x_train)}/{len(pack.x_val)}/"
              f"{len(pack.x_test)} | features {len(pack.feature_cols)} | training...", flush=True)

        t0 = time.time()
        model, fit_sec, tfm = C.fit_regressor(pack, hp, "MAE", tmode, extra=extra)
        print(f"[{target}] fit done in {fit_sec:.0f}s (best_iter={model.get_best_iteration()})",
              flush=True)

        raw_val = tfm.inverse(model.predict(pack.x_val))
        raw_test = tfm.inverse(model.predict(pack.x_test))
        final_test = np.maximum(calibrate(dc["method"], raw_val, pack.y_val, raw_test), 0.0)

        t1 = time.time()
        for _ in range(3):
            model.predict(pack.x_test[:20000])
        infer_us = (time.time() - t1) / 3 / min(len(pack.x_test), 20000) * 1e6

        size_kb = C.model_size_bytes(model) // 1024
        m = C.metric_pack(pack.y_test, final_test)
        r = dict(target=target, model="final_full", n_train=len(pack.x_train),
                 feature_set="baseline+history" if dh["use_history"] else "baseline",
                 calibration=dc["method"], **{k: round(m[k], 3) for k in FULL},
                 fit_sec=round(fit_sec, 1), model_size_kb=size_kb,
                 inference_us_per_row=round(infer_us, 2))
        rows.append(r)

        tag = "next" if target == C.NEXT_TARGET else "crm"
        model.save_model(str(FULL_DIR / f"final_model_{tag}_full.cbm"))
        (FULL_DIR / f"final_features_{tag}.json").write_text(json.dumps(
            {"features": pack.feature_cols, "cat_features": pack.cat_cols}, indent=2))
        (FULL_DIR / f"final_config_{tag}.json").write_text(json.dumps(
            {"target": target, "hp": hp, "target_mode": tmode, "extra": extra,
             "calibration": dc["method"], "use_history": dh["use_history"],
             "n_train": len(pack.x_train)}, indent=2, default=str))
        readme[target] = dict(n_train=len(pack.x_train), n_features=len(pack.feature_cols),
                              use_history=dh["use_history"], calibration=dc["method"],
                              target_mode=tmode, test_mae=r["mae"], test_r2=r["r2"],
                              model_size_kb=size_kb, fit_sec=round(fit_sec, 1))

        import pandas as pd
        pd.DataFrame(rows).to_csv(C.OUTPUT_DIR / "final_model_metrics_full.csv", index=False)
        (C.OUTPUT_DIR / "README_final_full.json").write_text(json.dumps(readme, indent=2, default=str))
        print(f"[{target}] DONE in {time.time() - t_all:.0f}s | test MAE={r['mae']} "
              f"R2={r['r2']} size={size_kb}KB -> saved", flush=True)

    print("ALL TARGETS DONE", flush=True)


if __name__ == "__main__":
    main()
