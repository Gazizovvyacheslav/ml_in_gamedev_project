"""Train a product-oriented final model on the FULL dataset with Quantile:alpha=0.40.

Same feature decisions as the MAE final model (history per decision, drift drop), but the
cautious quantile loss on the raw target (no p995 transform, no calibration). This is the
product/short-risk variant: it trades overall MAE for lower ProductMAE / EngagementRiskMAE /
small_mae. Saved separately (suffix _q040) so the MAE models are preserved.
"""

from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

import p2_common as C

TARGETS = [C.NEXT_TARGET, C.CRM_TARGET]
ALPHA = 0.40
LOSS = f"Quantile:alpha={ALPHA:.2f}"
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
        return hp, extra
    return dict(C.load_best_hp()), {}


def main():
    rows, readme = [], {}
    for target in TARGETS:
        t_all = time.time()
        dh = load_decision("history", target, {"use_history": False, "best_hist_features": []})
        dd = load_decision("drift", target, {"drop_features": []})
        hp, extra = hp_for(target)

        pb = C.get_aug_pack(target, base_only=True, max_rows=0)
        feats = list(pb.feature_cols) + (dh["best_hist_features"] if dh["use_history"] else [])
        feats = [c for c in feats if c not in set(dd["drop_features"])]
        pack = C.get_aug_pack(target, keep_features=feats, max_rows=0)
        print(f"[{target}] rows train/val/test = {len(pack.x_train)}/{len(pack.x_val)}/"
              f"{len(pack.x_test)} | features {len(pack.feature_cols)} | loss {LOSS} | training...",
              flush=True)

        model, fit_sec, tfm = C.fit_regressor(pack, hp, LOSS, target_mode=None, extra=extra)
        print(f"[{target}] fit done in {fit_sec:.0f}s (best_iter={model.get_best_iteration()})",
              flush=True)

        ev = C.eval_split(model, pack, tfm)
        t1 = time.time()
        for _ in range(3):
            model.predict(pack.x_test[:20000])
        infer_us = (time.time() - t1) / 3 / min(len(pack.x_test), 20000) * 1e6
        size_kb = C.model_size_bytes(model) // 1024

        r = dict(target=target, model="final_full_q040", loss=LOSS, n_train=len(pack.x_train),
                 feature_set="baseline+history" if dh["use_history"] else "baseline",
                 **{k: round(ev[f"test_{k}"], 3) for k in FULL},
                 fit_sec=round(fit_sec, 1), model_size_kb=size_kb,
                 inference_us_per_row=round(infer_us, 2))
        rows.append(r)

        tag = "next" if target == C.NEXT_TARGET else "crm"
        model.save_model(str(FULL_DIR / f"final_model_{tag}_full_q040.cbm"))
        (FULL_DIR / f"final_config_{tag}_q040.json").write_text(json.dumps(
            {"target": target, "hp": hp, "loss": LOSS, "target_mode": "raw", "extra": extra,
             "use_history": dh["use_history"], "n_train": len(pack.x_train)}, indent=2, default=str))
        readme[target] = dict(n_train=len(pack.x_train), n_features=len(pack.feature_cols),
                              loss=LOSS, use_history=dh["use_history"],
                              test_mae=r["mae"], test_product_mae=r["product_mae"],
                              test_engagement_risk_mae=r["engagement_risk_mae"],
                              test_small_mae=r["small_mae"], test_r2=r["r2"],
                              model_size_kb=size_kb, fit_sec=round(fit_sec, 1))

        pd.DataFrame(rows).to_csv(C.OUTPUT_DIR / "final_model_metrics_full_q040.csv", index=False)
        (C.OUTPUT_DIR / "README_final_full_q040.json").write_text(json.dumps(readme, indent=2, default=str))
        print(f"[{target}] DONE in {time.time() - t_all:.0f}s | test MAE={r['mae']} "
              f"ProductMAE={r['product_mae']} EngRisk={r['engagement_risk_mae']} "
              f"small={r['small_mae']} -> saved", flush=True)

    print("ALL TARGETS DONE", flush=True)


if __name__ == "__main__":
    main()
