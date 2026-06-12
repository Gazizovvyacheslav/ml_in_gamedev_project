"""exp7 - protocol strategy check on the best base config, both targets.

team_modeling_protocol.txt requires that for the 1-2 best configs we re-run the three
canonical strategies and report the full metric set on both targets:
    * capped_target          (target_mode = p995, MAE loss)
    * Quantile:alpha=0.40    (balanced product variant)
    * Quantile:alpha=0.35    (cautious short-risk variant)

The CRM target is built on the fly by the shared pipeline; if it cannot be computed in a
reasonable time it is skipped (CRM is Participant 1's primary deliverable).
"""

from __future__ import annotations

import p2_common as C

STRATEGIES = [
    dict(tag="capped_target", loss="MAE", mode="p995"),
    dict(tag="quantile_040", loss="Quantile:alpha=0.40", mode=None),
    dict(tag="quantile_035", loss="Quantile:alpha=0.35", mode=None),
]


def run_for_target(packs, target, hp):
    p = packs[target]
    rows = []
    for s in STRATEGIES:
        model, fit_sec, tfm = C.fit_regressor(
            p, hp, loss_function=s["loss"], target_mode=s["mode"]
        )
        r = dict(
            experiment="final_strategies",
            target=target,
            model_name=s["tag"],
            loss_function=s["loss"],
            target_mode=str(s["mode"]),
            fit_sec=fit_sec,
        )
        r.update(C.eval_split(model, p, tfm))
        rows.append(r)
        print(
            f"[{target}] {s['tag']}: val_mae={r['val_mae']:.2f} "
            f"product={r['val_product_mae']:.2f} eng_risk={r['val_engagement_risk_mae']:.2f} "
            f"small={r['val_small_mae']:.1f}"
        )
    return rows


def main():
    hp = C.load_best_hp()
    rows = []

    packs_next = C.get_pack(target_cols=(C.NEXT_TARGET,))
    rows += run_for_target(packs_next, C.NEXT_TARGET, hp)

    try:
        packs_crm = C.get_pack(target_cols=(C.CRM_TARGET,))
        rows += run_for_target(packs_crm, C.CRM_TARGET, hp)
    except Exception as e:
        print(f"[crm] skipped: {type(e).__name__}: {e}")

    C.save_results(rows, "exp7_final_strategies")


if __name__ == "__main__":
    main()
