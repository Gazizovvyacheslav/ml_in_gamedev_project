"""exp18 - explain the validation vs test gap, both targets.

In the earlier blocks val MAE (~522) was consistently worse than test MAE (~429) for the
next-session target. This script characterises train/val/test by segment composition and
target distribution to show that the test period is simply "easier" (fewer long sessions,
lower mean target), rather than the model peeking at test.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import p2_common as C

TARGETS = [C.NEXT_TARGET, C.CRM_TARGET]


def describe(y):
    y = np.asarray(y, float)
    small = float((y <= 300).mean())
    normal = float(((y > 300) & (y <= 1200)).mean())
    long_ = float((y > 1200).mean())
    return dict(
        n=len(y),
        mean=float(y.mean()),
        median=float(np.median(y)),
        p90=float(np.percentile(y, 90)),
        p99=float(np.percentile(y, 99)),
        small_share=small,
        normal_share=normal,
        long_share=long_,
    )


def main():
    rows = []
    for target in TARGETS:
        pack = C.get_aug_pack(target, base_only=True)
        for split, y in [
            ("train", pack.y_train),
            ("val", pack.y_val),
            ("test", pack.y_test),
        ]:
            r = dict(experiment="val_test_gap", target=target, split=split)
            r.update(describe(y))
            rows.append(r)
        print(f"\n== {target} ==")
        sub = pd.DataFrame([r for r in rows if r["target"] == target])
        print(
            sub[
                ["split", "n", "mean", "median", "p90", "long_share", "small_share"]
            ].to_string(index=False)
        )

    res = C.save_results(rows, "exp18_val_test_gap")

    nxt = res[res.target == C.NEXT_TARGET].set_index("split")
    if {"val", "test"}.issubset(nxt.index):
        print(
            f"\nnext-session: val long_share={nxt.loc['val', 'long_share']:.3f} "
            f"vs test long_share={nxt.loc['test', 'long_share']:.3f}; "
            f"val mean={nxt.loc['val', 'mean']:.0f} vs test mean={nxt.loc['test', 'mean']:.0f} "
            "-> test period has fewer long sessions / lower mean, hence lower MAE."
        )


if __name__ == "__main__":
    main()
