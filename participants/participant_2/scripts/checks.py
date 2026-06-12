"""Anti-leak / chronology / no-NaN checks for Participant 2's setup.

Prints the three PASS/FAIL lines required by the team protocol and stores the result,
the train/val/test time ranges, the sample size and the feature list to
outputs/anti_leak.json.
"""

from __future__ import annotations

import json

import pandas as pd

import p2_common as C
from preprocessing.preprocessing import (
    build_feature_list,
    time_split,
    DEFAULT_TIME_COL,
)


def main():
    df = C.load_data(C.DATA_PATH)
    x = df.copy()
    x[DEFAULT_TIME_COL] = pd.to_datetime(x[DEFAULT_TIME_COL], errors="coerce")
    x = x[x[DEFAULT_TIME_COL].notna() & x[C.NEXT_TARGET].notna()].copy()
    x = x.sort_values(DEFAULT_TIME_COL).reset_index(drop=True)
    x = x.tail(C.MAX_ROWS).reset_index(drop=True)

    tr, va, te = time_split(x, max_rows=0)
    feats = build_feature_list(x, target_col=C.NEXT_TARGET)

    bad = [
        c
        for c in feats
        if c.lower().startswith("target")
        or c.lower().startswith("future_")
        or c
        in {
            "appmetrica_device_id",
            "installation_id",
            "session_id",
            "start",
            "end",
            "session_date",
            "install_datetime",
            "prev_session_end",
            "duration_hms",
        }
    ]
    leak_pass = len(bad) == 0

    chrono_pass = (
        tr[DEFAULT_TIME_COL].max()
        < va[DEFAULT_TIME_COL].min()
        < te[DEFAULT_TIME_COL].min()
    )

    pack = C.prepare_for_targets(df, target_cols=[C.NEXT_TARGET], max_rows=C.MAX_ROWS)[
        C.NEXT_TARGET
    ]
    nan_pass = not (
        pack.x_train.isna().any().any()
        or pack.x_val.isna().any().any()
        or pack.x_test.isna().any().any()
    )

    print(
        f"{'PASS' if leak_pass else 'FAIL'} leakage check"
        + ("" if leak_pass else f" (bad: {bad})")
    )
    print(f"{'PASS' if chrono_pass else 'FAIL'} chronology check")
    print(f"{'PASS' if nan_pass else 'FAIL'} no-NaN check")

    meta = dict(
        sample_rows=int(len(x)),
        train_rows=int(len(tr)),
        val_rows=int(len(va)),
        test_rows=int(len(te)),
        train_time=[str(tr[DEFAULT_TIME_COL].min()), str(tr[DEFAULT_TIME_COL].max())],
        val_time=[str(va[DEFAULT_TIME_COL].min()), str(va[DEFAULT_TIME_COL].max())],
        test_time=[str(te[DEFAULT_TIME_COL].min()), str(te[DEFAULT_TIME_COL].max())],
        n_features=len(feats),
        n_num=len(pack.num_cols),
        n_cat=len(pack.cat_cols),
        features=feats,
        cat_features=pack.cat_cols,
        leakage_check="PASS" if leak_pass else "FAIL",
        chronology_check="PASS" if chrono_pass else "FAIL",
        no_nan_check="PASS" if nan_pass else "FAIL",
    )
    with open(C.OUTPUT_DIR / "anti_leak.json", "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"[saved] {C.OUTPUT_DIR / 'anti_leak.json'}")


if __name__ == "__main__":
    main()
