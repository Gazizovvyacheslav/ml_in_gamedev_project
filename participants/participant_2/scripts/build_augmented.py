"""Build the augmented dataset once: + CRM target + history features.

Saved to outputs/sessions_augmented.parquet so every block (history / calibration / drift /
final) reads ready data instead of recomputing the slow per-player aggregations and the CRM
target each time.
"""

from __future__ import annotations

import time

import p2_common as C
from preprocessing.preprocessing import add_crm_target_7d
from history_features import add_history_features

AUG_PATH = C.OUTPUT_DIR / "sessions_augmented.parquet"


def main():
    t0 = time.time()
    df = C.load_data(C.DATA_PATH)
    print(f"loaded {len(df)} rows in {time.time() - t0:.1f}s")

    t1 = time.time()
    df = add_crm_target_7d(df, target_col=C.CRM_TARGET)
    print(
        f"+CRM target ({df[C.CRM_TARGET].notna().sum()} observed) in {time.time() - t1:.1f}s"
    )

    t2 = time.time()
    df = add_history_features(df)
    print(f"+history features in {time.time() - t2:.1f}s")

    df.to_parquet(AUG_PATH, index=False)
    print(
        f"[saved] {AUG_PATH}  ({len(df)} rows, {df.shape[1]} cols) total {time.time() - t0:.1f}s"
    )


if __name__ == "__main__":
    main()
