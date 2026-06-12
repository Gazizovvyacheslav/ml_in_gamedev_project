"""Time-aware history features (Participant 2, block 1).

Every feature is computed strictly from a player's PAST sessions relative to the current
row (shift(1) before any rolling), so there is no future leakage. Names are prefixed
`hist_` so the anti-leak blacklist (which drops target*/future_*) leaves them as features.

The dataset already ships several past-aggregates (past_sessions_count,
avg_past_5_sessions_duration_sec, median_past_session_duration_sec, ...). Here we add the
ones that are missing: last-3 window stats, EWMA, recent trend, time since previous
session, and time-window (1/3/7d) session counts and playtime.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

PLAYER = "appmetrica_device_id"
TIME = "start"
DUR = "duration_seconds"

HIST_COLS = [
    "hist_prev_dur",
    "hist_mean_last3",
    "hist_mean_last5",
    "hist_median_last3",
    "hist_median_last5",
    "hist_max_last5",
    "hist_ewma5",
    "hist_trend_recent",
    "hist_time_since_prev_sec",
    "hist_session_number",
    "hist_sessions_last_1d",
    "hist_sessions_last_3d",
    "hist_sessions_last_7d",
    "hist_playtime_last_7d",
]


def add_history_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return df with hist_* columns added (past-only, no leakage)."""
    df = df.copy()
    df[TIME] = pd.to_datetime(df[TIME], errors="coerce")
    df = df.sort_values([PLAYER, TIME]).reset_index(drop=True)
    g = df.groupby(PLAYER, sort=False)

    pdur = g[DUR].shift(1)
    df["hist_prev_dur"] = pdur

    roll = pdur.groupby(df[PLAYER], sort=False)
    df["hist_mean_last3"] = roll.transform(lambda s: s.rolling(3, min_periods=1).mean())
    df["hist_mean_last5"] = roll.transform(lambda s: s.rolling(5, min_periods=1).mean())
    df["hist_median_last3"] = roll.transform(
        lambda s: s.rolling(3, min_periods=1).median()
    )
    df["hist_median_last5"] = roll.transform(
        lambda s: s.rolling(5, min_periods=1).median()
    )
    df["hist_max_last5"] = roll.transform(lambda s: s.rolling(5, min_periods=1).max())
    df["hist_ewma5"] = roll.transform(lambda s: s.ewm(span=5, adjust=False).mean())

    df["hist_trend_recent"] = df["hist_prev_dur"] - df["hist_mean_last5"]

    prev_start = g[TIME].shift(1)
    df["hist_time_since_prev_sec"] = (df[TIME] - prev_start).dt.total_seconds()

    df["hist_session_number"] = g.cumcount()

    s = df[[PLAYER, TIME, DUR]].copy()
    s["ones"] = 1.0
    idx = s.set_index(TIME)
    for win, days in [("1d", 1), ("3d", 3), ("7d", 7)]:
        cnt = (
            idx.groupby(PLAYER, sort=False)["ones"]
            .rolling(f"{days}D")
            .sum()
            .reset_index(level=0, drop=True)
        )
        cnt = cnt.to_numpy() - 1.0
        df[f"hist_sessions_last_{win}"] = np.maximum(cnt, 0.0)
    play7 = (
        idx.groupby(PLAYER, sort=False)[DUR]
        .rolling("7D")
        .sum()
        .reset_index(level=0, drop=True)
    )
    df["hist_playtime_last_7d"] = np.maximum(play7.to_numpy() - df[DUR].to_numpy(), 0.0)

    return df


if __name__ == "__main__":
    import time
    import pyarrow.parquet as pq

    path = "/Users/ilyakravchuk/Desktop/сдфгву/sessions_preprocessed.parquet"
    df = pq.ParquetFile(path).read().to_pandas()
    sub = df[df[PLAYER].isin(df[PLAYER].drop_duplicates().head(2000))].copy()
    t0 = time.time()
    out = add_history_features(sub)
    print(f"rows={len(out)} time={time.time() - t0:.1f}s")
    print(out[[PLAYER, TIME, DUR] + HIST_COLS].head(8).to_string())
    print("\nNaN share per hist col:")
    print(out[HIST_COLS].isna().mean().round(3).to_string())
