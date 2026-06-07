from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, median_absolute_error, r2_score


DEFAULT_DATA_PATH = Path("/Users/avals282006/Downloads/sessions_preprocessed.parquet")
DEFAULT_TARGET = "target_next_session_length_sec"
CRM_TARGET = "future_sessions_mean_playtime_7d"
DEFAULT_TARGETS = [DEFAULT_TARGET, CRM_TARGET]
DEFAULT_TIME_COL = "start"


ID_COLS = ["appmetrica_device_id", "installation_id", "session_id"]
TIME_COLS = ["start", "end", "session_date", "install_datetime", "prev_session_end"]
SERVICE_COLS = ["duration_hms"]


@dataclass
class PreparedData:
    x_train: pd.DataFrame
    x_val: pd.DataFrame
    x_test: pd.DataFrame
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    feature_cols: list[str]
    num_cols: list[str]
    cat_cols: list[str]
    target_col: str
    time_col: str


class TargetTransform:
    def __init__(self, mode: str = "raw"):
        self.mode = mode
        self.cap_995: float | None = None

    def fit(self, y: np.ndarray):
        y = np.asarray(y, dtype=float)
        self.cap_995 = float(np.percentile(y, 99.5))
        return self

    def transform(self, y: np.ndarray):
        y = np.asarray(y, dtype=float)
        if self.mode == "raw":
            return y
        if self.mode == "p995":
            return np.minimum(y, self.cap_995)
        if self.mode == "log1p_p995":
            return np.log1p(np.minimum(np.maximum(y, 0.0), self.cap_995))
        raise ValueError(f"Unknown mode: {self.mode}")

    def inverse(self, y_pred: np.ndarray):
        z = np.asarray(y_pred, dtype=float)
        if self.mode in ("raw", "p995"):
            out = z
        elif self.mode == "log1p_p995":
            out = np.expm1(np.clip(z, -12, 12))
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
        return np.clip(np.maximum(out, 0.0), 0.0, 1e7)


def load_data(path: str | Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    df = pd.read_parquet(path)
    return df


def add_crm_target_7d(
    df: pd.DataFrame,
    target_col: str = CRM_TARGET,
    installation_col: str = "installation_id",
    time_col: str = DEFAULT_TIME_COL,
    duration_col: str = "duration_seconds",
    horizon_days: int = 7,
) -> pd.DataFrame:
    x = df.copy()
    x[time_col] = pd.to_datetime(x[time_col], errors="coerce")
    x = x[x[time_col].notna() & x[installation_col].notna()].copy()
    x = x.sort_values([installation_col, time_col]).reset_index(drop=True)

    sec_day = 24 * 3600
    delta_ns = np.int64(horizon_days * sec_day * 1_000_000_000)
    dur = pd.to_numeric(x[duration_col], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)

    fut_play = np.zeros(len(x), dtype=np.float64)
    fut_cnt = np.zeros(len(x), dtype=np.int32)

    for _, idx in x.groupby(installation_col, sort=False).indices.items():
        idx = np.asarray(idx, dtype=np.int64)
        t = x.iloc[idx][time_col].values.astype("datetime64[ns]").astype(np.int64)
        d = dur[idx]
        n = len(idx)

        nxt = np.arange(n, dtype=np.int64) + 1
        r = np.searchsorted(t, t + delta_ns, side="right")
        c = np.maximum(r - nxt, 0)
        fut_cnt[idx] = c

        csum = np.concatenate(([0.0], np.cumsum(d)))
        s = csum[r] - csum[np.minimum(nxt, n)]
        fut_play[idx] = s

    max_t_ns = x[time_col].max().value
    cut_ns = max_t_ns - delta_ns
    observed = x[time_col].values.astype("datetime64[ns]").astype(np.int64) <= cut_ns
    mean7 = np.divide(fut_play, fut_cnt, out=np.zeros_like(fut_play), where=fut_cnt > 0)
    x[target_col] = np.where(observed, mean7, np.nan)
    return x


def time_split(df: pd.DataFrame, time_col: str = DEFAULT_TIME_COL, max_rows: int = 60000):
    x = df.copy()
    x[time_col] = pd.to_datetime(x[time_col], errors="coerce")
    x = x[x[time_col].notna()].copy()
    x = x.sort_values(time_col).reset_index(drop=True)
    if max_rows and len(x) > max_rows:
        x = x.tail(max_rows).copy().reset_index(drop=True)

    n = len(x)
    i1 = int(n * 0.70)
    i2 = int(n * 0.85)
    return x.iloc[:i1].copy(), x.iloc[i1:i2].copy(), x.iloc[i2:].copy()


def build_feature_list(df: pd.DataFrame, target_col: str = DEFAULT_TARGET):
    drop = set(ID_COLS + TIME_COLS + SERVICE_COLS + [target_col])
    for c in df.columns:
        cl = c.lower()
        if cl.startswith("target") and c != target_col:
            drop.add(c)
        if cl.startswith("future_"):
            drop.add(c)
    feature_cols = [c for c in df.columns if c not in drop]
    return feature_cols


def preprocess_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = DEFAULT_TARGET,
) -> PreparedData:
    x_train = train_df[feature_cols].copy()
    x_val = val_df[feature_cols].copy()
    x_test = test_df[feature_cols].copy()

    y_train = train_df[target_col].astype(float).values
    y_val = val_df[target_col].astype(float).values
    y_test = test_df[target_col].astype(float).values

    num_cols = x_train.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    cat_cols = [c for c in x_train.columns if c not in num_cols]

    if num_cols:
        med = x_train[num_cols].median()
        x_train[num_cols] = x_train[num_cols].fillna(med)
        x_val[num_cols] = x_val[num_cols].fillna(med)
        x_test[num_cols] = x_test[num_cols].fillna(med)

    if cat_cols:
        x_train[cat_cols] = x_train[cat_cols].astype("object").fillna("unknown")
        x_val[cat_cols] = x_val[cat_cols].astype("object").fillna("unknown")
        x_test[cat_cols] = x_test[cat_cols].astype("object").fillna("unknown")

    return PreparedData(
        x_train=x_train,
        x_val=x_val,
        x_test=x_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        feature_cols=feature_cols,
        num_cols=num_cols,
        cat_cols=cat_cols,
        target_col=target_col,
        time_col=DEFAULT_TIME_COL,
    )


def prepare_for_target(
    df: pd.DataFrame,
    target_col: str,
    time_col: str = DEFAULT_TIME_COL,
    max_rows: int = 60000,
) -> PreparedData:
    x = df.copy()
    x[time_col] = pd.to_datetime(x[time_col], errors="coerce")
    x = x[x[time_col].notna()].copy()
    x = x[x[target_col].notna()].copy()
    x = x.sort_values(time_col).reset_index(drop=True)
    if max_rows and len(x) > max_rows:
        x = x.tail(max_rows).copy().reset_index(drop=True)

    tr, va, te = time_split(x, time_col=time_col, max_rows=0)
    feats = build_feature_list(x, target_col=target_col)
    out = preprocess_splits(tr, va, te, feats, target_col=target_col)
    out.time_col = time_col
    return out


def prepare_for_targets(
    df: pd.DataFrame,
    target_cols: list[str] | None = None,
    time_col: str = DEFAULT_TIME_COL,
    max_rows: int = 60000,
) -> dict[str, PreparedData]:
    t_cols = target_cols or DEFAULT_TARGETS
    x = df.copy()
    if CRM_TARGET in t_cols and CRM_TARGET not in x.columns:
        x = add_crm_target_7d(x, target_col=CRM_TARGET, time_col=time_col)

    out: dict[str, PreparedData] = {}
    for t in t_cols:
        out[t] = prepare_for_target(x, target_col=t, time_col=time_col, max_rows=max_rows)
    return out


def product_mae(y_true, y_pred, cap=1200.0, tail_weight=0.2, over_weight=2.0, under_weight=1.0):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    yt = np.minimum(y_true, cap)
    yp = np.minimum(y_pred, cap)
    e = yp - yt
    w_tail = np.where(y_true > cap, tail_weight, 1.0)
    w_dir = np.where(e > 0, over_weight, under_weight)
    w = w_tail * w_dir
    return float(np.sum(np.abs(e) * w) / np.sum(w))


def engagement_risk_mae(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    e = np.abs(y_true - y_pred)
    w = np.where(y_true <= 300, 1.0, np.where(y_true <= 1200, 0.67, 0.05))
    return float(np.sum(w * e) / np.sum(w))


def regression_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.maximum(np.asarray(y_pred, dtype=float), 0.0)
    ae = np.abs(y_true - y_pred)

    out = {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "medae": float(median_absolute_error(y_true, y_pred)),
        "p70_abs_error": float(np.percentile(ae, 70)),
        "p90_abs_error": float(np.percentile(ae, 90)),
        "r2": float(r2_score(y_true, y_pred)),
        "product_mae": product_mae(y_true, y_pred),
        "engagement_risk_mae": engagement_risk_mae(y_true, y_pred),
    }

    small = y_true <= 300
    normal = (y_true > 300) & (y_true <= 1200)
    long_ = y_true > 1200
    out["small_mae"] = float(np.mean(ae[small])) if small.any() else np.nan
    out["normal_mae"] = float(np.mean(ae[normal])) if normal.any() else np.nan
    out["long_mae"] = float(np.mean(ae[long_])) if long_.any() else np.nan
    return out
