"""Shared helpers for Participant 2 (CatBoost feature-selection / categories / tree-structure tuning).

Reuses the team-wide pipeline from ``preprocessing.preprocessing`` so that results are
directly comparable with the other participants (same time split, same anti-leak rules,
same metric definitions).

Participant 2 mandatory experiments (distribution_boosting_extra_tuning.txt):
    1. SHAP recursive feature selection vs manual top-k pruning
    2. Bootstrap MVS vs Bernoulli / Bayesian
    3. CTR tuning of categorical features
    4. Tree structure: SymmetricTree / Depthwise / Lossguide
    5. Numeric feature quantization
    6. (optional) rsm
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


_here = Path(__file__).resolve()
_repo_root = None
for p in [_here, *_here.parents]:
    if p.name == "ml_in_gamedev_project":
        _repo_root = p
        break
if _repo_root is None:
    raise RuntimeError("could not locate ml_in_gamedev_project repo root")
sys.path.append(str(_repo_root))

from preprocessing.preprocessing import (
    load_data,
    prepare_for_targets,
    regression_metrics,
    TargetTransform,
)


DATA_PATH = Path("/Users/ilyakravchuk/Desktop/сдфгву/sessions_preprocessed.parquet")

NEXT_TARGET = "target_next_session_length_sec"
CRM_TARGET = "future_sessions_mean_playtime_7d"


MAX_ROWS = 30000

_p2_dir = next(
    (p for p in [_here, *_here.parents] if p.name == "participant_2"), _here.parent
)
OUTPUT_DIR = _p2_dir / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


EVENT_COLS = [
    "most_common_event_name",
    "most_common_connection_type",
    "most_common_event_count",
    "most_common_event_share",
    "events_total",
    "unique_events_count",
    "events_span_sec",
    "first_event_timestamp",
    "last_event_timestamp",
    "unique_device_models",
    "unique_device_types",
    "unique_app_versions",
    "unique_countries",
    "unique_cities",
    "unique_connection_types",
]

INSTALL_COLS = [
    "publisher_name",
    "tracker_name",
    "is_reinstallation",
    "is_reattribution",
    "attributed_touch_type",
    "country_iso_code",
    "device_type",
    "app_version_name",
    "connection_type",
    "time_since_install_sec",
    "traffic_source",
    "install_hour",
    "install_dayofweek",
    "install_country",
    "install_device_type",
    "install_app_version_name",
    "install_connection_type",
    "is_organic",
    "time_since_install_sec_is_missing",
    "install_hour_is_missing",
    "install_dayofweek_is_missing",
]


def feature_groups(all_cols):
    """Split the feature list into session / install / event groups."""
    event = [c for c in all_cols if c in EVENT_COLS]
    install = [c for c in all_cols if c in INSTALL_COLS]
    used = set(event) | set(install)
    session = [c for c in all_cols if c not in used]
    return dict(session=session, install=install, event=event)


BASE_PARAMS = dict(
    od_type="Iter",
    eval_metric="MAE",
    random_seed=42,
    verbose=False,
    thread_count=-1,
)


BASE_HP = dict(
    depth=6,
    learning_rate=0.05,
    l2_leaf_reg=5.0,
    iterations=600,
    od_wait=80,
)


def get_pack(target_cols=(NEXT_TARGET,), max_rows: int = MAX_ROWS):
    """Load the dataset once and build PreparedData packs for the requested targets."""
    df = load_data(DATA_PATH)
    return prepare_for_targets(df, target_cols=list(target_cols), max_rows=max_rows)


AUG_PATH = OUTPUT_DIR / "sessions_augmented.parquet"
_AUG_CACHE = {}


def load_augmented():
    """Augmented dataset (raw + CRM target + hist_* features), cached in-process."""
    import pandas as pd

    if "df" not in _AUG_CACHE:
        _AUG_CACHE["df"] = pd.read_parquet(AUG_PATH)
    return _AUG_CACHE["df"]


def get_aug_pack(
    target,
    max_rows: int = MAX_ROWS,
    base_only: bool = False,
    drop_features=None,
    keep_features=None,
):
    """Build a PreparedData pack from the augmented dataset for one target.

    base_only=True  -> drop hist_* (baseline feature set)
    drop_features   -> additionally drop these columns (drift removal)
    keep_features   -> restrict to exactly this feature list (overrides base_only)
    """
    import pandas as pd
    from preprocessing.preprocessing import (
        time_split,
        build_feature_list,
        preprocess_splits,
        DEFAULT_TIME_COL,
    )

    x = load_augmented().copy()
    x[DEFAULT_TIME_COL] = pd.to_datetime(x[DEFAULT_TIME_COL], errors="coerce")
    x = x[x[DEFAULT_TIME_COL].notna() & x[target].notna()].copy()
    x = x.sort_values(DEFAULT_TIME_COL).reset_index(drop=True)
    if max_rows and len(x) > max_rows:
        x = x.tail(max_rows).reset_index(drop=True)

    tr, va, te = time_split(x, max_rows=0)
    feats = build_feature_list(x, target_col=target)
    if keep_features is not None:
        feats = [c for c in keep_features if c in x.columns]
    else:
        if base_only:
            feats = [c for c in feats if not c.startswith("hist_")]
        if drop_features:
            drop = set(drop_features)
            feats = [c for c in feats if c not in drop]
    return preprocess_splits(tr, va, te, feats, target_col=target)


def metric_pack(y_true, y_pred) -> dict:
    """Full team metric set (regression_metrics) + WMAPE.

    Keys: mae, medae, p70_abs_error, p90_abs_error, r2, product_mae,
    engagement_risk_mae, small_mae, normal_mae, long_mae, wmape.
    """
    m = regression_metrics(y_true, y_pred)
    yt = np.asarray(y_true, dtype=float)
    yp = np.maximum(np.asarray(y_pred, dtype=float), 0.0)
    den = np.abs(yt).sum()
    m["wmape"] = float(np.abs(yt - yp).sum() / den) if den > 0 else float("nan")
    return m


class _SubPack:
    """Lightweight view of a PreparedData pack restricted to a subset of columns."""

    pass


def subset_pack(pack, keep_cols):
    keep_cols = [c for c in keep_cols if c in pack.x_train.columns]
    q = _SubPack()
    q.x_train = pack.x_train[keep_cols]
    q.x_val = pack.x_val[keep_cols]
    q.x_test = pack.x_test[keep_cols]
    q.y_train, q.y_val, q.y_test = pack.y_train, pack.y_val, pack.y_test
    q.cat_cols = [c for c in pack.cat_cols if c in keep_cols]
    q.feature_cols = list(keep_cols)
    return q


def eval_split(model, pack, tfm: TargetTransform | None = None) -> dict:
    """Predict on val/test of a PreparedData pack and return prefixed metrics + fit_sec."""

    def _pred(x):
        z = model.predict(x)
        return (
            tfm.inverse(z) if tfm is not None else np.maximum(np.asarray(z, float), 0.0)
        )

    out = {}
    for split, x, y in [
        ("val", pack.x_val, pack.y_val),
        ("test", pack.x_test, pack.y_test),
    ]:
        m = metric_pack(y, _pred(x))
        for k, v in m.items():
            out[f"{split}_{k}"] = v
    return out


def model_size_bytes(model) -> int:
    """Serialise a fitted CatBoost model to a temp file to measure on-disk size."""
    import tempfile
    import os

    with tempfile.NamedTemporaryFile(suffix=".cbm", delete=False) as fh:
        path = fh.name
    try:
        model.save_model(path)
        return os.path.getsize(path)
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


BEST_CONFIG_PATH = OUTPUT_DIR / "best_base_config.json"


def load_best_hp() -> dict:
    """Best base config found by exp0; falls back to BASE_HP if the sweep hasn't run."""
    import json

    if BEST_CONFIG_PATH.exists():
        with open(BEST_CONFIG_PATH) as fh:
            d = json.load(fh)
        return d.get("hp", dict(BASE_HP))
    return dict(BASE_HP)


def save_results(rows: list[dict], name: str) -> pd.DataFrame:
    """Write an experiment's rows to outputs/<name>.csv and return the DataFrame."""
    res = pd.DataFrame(rows)
    path = OUTPUT_DIR / f"{name}.csv"
    res.to_csv(path, index=False)
    print(f"[saved] {path}  ({len(res)} rows)")
    return res


def fit_regressor(
    pack,
    hp: dict,
    loss_function: str = "MAE",
    target_mode: str | None = None,
    cat_features=None,
    extra: dict | None = None,
    use_eval_set: bool = True,
):
    """Fit one CatBoostRegressor and return (model, fit_sec, tfm).

    target_mode in {None, 'p995', 'log1p_p995'} applies a TargetTransform; otherwise the
    raw target is used (appropriate for quantile losses).
    """
    from catboost import CatBoostRegressor

    cat_features = pack.cat_cols if cat_features is None else cat_features
    params = dict(BASE_PARAMS)
    params.update(hp)
    if extra:
        params.update(extra)

    tfm = None
    y_train = pack.y_train
    if target_mode:
        tfm = TargetTransform(mode=target_mode).fit(pack.y_train)
        y_train = tfm.transform(pack.y_train)

    model = CatBoostRegressor(loss_function=loss_function, **params)

    fit_kw = dict(cat_features=cat_features)
    if use_eval_set:
        y_val = tfm.transform(pack.y_val) if tfm is not None else pack.y_val
        fit_kw["eval_set"] = (pack.x_val, y_val)
        fit_kw["use_best_model"] = True

    t0 = time.time()
    model.fit(pack.x_train, y_train, **fit_kw)
    fit_sec = time.time() - t0
    return model, fit_sec, tfm
