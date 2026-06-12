"""Generate REPORT.md from the experiment CSVs (Participant 2)."""

from __future__ import annotations

import json

import pandas as pd

import p2_common as C

OUT = C.OUTPUT_DIR


def _csv(name):
    p = OUT / f"{name}.csv"
    return pd.read_csv(p) if p.exists() else None


def _tbl(df, cols):
    df = df[[c for c in cols if c in df.columns]].copy()
    for c in df.columns:
        if df[c].dtype.kind == "f":
            df[c] = df[c].round(2)
    return df.to_markdown(index=False)


def main():
    meta = (
        json.loads((OUT / "anti_leak.json").read_text())
        if (OUT / "anti_leak.json").exists()
        else {}
    )
    L = []
    A = L.append
    A("# Участник 2 — Boosting, признаки и архитектура CatBoost\n")
    A(
        "Все эксперименты Участника 2 из обоих ТЗ: "
        "`distribution_boosting_extra_tuning.txt` (доп. тюнинг CatBoost, разделы 1-6) и "
        "`distribution_of_responsoblities.txt` (LightGBM, основной CatBoost sweep, feature "
        "ablation, каскадная модель, коррекция остатков, Ridge). Без API. Общий пайплайн, "
        "тайм-сплит и метрики взяты из `preprocessing/preprocessing.py` и "
        "`team_modeling_protocol.txt`.\n"
    )

    if meta:
        A("## Протокол и проверки\n")
        A(f"- Таргет: `{C.NEXT_TARGET}` (next-session). CRM-таргет — зона Участника 1.")
        A(
            f"- Выборка: последние **{meta.get('sample_rows')}** строк по времени; "
            f"split 70/15/15 = {meta.get('train_rows')}/{meta.get('val_rows')}/{meta.get('test_rows')}."
        )
        A(
            f"- Признаков: {meta.get('n_features')} ({meta.get('n_num')} числовых, "
            f"{meta.get('n_cat')} категориальных)."
        )
        A(
            f"- Anti-leak: **{meta.get('leakage_check')}**, chronology: "
            f"**{meta.get('chronology_check')}**, no-NaN: **{meta.get('no_nan_check')}**."
        )
        A(
            f"- Train: {meta.get('train_time', ['', ''])[0]} → {meta.get('train_time', ['', ''])[1]}"
        )
        A(
            f"- Test:  {meta.get('test_time', ['', ''])[0]} → {meta.get('test_time', ['', ''])[1]}\n"
        )

    if (OUT / "best_base_config.json").exists():
        bc = json.loads((OUT / "best_base_config.json").read_text())
        A("## Базовая конфигурация (exp0)\n")
        A(
            f"Лучшая по validation MAE: `{bc['hp']}`, target_mode=`{bc['target_mode']}`, "
            f"val_mae={bc['val_mae']:.2f}, test_mae={bc['test_mae']:.2f}. "
            "Все эксперименты ниже варьируют по одной оси вокруг неё.\n"
        )

    sections = [
        (
            "exp1_shap_feature_selection",
            "1. SHAP feature selection vs ручной top-k",
            [
                "model_name",
                "selection",
                "n_features",
                "val_mae",
                "test_mae",
                "fit_sec",
                "model_size_kb",
            ],
            "val_mae",
        ),
        (
            "exp2_bootstrap_mvs",
            "2. Bootstrap: MVS vs Bernoulli / Bayesian",
            [
                "model_name",
                "val_mae",
                "val_mae_std",
                "test_mae",
                "val_long_mae",
                "fit_sec",
            ],
            "val_mae",
        ),
        (
            "exp3_ctr_tuning",
            "3. CTR-тюнинг категориальных признаков",
            ["model_name", "val_mae", "test_mae", "fit_sec", "model_size_kb"],
            "val_mae",
        ),
        (
            "exp4_tree_structure",
            "4. Структура деревьев: Symmetric / Depthwise / Lossguide",
            [
                "model_name",
                "val_mae",
                "test_mae",
                "val_small_mae",
                "val_normal_mae",
                "val_long_mae",
                "fit_sec",
                "model_size_kb",
            ],
            "val_mae",
        ),
        (
            "exp5_quantization",
            "5. Квантизация числовых признаков",
            [
                "model_name",
                "val_mae",
                "test_mae",
                "val_long_mae",
                "fit_sec",
                "model_size_kb",
            ],
            "val_mae",
        ),
        (
            "exp6_rsm",
            "6. (запас) rsm — random subspace",
            ["model_name", "val_mae", "test_mae", "fit_sec"],
            "val_mae",
        ),
        (
            "exp7_final_strategies",
            "7. Протокольные стратегии на лучшем конфиге (оба таргета)",
            [
                "target",
                "model_name",
                "loss_function",
                "val_mae",
                "val_product_mae",
                "val_engagement_risk_mae",
                "val_small_mae",
                "test_mae",
            ],
            None,
        ),
        (
            "exp8_main_catboost_sweep",
            "8. Основной CatBoost sweep по MAE (best_by_mae)",
            [
                "model_name",
                "depth",
                "learning_rate",
                "l2_leaf_reg",
                "iterations",
                "bootstrap",
                "target_mode",
                "clip_mode",
                "val_mae",
                "test_mae",
                "fit_sec",
            ],
            "val_mae",
        ),
        (
            "exp9_feature_ablation",
            "9. Feature ablation групп признаков",
            [
                "model_name",
                "n_features",
                "val_mae",
                "test_mae",
                "fit_sec",
                "model_size_kb",
            ],
            "val_mae",
        ),
        (
            "exp10_cascade",
            "10. Каскад «классификация → регрессия» (hard/soft/hybrid)",
            [
                "model_name",
                "val_mae",
                "val_product_mae",
                "val_engagement_risk_mae",
                "val_small_mae",
                "val_normal_mae",
                "val_long_mae",
                "test_mae",
            ],
            "val_mae",
        ),
        (
            "exp11_residual_correction",
            "11. Коррекция остатков базовой модели",
            [
                "model_name",
                "val_mae",
                "test_mae",
                "val_small_mae",
                "val_normal_mae",
                "val_long_mae",
                "val_product_mae",
            ],
            "val_mae",
        ),
        (
            "exp12_ridge",
            "12. Ridge baseline vs Dummy (оба таргета)",
            ["target", "model_name", "val_mae", "test_mae", "val_r2", "val_small_mae"],
            None,
        ),
        (
            "exp13_lightgbm",
            "13. LightGBM как альтернативный бустинг",
            [
                "model_name",
                "objective",
                "n_estimators",
                "learning_rate",
                "num_leaves",
                "val_mae",
                "test_mae",
                "fit_sec",
            ],
            "val_mae",
        ),
        (
            "exp18_val_test_gap",
            "14. Анализ val-test разрыва",
            [
                "target",
                "split",
                "n",
                "mean",
                "median",
                "p90",
                "long_share",
                "small_share",
            ],
            None,
        ),
        (
            "exp14_history_features",
            "15. Time-aware history-признаки (оба таргета)",
            [
                "target",
                "model_name",
                "n_features",
                "val_mae",
                "val_r2",
                "val_product_mae",
                "val_small_mae",
                "test_mae",
            ],
            None,
        ),
        (
            "exp15_calibration",
            "16. Regression calibration (оба таргета)",
            [
                "target",
                "model_name",
                "val_mae",
                "val_r2",
                "val_product_mae",
                "val_small_mae",
                "val_long_mae",
                "test_mae",
            ],
            None,
        ),
        (
            "exp16_drift",
            "17. Feature drift detection (оба таргета)",
            ["target", "model_name", "n_features", "n_dropped", "val_mae", "test_mae"],
            None,
        ),
    ]

    for name, title, cols, sortby in sections:
        df = _csv(name)
        if df is None:
            continue
        A(f"## {title}\n")
        view = df.sort_values(sortby) if sortby and sortby in df.columns else df
        A(_tbl(view, cols))
        A("")

    fm = OUT / "final_model_metrics.csv"
    if fm.exists():
        A("## 18. Финальная техническая модель (полный набор метрик, test)\n")
        df = pd.read_csv(fm)
        A(
            _tbl(
                df,
                [
                    "target",
                    "model_name",
                    "feature_set",
                    "calibration",
                    "drift_filter",
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
                    "model_size_kb",
                    "inference_us_per_row",
                ],
            )
        )
        A("")
        rd = OUT / "README_final.json"
        if rd.exists():
            import json as _json

            readme = _json.loads(rd.read_text())
            for t, d in readme.items():
                A(
                    f"- **{t}**: признаков {d['n_features']}, history={d['use_history']}, "
                    f"drift_filter={d['drift_filter']}, calibration={d['calibration']}; "
                    f"baseline test MAE {d['baseline_test_mae']} → final {d['final_test_mae']} "
                    f"(Δ {d['improvement_sec']:+}), {d['model_size_kb']} КБ, "
                    f"{d['inference_us_per_row']} мкс/строку."
                )
            A("")

    A("## Итоговый CSV\n")
    A(
        "`outputs/participant2_results.csv` — все запуски в схеме "
        "`team_modeling_protocol.txt` §16 (val_* / test_* метрики, params, fit_sec, status).\n"
    )

    (C.OUTPUT_DIR.parent / "REPORT.md").write_text("\n".join(L))
    print(f"[saved] {C.OUTPUT_DIR.parent / 'REPORT.md'}")


if __name__ == "__main__":
    main()
