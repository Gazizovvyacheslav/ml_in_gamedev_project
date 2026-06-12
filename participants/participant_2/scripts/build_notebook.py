"""Generate the Participant 2 analysis notebook (loads precomputed CSVs, renders tables
and charts). Kept light on purpose: re-running the heavy sweeps lives in exp0..exp7.py.
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
NB = HERE / "7_catboost_extra_tuning_participant2.ipynb"


def md(text):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": text.splitlines(keepends=True),
    }


def code(text):
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": text.splitlines(keepends=True),
    }


cells = []

cells.append(
    md(
        """# Участник 2 — Feature Selection, категории и структура деревьев CatBoost

Дополнительные boosting-эксперименты (`distribution_boosting_extra_tuning.txt`, раздел «Участник 2»).
Таргет — `target_next_session_length_sec`. Общий пайплайн, тайм-сплит 70/15/15 и метрики взяты из
`preprocessing/preprocessing.py` и `team_modeling_protocol.txt`, чтобы результаты были сравнимы с
другими участниками.

**Состав:**
1. SHAP feature selection vs ручной top-k
2. Bootstrap: MVS vs Bernoulli / Bayesian
3. CTR-тюнинг категориальных признаков
4. Структура деревьев: SymmetricTree / Depthwise / Lossguide
5. Квантизация числовых признаков
6. (запас) rsm
7. Протокольные стратегии (capped / Quantile 0.40 / Quantile 0.35) на обоих таргетах

> Тяжёлые прогоны вынесены в скрипты `exp0…exp7.py` (запускаются один раз и сохраняют CSV в
> `outputs/`). Этот ноутбук загружает готовые результаты и строит сводки/графики. Чтобы
> пересчитать с нуля — запустите `python exp0_base_sweep.py && … && python aggregate.py`.
"""
    )
)

cells.append(
    code(
        """import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

OUT = Path('outputs')
pd.set_option('display.max_columns', 200)

def load(name):
    return pd.read_csv(OUT / f'{name}.csv')

meta = json.loads((OUT / 'anti_leak.json').read_text())
best = json.loads((OUT / 'best_base_config.json').read_text())
print('anti-leak:', meta['leakage_check'], '| chronology:', meta['chronology_check'],
      '| no-NaN:', meta['no_nan_check'])
print('sample rows:', meta['sample_rows'],
      '| train/val/test:', meta['train_rows'], meta['val_rows'], meta['test_rows'])
print('features:', meta['n_features'], '(num', meta['n_num'], '/ cat', meta['n_cat'], ')')
print('base config:', best['hp'], '| target_mode', best['target_mode'],
      '| val_mae', round(best['val_mae'], 2))
"""
    )
)

cells.append(
    md(
        """## Замечание про шум таргета

Для `target_next_session_length_sec` validation MAE у всех конфигов лежит в коридоре ~521–525 сек
(R²≈0.01). Поэтому выигрыш архитектурных правок по MAE невелик, и их главная ценность —
**компактность и скорость** модели, а не точность. Это согласуется с `team_modeling_protocol.txt`.
"""
    )
)

cells.append(md("## 1. SHAP feature selection vs ручной top-k"))
cells.append(
    code(
        """d1 = load('exp1_shap_feature_selection').sort_values('val_mae')
display(d1[['model_name','selection','n_features','val_mae','test_mae','fit_sec','model_size_kb']])

order = ['shap_k40','shap_k60','full','topk_40','topk_60']
dd = d1.set_index('model_name').loc[[m for m in order if m in d1.model_name.values]].reset_index()
fig, ax1 = plt.subplots(figsize=(7,3.4))
cols = ['#2b6cb0' if 'shap' in m else ('#a0aec0' if m=='full' else '#dd6b20') for m in dd.model_name]
ax1.bar(dd.model_name, dd.val_mae, color=cols); ax1.set_ylabel('val MAE, сек')
ax1.set_ylim(dd.val_mae.min()-1.5, dd.val_mae.max()+1.0)
ax2 = ax1.twinx(); ax2.plot(dd.model_name, dd.model_size_kb, 'o-', color='#2f855a')
ax2.set_ylabel('размер, КБ', color='#2f855a'); ax2.grid(False)
ax1.set_title('SHAP отбор vs ручной top-k'); plt.tight_layout(); plt.show()
"""
    )
)
cells.append(
    md(
        """**Вывод.** `RecursiveByShapValues` на 40 признаках (из 73) даёт лучший val MAE (521.6), обходя и
полную модель (522.5), и ручной top-k (523.3), при модели ~в 3.5 раза меньше (199 КБ против 712 КБ).
Список 40 признаков сохранён в `outputs/feature_set_shap_40.json` — можно использовать для ускорения
inference без потери качества."""
    )
)

cells.append(md("## 2. Bootstrap: MVS vs Bernoulli / Bayesian"))
cells.append(
    code(
        """d2 = load('exp2_bootstrap_mvs').sort_values('val_mae')
display(d2[['model_name','val_mae','val_mae_std','test_mae','val_long_mae','fit_sec']])
"""
    )
)
cells.append(
    md(
        """**Вывод.** MVS **не даёт** выигрыша по MAE и не ускоряет заметно обучение. Самый стабильный по
сидам (42/52/62) — Bayesian (std 0.18). На long-tail различия в пределах шума. → MVS не внедрять."""
    )
)

cells.append(md("## 3. CTR-тюнинг категориальных признаков"))
cells.append(
    code(
        """d3 = load('exp3_ctr_tuning').sort_values('val_mae')
display(d3[['model_name','one_hot_max_size','max_ctr_complexity','ctr_target_border_count',
            'val_mae','test_mae','fit_sec','model_size_kb']].head(10))

fig, ax = plt.subplots(figsize=(7,3.4))
sc = ax.scatter(d3.model_size_kb, d3.val_mae, c=d3.max_ctr_complexity, cmap='coolwarm',
                s=60, edgecolor='k', linewidth=0.4)
ax.set_xlabel('размер модели, КБ'); ax.set_ylabel('val MAE, сек')
ax.set_title('CTR: качество vs размер (цвет = ctr_complexity)')
plt.colorbar(sc, label='max_ctr_complexity'); plt.tight_layout(); plt.show()
"""
    )
)
cells.append(
    md(
        """**Вывод.** Лучший — простой `one_hot=10, ctr_complexity=1, ctr_border=1` (522.7). Комбинации
категорий (mcc=2) и тонкая квантизация таргета для CTR пользы не дают и раздувают модель (до 733 КБ)."""
    )
)

cells.append(md("## 4. Структура деревьев: Symmetric / Depthwise / Lossguide"))
cells.append(
    code(
        """d4 = load('exp4_tree_structure').sort_values('val_mae')
display(d4[['model_name','val_mae','test_mae','val_small_mae','val_normal_mae','val_long_mae',
            'fit_sec','model_size_kb']])

dd = d4.sort_values('val_mae')
fig, ax = plt.subplots(figsize=(7,3.6))
cols = ['#2b6cb0' if m.startswith('depthwise') else ('#dd6b20' if m=='symmetric' else '#a0aec0')
        for m in dd.model_name]
ax.barh(dd.model_name, dd.val_mae, color=cols); ax.invert_yaxis()
ax.set_xlim(dd.val_mae.min()-0.6, dd.val_mae.max()+0.4); ax.set_xlabel('val MAE, сек')
ax.set_title('Depthwise (синий) vs Symmetric (оранж.) vs Lossguide'); plt.tight_layout(); plt.show()
"""
    )
)
cells.append(
    md(
        """**Вывод (главная находка).** `Depthwise, min_data_in_leaf=20` — лучший val MAE (522.3) и лучший
long_mae (1943), при этом модель вдвое меньше и быстрее симметричной (359 КБ / 2.0 с против
712 КБ / 4.0 с). Рекомендация: перейти на Depthwise."""
    )
)

cells.append(md("## 5. Квантизация числовых признаков"))
cells.append(
    code(
        """d5 = load('exp5_quantization').sort_values('val_mae')
display(d5[['model_name','val_mae','test_mae','val_long_mae','fit_sec','model_size_kb']].head(10))
"""
    )
)
cells.append(
    md(
        """**Вывод.** Дефолтный `border_count=254, GreedyLogSum` оптимален по val MAE. Очень тонкая
квантизация (512, `per_float=1024`) на длинном хвосте практически не помогает. → оставить дефолт."""
    )
)

cells.append(md("## 6. (запас) rsm — random subspace"))
cells.append(
    code(
        """d6 = load('exp6_rsm').sort_values('val_mae')
d6['gap'] = d6['val_mae'] - d6['test_mae']
display(d6[['model_name','val_mae','test_mae','gap','fit_sec']])
"""
    )
)
cells.append(
    md(
        """**Вывод.** Разрыв val–test почти одинаков (~+93 сек) для всех значений rsm — дополнительной
регуляризации здесь нет; rsm полезен скорее для разнообразия моделей в ансамбле."""
    )
)

cells.append(md("## 7. Протокольные стратегии на лучшем конфиге (оба таргета)"))
cells.append(
    code(
        """d7 = load('exp7_final_strategies')
display(d7[['target','model_name','loss_function','val_mae','val_product_mae',
            'val_engagement_risk_mae','val_small_mae','test_mae']])

dn = d7[d7.target=='target_next_session_length_sec'].set_index('model_name')
order = ['capped_target','quantile_040','quantile_035']
dn = dn.loc[[m for m in order if m in dn.index]]
metrics = ['val_mae','val_product_mae','val_engagement_risk_mae','val_small_mae']
labels = ['MAE','ProductMAE','EngRiskMAE','small MAE']
x = np.arange(len(metrics)); w = 0.25
fig, ax = plt.subplots(figsize=(7,3.6))
for i,m in enumerate(order):
    ax.bar(x+(i-1)*w, [dn.loc[m,k] for k in metrics], w, label=m)
ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylabel('ошибка, сек')
ax.set_title('Next-session: trade-off стратегий'); ax.legend(fontsize=8)
plt.tight_layout(); plt.show()
"""
    )
)
cells.append(
    md(
        """**Вывод.** `capped_target` лучший по общему MAE; `Quantile:alpha=0.35` — лучшая short-risk модель
(минимальные ProductMAE / EngagementRiskMAE / small_mae). CRM-таргет заметно стабильнее next-session
(test MAE ~272 против ~429). Подтверждает командные результаты протокола."""
    )
)

cells.append(
    md(
        """# Часть II — основной CatBoost, LightGBM, каскад, остатки, Ridge

Эксперименты из второго ТЗ (`distribution_of_responsoblities.txt`): поиск лучшей технической
модели по validation MAE, сравнение групп признаков, альтернативный бустинг и более сложные
архитектуры. API в объём не входит."""
    )
)

cells.append(md("## 9. Основной CatBoost sweep по MAE (best_by_mae)"))
cells.append(
    code(
        """d8 = load('exp8_main_catboost_sweep').sort_values('val_mae')
display(d8[['model_name','depth','learning_rate','l2_leaf_reg','iterations','bootstrap',
            'target_mode','clip_mode','val_mae','test_mae','fit_sec']].head(8))
"""
    )
)
cells.append(
    md(
        """**Вывод.** Лучшая техническая модель — `depth=7, lr=0.03, l2=5, log1p_p995, Bernoulli`
(val MAE 521.8 / test 428.4). Широкий sweep дал лишь ~0.7 сек к компактному базовому — снова
упёрлись в шум таргета."""
    )
)

cells.append(md("## 10. Feature ablation групп признаков"))
cells.append(
    code(
        """d9 = load('exp9_feature_ablation').sort_values('val_mae')
display(d9[['model_name','n_features','val_mae','test_mae','fit_sec','model_size_kb']])

fig, ax1 = plt.subplots(figsize=(7,3.4))
order = ['session_only','session_install','session_install_events','top_k_40','top_k_60','top_k_73']
dd = d9.set_index('model_name').loc[[m for m in order if m in d9.model_name.values]].reset_index()
ax1.bar(dd.model_name, dd.val_mae, color='#2b6cb0'); ax1.set_ylabel('val MAE, сек', color='#2b6cb0')
ax1.set_ylim(dd.val_mae.min()-1, dd.val_mae.max()+1); ax1.tick_params(axis='x', rotation=25)
ax2 = ax1.twinx(); ax2.plot(dd.model_name, dd.model_size_kb, 'o-', color='#dd6b20')
ax2.set_ylabel('размер, КБ', color='#dd6b20'); ax2.grid(False); plt.tight_layout(); plt.show()
"""
    )
)
cells.append(
    md(
        """**Вывод.** Event-признаки почти не помогают: `session_only` (37) даёт val MAE 522.9 против 522.5
у полного набора (73), но модель в ~4 раза меньше (172 vs 712 КБ) и в ~8 раз быстрее. Для inference
event-группу можно отбрасывать почти без потерь."""
    )
)

cells.append(md("## 11. Каскад «классификация → регрессия» (hard / soft / hybrid)"))
cells.append(
    code(
        """d10 = load('exp10_cascade')
display(d10[['model_name','val_mae','val_product_mae','val_engagement_risk_mae',
             'val_small_mae','val_normal_mae','val_long_mae','test_mae']])
"""
    )
)
cells.append(
    md(
        """**Вывод (честный отрицательный).** Ни один режим не обходит одиночную модель по общему MAE
(baseline 522.5 против hard 609 / soft 563 / hybrid 527) — ошибки классификатора (acc≈0.51)
распространяются на регрессию. Полезен только `soft routing` для long-tail (long MAE 1693 vs 1955)."""
    )
)

cells.append(md("## 12. Коррекция остатков базовой модели"))
cells.append(
    code(
        """d11 = load('exp11_residual_correction')
display(d11[['model_name','val_mae','test_mae','val_small_mae','val_normal_mae',
             'val_long_mae','val_product_mae']])
"""
    )
)
cells.append(
    md(
        """**Вывод.** По общему MAE коррекция не помогает (522.5 → 528.6), но подтверждает систематику:
корректор сдвигает прогноз вниз (−36 сек) и улучшает small_mae (207.5 → 186.7) ценой normal/long.
Завышение коротких сессий реально есть, но глобально не «вычитается» из-за шума. Это диагностика."""
    )
)

cells.append(md("## 13. LightGBM как альтернативный бустинг"))
cells.append(
    code(
        """d13 = load('exp13_lightgbm').sort_values('val_mae')
display(d13[['model_name','objective','n_estimators','learning_rate','num_leaves',
             'val_mae','test_mae','fit_sec']].head(8))
print('CatBoost reference (exp8) test_mae=428.42  |  best LightGBM test_mae=%.2f' % d13.test_mae.min())
"""
    )
)
cells.append(
    md(
        """**Вывод.** Лучший LightGBM val MAE 524.7 / test 431.2 за ~2 с — сопоставим, но чуть хуже CatBoost
(521.8 / 428.4); CatBoost нативно работает с категориями и удобнее как единое семейство. Отказ от
LightGBM — результат сравнения, а не исходное предположение."""
    )
)

cells.append(md("## 14. Ridge baseline vs Dummy (оба таргета)"))
cells.append(
    code(
        """d12 = load('exp12_ridge')
for t in d12.target.unique():
    print('==', t, '==')
    display(d12[d12.target==t].sort_values('val_mae')[['model_name','val_mae','test_mae',
            'val_r2','val_small_mae']].head(5))
"""
    )
)
cells.append(
    md(
        """**Вывод.** На next-session Ridge не обходит даже `dummy_median` (581 против 555 и 521.8 у
CatBoost) — линейная модель почти бесполезна на шумном таргете. На CRM Ridge разумнее (test 331.8),
но CatBoost сильнее (~272). Это обосновывает пользу нелинейных взаимодействий CatBoost."""
    )
)

cells.append(
    md(
        """# Часть III — доработка по ревью: оба таргета, полные метрики, финальная модель

Блоки строго в технической зоне Участника 2 для **обоих таргетов**, с полным набором метрик
(MAE, MedAE, P70, P90, R², small/normal/long, ProductMAE, EngagementRiskMAE, WMAPE). Выбор по
validation; калибраторы учатся на calibration-split; test — только финальная оценка."""
    )
)

cells.append(md("## 15. Анализ val–test разрыва"))
cells.append(
    code(
        """d18 = load('exp18_val_test_gap')
display(d18[['target','split','n','mean','median','p90','long_share','small_share']])
"""
    )
)
cells.append(
    md(
        """**Вывод.** На next-session test-период объективно легче: доля long-сессий 0.122 против 0.158 в
val, среднее 537 против 661 сек — отсюда и более низкий test MAE. Модель не подглядывает в test.
Для CRM, наоборот, val чуть легче test."""
    )
)

cells.append(md("## 16. Time-aware history-признаки (оба таргета)"))
cells.append(
    code(
        """d14 = load('exp14_history_features')
display(d14[['target','model_name','n_features','val_mae','val_r2','val_product_mae',
             'val_small_mae','test_mae']])
"""
    )
)
cells.append(
    md(
        """**Вывод (по таргетам по-разному).** 14 past-only признаков (EWMA, медиана/среднее последних 3/5,
тренд, время с прошлой сессии, активность за 1/3/7 дней). На **next-session** даёт маржинальный плюс
(522.5 → 522.2; лучшие: `hist_ewma5`, `hist_median_last5`, `hist_time_since_prev_sec`) → включаем.
На **CRM** не помогает (238.7 → 238.9) → не включаем."""
    )
)

cells.append(md("## 17. Regression calibration (оба таргета)"))
cells.append(
    code(
        """d15 = load('exp15_calibration')
display(d15[['target','model_name','val_mae','val_r2','val_product_mae','val_small_mae',
             'val_long_mae','test_mae']])
"""
    )
)
cells.append(
    md(
        """**Вывод.** Калибровка (bin/isotonic/segment) улучшает `long_mae` и R² (next: long 1982→1608,
R² 0.01→0.10), но резко ухудшает small/normal и общий MAE — тянет прогноз к условному *среднему*,
тогда как MAE оптимизирует *медиану*. Для MAE-модели не включаем (выбор = `raw`)."""
    )
)

cells.append(md("## 18. Feature drift detection (оба таргета)"))
cells.append(
    code(
        """d16 = load('exp16_drift')
display(d16[['target','model_name','n_features','n_dropped','val_mae','test_mae']])
ds = pd.read_csv('outputs/drift_scores_target_next_session_length_sec.csv')
display(ds.head(8)[['feature','kind','psi','mean_shift']])
"""
    )
)
cells.append(
    md(
        """**Вывод.** Удаление drift-heavy признаков (top-5/top-10/strong PSI>0.25) не улучшает validation
MAE ни на одном таргете → в финальной модели оставляем все признаки. Drift-скоры сохранены в
`drift_scores_<target>.csv`."""
    )
)

cells.append(md("## 19. Финальная техническая модель (полные метрики, test)"))
cells.append(
    code(
        """fm = pd.read_csv('outputs/final_model_metrics.csv')
display(fm[['target','model_name','feature_set','calibration','drift_filter','mae','medae',
            'p70_abs_error','p90_abs_error','r2','small_mae','normal_mae','long_mae',
            'product_mae','engagement_risk_mae','wmape','model_size_kb','inference_us_per_row']])
print(json.dumps(json.load(open('outputs/README_final.json')), ensure_ascii=False, indent=2)[:1200])
"""
    )
)
cells.append(
    md(
        """**Вывод.** Финальная модель = best_by_mae гиперпараметры + (baseline + полезные history −
drift-heavy) + калибровка по решению блока 17, обучена для обоих таргетов. Артефакты сохранены в
`outputs/final_models/` (модель `.cbm`, список признаков, конфиг, README). Сравнение с baseline — в
таблице выше."""
    )
)

cells.append(
    md(
        """## Итоговые рекомендации Участника 2

1. **Архитектура:** `grow_policy=Depthwise, min_data_in_leaf=20` — тот же/лучший MAE при вдвое
   меньшей и более быстрой модели.
2. **Признаки:** SHAP-отбор 40 признаков вместо 73 — качество не падает, модель компактнее.
3. **Категории:** простые CTR-настройки (`ctr_complexity=1`); комбинации не оправданы.
4. **Квантизация:** дефолт `border_count=254, GreedyLogSum`.
5. **Bootstrap:** MVS не внедрять; для стабильности — Bayesian.
6. **Стратегии:** техническая точность — `capped_target`; short-risk — `Quantile:alpha=0.35`.
7. **best_by_mae:** CatBoost `depth=7, lr=0.03, l2=5, log1p_p995, Bernoulli` (val 521.8 / test 428.4).
8. **LightGBM** сопоставим, но чуть хуже CatBoost → остаёмся на CatBoost (по результату сравнения).
9. **Каскад / коррекция остатков** не улучшают общий MAE; точечно полезны (soft — long-tail,
   корректор — диагностика завышения коротких сессий).
10. **Ridge** не обходит median-baseline → нелинейный бустинг оправдан.

Артефакты: `outputs/participant2_results.csv` (схема протокола §16), per-experiment CSV,
`anti_leak.json`, `best_base_config.json`, `feature_set_shap_40.json`. Сводный PDF —
`Participant2_Report.pdf`.
"""
    )
)

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.13"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

NB.write_text(json.dumps(nb, ensure_ascii=False, indent=1))
print(f"[saved] {NB}")


if __name__ == "__main__":
    pass
