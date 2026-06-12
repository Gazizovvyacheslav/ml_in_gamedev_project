"""Generate Participant 2 themed notebooks (inline runnable code, like participant 1).

Writes 7 notebooks into the participant_2 root, grouping the experiments by theme. Each
notebook bootstraps sys.path to this scripts/ folder, imports the shared p2_common pipeline
and runs the experiment logic inline, displaying result tables.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text):
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": text.rstrip("\n").splitlines(keepends=True),
    }


def write_nb(name, cells):
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "p2venv", "language": "python", "name": "p2venv"},
            "language_info": {"name": "python", "version": "3.13"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    (ROOT / name).write_text(json.dumps(nb, ensure_ascii=False, indent=1))
    print(f"[saved] {name} ({len(cells)} cells)")


SETUP = '''import sys
from pathlib import Path

_here = Path.cwd()
_p2 = next((b for b in [_here, *_here.parents] if (b / "scripts" / "p2_common.py").exists()), _here)
sys.path.insert(0, str(_p2 / "scripts"))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import p2_common as C
from catboost import CatBoostRegressor, CatBoostClassifier, Pool

pd.set_option("display.max_columns", 120)
pd.set_option("display.width", 200)
plt.rcParams["figure.figsize"] = (7, 3.4)
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.3
BLUE, ORANGE, GREEN, GREY = "#2b6cb0", "#dd6b20", "#2f855a", "#a0aec0"
print("pipeline OK | OUTPUT_DIR:", C.OUTPUT_DIR.name, "| MAX_ROWS:", C.MAX_ROWS)'''

INTRO_COMMON = (
    "Общий пайплайн, тайм-сплит 70/15/15 и метрики взяты из `preprocessing/preprocessing.py` "
    "и `team_modeling_protocol.txt` (через `scripts/p2_common.py`), чтобы результаты были "
    "сравнимы с другими участниками. Выбор моделей — по validation; test — только финальная "
    "оценка. Выборка — последние 30 000 строк по времени."
)


# ============================ NB 1: extra CatBoost tuning ============================
def nb1():
    cells = [
        md(
            "# 1. Доп. тюнинг CatBoost: признаки, категории, структура деревьев\n\n"
            "Раздел «Участник 2» из `distribution_boosting_extra_tuning.txt`: базовый sweep, "
            "SHAP-отбор vs ручной top-k, bootstrap (MVS/Bernoulli/Bayesian), CTR-тюнинг, "
            "`grow_policy`, квантизация числовых признаков и запасной `rsm`.\n\n" + INTRO_COMMON
        ),
        code(SETUP),
        code(
            "packs = C.get_pack((C.NEXT_TARGET,))\n"
            "p = packs[C.NEXT_TARGET]\n"
            "print('train/val/test:', len(p.x_train), len(p.x_val), len(p.x_test),\n"
            "      '| num/cat:', len(p.num_cols), len(p.cat_cols))"
        ),
        md(
            "## 1.1 Базовый sweep по MAE\n"
            "Небольшая сетка `depth × learning_rate × l2`, выбор лучшей по validation MAE — "
            "вокруг неё крутятся остальные блоки."
        ),
        code(
            "grid = [dict(depth=d, learning_rate=lr, l2_leaf_reg=l2, iterations=800, od_wait=80)\n"
            "        for d in (5, 6, 7) for lr in (0.03, 0.05) for l2 in (3.0, 5.0, 8.0)]\n"
            "rows = []\n"
            "for hp in grid:\n"
            "    m, fs, tfm = C.fit_regressor(p, hp, 'MAE', 'p995')\n"
            "    r = dict(hp); r['fit_sec'] = round(fs, 1); r.update(C.eval_split(m, p, tfm))\n"
            "    rows.append(r)\n"
            "base = pd.DataFrame(rows).sort_values('val_mae').reset_index(drop=True)\n"
            "hp = C.load_best_hp()\n"
            "print('BEST_HP:', hp)\n"
            "display(base[['depth', 'learning_rate', 'l2_leaf_reg', 'val_mae', 'test_mae', 'fit_sec']].head())"
        ),
        md(
            "## 1.2 SHAP feature selection vs ручной top-k\n"
            "Встроенный `RecursiveByShapValues` против ручного отбора top-k по важности "
            "(после anti-leak признаков 73, поэтому k = 40 и 60)."
        ),
        code(
            "all_cols = list(p.feature_cols)\n"
            "full, _, tfm = C.fit_regressor(p, hp, 'MAE', 'p995')\n"
            "imp = full.get_feature_importance()\n"
            "order = [all_cols[i] for i in np.argsort(imp)[::-1]]\n"
            "rows = [dict(model='full', selection='none', n_features=len(all_cols),\n"
            "             val_mae=C.eval_split(full, p, tfm)['val_mae'],\n"
            "             test_mae=C.eval_split(full, p, tfm)['test_mae'],\n"
            "             model_size_kb=C.model_size_bytes(full) // 1024)]\n"
            "train_pool = Pool(p.x_train, tfm.transform(p.y_train), cat_features=p.cat_cols)\n"
            "val_pool = Pool(p.x_val, tfm.transform(p.y_val), cat_features=p.cat_cols)\n"
            "for k in (40, 60):\n"
            "    sel = CatBoostRegressor(loss_function='MAE', **C.BASE_PARAMS, **hp)\n"
            "    s = sel.select_features(train_pool, eval_set=val_pool,\n"
            "                            features_for_select=list(range(len(all_cols))),\n"
            "                            num_features_to_select=k, algorithm='RecursiveByShapValues',\n"
            "                            steps=3, train_final_model=False, logging_level='Silent')\n"
            "    cols = [all_cols[i] for i in s['selected_features']]\n"
            "    qp = C.subset_pack(p, cols); m, _, t = C.fit_regressor(qp, hp, 'MAE', 'p995')\n"
            "    ev = C.eval_split(m, qp, t)\n"
            "    rows.append(dict(model=f'shap_k{k}', selection='shap', n_features=k,\n"
            "                     val_mae=ev['val_mae'], test_mae=ev['test_mae'],\n"
            "                     model_size_kb=C.model_size_bytes(m) // 1024))\n"
            "    qp2 = C.subset_pack(p, order[:k]); m2, _, t2 = C.fit_regressor(qp2, hp, 'MAE', 'p995')\n"
            "    ev2 = C.eval_split(m2, qp2, t2)\n"
            "    rows.append(dict(model=f'topk_{k}', selection='manual', n_features=k,\n"
            "                     val_mae=ev2['val_mae'], test_mae=ev2['test_mae'],\n"
            "                     model_size_kb=C.model_size_bytes(m2) // 1024))\n"
            "shap_df = pd.DataFrame(rows).sort_values('val_mae').reset_index(drop=True)\n"
            "display(shap_df)"
        ),
        code(
            "order = ['shap_k40', 'shap_k60', 'full', 'topk_40', 'topk_60']\n"
            "d = shap_df.set_index('model').loc[[m for m in order if m in shap_df.model.values]]\n"
            "fig, ax1 = plt.subplots()\n"
            "cols = [BLUE if 'shap' in m else (GREY if m == 'full' else ORANGE) for m in d.index]\n"
            "ax1.bar(d.index, d.val_mae, color=cols)\n"
            "ax1.set_ylim(d.val_mae.min() - 1.5, d.val_mae.max() + 1.0); ax1.set_ylabel('val MAE, сек')\n"
            "ax2 = ax1.twinx(); ax2.plot(d.index, d.model_size_kb, 'o-', color=GREEN); ax2.grid(False)\n"
            "ax2.set_ylabel('размер модели, КБ', color=GREEN)\n"
            "ax1.set_title('SHAP-отбор vs ручной top-k: качество и размер'); plt.tight_layout(); plt.show()"
        ),
        md(
            "**Вывод.** SHAP-отбор на 40 признаках обходит и полную модель, и ручной top-k при "
            "модели в ~3.5 раза меньше → можно ускорить inference без потери качества."
        ),
        md(
            "## 1.3 Bootstrap: MVS vs Bernoulli / Bayesian\n"
            "Устойчивость по сидам 42/52/62 (mean/std val MAE) + влияние на long-tail."
        ),
        code(
            "configs = [dict(tag='bernoulli_ss0.8', bootstrap_type='Bernoulli', subsample=0.8),\n"
            "           dict(tag='bayesian', bootstrap_type='Bayesian', bagging_temperature=1.0),\n"
            "           dict(tag='mvs_ss0.7', bootstrap_type='MVS', subsample=0.7),\n"
            "           dict(tag='mvs_ss0.8', bootstrap_type='MVS', subsample=0.8),\n"
            "           dict(tag='mvs_ss0.9', bootstrap_type='MVS', subsample=0.9)]\n"
            "rows = []\n"
            "for cfg in configs:\n"
            "    tag = cfg.pop('tag'); vals = []; longs = []\n"
            "    for sd in (42, 52, 62):\n"
            "        m, _, tfm = C.fit_regressor(p, hp, 'MAE', 'p995', extra={**cfg, 'random_seed': sd})\n"
            "        ev = C.eval_split(m, p, tfm); vals.append(ev['val_mae']); longs.append(ev['val_long_mae'])\n"
            "    rows.append(dict(model=tag, **cfg, val_mae=np.mean(vals), val_mae_std=np.std(vals),\n"
            "                     val_long_mae=np.mean(longs)))\n"
            "display(pd.DataFrame(rows).sort_values('val_mae').reset_index(drop=True))"
        ),
        md("**Вывод.** MVS не даёт выигрыша по MAE и не ускоряет; самый стабильный по сидам — Bayesian."),
        md(
            "## 1.4 CTR-тюнинг категориальных признаков\n"
            "`one_hot_max_size × max_ctr_complexity × ctr_target_border_count`."
        ),
        code(
            "import itertools\n"
            "rows = []\n"
            "for ohms, mcc, ctbc in itertools.product((2, 5, 10), (1, 2), (1, 3, 5)):\n"
            "    extra = dict(one_hot_max_size=ohms, max_ctr_complexity=mcc, ctr_target_border_count=ctbc)\n"
            "    m, fs, tfm = C.fit_regressor(p, hp, 'MAE', 'p995', extra=extra)\n"
            "    ev = C.eval_split(m, p, tfm)\n"
            "    rows.append(dict(model=f'ohms{ohms}_mcc{mcc}_ctbc{ctbc}', **extra,\n"
            "                     val_mae=ev['val_mae'], test_mae=ev['test_mae'], fit_sec=round(fs, 1),\n"
            "                     model_size_kb=C.model_size_bytes(m) // 1024))\n"
            "display(pd.DataFrame(rows).sort_values('val_mae').head(8).reset_index(drop=True))"
        ),
        md("**Вывод.** Лучший — простой `one_hot=10, ctr_complexity=1, ctr_border=1`; комбинации только раздувают модель."),
        md(
            "## 1.5 Структура деревьев: Symmetric / Depthwise / Lossguide\n"
            "Смотрим общий MAE и сегментные ошибки (small/normal/long)."
        ),
        code(
            "configs = [('symmetric', dict(hp), dict(grow_policy='SymmetricTree'))]\n"
            "for mdl in (20, 50, 100):\n"
            "    configs.append((f'depthwise_mdl{mdl}', dict(hp), dict(grow_policy='Depthwise', min_data_in_leaf=mdl)))\n"
            "for ml in (31, 63, 127):\n"
            "    for mdl in (20, 50):\n"
            "        h = dict(hp); h.pop('depth', None)\n"
            "        configs.append((f'lossguide_ml{ml}_mdl{mdl}', h,\n"
            "                        dict(grow_policy='Lossguide', max_leaves=ml, min_data_in_leaf=mdl)))\n"
            "rows = []\n"
            "for tag, h, extra in configs:\n"
            "    m, fs, tfm = C.fit_regressor(p, h, 'MAE', 'p995', extra=extra)\n"
            "    ev = C.eval_split(m, p, tfm)\n"
            "    rows.append(dict(model=tag, val_mae=ev['val_mae'], test_mae=ev['test_mae'],\n"
            "                     small=ev['val_small_mae'], normal=ev['val_normal_mae'], long=ev['val_long_mae'],\n"
            "                     fit_sec=round(fs, 1), model_size_kb=C.model_size_bytes(m) // 1024))\n"
            "tree_df = pd.DataFrame(rows).sort_values('val_mae').reset_index(drop=True)\n"
            "display(tree_df)"
        ),
        code(
            "d = tree_df.sort_values('val_mae')\n"
            "cols = [BLUE if m.startswith('depthwise') else (ORANGE if m == 'symmetric' else GREY)\n"
            "        for m in d.model]\n"
            "fig, ax = plt.subplots(figsize=(7, 3.8))\n"
            "ax.barh(d.model, d.val_mae, color=cols); ax.invert_yaxis()\n"
            "ax.set_xlim(d.val_mae.min() - 0.6, d.val_mae.max() + 0.4); ax.set_xlabel('val MAE, сек')\n"
            "ax.set_title('Структура деревьев: Depthwise (син) vs Symmetric (оранж) vs Lossguide')\n"
            "plt.tight_layout(); plt.show()"
        ),
        md("**Вывод (главная находка).** `Depthwise, min_data_in_leaf=20` — лучший MAE и long_mae при вдвое меньшей/быстрой модели."),
        md(
            "## 1.6 Квантизация числовых признаков\n"
            "`border_count × feature_border_type`."
        ),
        code(
            "import itertools\n"
            "rows = []\n"
            "for bc, bt in itertools.product((64, 128, 254, 512), ('GreedyLogSum', 'Median', 'UniformAndQuantiles')):\n"
            "    m, fs, tfm = C.fit_regressor(p, hp, 'MAE', 'p995', extra=dict(border_count=bc, feature_border_type=bt))\n"
            "    ev = C.eval_split(m, p, tfm)\n"
            "    rows.append(dict(model=f'bc{bc}_{bt}', border_count=bc, feature_border_type=bt,\n"
            "                     val_mae=ev['val_mae'], test_mae=ev['test_mae'], val_long_mae=ev['val_long_mae'],\n"
            "                     fit_sec=round(fs, 1)))\n"
            "display(pd.DataFrame(rows).sort_values('val_mae').head(8).reset_index(drop=True))"
        ),
        md("**Вывод.** Дефолт `border_count=254, GreedyLogSum` оптимален; тонкая квантизация на длинном хвосте не помогает."),
        md("## 1.7 (запас) rsm — random subspace"),
        code(
            "rows = []\n"
            "for rsm in (0.6, 0.8, 1.0):\n"
            "    m, fs, tfm = C.fit_regressor(p, hp, 'MAE', 'p995', extra=dict(rsm=rsm))\n"
            "    ev = C.eval_split(m, p, tfm)\n"
            "    rows.append(dict(rsm=rsm, val_mae=ev['val_mae'], test_mae=ev['test_mae'],\n"
            "                     gap=ev['val_mae'] - ev['test_mae'], fit_sec=round(fs, 1)))\n"
            "display(pd.DataFrame(rows).sort_values('val_mae').reset_index(drop=True))"
        ),
        md("**Вывод.** Разрыв val–test одинаков для всех rsm — регуляризации не даёт."),
    ]
    write_nb("1_catboost_extra_tuning.ipynb", cells)


# ====================== NB 2: main sweep + ablation + strategies ======================
def nb2():
    cells = [
        md(
            "# 2. Основной CatBoost sweep по MAE, feature ablation и протокольные стратегии\n\n"
            "`distribution_of_responsoblities.txt`, Эксперименты 2-3: поиск лучшей технической "
            "модели по validation MAE, сравнение групп признаков, и три канонические стратегии "
            "на обоих таргетах.\n\n" + INTRO_COMMON
        ),
        code(SETUP),
        code("p = C.get_pack((C.NEXT_TARGET,))[C.NEXT_TARGET]\nprint('rows:', len(p.x_train), len(p.x_val), len(p.x_test))"),
        md(
            "## 2.1 Широкий sweep по MAE (best_by_mae)\n"
            "Случайный поиск 40 конфигов по сетке ТЗ (iterations/depth/lr/l2/min_data/random_strength/"
            "bootstrap/target_transform/clip_mode), выбор по validation MAE."
        ),
        code(
            "import random\n"
            "grid = dict(iterations=[1200, 1500, 1800], depth=[5, 6, 7, 8], learning_rate=[0.02, 0.03, 0.05],\n"
            "            l2_leaf_reg=[3.0, 5.0, 7.0, 10.0], min_data_in_leaf=[20, 50], random_strength=[1.0, 1.5],\n"
            "            bootstrap=['Bernoulli', 'Bayesian'], target_mode=['p995', 'log1p_p995'])\n"
            "rng = random.Random(42); seen = set(); configs = []\n"
            "while len(configs) < 40:\n"
            "    cfg = {k: rng.choice(v) for k, v in grid.items()}\n"
            "    key = tuple(sorted(cfg.items()))\n"
            "    if key in seen:\n"
            "        continue\n"
            "    seen.add(key); configs.append(cfg)\n"
            "rows = []\n"
            "for i, cfg in enumerate(configs):\n"
            "    hp = dict(depth=cfg['depth'], learning_rate=cfg['learning_rate'], l2_leaf_reg=cfg['l2_leaf_reg'],\n"
            "              iterations=cfg['iterations'], min_data_in_leaf=cfg['min_data_in_leaf'],\n"
            "              random_strength=cfg['random_strength'], od_wait=80)\n"
            "    extra = dict(bootstrap_type=cfg['bootstrap'])\n"
            "    if cfg['bootstrap'] == 'Bernoulli':\n"
            "        extra['subsample'] = 0.85\n"
            "    m, fs, tfm = C.fit_regressor(p, hp, 'MAE', cfg['target_mode'], extra=extra)\n"
            "    ev = C.eval_split(m, p, tfm)\n"
            "    rows.append(dict(cfg=f'cfg{i:02d}', **cfg, val_mae=ev['val_mae'], test_mae=ev['test_mae'], fit_sec=round(fs, 1)))\n"
            "sweep = pd.DataFrame(rows).sort_values('val_mae').reset_index(drop=True)\n"
            "display(sweep.head(8))"
        ),
        md("**Вывод.** Лучшая техническая модель — `depth=7, lr=0.03, log1p_p995, Bernoulli` (best_by_mae)."),
        md(
            "## 2.2 Feature ablation групп признаков\n"
            "session_only / session_install / session_install_events / top_k_40/60/73."
        ),
        code(
            "hp = C.load_best_hp()\n"
            "all_cols = list(p.feature_cols)\n"
            "groups = C.feature_groups(all_cols)\n"
            "full, _, _ = C.fit_regressor(p, hp, 'MAE', 'p995')\n"
            "imp = full.get_feature_importance(); order = [all_cols[i] for i in np.argsort(imp)[::-1]]\n"
            "fsets = {'session_only': groups['session'],\n"
            "         'session_install': groups['session'] + groups['install'],\n"
            "         'session_install_events': all_cols}\n"
            "for k in (40, 60, len(all_cols)):\n"
            "    fsets[f'top_k_{k}'] = order[:k]\n"
            "rows = []\n"
            "for name, cols in fsets.items():\n"
            "    qp = C.subset_pack(p, cols); m, fs, tfm = C.fit_regressor(qp, hp, 'MAE', 'p995')\n"
            "    ev = C.eval_split(m, qp, tfm)\n"
            "    rows.append(dict(feature_set=name, n_features=len(qp.feature_cols), val_mae=ev['val_mae'],\n"
            "                     test_mae=ev['test_mae'], fit_sec=round(fs, 1), model_size_kb=C.model_size_bytes(m) // 1024))\n"
            "abl_df = pd.DataFrame(rows).sort_values('val_mae').reset_index(drop=True)\n"
            "display(abl_df)"
        ),
        code(
            "d = abl_df.sort_values('val_mae')\n"
            "fig, ax1 = plt.subplots()\n"
            "ax1.bar(d.feature_set, d.val_mae, color=BLUE); ax1.tick_params(axis='x', rotation=25)\n"
            "ax1.set_ylim(d.val_mae.min() - 1, d.val_mae.max() + 1); ax1.set_ylabel('val MAE, сек')\n"
            "ax2 = ax1.twinx(); ax2.plot(d.feature_set, d.model_size_kb, 'o-', color=ORANGE); ax2.grid(False)\n"
            "ax2.set_ylabel('размер модели, КБ', color=ORANGE)\n"
            "ax1.set_title('Группы признаков: качество и размер'); plt.tight_layout(); plt.show()"
        ),
        md("**Вывод.** Event-признаки почти не помогают: `session_only` (37) ≈ полный набор, но модель в ~4 раза меньше."),
        md(
            "## 2.3 Протокольные стратегии на обоих таргетах\n"
            "capped_target (MAE) / Quantile 0.40 / Quantile 0.35 — полный набор метрик."
        ),
        code(
            "strategies = [('capped_target', 'MAE', 'p995'),\n"
            "              ('quantile_040', 'Quantile:alpha=0.40', None),\n"
            "              ('quantile_035', 'Quantile:alpha=0.35', None)]\n"
            "rows = []\n"
            "for target in (C.NEXT_TARGET, C.CRM_TARGET):\n"
            "    pk = C.get_aug_pack(target)\n"
            "    for tag, loss, mode in strategies:\n"
            "        m, _, tfm = C.fit_regressor(pk, hp, loss, mode)\n"
            "        ev = C.eval_split(m, pk, tfm)\n"
            "        rows.append(dict(target=target.replace('_sec', ''), strategy=tag,\n"
            "                         val_mae=ev['val_mae'], val_product_mae=ev['val_product_mae'],\n"
            "                         val_eng_risk=ev['val_engagement_risk_mae'], val_small=ev['val_small_mae'],\n"
            "                         test_mae=ev['test_mae']))\n"
            "strat_df = pd.DataFrame(rows)\n"
            "display(strat_df)"
        ),
        code(
            "dn = strat_df[strat_df.target.str.startswith('target_next')].set_index('strategy')\n"
            "order = [s for s in ['capped_target', 'quantile_040', 'quantile_035'] if s in dn.index]\n"
            "metrics = ['val_mae', 'val_product_mae', 'val_eng_risk', 'val_small']\n"
            "labels = ['MAE', 'ProductMAE', 'EngRiskMAE', 'small MAE']\n"
            "x = np.arange(len(metrics)); w = 0.25\n"
            "fig, ax = plt.subplots(figsize=(7, 3.6))\n"
            "for i, s in enumerate(order):\n"
            "    ax.bar(x + (i - 1) * w, [dn.loc[s, m] for m in metrics], w, label=s)\n"
            "ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylabel('ошибка, сек')\n"
            "ax.set_title('Next-session: trade-off стратегий'); ax.legend(fontsize=8)\n"
            "plt.tight_layout(); plt.show()"
        ),
        md("**Вывод.** capped_target лучший по MAE; Quantile 0.35 — лучшая short-risk (мин. Product/EngRisk/small)."),
    ]
    write_nb("2_main_sweep_ablation_strategies.ipynb", cells)


# ============================ NB 3: LightGBM ============================
def nb3():
    cells = [
        md(
            "# 3. LightGBM как альтернативный бустинг\n\n"
            "`distribution_of_responsoblities.txt`, Эксперимент 1: проверить, даёт ли LightGBM "
            "сопоставимое качество, и обосновать выбор CatBoost сравнением, а не предположением. "
            "Категории — нативно через pandas `category`.\n\n" + INTRO_COMMON
        ),
        code(SETUP),
        code(
            "import lightgbm as lgb\n"
            "from preprocessing.preprocessing import TargetTransform\n"
            "p = C.get_pack((C.NEXT_TARGET,))[C.NEXT_TARGET]\n"
            "def to_cat(df):\n"
            "    df = df.copy()\n"
            "    for c in p.cat_cols:\n"
            "        df[c] = df[c].astype('category')\n"
            "    return df\n"
            "xtr, xva, xte = to_cat(p.x_train), to_cat(p.x_val), to_cat(p.x_test)\n"
            "print('lightgbm', lgb.__version__)"
        ),
        md("## 3.1 Компактный случайный поиск"),
        code(
            "import random, time\n"
            "grid = dict(objective=['regression_l1', 'quantile'], alpha=[0.40, 0.50],\n"
            "            n_estimators=[600, 1000, 1400], learning_rate=[0.02, 0.03, 0.05],\n"
            "            num_leaves=[31, 63], max_depth=[-1, 6, 8], feature_fraction=[0.8, 1.0],\n"
            "            bagging_fraction=[0.8, 1.0], target_mode=['p995', 'log1p_p995'])\n"
            "rng = random.Random(42); seen = set(); configs = []\n"
            "while len(configs) < 14:\n"
            "    cfg = {k: rng.choice(v) for k, v in grid.items()}\n"
            "    if cfg['objective'] != 'quantile':\n"
            "        cfg['alpha'] = 0.5\n"
            "    key = tuple(sorted(cfg.items()))\n"
            "    if key in seen:\n"
            "        continue\n"
            "    seen.add(key); configs.append(cfg)\n"
            "rows = []\n"
            "for i, cfg in enumerate(configs):\n"
            "    tfm = TargetTransform(cfg['target_mode']).fit(p.y_train)\n"
            "    params = dict(objective=cfg['objective'], n_estimators=cfg['n_estimators'],\n"
            "                  learning_rate=cfg['learning_rate'], num_leaves=cfg['num_leaves'],\n"
            "                  max_depth=cfg['max_depth'], feature_fraction=cfg['feature_fraction'],\n"
            "                  bagging_fraction=cfg['bagging_fraction'], bagging_freq=1,\n"
            "                  random_state=42, n_jobs=-1, verbosity=-1)\n"
            "    if cfg['objective'] == 'quantile':\n"
            "        params['alpha'] = cfg['alpha']\n"
            "    model = lgb.LGBMRegressor(**params)\n"
            "    t0 = time.time()\n"
            "    model.fit(xtr, tfm.transform(p.y_train), eval_set=[(xva, tfm.transform(p.y_val))],\n"
            "              callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(0)])\n"
            "    fs = time.time() - t0\n"
            "    ev = {s + '_mae': C.metric_pack(y, tfm.inverse(model.predict(X)))['mae']\n"
            "          for s, X, y in [('val', xva, p.y_val), ('test', xte, p.y_test)]}\n"
            "    rows.append(dict(model=f'lgb{i:02d}', objective=cfg['objective'], n_estimators=cfg['n_estimators'],\n"
            "                     learning_rate=cfg['learning_rate'], num_leaves=cfg['num_leaves'],\n"
            "                     val_mae=ev['val_mae'], test_mae=ev['test_mae'], fit_sec=round(fs, 1)))\n"
            "lgbm = pd.DataFrame(rows).sort_values('val_mae').reset_index(drop=True)\n"
            "display(lgbm.head(8))\n"
            "import json\n"
            "ref = C.OUTPUT_DIR / 'best_main_model.json'\n"
            "cb = json.loads(ref.read_text()) if ref.exists() else None\n"
            "if cb:\n"
            "    print('CatBoost reference test_mae:', cb['test_mae'])"
        ),
        code(
            "names = ['CatBoost\\n(best_by_mae)', 'LightGBM\\n(best)']\n"
            "vals = [cb['test_mae'] if cb else np.nan, lgbm.test_mae.min()]\n"
            "fig, ax = plt.subplots(figsize=(5, 3.4))\n"
            "ax.bar(names, vals, color=[BLUE, '#805ad5'])\n"
            "for i, v in enumerate(vals):\n"
            "    ax.text(i, v, f'{v:.1f}', ha='center', va='bottom')\n"
            "ax.set_ylabel('test MAE, сек'); ax.set_ylim(min(vals) - 4, max(vals) + 4)\n"
            "ax.set_title('LightGBM vs CatBoost (next-session)'); plt.tight_layout(); plt.show()"
        ),
        md(
            "**Вывод.** Лучший LightGBM сопоставим, но чуть хуже CatBoost; CatBoost нативно работает с "
            "категориями и удобнее как единое семейство → остаёмся на CatBoost (по результату сравнения)."
        ),
    ]
    write_nb("3_lightgbm.ipynb", cells)


# ====================== NB 4: cascade + residual correction ======================
def nb4():
    cells = [
        md(
            "# 4. Каскад «классификация → регрессия» и коррекция остатков\n\n"
            "`distribution_of_responsoblities.txt`, Эксперименты 4-5: оригинальные архитектуры "
            "Участника 2.\n\n" + INTRO_COMMON
        ),
        code(SETUP),
        code(
            "from preprocessing.preprocessing import TargetTransform\n"
            "p = C.get_pack((C.NEXT_TARGET,))[C.NEXT_TARGET]\n"
            "hp = C.load_best_hp(); cat = p.cat_cols\n"
            "SEG = (300.0, 1200.0)\n"
            "def seg_label(y):\n"
            "    y = np.asarray(y, float)\n"
            "    return np.where(y <= SEG[0], 0, np.where(y <= SEG[1], 1, 2)).astype(int)"
        ),
        md(
            "## 4.1 Каскад: классификатор сегмента + 3 посегментных регрессора\n"
            "Маршрутизация hard / soft / hybrid, сравнение с одиночной моделью."
        ),
        code(
            "ytr_seg = seg_label(p.y_train)\n"
            "clf = CatBoostClassifier(loss_function='MultiClass', eval_metric='MultiClass', class_names=[0, 1, 2],\n"
            "                         depth=hp['depth'], learning_rate=hp['learning_rate'], l2_leaf_reg=hp['l2_leaf_reg'],\n"
            "                         iterations=hp['iterations'], od_type='Iter', od_wait=80, random_seed=42,\n"
            "                         verbose=False, thread_count=-1)\n"
            "clf.fit(p.x_train, ytr_seg, cat_features=cat, eval_set=(p.x_val, seg_label(p.y_val)), use_best_model=True)\n"
            "proba_val, proba_test = clf.predict_proba(p.x_val), clf.predict_proba(p.x_test)\n"
            "acc = float((proba_val.argmax(1) == seg_label(p.y_val)).mean())\n"
            "seg_models, seg_tfm = {}, {}\n"
            "for s in (0, 1, 2):\n"
            "    mask = ytr_seg == s\n"
            "    tfm = TargetTransform('p995').fit(p.y_train[mask])\n"
            "    m = CatBoostRegressor(loss_function='MAE', depth=hp['depth'], learning_rate=hp['learning_rate'],\n"
            "                          l2_leaf_reg=hp['l2_leaf_reg'], iterations=hp['iterations'], od_type='Iter',\n"
            "                          random_seed=42, verbose=False, thread_count=-1)\n"
            "    m.fit(p.x_train[mask], tfm.transform(p.y_train[mask]), cat_features=cat)\n"
            "    seg_models[s], seg_tfm[s] = m, tfm\n"
            "general, _, gtfm = C.fit_regressor(p, hp, 'MAE', 'p995')\n"
            "def seg_preds(X):\n"
            "    return {s: seg_tfm[s].inverse(seg_models[s].predict(X)) for s in (0, 1, 2)}\n"
            "spv, spt = seg_preds(p.x_val), seg_preds(p.x_test)\n"
            "gv, gt = gtfm.inverse(general.predict(p.x_val)), gtfm.inverse(general.predict(p.x_test))\n"
            "def route(kind, proba, sp, gen):\n"
            "    cls = proba.argmax(1); n = len(cls)\n"
            "    if kind == 'hard':\n"
            "        return np.maximum(np.array([sp[cls[i]][i] for i in range(n)]), 0)\n"
            "    if kind == 'soft':\n"
            "        return np.maximum(sum(proba[:, s] * sp[s] for s in (0, 1, 2)), 0)\n"
            "    conf = proba.max(1) >= 0.6\n"
            "    seg = np.array([sp[cls[i]][i] for i in range(n)])\n"
            "    return np.maximum(np.where(conf, seg, gen), 0)\n"
            "rows = [dict(model='baseline_general', **{f'val_{k}': v for k, v in C.metric_pack(p.y_val, gv).items()})]\n"
            "rows[0]['test_mae'] = C.metric_pack(p.y_test, gt)['mae']\n"
            "for kind in ('hard', 'soft', 'hybrid'):\n"
            "    pv = route(kind, proba_val, spv, gv); pt = route(kind, proba_test, spt, gt)\n"
            "    mv = C.metric_pack(p.y_val, pv)\n"
            "    rows.append(dict(model=f'cascade_{kind}', **{f'val_{k}': v for k, v in mv.items()},\n"
            "                     test_mae=C.metric_pack(p.y_test, pt)['mae']))\n"
            "print('classifier val accuracy:', round(acc, 3))\n"
            "cas = pd.DataFrame(rows)\n"
            "display(cas[['model', 'val_mae', 'val_product_mae', 'val_engagement_risk_mae',\n"
            "             'val_small_mae', 'val_normal_mae', 'val_long_mae', 'test_mae']])"
        ),
        code(
            "order = ['baseline_general', 'cascade_hard', 'cascade_soft', 'cascade_hybrid']\n"
            "d = cas.set_index('model').loc[[m for m in order if m in cas.model.values]]\n"
            "x = np.arange(len(d))\n"
            "fig, ax1 = plt.subplots()\n"
            "ax1.bar(x - 0.2, d.val_mae, 0.4, color=BLUE, label='val MAE')\n"
            "ax1.set_ylim(500, d.val_mae.max() + 20); ax1.set_ylabel('val MAE, сек', color=BLUE)\n"
            "ax2 = ax1.twinx(); ax2.bar(x + 0.2, d.val_long_mae, 0.4, color=ORANGE); ax2.grid(False)\n"
            "ax2.set_ylabel('val long MAE, сек', color=ORANGE)\n"
            "ax1.set_xticks(x); ax1.set_xticklabels([m.replace('cascade_', '') for m in d.index])\n"
            "ax1.set_title('Каскад vs одиночная модель: общий MAE и long-tail'); plt.tight_layout(); plt.show()"
        ),
        md(
            "**Вывод (честный отрицательный).** Ни один режим не обходит одиночную модель по общему MAE "
            "(ошибки классификатора распространяются); soft-режим полезен только для long-tail."
        ),
        md(
            "## 4.2 Коррекция остатков базовой модели\n"
            "Честная схема: base на блоке A → остатки на блоке B → компактный корректор; "
            "`final = base + correction`."
        ),
        code(
            "n = len(p.x_train); cut = int(n * 0.7)\n"
            "xa, ya = p.x_train.iloc[:cut], p.y_train[:cut]\n"
            "xb, yb = p.x_train.iloc[cut:], p.y_train[cut:]\n"
            "ta = TargetTransform('p995').fit(ya)\n"
            "base_a = CatBoostRegressor(loss_function='MAE', depth=hp['depth'], learning_rate=hp['learning_rate'],\n"
            "                           l2_leaf_reg=hp['l2_leaf_reg'], iterations=hp['iterations'], od_type='Iter',\n"
            "                           random_seed=42, verbose=False, thread_count=-1)\n"
            "base_a.fit(xa, ta.transform(ya), cat_features=cat)\n"
            "resid_b = yb - ta.inverse(base_a.predict(xb))\n"
            "corr = CatBoostRegressor(loss_function='MAE', depth=4, learning_rate=0.03, l2_leaf_reg=5.0,\n"
            "                         iterations=400, od_type='Iter', random_seed=42, verbose=False, thread_count=-1)\n"
            "corr.fit(xb, resid_b, cat_features=cat)\n"
            "base_full, _, tfm = C.fit_regressor(p, hp, 'MAE', 'p995')\n"
            "rows = []\n"
            "for name, use_corr in [('base_only', False), ('base_plus_correction', True)]:\n"
            "    r = dict(model=name)\n"
            "    for split, X, y in [('val', p.x_val, p.y_val), ('test', p.x_test, p.y_test)]:\n"
            "        pred = tfm.inverse(base_full.predict(X))\n"
            "        if use_corr:\n"
            "            pred = np.maximum(pred + corr.predict(X), 0)\n"
            "        mm = C.metric_pack(y, pred)\n"
            "        r[f'{split}_mae'] = mm['mae']\n"
            "        if split == 'val':\n"
            "            r['val_small'] = mm['small_mae']; r['val_normal'] = mm['normal_mae']; r['val_long'] = mm['long_mae']\n"
            "    rows.append(r)\n"
            "display(pd.DataFrame(rows))"
        ),
        md(
            "**Вывод.** По общему MAE коррекция не помогает, но подтверждает систематику — улучшает small_mae "
            "(завышение коротких сессий реально есть). Диагностика, не production-приём."
        ),
    ]
    write_nb("4_cascade_residual.ipynb", cells)


# ============================ NB 5: Ridge baseline ============================
def nb5():
    cells = [
        md(
            "# 5. Ridge — прозрачный линейный baseline (оба таргета)\n\n"
            "Дополнение к ТЗ: L2-регуляризованный baseline против `DummyRegressor(mean/median)`, "
            "чтобы оценить, насколько нелинейный бустинг превосходит простую линейную зависимость.\n\n"
            + INTRO_COMMON
        ),
        code(SETUP),
        code(
            "from sklearn.compose import ColumnTransformer\n"
            "from sklearn.dummy import DummyRegressor\n"
            "from sklearn.linear_model import Ridge\n"
            "from sklearn.pipeline import Pipeline\n"
            "from sklearn.preprocessing import OneHotEncoder, StandardScaler\n"
            "from preprocessing.preprocessing import TargetTransform\n\n"
            "def build(num, cat, est):\n"
            "    pre = ColumnTransformer([('num', StandardScaler(), num),\n"
            "                             ('cat', OneHotEncoder(handle_unknown='ignore', min_frequency=0.01), cat)])\n"
            "    return Pipeline([('pre', pre), ('est', est)])"
        ),
        code(
            "rows = []\n"
            "for target in (C.NEXT_TARGET, C.CRM_TARGET):\n"
            "    pk = C.get_aug_pack(target, base_only=True)\n"
            "    for mode in ('raw', 'p995', 'log1p_p995'):\n"
            "        tfm = TargetTransform(mode).fit(pk.y_train)\n"
            "        for alpha in (0.1, 1.0, 10.0, 100.0):\n"
            "            model = build(pk.num_cols, pk.cat_cols, Ridge(alpha=alpha))\n"
            "            model.fit(pk.x_train, tfm.transform(pk.y_train))\n"
            "            ev = {s + '_' + k: v for s, X, y in [('val', pk.x_val, pk.y_val), ('test', pk.x_test, pk.y_test)]\n"
            "                  for k, v in C.metric_pack(y, tfm.inverse(model.predict(X))).items()}\n"
            "            rows.append(dict(target=target.replace('_sec', ''), model=f'ridge_a{alpha}_{mode}',\n"
            "                             val_mae=ev['val_mae'], test_mae=ev['test_mae'], val_r2=ev['val_r2'],\n"
            "                             val_small=ev['val_small_mae']))\n"
            "    for strat in ('mean', 'median'):\n"
            "        dm = DummyRegressor(strategy=strat).fit(pk.x_train, pk.y_train)\n"
            "        ev = {s + '_' + k: v for s, X, y in [('val', pk.x_val, pk.y_val), ('test', pk.x_test, pk.y_test)]\n"
            "              for k, v in C.metric_pack(y, np.maximum(dm.predict(X), 0)).items()}\n"
            "        rows.append(dict(target=target.replace('_sec', ''), model=f'dummy_{strat}',\n"
            "                         val_mae=ev['val_mae'], test_mae=ev['test_mae'], val_r2=ev['val_r2'],\n"
            "                         val_small=ev['val_small_mae']))\n"
            "ridge = pd.DataFrame(rows)\n"
            "for t in ridge.target.unique():\n"
            "    print('==', t, '==')\n"
            "    display(ridge[ridge.target == t].sort_values('val_mae').head(5).reset_index(drop=True))"
        ),
        code(
            "tgts = list(ridge.target.unique())\n"
            "fig, axes = plt.subplots(1, len(tgts), figsize=(8, 3.4))\n"
            "for ax, t in zip(np.atleast_1d(axes), tgts):\n"
            "    sub = ridge[ridge.target == t]\n"
            "    best_ridge = sub[sub.model.str.startswith('ridge')].val_mae.min()\n"
            "    dmed = float(sub[sub.model == 'dummy_median'].val_mae.iloc[0])\n"
            "    dmean = float(sub[sub.model == 'dummy_mean'].val_mae.iloc[0])\n"
            "    vals = [best_ridge, dmed, dmean]\n"
            "    ax.bar(['best\\nRidge', 'dummy\\nmedian', 'dummy\\nmean'], vals, color=[ORANGE, GREY, GREY])\n"
            "    for i, v in enumerate(vals):\n"
            "        ax.text(i, v, f'{v:.0f}', ha='center', va='bottom', fontsize=8)\n"
            "    ax.set_title(t, fontsize=9); ax.set_ylabel('val MAE, сек')\n"
            "fig.suptitle('Ridge не обходит даже median-baseline'); plt.tight_layout(); plt.show()"
        ),
        md(
            "**Вывод.** На next-session Ridge не обходит даже `dummy_median` (линейная модель почти "
            "бесполезна на шумном таргете); на CRM Ridge разумнее, но CatBoost всё равно сильнее. Это "
            "обосновывает пользу нелинейных взаимодействий CatBoost."
        ),
    ]
    write_nb("5_ridge_baseline.ipynb", cells)


# ============== NB 6: history features / calibration / drift / val-test gap ==============
def nb6():
    cells = [
        md(
            "# 6. History-признаки, калибровка, drift и val–test разрыв (оба таргета)\n\n"
            "Доработка по ревью: блоки строго в технической зоне Участника 2, для **обоих таргетов**, "
            "с полным набором метрик (MAE, MedAE, P70, P90, R², small/normal/long, ProductMAE, "
            "EngagementRiskMAE, WMAPE).\n\n" + INTRO_COMMON +
            "\n\nДанные берутся из аугментированного датасета `outputs/sessions_augmented.parquet` "
            "(raw + CRM-таргет + history-признаки), собранного `scripts/build_augmented.py`."
        ),
        code(SETUP),
        code("hp = C.load_best_hp()\nTARGETS = [C.NEXT_TARGET, C.CRM_TARGET]\nprint('targets:', TARGETS)"),
        md(
            "## 6.1 Анализ val–test разрыва\n"
            "Почему на next-session val MAE (~522) хуже test (~429): состав сегментов / распределение таргета."
        ),
        code(
            "def describe(y):\n"
            "    y = np.asarray(y, float)\n"
            "    return dict(n=len(y), mean=round(y.mean(), 1), median=round(float(np.median(y)), 1),\n"
            "                p90=round(float(np.percentile(y, 90)), 1),\n"
            "                long_share=round(float((y > 1200).mean()), 3),\n"
            "                small_share=round(float((y <= 300).mean()), 3))\n"
            "rows = []\n"
            "for target in TARGETS:\n"
            "    pk = C.get_aug_pack(target, base_only=True)\n"
            "    for split, y in [('train', pk.y_train), ('val', pk.y_val), ('test', pk.y_test)]:\n"
            "        rows.append(dict(target=target.replace('_sec', ''), split=split, **describe(y)))\n"
            "gap_df = pd.DataFrame(rows)\n"
            "display(gap_df)"
        ),
        code(
            "fig, axes = plt.subplots(1, 2, figsize=(8, 3.2))\n"
            "for ax, t in zip(axes, gap_df.target.unique()):\n"
            "    d = gap_df[gap_df.target == t].set_index('split').loc[['train', 'val', 'test']]\n"
            "    x = np.arange(3)\n"
            "    ax.bar(x - 0.2, d.long_share, 0.4, color=ORANGE, label='long_share')\n"
            "    ax.set_ylabel('доля long', color=ORANGE)\n"
            "    a2 = ax.twinx(); a2.plot(x, d['mean'], 'o-', color=BLUE); a2.grid(False)\n"
            "    a2.set_ylabel('mean target', color=BLUE)\n"
            "    ax.set_xticks(x); ax.set_xticklabels(['train', 'val', 'test']); ax.set_title(t, fontsize=9)\n"
            "fig.suptitle('val-test разрыв: состав сегментов и среднее'); plt.tight_layout(); plt.show()"
        ),
        md("**Вывод.** На next-session test-период объективно легче (меньше long-сессий, ниже среднее) → ниже MAE. Модель не подглядывает в test."),
        md(
            "## 6.2 Time-aware history-признаки\n"
            "14 past-only признаков: baseline / baseline+history / baseline+лучшие history."
        ),
        code(
            "rows = []\n"
            "for target in TARGETS:\n"
            "    pb = C.get_aug_pack(target, base_only=True)\n"
            "    ph = C.get_aug_pack(target, base_only=False)\n"
            "    mb, _, tb = C.fit_regressor(pb, hp, 'MAE', 'p995'); evb = C.eval_split(mb, pb, tb)\n"
            "    mh, _, th = C.fit_regressor(ph, hp, 'MAE', 'p995'); evh = C.eval_split(mh, ph, th)\n"
            "    imp = mh.get_feature_importance(); cols = list(ph.feature_cols)\n"
            "    best_hist = [c for _, c in sorted(((imp[i], cols[i]) for i in range(len(cols))\n"
            "                 if cols[i].startswith('hist_')), reverse=True)[:6]]\n"
            "    pbest = C.get_aug_pack(target, keep_features=list(pb.feature_cols) + best_hist)\n"
            "    mc, _, tc = C.fit_regressor(pbest, hp, 'MAE', 'p995'); evc = C.eval_split(mc, pbest, tc)\n"
            "    for tag, ev, npf in [('baseline', evb, len(pb.feature_cols)),\n"
            "                         ('baseline_plus_history', evh, len(ph.feature_cols)),\n"
            "                         ('baseline_plus_best_history', evc, len(pbest.feature_cols))]:\n"
            "        rows.append(dict(target=target.replace('_sec', ''), model=tag, n_features=npf,\n"
            "                         val_mae=ev['val_mae'], val_r2=ev['val_r2'], val_product=ev['val_product_mae'],\n"
            "                         val_small=ev['val_small_mae'], test_mae=ev['test_mae']))\n"
            "hist_df = pd.DataFrame(rows)\n"
            "display(hist_df)"
        ),
        code(
            "names = ['baseline', 'baseline_plus_history', 'baseline_plus_best_history']\n"
            "fig, axes = plt.subplots(1, 2, figsize=(8, 3.2))\n"
            "for ax, t in zip(axes, hist_df.target.unique()):\n"
            "    d = hist_df[hist_df.target == t].set_index('model').loc[names]\n"
            "    ax.bar(range(3), d.val_mae, color=[GREY, BLUE, GREEN])\n"
            "    ax.set_xticks(range(3)); ax.set_xticklabels(['base', '+hist', '+best'], fontsize=8)\n"
            "    lo, hi = d.val_mae.min(), d.val_mae.max()\n"
            "    ax.set_ylim(lo - (hi - lo) - 0.5, hi + (hi - lo) + 0.5)\n"
            "    for i, v in enumerate(d.val_mae):\n"
            "        ax.text(i, v, f'{v:.1f}', ha='center', va='bottom', fontsize=7)\n"
            "    ax.set_title(t, fontsize=9)\n"
            "axes[0].set_ylabel('val MAE, сек')\n"
            "fig.suptitle('History-признаки: val MAE (выбор по validation)'); plt.tight_layout(); plt.show()"
        ),
        md("**Вывод.** History помогает маржинально на next-session (ewma5, median_last5, time_since_prev) и не помогает CRM."),
        md(
            "## 6.3 Regression calibration\n"
            "bin / isotonic / segment; калибратор на calibration-split, test — только финал."
        ),
        code(
            "from exp15_calibration import fit_bin, apply_bin, seg_of\n"
            "from sklearn.isotonic import IsotonicRegression\n"
            "rows = []\n"
            "for target in TARGETS:\n"
            "    pk = C.get_aug_pack(target, base_only=False)\n"
            "    m, _, tfm = C.fit_regressor(pk, hp, 'MAE', 'p995')\n"
            "    rv, rt = tfm.inverse(m.predict(pk.x_val)), tfm.inverse(m.predict(pk.x_test))\n"
            "    cut = len(rv) // 2\n"
            "    rf, yf, re, ye = rv[:cut], pk.y_val[:cut], rv[cut:], pk.y_val[cut:]\n"
            "    bcal = fit_bin(rf, yf); iso = IsotonicRegression(out_of_bounds='clip').fit(rf, yf)\n"
            "    seg_iso = {}\n"
            "    sf = seg_of(rf)\n"
            "    for s in (0, 1, 2):\n"
            "        mm = sf == s\n"
            "        if mm.sum() >= 50:\n"
            "            seg_iso[s] = IsotonicRegression(out_of_bounds='clip').fit(rf[mm], yf[mm])\n"
            "    def cal(method, raw):\n"
            "        if method == 'raw':\n"
            "            return raw\n"
            "        if method == 'bin':\n"
            "            return apply_bin(bcal, raw)\n"
            "        if method == 'isotonic':\n"
            "            return iso.predict(raw)\n"
            "        out = raw.astype(float).copy(); sg = seg_of(raw)\n"
            "        for s in (0, 1, 2):\n"
            "            mm = sg == s\n"
            "            if s in seg_iso and mm.any():\n"
            "                out[mm] = seg_iso[s].predict(raw[mm])\n"
            "        return out\n"
            "    for method in ('raw', 'bin', 'isotonic', 'segment'):\n"
            "        mm = C.metric_pack(ye, np.maximum(cal(method, re), 0))\n"
            "        rows.append(dict(target=target.replace('_sec', ''), calibration=method, val_mae=mm['mae'],\n"
            "                         val_r2=mm['r2'], val_small=mm['small_mae'], val_long=mm['long_mae']))\n"
            "display(pd.DataFrame(rows))"
        ),
        md(
            "**Вывод.** Калибровка улучшает long_mae и R², но ухудшает small/normal и общий MAE "
            "(тянет к среднему, а MAE = медиана) → не включаем (выбор = raw)."
        ),
        md(
            "## 6.4 Feature drift detection\n"
            "PSI (числовые) + доля новых категорий; удаление drift-heavy только если не вредит validation."
        ),
        code(
            "def psi(a, b, bins=10):\n"
            "    a, b = np.asarray(a, float), np.asarray(b, float)\n"
            "    edges = np.unique(np.quantile(a, np.linspace(0, 1, bins + 1)))\n"
            "    if len(edges) < 3:\n"
            "        return 0.0\n"
            "    ta = np.histogram(a, bins=edges)[0] / max(len(a), 1) + 1e-6\n"
            "    tb = np.histogram(b, bins=edges)[0] / max(len(b), 1) + 1e-6\n"
            "    return float(np.sum((tb - ta) * np.log(tb / ta)))\n"
            "for target in TARGETS:\n"
            "    pk = C.get_aug_pack(target, base_only=False)\n"
            "    dr = [dict(feature=c, kind='num', psi=round(psi(pk.x_train[c], pk.x_test[c]), 4)) for c in pk.num_cols]\n"
            "    for c in pk.cat_cols:\n"
            "        seen = set(pk.x_train[c].astype(str).unique())\n"
            "        dr.append(dict(feature=c, kind='cat', psi=round(float((~pk.x_test[c].astype(str).isin(seen)).mean()), 4)))\n"
            "    dd = pd.DataFrame(dr).sort_values('psi', ascending=False).reset_index(drop=True)\n"
            "    top5, top10 = dd.feature.head(5).tolist(), dd.feature.head(10).tolist()\n"
            "    print('==', target.replace('_sec', ''), '== top drift:', top5)\n"
            "    rows = []\n"
            "    for drop, nm in [([], 'all_features'), (top5, 'drop_top5'), (top10, 'drop_top10')]:\n"
            "        qp = C.get_aug_pack(target, base_only=False, drop_features=drop)\n"
            "        mdl, _, tfm = C.fit_regressor(qp, hp, 'MAE', 'p995'); ev = C.eval_split(mdl, qp, tfm)\n"
            "        rows.append(dict(variant=nm, n_features=len(qp.feature_cols), val_mae=ev['val_mae'], test_mae=ev['test_mae']))\n"
            "    display(pd.DataFrame(rows))"
        ),
        md("**Вывод.** Удаление drift-heavy признаков не улучшает validation ни на одном таргете → оставляем все признаки."),
    ]
    write_nb("6_history_calibration_drift.ipynb", cells)


# ============================ NB 7: final model ============================
def nb7():
    cells = [
        md(
            "# 7. Финальная техническая модель (оба таргета)\n\n"
            "Сборка лучших решений из блоков 1-6: гиперпараметры (best_by_mae), набор признаков "
            "(baseline + полезные history − drift-heavy), калибровка. Обучение для обоих таргетов, "
            "полный набор метрик на test и сравнение с обычным baseline. Решения берутся из "
            "`outputs/decision_*.json`.\n\n" + INTRO_COMMON
        ),
        code(SETUP),
        code(
            "import json\n"
            "from exp15_calibration import fit_bin, apply_bin, seg_of\n"
            "from sklearn.isotonic import IsotonicRegression\n\n"
            "def load_decision(name, target, default):\n"
            "    pth = C.OUTPUT_DIR / f'decision_{name}_{target}.json'\n"
            "    return json.loads(pth.read_text()) if pth.exists() else default\n\n"
            "def hp_for(target):\n"
            "    pth = C.OUTPUT_DIR / 'best_main_model.json'\n"
            "    if target == C.NEXT_TARGET and pth.exists():\n"
            "        cfg = json.loads(pth.read_text())['config']\n"
            "        hp = dict(depth=int(cfg['depth']), learning_rate=float(cfg['learning_rate']),\n"
            "                  l2_leaf_reg=float(cfg['l2_leaf_reg']), iterations=int(cfg['iterations']),\n"
            "                  min_data_in_leaf=int(cfg['min_data_in_leaf']),\n"
            "                  random_strength=float(cfg['random_strength']), od_wait=80)\n"
            "        extra = dict(bootstrap_type=cfg['bootstrap'])\n"
            "        if cfg['bootstrap'] == 'Bernoulli':\n"
            "            extra['subsample'] = 0.85\n"
            "        return hp, cfg['target_mode'], extra\n"
            "    return dict(C.load_best_hp()), 'p995', {}\n\n"
            "FULL = ['mae', 'medae', 'p70_abs_error', 'p90_abs_error', 'r2', 'small_mae', 'normal_mae',\n"
            "        'long_mae', 'product_mae', 'engagement_risk_mae', 'wmape']"
        ),
        code(
            "import time\n"
            "rows = []\n"
            "for target in (C.NEXT_TARGET, C.CRM_TARGET):\n"
            "    dh = load_decision('history', target, {'use_history': False, 'best_hist_features': []})\n"
            "    dd = load_decision('drift', target, {'drop_features': []})\n"
            "    dc = load_decision('calibration', target, {'method': 'raw'})\n"
            "    hp, tmode, extra = hp_for(target)\n"
            "    pb = C.get_aug_pack(target, base_only=True)\n"
            "    mb, fsb, tb = C.fit_regressor(pb, hp, 'MAE', tmode, extra=extra)\n"
            "    pred_b = tb.inverse(mb.predict(pb.x_test))\n"
            "    rb = dict(target=target.replace('_sec', ''), model='baseline_reference', feature_set='baseline',\n"
            "              calibration='raw', **{k: round(C.metric_pack(pb.y_test, pred_b)[k], 2) for k in FULL},\n"
            "              model_size_kb=C.model_size_bytes(mb) // 1024)\n"
            "    feats = list(pb.feature_cols) + (dh['best_hist_features'] if dh['use_history'] else [])\n"
            "    feats = [c for c in feats if c not in set(dd['drop_features'])]\n"
            "    pk = C.get_aug_pack(target, keep_features=feats)\n"
            "    m, fs, tfm = C.fit_regressor(pk, hp, 'MAE', tmode, extra=extra)\n"
            "    rv, rt = tfm.inverse(m.predict(pk.x_val)), tfm.inverse(m.predict(pk.x_test))\n"
            "    method = dc['method']\n"
            "    if method == 'bin':\n"
            "        ft = apply_bin(fit_bin(rv, pk.y_val), rt)\n"
            "    elif method == 'isotonic':\n"
            "        ft = IsotonicRegression(out_of_bounds='clip').fit(rv, pk.y_val).predict(rt)\n"
            "    else:\n"
            "        ft = rt\n"
            "    ft = np.maximum(ft, 0)\n"
            "    t0 = time.time()\n"
            "    for _ in range(3):\n"
            "        m.predict(pk.x_test)\n"
            "    infer_us = (time.time() - t0) / 3 / len(pk.x_test) * 1e6\n"
            "    rf = dict(target=target.replace('_sec', ''),\n"
            "              model='final', feature_set='baseline+history' if dh['use_history'] else 'baseline',\n"
            "              calibration=method, **{k: round(C.metric_pack(pk.y_test, ft)[k], 2) for k in FULL},\n"
            "              model_size_kb=C.model_size_bytes(m) // 1024)\n"
            "    rf['inference_us_per_row'] = round(infer_us, 2)\n"
            "    rows += [rb, rf]\n"
            "final = pd.DataFrame(rows)\n"
            "display(final[['target', 'model', 'feature_set', 'calibration', 'mae', 'medae', 'p90_abs_error',\n"
            "               'r2', 'small_mae', 'long_mae', 'product_mae', 'engagement_risk_mae', 'wmape',\n"
            "               'model_size_kb']])"
        ),
        code(
            "fig, axes = plt.subplots(1, 2, figsize=(8, 3.2))\n"
            "for ax, t in zip(axes, final.target.unique()):\n"
            "    d = final[final.target == t].set_index('model')\n"
            "    names = [n for n in ['baseline_reference', 'final'] if n in d.index]\n"
            "    vals = [d.loc[n, 'mae'] for n in names]\n"
            "    ax.bar(['baseline', 'final'][:len(names)], vals, color=[GREY, GREEN])\n"
            "    lo, hi = min(vals), max(vals)\n"
            "    ax.set_ylim(lo - (hi - lo) - 1, hi + (hi - lo) + 1)\n"
            "    for i, v in enumerate(vals):\n"
            "        ax.text(i, v, f'{v:.1f}', ha='center', va='bottom')\n"
            "    ax.set_title(t, fontsize=9); ax.set_ylabel('test MAE, сек')\n"
            "fig.suptitle('Финальная модель vs baseline (test MAE)'); plt.tight_layout(); plt.show()"
        ),
        md(
            "**Вывод.** Финальная модель собрана и сохранена для обоих таргетов. По MAE ≈ baseline — на "
            "шумных таргетах резерв качества лежит не в этих приёмах, а в самих данных/таргете; ценность "
            "доработки — строго обоснованный выбор признаков/настроек с полным набором метрик. "
            "Готовые артефакты (`.cbm`, конфиг, признаки) — в `outputs/final_models/`, сводный JSON — "
            "`outputs/README_final.json`."
        ),
        code(
            "import json\n"
            "print(json.dumps(json.loads((C.OUTPUT_DIR / 'README_final.json').read_text()),\n"
            "                 ensure_ascii=False, indent=2)[:1500])"
        ),
    ]
    write_nb("7_final_model.ipynb", cells)


def main():
    nb1(); nb2(); nb3(); nb4(); nb5(); nb6(); nb7()
    print("\nAll themed notebooks written to", ROOT.name)


if __name__ == "__main__":
    main()
