# Участник 2 — Boosting, признаки и архитектура CatBoost

Все эксперименты Участника 2 из обоих ТЗ: `distribution_boosting_extra_tuning.txt` (доп. тюнинг CatBoost, разделы 1-6) и `distribution_of_responsoblities.txt` (LightGBM, основной CatBoost sweep, feature ablation, каскадная модель, коррекция остатков, Ridge). Без API. Общий пайплайн, тайм-сплит и метрики взяты из `preprocessing/preprocessing.py` и `team_modeling_protocol.txt`.

## Протокол и проверки

- Таргет: `target_next_session_length_sec` (next-session). CRM-таргет — зона Участника 1.
- Выборка: последние **30000** строк по времени; split 70/15/15 = 21000/4500/4500.
- Признаков: 73 (59 числовых, 14 категориальных).
- Anti-leak: **PASS**, chronology: **PASS**, no-NaN: **PASS**.
- Train: 2026-04-27 21:34:36 → 2026-04-29 23:13:26
- Test:  2026-04-30 11:32:42 → 2026-04-30 22:57:22

## Базовая конфигурация (exp0)

Лучшая по validation MAE: `{'depth': 6, 'learning_rate': 0.05, 'l2_leaf_reg': 3.0, 'iterations': 800, 'od_wait': 80}`, target_mode=`p995`, val_mae=522.53, test_mae=429.26. Все эксперименты ниже варьируют по одной оси вокруг неё.

## 1. SHAP feature selection vs ручной top-k

| model_name   | selection      |   n_features |   val_mae |   test_mae |   fit_sec |   model_size_kb |
|:-------------|:---------------|-------------:|----------:|-----------:|----------:|----------------:|
| shap_k40     | shap_recursive |           40 |    521.58 |     428.84 |      1.63 |             199 |
| shap_k60     | shap_recursive |           60 |    521.85 |     429.21 |      4.03 |             763 |
| full         | none           |           73 |    522.53 |     429.26 |      3.94 |             712 |
| topk_40      | manual_topk    |           40 |    523.27 |     429.54 |      2.1  |             265 |
| topk_60      | manual_topk    |           60 |    523.32 |     429.85 |      2.83 |             475 |

## 2. Bootstrap: MVS vs Bernoulli / Bayesian

| model_name      |   val_mae |   val_mae_std |   test_mae |   val_long_mae |   fit_sec |
|:----------------|----------:|--------------:|-----------:|---------------:|----------:|
| mvs_ss0.9       |    523.18 |          0.36 |     428.97 |        1967    |      3.3  |
| bernoulli_ss0.8 |    523.19 |          0.74 |     428.65 |        1955.29 |      3.83 |
| mvs_ss0.7       |    523.26 |          0.26 |     429.31 |        1962.04 |      3.23 |
| bayesian        |    523.41 |          0.18 |     428.89 |        1963.19 |      3.62 |
| mvs_ss0.8       |    523.88 |          1.15 |     429.36 |        1981.01 |      2.6  |

## 3. CTR-тюнинг категориальных признаков

| model_name        |   val_mae |   test_mae |   fit_sec |   model_size_kb |
|:------------------|----------:|-----------:|----------:|----------------:|
| ohms10_mcc1_ctbc1 |    522.71 |     429.01 |      1.52 |             339 |
| ohms10_mcc2_ctbc5 |    522.86 |     429.14 |      4.92 |             733 |
| ohms10_mcc2_ctbc3 |    523.03 |     429.07 |      3.09 |             531 |
| ohms5_mcc1_ctbc3  |    523.16 |     428.34 |      2.08 |             263 |
| ohms10_mcc2_ctbc1 |    523.25 |     428.66 |      1.71 |             342 |
| ohms2_mcc1_ctbc1  |    523.27 |     428.55 |      1.41 |             193 |
| ohms2_mcc2_ctbc5  |    523.38 |     428.12 |      7.96 |             662 |
| ohms5_mcc2_ctbc5  |    523.43 |     429.03 |      5.15 |             574 |
| ohms2_mcc1_ctbc3  |    523.45 |     428.43 |      2.36 |             243 |
| ohms10_mcc1_ctbc5 |    523.53 |     429.1  |      1.87 |             276 |
| ohms5_mcc1_ctbc5  |    523.54 |     430.17 |      3.54 |             356 |
| ohms5_mcc2_ctbc1  |    523.61 |     429.48 |      2.92 |             665 |
| ohms5_mcc1_ctbc1  |    523.74 |     428.59 |      1.07 |             163 |
| ohms10_mcc1_ctbc3 |    523.75 |     430.02 |      1.2  |             180 |
| ohms2_mcc1_ctbc5  |    523.76 |     428.83 |      3.48 |             257 |
| ohms2_mcc2_ctbc1  |    523.83 |     428.73 |      1.96 |             290 |
| ohms5_mcc2_ctbc3  |    524.11 |     428.54 |      3.39 |             518 |
| ohms2_mcc2_ctbc3  |    524.36 |     428.93 |      5.36 |             627 |

## 4. Структура деревьев: Symmetric / Depthwise / Lossguide

| model_name            |   val_mae |   test_mae |   val_small_mae |   val_normal_mae |   val_long_mae |   fit_sec |   model_size_kb |
|:----------------------|----------:|-----------:|----------------:|-----------------:|---------------:|----------:|----------------:|
| depthwise_mdl20       |    522.3  |     428.85 |          215.1  |           316.1  |        1943.07 |      2.02 |             359 |
| depthwise_mdl50       |    522.49 |     428.5  |          205.05 |           320.6  |        1966.35 |      1.61 |             259 |
| lossguide_ml63_mdl20  |    522.52 |     428.95 |          207.02 |           320.13 |        1961.32 |      3.07 |             401 |
| lossguide_ml127_mdl20 |    522.52 |     428.95 |          207.02 |           320.13 |        1961.32 |      2.9  |             401 |
| symmetric             |    522.53 |     429.26 |          207.51 |           322.12 |        1955.53 |      4.02 |             712 |
| lossguide_ml31_mdl50  |    523.39 |     428.37 |          205.69 |           319.99 |        1971.37 |      1.65 |             173 |
| lossguide_ml63_mdl50  |    523.62 |     428.29 |          214.66 |           314.05 |        1957.29 |      1.96 |             221 |
| lossguide_ml127_mdl50 |    523.62 |     428.29 |          214.66 |           314.05 |        1957.29 |      1.84 |             221 |
| depthwise_mdl100      |    523.84 |     428.73 |          204.81 |           316.46 |        1984.64 |      1.2  |             157 |
| lossguide_ml31_mdl20  |    523.84 |     427.7  |          205.57 |           322.18 |        1969.86 |      2.07 |             242 |

## 5. Квантизация числовых признаков

| model_name                |   val_mae |   test_mae |   val_long_mae |   fit_sec |   model_size_kb |
|:--------------------------|----------:|-----------:|---------------:|----------:|----------------:|
| bc254_GreedyLogSum        |    522.53 |     429.26 |        1955.53 |      3.7  |             712 |
| bc254_Median              |    522.72 |     428.53 |        1958.82 |      3.42 |             713 |
| bc128_Median              |    523.02 |     428.99 |        1960.21 |      3.45 |             651 |
| bc254_UniformAndQuantiles |    523.03 |     428.07 |        1964.6  |      2.38 |             493 |
| bc64_GreedyLogSum         |    523.06 |     428.61 |        1963.94 |      5.01 |             825 |
| bc128_GreedyLogSum        |    523.11 |     429.81 |        1959.05 |      4.51 |             772 |
| bc64_Median               |    523.37 |     428.23 |        1966.51 |      3.68 |             670 |
| bc64_UniformAndQuantiles  |    523.47 |     427.98 |        1964.4  |      3.95 |             756 |
| bc512_GreedyLogSum        |    523.76 |     430.64 |        1951.26 |      2.73 |             461 |
| per_float_top3_1024       |    523.81 |     429.05 |        1966.55 |      2.9  |             509 |
| bc128_UniformAndQuantiles |    524.43 |     428.76 |        1971.17 |      2.28 |             313 |
| bc512_UniformAndQuantiles |    524.5  |     429.48 |        1988.63 |      1.92 |             155 |
| bc512_Median              |    524.86 |     430.65 |        1945.29 |      4.13 |             744 |

## 6. (запас) rsm — random subspace

| model_name   |   val_mae |   test_mae |   fit_sec |
|:-------------|----------:|-----------:|----------:|
| rsm1.0       |    522.53 |     429.26 |      3.88 |
| rsm0.6       |    522.93 |     428.48 |      3.05 |
| rsm0.8       |    523.73 |     428.4  |      2.07 |

## 7. Протокольные стратегии на лучшем конфиге (оба таргета)

| target                           | model_name    | loss_function       |   val_mae |   val_product_mae |   val_engagement_risk_mae |   val_small_mae |   test_mae |
|:---------------------------------|:--------------|:--------------------|----------:|------------------:|--------------------------:|----------------:|-----------:|
| target_next_session_length_sec   | capped_target | MAE                 |    522.53 |            251.65 |                    261.87 |          207.51 |     429.26 |
| target_next_session_length_sec   | quantile_040  | Quantile:alpha=0.40 |    529.23 |            233.57 |                    242.72 |          158.7  |     431.11 |
| target_next_session_length_sec   | quantile_035  | Quantile:alpha=0.35 |    539.71 |            221    |                    230.36 |          117.66 |     435.98 |
| future_sessions_mean_playtime_7d | capped_target | MAE                 |    240.01 |            187.5  |                    190.79 |          216.94 |     272.19 |
| future_sessions_mean_playtime_7d | quantile_040  | Quantile:alpha=0.40 |    238.13 |            173.76 |                    177.74 |          176.43 |     271.74 |
| future_sessions_mean_playtime_7d | quantile_035  | Quantile:alpha=0.35 |    244.46 |            172.26 |                    176.04 |          157.58 |     279.77 |

## 8. Основной CatBoost sweep по MAE (best_by_mae)

| model_name   |   depth |   learning_rate |   l2_leaf_reg |   iterations | bootstrap   | target_mode   | clip_mode   |   val_mae |   test_mae |   fit_sec |
|:-------------|--------:|----------------:|--------------:|-------------:|:------------|:--------------|:------------|----------:|-----------:|----------:|
| cfg39        |       7 |            0.03 |             5 |         1200 | Bernoulli   | log1p_p995    | none        |    521.8  |     428.42 |      6.95 |
| cfg06        |       8 |            0.03 |            10 |         1200 | Bayesian    | log1p_p995    | none        |    522.13 |     427.23 |      8.13 |
| cfg25        |       6 |            0.03 |             3 |         1500 | Bernoulli   | log1p_p995    | p005_p995   |    522.47 |     428.19 |      7.78 |
| cfg24        |       5 |            0.02 |             3 |         1500 | Bernoulli   | log1p_p995    | p005_p995   |    522.57 |     427.81 |      9.02 |
| cfg21        |       5 |            0.05 |            10 |         1200 | Bayesian    | log1p_p995    | none        |    522.59 |     428.26 |      4.28 |
| cfg33        |       5 |            0.02 |             5 |         1500 | Bernoulli   | log1p_p995    | none        |    522.71 |     428.08 |      9.48 |
| cfg16        |       8 |            0.02 |             5 |         1200 | Bernoulli   | log1p_p995    | p005_p995   |    522.8  |     427.74 |     16.22 |
| cfg01        |       8 |            0.02 |             3 |         1800 | Bernoulli   | p995          | none        |    522.8  |     428.81 |      7.42 |
| cfg30        |       7 |            0.03 |             5 |         1200 | Bayesian    | p995          | none        |    522.85 |     428.62 |      5.32 |
| cfg32        |       7 |            0.02 |            10 |         1500 | Bayesian    | p995          | none        |    522.91 |     429.11 |      6.57 |
| cfg20        |       8 |            0.02 |             5 |         1800 | Bayesian    | log1p_p995    | none        |    522.94 |     428.61 |     14.46 |
| cfg29        |       5 |            0.05 |             3 |         1800 | Bayesian    | p995          | p005_p995   |    522.96 |     428.46 |      3.89 |
| cfg02        |       8 |            0.02 |            10 |         1800 | Bernoulli   | log1p_p995    | p005_p995   |    522.97 |     427.93 |     11.93 |
| cfg36        |       5 |            0.05 |             5 |         1200 | Bernoulli   | log1p_p995    | none        |    522.97 |     427.4  |      4.36 |
| cfg08        |       6 |            0.05 |             7 |         1500 | Bernoulli   | log1p_p995    | p005_p995   |    522.99 |     427.88 |      7.32 |
| cfg26        |       7 |            0.03 |            10 |         1500 | Bayesian    | p995          | none        |    523.01 |     428.99 |      4.75 |
| cfg13        |       7 |            0.03 |             5 |         1200 | Bayesian    | p995          | none        |    523.04 |     429.05 |      4.03 |
| cfg35        |       5 |            0.03 |             5 |         1200 | Bayesian    | log1p_p995    | none        |    523.12 |     428.31 |      6.53 |
| cfg09        |       5 |            0.02 |             7 |         1500 | Bayesian    | log1p_p995    | none        |    523.12 |     427.79 |      8.98 |
| cfg04        |       7 |            0.02 |            10 |         1500 | Bernoulli   | log1p_p995    | p005_p995   |    523.19 |     428    |     14.42 |
| cfg05        |       6 |            0.05 |             3 |         1800 | Bayesian    | p995          | none        |    523.21 |     428.26 |      3.76 |
| cfg37        |       7 |            0.03 |             3 |         1200 | Bernoulli   | p995          | p005_p995   |    523.22 |     428.97 |      4.72 |
| cfg18        |       6 |            0.02 |             7 |         1200 | Bernoulli   | p995          | none        |    523.23 |     428.95 |      6.23 |
| cfg11        |       8 |            0.02 |             3 |         1800 | Bernoulli   | log1p_p995    | none        |    523.26 |     427.17 |     11.62 |
| cfg34        |       7 |            0.05 |            10 |         1800 | Bernoulli   | log1p_p995    | p005_p995   |    523.39 |     427.68 |      5.03 |
| cfg03        |       6 |            0.02 |             7 |         1500 | Bayesian    | p995          | p005_p995   |    523.52 |     429.06 |      5.24 |
| cfg28        |       5 |            0.02 |             3 |         1800 | Bayesian    | p995          | none        |    523.65 |     428.54 |      6.2  |
| cfg23        |       8 |            0.02 |            10 |         1800 | Bayesian    | p995          | none        |    523.69 |     429.04 |      6.7  |
| cfg22        |       5 |            0.02 |            10 |         1800 | Bernoulli   | p995          | none        |    523.81 |     428.66 |      5.56 |
| cfg00        |       5 |            0.02 |             7 |         1800 | Bernoulli   | p995          | none        |    523.82 |     428.79 |      5.22 |
| cfg15        |       5 |            0.02 |             7 |         1500 | Bernoulli   | p995          | none        |    523.82 |     428.79 |      5.08 |
| cfg27        |       5 |            0.05 |             7 |         1800 | Bayesian    | p995          | none        |    523.83 |     429.15 |      2.51 |
| cfg38        |       8 |            0.05 |             5 |         1800 | Bernoulli   | log1p_p995    | none        |    523.84 |     427.62 |      4.44 |
| cfg07        |       7 |            0.05 |             3 |         1800 | Bernoulli   | log1p_p995    | p005_p995   |    523.88 |     428.78 |      6.59 |
| cfg17        |       6 |            0.05 |             7 |         1200 | Bayesian    | log1p_p995    | none        |    523.98 |     428.76 |      4.29 |
| cfg14        |       7 |            0.05 |             5 |         1800 | Bernoulli   | p995          | p005_p995   |    524.07 |     429.38 |      3.32 |
| cfg31        |       5 |            0.02 |             5 |         1500 | Bayesian    | p995          | none        |    524.07 |     428.95 |      2.89 |
| cfg10        |       6 |            0.02 |             7 |         1500 | Bayesian    | p995          | none        |    524.14 |     428.69 |      5.32 |
| cfg12        |       8 |            0.05 |            10 |         1500 | Bernoulli   | log1p_p995    | p005_p995   |    524.26 |     427.55 |      7.94 |
| cfg19        |       5 |            0.02 |             3 |         1800 | Bernoulli   | p995          | p005_p995   |    528.01 |     431.42 |      1.7  |

## 9. Feature ablation групп признаков

| model_name             |   n_features |   val_mae |   test_mae |   fit_sec |   model_size_kb |
|:-----------------------|-------------:|----------:|-----------:|----------:|----------------:|
| session_install_events |           73 |    522.53 |     429.26 |      3.92 |             712 |
| session_install        |           58 |    522.89 |     429.07 |      3.71 |             556 |
| session_only           |           37 |    522.95 |     428.37 |      0.5  |             172 |
| top_k_73               |           73 |    523.02 |     428.99 |      2.89 |             506 |
| top_k_40               |           40 |    523.27 |     429.54 |      2.08 |             265 |
| top_k_60               |           60 |    523.32 |     429.85 |      2.78 |             475 |

## 10. Каскад «классификация → регрессия» (hard/soft/hybrid)

| model_name       |   val_mae |   val_product_mae |   val_engagement_risk_mae |   val_small_mae |   val_normal_mae |   val_long_mae |   test_mae |
|:-----------------|----------:|------------------:|--------------------------:|----------------:|-----------------:|---------------:|-----------:|
| baseline_general |    522.53 |            251.65 |                    261.87 |          207.51 |           322.12 |        1955.53 |     429.26 |
| cascade_hybrid   |    527.05 |            254.44 |                    264.03 |          202.95 |           338.79 |        1962.5  |     430.16 |
| cascade_soft     |    563.41 |            349.27 |                    376.27 |          397.17 |           285.36 |        1693.31 |     470.13 |
| cascade_hard     |    609.39 |            282.45 |                    346.91 |          229.27 |           548.41 |        1946.26 |     474.35 |

## 11. Коррекция остатков базовой модели

| model_name           |   val_mae |   test_mae |   val_small_mae |   val_normal_mae |   val_long_mae |   val_product_mae |
|:---------------------|----------:|-----------:|----------------:|-----------------:|---------------:|------------------:|
| base_only            |    522.53 |     429.26 |          207.51 |           322.12 |        1955.53 |            251.65 |
| base_plus_correction |    528.59 |     433.1  |          186.72 |           355.66 |        1987.05 |            252.13 |

## 12. Ridge baseline vs Dummy (оба таргета)

| target                           | model_name              |   val_mae |   test_mae |   val_r2 |   val_small_mae |
|:---------------------------------|:------------------------|----------:|-----------:|---------:|----------------:|
| target_next_session_length_sec   | ridge_a0.1_raw          |    618.59 |     526.72 |     0.04 |          496    |
| target_next_session_length_sec   | ridge_a1.0_raw          |    618.49 |     526.7  |     0.04 |          495.94 |
| target_next_session_length_sec   | ridge_a10.0_raw         |    618.13 |     526.56 |     0.04 |          495.8  |
| target_next_session_length_sec   | ridge_a100.0_raw        |    617.26 |     525.68 |     0.04 |          496    |
| target_next_session_length_sec   | ridge_a0.1_p995         |    581.2  |     496.48 |     0.08 |          425.21 |
| target_next_session_length_sec   | ridge_a1.0_p995         |    581.17 |     496.47 |     0.08 |          425.2  |
| target_next_session_length_sec   | ridge_a10.0_p995        |    581.08 |     496.4  |     0.08 |          425.23 |
| target_next_session_length_sec   | ridge_a100.0_p995       |    580.77 |     496.12 |     0.08 |          425.9  |
| target_next_session_length_sec   | ridge_a0.1_log1p_p995   |    586.33 |     437.46 |    -5.41 |          134.84 |
| target_next_session_length_sec   | ridge_a1.0_log1p_p995   |    586.3  |     437.46 |    -5.41 |          134.82 |
| target_next_session_length_sec   | ridge_a10.0_log1p_p995  |    586.1  |     437.44 |    -5.4  |          134.73 |
| target_next_session_length_sec   | ridge_a100.0_log1p_p995 |    584.79 |     437.34 |    -5.38 |          134.03 |
| target_next_session_length_sec   | dummy_mean              |    644.61 |     588.22 |    -0    |          571.18 |
| target_next_session_length_sec   | dummy_median            |    555.43 |     457.69 |    -0.11 |          196.6  |
| future_sessions_mean_playtime_7d | ridge_a0.1_raw          |    310.1  |     334.78 |     0.21 |          352.49 |
| future_sessions_mean_playtime_7d | ridge_a1.0_raw          |    310.08 |     334.75 |     0.21 |          352.47 |
| future_sessions_mean_playtime_7d | ridge_a10.0_raw         |    309.91 |     334.54 |     0.21 |          352.38 |
| future_sessions_mean_playtime_7d | ridge_a100.0_raw        |    309.29 |     333.79 |     0.21 |          352.77 |
| future_sessions_mean_playtime_7d | ridge_a0.1_p995         |    308.26 |     332.61 |     0.21 |          349.57 |
| future_sessions_mean_playtime_7d | ridge_a1.0_p995         |    308.24 |     332.58 |     0.21 |          349.56 |
| future_sessions_mean_playtime_7d | ridge_a10.0_p995        |    308.1  |     332.39 |     0.22 |          349.47 |
| future_sessions_mean_playtime_7d | ridge_a100.0_p995       |    307.62 |     331.76 |     0.22 |          349.82 |
| future_sessions_mean_playtime_7d | ridge_a0.1_log1p_p995   |    345.63 |     371.7  |    -0.71 |          144.97 |
| future_sessions_mean_playtime_7d | ridge_a1.0_log1p_p995   |    345.38 |     371.54 |    -0.7  |          144.93 |
| future_sessions_mean_playtime_7d | ridge_a10.0_log1p_p995  |    343.13 |     370.22 |    -0.61 |          144.63 |
| future_sessions_mean_playtime_7d | ridge_a100.0_log1p_p995 |    332.41 |     364.76 |    -0.27 |          143.42 |
| future_sessions_mean_playtime_7d | dummy_mean              |    365.81 |     366.1  |    -0.01 |          461.66 |
| future_sessions_mean_playtime_7d | dummy_median            |    344.33 |     354.91 |    -0.03 |          334.5  |

## 13. LightGBM как альтернативный бустинг

| model_name   | objective     |   n_estimators |   learning_rate |   num_leaves |   val_mae |   test_mae |   fit_sec |
|:-------------|:--------------|---------------:|----------------:|-------------:|----------:|-----------:|----------:|
| lgb12        | regression_l1 |           1400 |            0.02 |           63 |    524.69 |     431.25 |      2    |
| lgb09        | regression_l1 |           1000 |            0.03 |           31 |    525.16 |     430.06 |      1.61 |
| lgb13        | regression_l1 |           1400 |            0.03 |           31 |    525.29 |     430.26 |      1.4  |
| lgb03        | regression_l1 |           1000 |            0.02 |           31 |    525.48 |     430.52 |      1.98 |
| lgb00        | regression_l1 |           1400 |            0.03 |           31 |    525.68 |     431.27 |      1.54 |
| lgb10        | quantile      |            600 |            0.03 |           31 |    525.69 |     429.78 |      1.7  |
| lgb05        | regression_l1 |            600 |            0.05 |           31 |    525.81 |     430.43 |      0.89 |
| lgb08        | quantile      |           1400 |            0.05 |           31 |    525.86 |     431.09 |      1.04 |
| lgb02        | regression_l1 |           1400 |            0.03 |           31 |    525.98 |     429.92 |      1.86 |
| lgb06        | quantile      |           1000 |            0.05 |           63 |    526.69 |     431.19 |      1.91 |
| lgb11        | quantile      |            600 |            0.05 |           63 |    534.24 |     432.68 |      1.86 |
| lgb01        | quantile      |            600 |            0.02 |           31 |    534.61 |     433.2  |      2.34 |
| lgb04        | quantile      |           1400 |            0.03 |           31 |    535.03 |     433.87 |      1.42 |
| lgb07        | quantile      |           1400 |            0.05 |           31 |    535.04 |     432.99 |      1.18 |

## 14. Анализ val-test разрыва

| target                           | split   |     n |   mean |   median |     p90 |   long_share |   small_share |
|:---------------------------------|:--------|------:|-------:|---------:|--------:|-------------:|--------------:|
| target_next_session_length_sec   | train   | 21000 | 676.58 |   302    | 1633    |         0.16 |          0.5  |
| target_next_session_length_sec   | val     |  4500 | 660.82 |   300.5  | 1705.1  |         0.16 |          0.5  |
| target_next_session_length_sec   | test    |  4500 | 537.26 |   234.5  | 1364.2  |         0.12 |          0.56 |
| future_sessions_mean_playtime_7d | train   | 21000 | 596.18 |   469.03 | 1224.72 |         0.11 |          0.31 |
| future_sessions_mean_playtime_7d | val     |  4500 | 554.82 |   448.68 | 1124.41 |         0.09 |          0.34 |
| future_sessions_mean_playtime_7d | test    |  4500 | 597.3  |   489.4  | 1233.19 |         0.11 |          0.28 |

## 15. Time-aware history-признаки (оба таргета)

| target                           | model_name                 |   n_features |   val_mae |   val_r2 |   val_product_mae |   val_small_mae |   test_mae |
|:---------------------------------|:---------------------------|-------------:|----------:|---------:|------------------:|----------------:|-----------:|
| target_next_session_length_sec   | baseline                   |           73 |    522.53 |     0.02 |            251.65 |          207.51 |     429.26 |
| target_next_session_length_sec   | baseline_plus_history      |           87 |    522.16 |     0.02 |            250.49 |          206.18 |     428.17 |
| target_next_session_length_sec   | baseline_plus_best_history |           79 |    523.01 |     0.02 |            251.04 |          206.15 |     428.49 |
| future_sessions_mean_playtime_7d | baseline                   |           73 |    238.66 |     0.37 |            187.46 |          219.81 |     273.34 |
| future_sessions_mean_playtime_7d | baseline_plus_history      |           87 |    238.88 |     0.37 |            188.09 |          222.37 |     272.16 |
| future_sessions_mean_playtime_7d | baseline_plus_best_history |           79 |    241.15 |     0.36 |            189.87 |          225.03 |     274.35 |

## 16. Regression calibration (оба таргета)

| target                           | model_name   |   val_mae |   val_r2 |   val_product_mae |   val_small_mae |   val_long_mae |   test_mae |
|:---------------------------------|:-------------|----------:|---------:|------------------:|----------------:|---------------:|-----------:|
| target_next_session_length_sec   | raw          |    521.06 |     0.01 |            251.15 |          208.43 |        1982.13 |     428.17 |
| target_next_session_length_sec   | bin          |    587.77 |     0.07 |            394.98 |          453.08 |        1621.68 |     502.63 |
| target_next_session_length_sec   | isotonic     |    581.27 |     0.1  |            390.07 |          446.6  |        1608.03 |     499.77 |
| target_next_session_length_sec   | segment      |    582.89 |     0.09 |            389.98 |          446.22 |        1627.78 |     499.7  |
| future_sessions_mean_playtime_7d | raw          |    233.27 |     0.4  |            183.24 |          220.55 |         807.19 |     272.16 |
| future_sessions_mean_playtime_7d | bin          |    248.31 |     0.37 |            199.78 |          252.47 |         800.22 |     286.17 |
| future_sessions_mean_playtime_7d | isotonic     |    246.34 |     0.39 |            197.67 |          247.31 |         812.35 |     280.48 |
| future_sessions_mean_playtime_7d | segment      |    246.19 |     0.39 |            197.74 |          247.33 |         800.99 |     281.35 |

## 17. Feature drift detection (оба таргета)

| target                           | model_name              |   n_features |   n_dropped |   val_mae |   test_mae |
|:---------------------------------|:------------------------|-------------:|------------:|----------:|-----------:|
| target_next_session_length_sec   | all_features            |           87 |           0 |    522.16 |     428.17 |
| target_next_session_length_sec   | drop_top5_drift         |           82 |           5 |    522.3  |     429.27 |
| target_next_session_length_sec   | drop_top10_drift        |           77 |          10 |    523.01 |     428.58 |
| target_next_session_length_sec   | drop_strong_psi>0.25_n5 |           82 |           5 |    522.3  |     429.27 |
| future_sessions_mean_playtime_7d | all_features            |           87 |           0 |    238.88 |     272.16 |
| future_sessions_mean_playtime_7d | drop_top5_drift         |           82 |           5 |    240.65 |     273.59 |
| future_sessions_mean_playtime_7d | drop_top10_drift        |           77 |          10 |    243.84 |     274.59 |
| future_sessions_mean_playtime_7d | drop_strong_psi>0.25_n4 |           83 |           4 |    239.84 |     272.26 |

## 18. Финальная техническая модель (полный набор метрик, test)

| target                           | model_name         | feature_set      | calibration   | drift_filter   |    mae |   medae |   p70_abs_error |   p90_abs_error |   r2 |   small_mae |   normal_mae |   long_mae |   product_mae |   engagement_risk_mae |   wmape |   model_size_kb |   inference_us_per_row |
|:---------------------------------|:-------------------|:-----------------|:--------------|:---------------|-------:|--------:|----------------:|----------------:|-----:|------------:|-------------:|-----------:|--------------:|----------------------:|--------:|----------------:|-----------------------:|
| target_next_session_length_sec   | baseline_reference | baseline         | raw           | none           | 428.42 |  206.2  |          375.6  |          997.44 | 0.04 |      185.8  |       316.36 |    1833.95 |        228.68 |                234.44 |    0.8  |            1593 |                   0    |
| target_next_session_length_sec   | final              | baseline+history | raw           | all_features   | 428.37 |  206.13 |          369.92 |          998.06 | 0.03 |      181.15 |       319.27 |    1847.23 |        226.34 |                232    |    0.8  |            1606 |                   2.86 |
| future_sessions_mean_playtime_7d | baseline_reference | baseline         | raw           | none           | 273.34 |  159.98 |          285.16 |          645.01 | 0.26 |      253.34 |       177.19 |     876.55 |        206.65 |                212.94 |    0.46 |            1483 |                   0    |
| future_sessions_mean_playtime_7d | final              | baseline         | raw           | all_features   | 273.34 |  159.98 |          285.16 |          645.01 | 0.26 |      253.34 |       177.19 |     876.55 |        206.65 |                212.94 |    0.46 |            1483 |                   2.31 |

- **target_next_session_length_sec**: признаков 87, history=True, drift_filter=all_features, calibration=raw; baseline test MAE 428.423 → final 428.367 (Δ +0.06), 1606 КБ, 2.86 мкс/строку.
- **future_sessions_mean_playtime_7d**: признаков 73, history=False, drift_filter=all_features, calibration=raw; baseline test MAE 273.337 → final 273.337 (Δ +0.0), 1483 КБ, 2.31 мкс/строку.

## Итоговый CSV

`outputs/participant2_results.csv` — все запуски в схеме `team_modeling_protocol.txt` §16 (val_* / test_* метрики, params, fit_sec, status).
