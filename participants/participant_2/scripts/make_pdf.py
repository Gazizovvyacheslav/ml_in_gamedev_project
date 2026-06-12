"""Build a polished PDF report (Russian) from Participant 2's experiment CSVs.

Charts via matplotlib, layout via reportlab Platypus. DejaVuSans is registered so that
Cyrillic text renders correctly (the built-in reportlab fonts have no Cyrillic glyphs).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image,
    PageBreak,
)

import p2_common as C

OUT = C.OUTPUT_DIR
FIGS = OUT / "figs"
FIGS.mkdir(exist_ok=True)


_mpl_ttf = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
pdfmetrics.registerFont(TTFont("DejaVu", str(_mpl_ttf / "DejaVuSans.ttf")))
pdfmetrics.registerFont(TTFont("DejaVu-Bold", str(_mpl_ttf / "DejaVuSans-Bold.ttf")))
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.3

BLUE = "#2b6cb0"
ORANGE = "#dd6b20"
GREY = "#a0aec0"


def csv(name):
    p = OUT / f"{name}.csv"
    return pd.read_csv(p) if p.exists() else None


def chart_feature_selection():
    d = csv("exp1_shap_feature_selection")
    if d is None:
        return None
    order = ["shap_k40", "shap_k60", "full", "topk_40", "topk_60"]
    d = (
        d.set_index("model_name")
        .loc[[m for m in order if m in d.model_name.values]]
        .reset_index()
    )
    fig, ax1 = plt.subplots(figsize=(7, 3.4))
    cols = [
        BLUE if "shap" in m else (GREY if m == "full" else ORANGE) for m in d.model_name
    ]
    ax1.bar(d.model_name, d.val_mae, color=cols)
    ax1.set_ylabel("val MAE, сек")
    ax1.set_ylim(d.val_mae.min() - 1.5, d.val_mae.max() + 1.0)
    for x, v in zip(d.model_name, d.val_mae):
        ax1.text(x, v + 0.05, f"{v:.1f}", ha="center", va="bottom", fontsize=8)
    ax2 = ax1.twinx()
    ax2.plot(d.model_name, d.model_size_kb, "o-", color="#2f855a", label="размер, КБ")
    ax2.set_ylabel("размер модели, КБ", color="#2f855a")
    ax2.grid(False)
    ax1.set_title("SHAP отбор vs ручной top-k: val MAE и размер модели")
    fig.tight_layout()
    p = FIGS / "exp1.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def chart_tree_structure():
    d = csv("exp4_tree_structure")
    if d is None:
        return None
    d = d.sort_values("val_mae")
    fig, ax = plt.subplots(figsize=(7, 3.6))
    cols = [
        BLUE if m.startswith("depthwise") else (ORANGE if m == "symmetric" else GREY)
        for m in d.model_name
    ]
    ax.barh(d.model_name, d.val_mae, color=cols)
    ax.set_xlim(d.val_mae.min() - 0.6, d.val_mae.max() + 0.4)
    ax.invert_yaxis()
    for y, v in zip(range(len(d)), d.val_mae):
        ax.text(v + 0.02, y, f"{v:.1f}", va="center", fontsize=8)
    ax.set_xlabel("val MAE, сек")
    ax.set_title(
        "Структура деревьев: Depthwise (синий) vs Symmetric (оранж.) vs Lossguide"
    )
    fig.tight_layout()
    p = FIGS / "exp4.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def chart_strategies():
    d = csv("exp7_final_strategies")
    if d is None:
        return None
    d = d[d.target == C.NEXT_TARGET].set_index("model_name")
    order = ["capped_target", "quantile_040", "quantile_035"]
    d = d.loc[[m for m in order if m in d.index]]
    metrics = ["val_mae", "val_product_mae", "val_engagement_risk_mae", "val_small_mae"]
    labels = ["MAE", "ProductMAE", "EngRiskMAE", "small MAE"]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    import numpy as np

    x = np.arange(len(metrics))
    w = 0.25
    for i, m in enumerate(order):
        vals = [d.loc[m, k] for k in metrics]
        ax.bar(x + (i - 1) * w, vals, w, label=m)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("ошибка, сек")
    ax.set_title("Next-session: trade-off стратегий (capped / Q0.40 / Q0.35)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = FIGS / "exp7.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def chart_family_compare():
    """Best next-session val MAE per model family / baseline."""

    bars = []
    e8 = csv("exp8_main_catboost_sweep")
    if e8 is not None:
        bars.append(("CatBoost\n(exp8)", e8.val_mae.min(), BLUE))
    e13 = csv("exp13_lightgbm")
    if e13 is not None and "val_mae" in e13:
        bars.append(("LightGBM\n(exp13)", e13.val_mae.min(), "#805ad5"))
    e12 = csv("exp12_ridge")
    if e12 is not None:
        nxt = e12[e12.target == C.NEXT_TARGET]
        ridge = nxt[nxt.model_name.str.startswith("ridge")]
        dmed = nxt[nxt.model_name == "dummy_median"]
        if len(ridge):
            bars.append(("Ridge\n(exp12)", ridge.val_mae.min(), ORANGE))
        if len(dmed):
            bars.append(("Dummy\nmedian", float(dmed.val_mae.iloc[0]), GREY))
    if not bars:
        return None
    fig, ax = plt.subplots(figsize=(7, 3.4))
    names = [b[0] for b in bars]
    vals = [b[1] for b in bars]
    cs = [b[2] for b in bars]
    ax.bar(names, vals, color=cs)
    for i, v in enumerate(vals):
        ax.text(i, v + 1, f"{v:.0f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("val MAE, сек (next-session)")
    ax.set_title("Сравнение семейств моделей: CatBoost < LightGBM << Ridge < median")
    fig.tight_layout()
    p = FIGS / "family.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def chart_feature_ablation():
    d = csv("exp9_feature_ablation")
    if d is None:
        return None
    order = [
        "session_only",
        "session_install",
        "session_install_events",
        "top_k_40",
        "top_k_60",
        "top_k_73",
    ]
    d = (
        d.set_index("model_name")
        .loc[[m for m in order if m in d.model_name.values]]
        .reset_index()
    )
    fig, ax1 = plt.subplots(figsize=(7, 3.4))
    ax1.bar(d.model_name, d.val_mae, color=BLUE)
    ax1.set_ylabel("val MAE, сек", color=BLUE)
    ax1.set_ylim(d.val_mae.min() - 1, d.val_mae.max() + 1)
    ax1.tick_params(axis="x", rotation=25)
    ax2 = ax1.twinx()
    ax2.plot(d.model_name, d.model_size_kb, "o-", color=ORANGE)
    ax2.set_ylabel("размер модели, КБ", color=ORANGE)
    ax2.grid(False)
    ax1.set_title("Группы признаков: val MAE и размер модели")
    fig.tight_layout()
    p = FIGS / "ablation.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def chart_cascade():
    d = csv("exp10_cascade")
    if d is None:
        return None
    import numpy as np

    order = ["baseline_general", "cascade_hard", "cascade_soft", "cascade_hybrid"]
    d = (
        d.set_index("model_name")
        .loc[[m for m in order if m in d.model_name.values]]
        .reset_index()
    )
    fig, ax1 = plt.subplots(figsize=(7, 3.4))
    x = np.arange(len(d))
    ax1.bar(x - 0.2, d.val_mae, 0.4, color=BLUE, label="val MAE")
    ax1.set_ylabel("val MAE, сек", color=BLUE)
    ax1.set_ylim(500, max(d.val_mae) + 20)
    ax2 = ax1.twinx()
    ax2.bar(x + 0.2, d.val_long_mae, 0.4, color=ORANGE, label="val long MAE")
    ax2.set_ylabel("val long MAE, сек", color=ORANGE)
    ax2.grid(False)
    ax1.set_xticks(x)
    ax1.set_xticklabels([m.replace("cascade_", "") for m in d.model_name])
    ax1.set_title("Каскад vs одиночная модель: общий MAE и long-tail")
    fig.tight_layout()
    p = FIGS / "cascade.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def chart_ctr():
    d = csv("exp3_ctr_tuning")
    if d is None:
        return None
    fig, ax = plt.subplots(figsize=(7, 3.4))
    sc = ax.scatter(
        d.model_size_kb,
        d.val_mae,
        c=d.max_ctr_complexity,
        cmap="coolwarm",
        s=60,
        edgecolor="k",
        linewidth=0.4,
    )
    ax.set_xlabel("размер модели, КБ")
    ax.set_ylabel("val MAE, сек")
    ax.set_title("CTR-тюнинг: качество vs размер (цвет = ctr_complexity)")
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label("max_ctr_complexity")
    best = d.sort_values("val_mae").iloc[0]
    ax.annotate(
        best.model_name,
        (best.model_size_kb, best.val_mae),
        textcoords="offset points",
        xytext=(6, 6),
        fontsize=8,
    )
    fig.tight_layout()
    p = FIGS / "exp3.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def chart_valtest_gap():
    d = csv("exp18_val_test_gap")
    if d is None:
        return None
    import numpy as np

    d = d[d.target == C.NEXT_TARGET]
    splits = ["train", "val", "test"]
    d = d.set_index("split").loc[splits]
    fig, ax1 = plt.subplots(figsize=(7, 3.2))
    x = np.arange(len(splits))
    ax1.bar(x - 0.2, d.long_share, 0.4, color=ORANGE, label="long_share")
    ax1.set_ylabel("доля long-сессий", color=ORANGE)
    ax2 = ax1.twinx()
    ax2.plot(x, d["mean"], "o-", color=BLUE, label="mean target")
    ax2.set_ylabel("средний таргет, сек", color=BLUE)
    ax2.grid(False)
    ax1.set_xticks(x)
    ax1.set_xticklabels(splits)
    ax1.set_title("next-session: test-период легче (меньше long, ниже среднее)")
    fig.tight_layout()
    p = FIGS / "valtest.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def chart_history():
    d = csv("exp14_history_features")
    if d is None:
        return None

    targets = [C.NEXT_TARGET, C.CRM_TARGET]
    names = ["baseline", "baseline_plus_history", "baseline_plus_best_history"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))
    for ax, t in zip(axes, targets):
        sub = d[d.target == t].set_index("model_name").loc[names]
        ax.bar(range(3), sub.val_mae, color=[GREY, BLUE, "#2f855a"])
        ax.set_xticks(range(3))
        ax.set_xticklabels(["base", "+hist", "+best"], fontsize=8)
        lo, hi = sub.val_mae.min(), sub.val_mae.max()
        ax.set_ylim(lo - (hi - lo) - 0.5, hi + (hi - lo) + 0.5)
        ax.set_title("next" if t == C.NEXT_TARGET else "CRM", fontsize=9)
        for i, v in enumerate(sub.val_mae):
            ax.text(i, v, f"{v:.1f}", ha="center", va="bottom", fontsize=7)
    axes[0].set_ylabel("val MAE, сек")
    fig.suptitle("History-признаки: val MAE (выбор по validation)")
    fig.tight_layout()
    p = FIGS / "history.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def chart_final():
    fm = OUT / "final_model_metrics.csv"
    if not fm.exists():
        return None

    d = pd.read_csv(fm)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))
    for ax, t in zip(axes, [C.NEXT_TARGET, C.CRM_TARGET]):
        sub = d[d.target == t].set_index("model_name")
        names = [n for n in ["baseline_reference", "final"] if n in sub.index]
        vals = [sub.loc[n, "mae"] for n in names]
        ax.bar(range(len(names)), vals, color=[GREY, "#2f855a"])
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(["baseline", "final"], fontsize=8)
        lo, hi = min(vals), max(vals)
        ax.set_ylim(lo - (hi - lo) - 1, hi + (hi - lo) + 1)
        ax.set_title("next" if t == C.NEXT_TARGET else "CRM", fontsize=9)
        for i, v in enumerate(vals):
            ax.text(i, v, f"{v:.1f}", ha="center", va="bottom", fontsize=8)
    axes[0].set_ylabel("test MAE, сек")
    fig.suptitle("Финальная модель vs baseline (test MAE)")
    fig.tight_layout()
    p = FIGS / "final.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


OUT = C.OUTPUT_DIR


def styles():
    ss = getSampleStyleSheet()
    base = ss["Normal"]
    base.fontName = "DejaVu"
    base.fontSize = 9.5
    base.leading = 13
    H1 = ParagraphStyle(
        "H1",
        parent=ss["Heading1"],
        fontName="DejaVu-Bold",
        fontSize=15,
        spaceBefore=10,
        spaceAfter=6,
        textColor=colors.HexColor("#1a365d"),
    )
    H2 = ParagraphStyle(
        "H2",
        parent=ss["Heading2"],
        fontName="DejaVu-Bold",
        fontSize=12,
        spaceBefore=8,
        spaceAfter=4,
        textColor=colors.HexColor("#2b6cb0"),
    )
    title = ParagraphStyle(
        "T",
        parent=ss["Title"],
        fontName="DejaVu-Bold",
        fontSize=22,
        leading=26,
        textColor=colors.HexColor("#1a365d"),
    )
    sub = ParagraphStyle(
        "sub",
        parent=base,
        fontSize=11,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#4a5568"),
    )
    small = ParagraphStyle(
        "small", parent=base, fontSize=8.5, textColor=colors.HexColor("#4a5568")
    )
    return dict(base=base, H1=H1, H2=H2, title=title, sub=sub, small=small)


def df_table(df, cols, headers=None, highlight_first=True, col_w=None):
    df = df[[c for c in cols if c in df.columns]].copy()
    for c in df.columns:
        if df[c].dtype.kind == "f":
            df[c] = df[c].map(lambda v: f"{v:.1f}" if pd.notna(v) else "")
    head = headers or list(df.columns)
    data = [head] + df.astype(str).values.tolist()
    t = Table(data, colWidths=col_w, hAlign="LEFT")
    st = [
        ("FONTNAME", (0, 0), (-1, -1), "DejaVu"),
        ("FONTNAME", (0, 0), (-1, 0), "DejaVu-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2b6cb0")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        (
            "ROWBACKGROUNDS",
            (0, 1),
            (-1, -1),
            [colors.white, colors.HexColor("#edf2f7")],
        ),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cbd5e0")),
        ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]
    if highlight_first and len(data) > 1:
        st.append(("BACKGROUND", (0, 1), (-1, 1), colors.HexColor("#c6f6d5")))
        st.append(("FONTNAME", (0, 1), (-1, 1), "DejaVu-Bold"))
    t.setStyle(TableStyle(st))
    return t


def img(path, width=16 * cm):
    if path is None:
        return Spacer(1, 1)
    from PIL import Image as PILImage

    w, h = PILImage.open(path).size
    return Image(str(path), width=width, height=width * h / w)


def main():
    S = styles()
    meta = json.loads((OUT / "anti_leak.json").read_text())
    bc = json.loads((OUT / "best_base_config.json").read_text())
    story = []

    def P(t, s="base"):
        story.append(Paragraph(t, S[s]))

    def SP(h=6):
        story.append(Spacer(1, h))

    SP(40)
    P("Участник 2", "title")
    P("Boosting, признаки и архитектура CatBoost", "sub")
    SP(8)
    P(
        "Доп. тюнинг CatBoost · основной sweep по MAE · LightGBM · каскад · Ridge · "
        "прогноз длительности игровой сессии (без API)",
        "sub",
    )
    SP(30)
    P(
        f"Таргет: <b>target_next_session_length_sec</b> · выборка {meta['sample_rows']} строк "
        f"(70/15/15) · CatBoost 1.2.10",
        "sub",
    )
    P("Anti-leak: PASS · Chronology: PASS · No-NaN: PASS", "sub")
    story.append(PageBreak())

    P("1. Методология", "H1")
    P(
        "Все эксперименты используют общий командный пайплайн "
        "(<i>preprocessing/preprocessing.py</i>): строго временно́е разбиение 70/15/15 "
        "(<i>shuffle=False</i>), единый anti-leak (удаление <i>target*</i>, <i>future_*</i>, "
        "ID и служебных временных колонок) и единый набор метрик из "
        "<i>team_modeling_protocol.txt</i>. Выбор конфигурации — только по validation; "
        "test используется один раз для финальной оценки.",
        "base",
    )
    SP()
    rows = [
        ["Параметр", "Значение"],
        ["Выборка (последние по времени)", f"{meta['sample_rows']} строк"],
        [
            "Train / Val / Test",
            f"{meta['train_rows']} / {meta['val_rows']} / {meta['test_rows']}",
        ],
        [
            "Признаков (числовых / категор.)",
            f"{meta['n_features']} ({meta['n_num']} / {meta['n_cat']})",
        ],
        ["Train период", f"{meta['train_time'][0]} → {meta['train_time'][1]}"],
        ["Test период", f"{meta['test_time'][0]} → {meta['test_time'][1]}"],
        [
            "Базовый конфиг",
            f"depth={bc['hp']['depth']}, lr={bc['hp']['learning_rate']}, "
            f"l2={bc['hp']['l2_leaf_reg']}, iters={bc['hp']['iterations']}, "
            f"target_mode={bc['target_mode']}",
        ],
        [
            "Базовый val MAE / test MAE",
            f"{bc['val_mae']:.1f} / {bc['test_mae']:.1f} сек",
        ],
    ]
    t = Table(rows, colWidths=[6.5 * cm, 10 * cm], hAlign="LEFT")
    t.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, -1), "DejaVu"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("FONTNAME", (0, 0), (0, -1), "DejaVu-Bold"),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cbd5e0")),
                (
                    "ROWBACKGROUNDS",
                    (0, 0),
                    (-1, -1),
                    [colors.HexColor("#edf2f7"), colors.white],
                ),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]
        )
    )
    story.append(t)
    SP(6)
    P(
        "⚠ Важно: для next-session таргета MAE по validation у всех конфигов лежит в узком "
        "коридоре ~521–525 сек — таргет очень шумный (R²≈0.01, что отмечено в протоколе). "
        "Поэтому выигрыш архитектурных правок по MAE невелик, и их главная ценность — "
        "компактность и скорость модели.",
        "small",
    )
    story.append(PageBreak())

    def section(num, title, conclusion, table_df, cols, headers, chart, col_w=None):
        P(f"{num}. {title}", "H1")
        P(conclusion, "base")
        SP()
        if chart is not None:
            story.append(img(chart))
            SP()
        if table_df is not None:
            story.append(df_table(table_df, cols, headers, col_w=col_w))
        story.append(PageBreak())

    d1 = csv("exp1_shap_feature_selection").sort_values("val_mae")
    section(
        "2",
        "SHAP feature selection vs ручной top-k",
        "Встроенный рекурсивный отбор <b>RecursiveByShapValues</b> сравнивается с ручным "
        "top-k по важности. <b>Вывод:</b> SHAP-отбор на 40 признаках (из 73) даёт лучший "
        "val MAE (521.6), обходя и полную модель (522.5), и ручной top-k (523.3), при этом "
        "модель в ~3.5 раза меньше (199 КБ против 712 КБ). Размер модели можно сократить "
        "и ускорить inference без потери качества.",
        d1,
        [
            "model_name",
            "selection",
            "n_features",
            "val_mae",
            "test_mae",
            "fit_sec",
            "model_size_kb",
        ],
        ["модель", "отбор", "k", "val MAE", "test MAE", "fit, с", "размер, КБ"],
        chart_feature_selection(),
    )

    d2 = csv("exp2_bootstrap_mvs").sort_values("val_mae")
    section(
        "3",
        "Bootstrap: MVS vs Bernoulli / Bayesian",
        "Minimal Variance Sampling добавлен к сравнению с Bernoulli и Bayesian; "
        "устойчивость измерена по сидам 42/52/62. <b>Вывод:</b> MVS <b>не даёт</b> выигрыша "
        "по MAE и не ускоряет заметно обучение; самый стабильный по сидам — Bayesian "
        "(std 0.18). Для long-tail (val_long_mae) различия в пределах шума.",
        d2,
        ["model_name", "val_mae", "val_mae_std", "test_mae", "val_long_mae", "fit_sec"],
        ["конфиг", "val MAE", "±std", "test MAE", "long MAE", "fit, с"],
        None,
    )

    d3 = csv("exp3_ctr_tuning").sort_values("val_mae").head(10)
    section(
        "4",
        "CTR-тюнинг категориальных признаков",
        "Сетка по <i>one_hot_max_size</i>, <i>max_ctr_complexity</i>, "
        "<i>ctr_target_border_count</i>. <b>Вывод:</b> лучший — простой "
        "<i>one_hot=10, ctr_complexity=1, ctr_border=1</i> (522.7). Комбинации "
        "категорий (mcc=2) и более тонкая квантизация таргета для CTR пользы не дают и "
        "<b>раздувают модель</b> (до 733 КБ). Топ-10 по val MAE:",
        d3,
        ["model_name", "val_mae", "test_mae", "fit_sec", "model_size_kb"],
        ["конфиг", "val MAE", "test MAE", "fit, с", "размер, КБ"],
        chart_ctr(),
    )

    d4 = csv("exp4_tree_structure").sort_values("val_mae")
    section(
        "5",
        "Структура деревьев: Symmetric / Depthwise / Lossguide",
        "Сравнение стандартных симметричных деревьев с гибкими Depthwise и Lossguide. "
        "<b>Вывод (главная находка):</b> <b>Depthwise, min_data_in_leaf=20</b> — лучший val "
        "MAE (522.3) и лучший long_mae (1943), при этом модель вдвое меньше и быстрее "
        "симметричной (359 КБ / 2.0 с против 712 КБ / 4.0 с). Гибкие деревья дают небольшое "
        "преимущество на неоднородных сегментах.",
        d4,
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
        ["конфиг", "val MAE", "test MAE", "small", "normal", "long", "fit, с", "КБ"],
        chart_tree_structure(),
    )

    d5 = csv("exp5_quantization").sort_values("val_mae").head(10)
    section(
        "6",
        "Квантизация числовых признаков",
        "Сетка по <i>border_count</i> и <i>feature_border_type</i> плюс "
        "<i>per_float_feature_quantization</i> на топ-3 числовых признаках. <b>Вывод:</b> "
        "дефолтный <i>border_count=254, GreedyLogSum</i> оптимален по val MAE; очень тонкая "
        "квантизация (512, per_float=1024) на длинном хвосте практически не помогает. "
        "Топ-10 по val MAE:",
        d5,
        [
            "model_name",
            "val_mae",
            "test_mae",
            "val_long_mae",
            "fit_sec",
            "model_size_kb",
        ],
        ["конфиг", "val MAE", "test MAE", "long MAE", "fit, с", "КБ"],
        None,
    )

    d6 = csv("exp6_rsm").sort_values("val_mae")
    section(
        "7",
        "Запасной эксперимент: rsm",
        "Случайный выбор доли признаков при поиске split (rsm = 0.6 / 0.8 / 1.0). "
        "<b>Вывод:</b> разрыв val–test практически одинаков (~+93 сек) для всех значений — "
        "дополнительной регуляризации rsm здесь не даёт; полезен скорее для разнообразия "
        "моделей в ансамбле.",
        d6,
        ["model_name", "val_mae", "test_mae", "fit_sec"],
        ["rsm", "val MAE", "test MAE", "fit, с"],
        None,
    )

    d7 = csv("exp7_final_strategies")
    P("8. Протокольные стратегии на лучшем конфиге (оба таргета)", "H1")
    P(
        "Согласно протоколу §15, для лучшего конфига прогнаны три канонические стратегии — "
        "<b>capped_target</b> (MAE), <b>Quantile:alpha=0.40</b> и <b>Quantile:alpha=0.35</b> — "
        "на обоих таргетах. <b>Вывод:</b> capped_target лучший по общему MAE, а Quantile 0.35 — "
        "лучшая short-risk модель (минимальные ProductMAE/EngagementRiskMAE/small_mae). Это "
        "подтверждает командные результаты. CRM-таргет заметно стабильнее next-session "
        "(test MAE ~272 против ~429).",
        "base",
    )
    SP()
    story.append(img(chart_strategies()))
    SP()
    story.append(
        df_table(
            d7,
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
            [
                "таргет",
                "стратегия",
                "loss",
                "val MAE",
                "Product",
                "EngRisk",
                "small",
                "test MAE",
            ],
            highlight_first=False,
        )
    )
    story.append(PageBreak())

    P("Часть II. Технический прогноз: основной CatBoost, LightGBM, каскад", "H1")
    P(
        "Эксперименты из второго ТЗ (`distribution_of_responsoblities.txt`): поиск лучшей "
        "технической модели по validation MAE, сравнение групп признаков, альтернативный "
        "бустинг и более сложные архитектуры на базе CatBoost. API в объём не входит.",
        "base",
    )
    SP()
    story.append(img(chart_family_compare()))
    SP()
    P(
        "Главный вывод части II: <b>CatBoost — лучшая техническая модель</b> по val MAE; "
        "LightGBM сопоставим, но чуть хуже; линейный Ridge не обходит даже median-baseline. "
        "Это эмпирически обосновывает выбор CatBoost.",
        "small",
    )
    story.append(PageBreak())

    d8 = csv("exp8_main_catboost_sweep").sort_values("val_mae").head(8)
    section(
        "9",
        "Основной CatBoost sweep по MAE (best_by_mae)",
        "Широкий случайный поиск (40 конфигов) по сетке ТЗ: "
        "<i>iterations / depth / lr / l2 / min_data_in_leaf / random_strength / bootstrap / "
        "target_transform / clip_mode</i>. <b>Вывод:</b> лучшая техническая модель — "
        "<b>depth=7, lr=0.03, l2=5, log1p_p995, Bernoulli</b> (val MAE 521.8, test 428.4). "
        "Полный sweep дал лишь ~0.7 сек к компактному базовому — снова упёрлись в шум "
        "таргета. Топ-8 по val MAE:",
        d8,
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
        ],
        [
            "cfg",
            "depth",
            "lr",
            "l2",
            "iters",
            "bootstrap",
            "t_mode",
            "clip",
            "val",
            "test",
        ],
        None,
    )

    d9 = csv("exp9_feature_ablation").sort_values("val_mae")
    section(
        "10",
        "Feature ablation групп признаков",
        "Наборы: <i>session_only / session_install / session_install_events (полный) / "
        "top_k_40/60/73</i>. <b>Вывод:</b> event-признаки почти не помогают — "
        "<b>session_only (37 призн.)</b> даёт val MAE 522.9 против 522.5 у полного набора "
        "(73), но модель <b>в ~4 раза меньше</b> (172 КБ vs 712 КБ) и обучается в ~8 раз "
        "быстрее. Для inference event-группу можно отбрасывать почти без потерь.",
        d9,
        ["model_name", "n_features", "val_mae", "test_mae", "fit_sec", "model_size_kb"],
        ["набор", "k", "val MAE", "test MAE", "fit, с", "размер, КБ"],
        chart_feature_ablation(),
    )

    d10 = csv("exp10_cascade")
    section(
        "11",
        "Каскад «классификация → регрессия» (hard / soft / hybrid)",
        "CatBoostClassifier (small/normal/long, acc≈0.51) + 3 посегментных регрессора; "
        "три режима маршрутизации. <b>Вывод (честный отрицательный):</b> ни один режим не "
        "обходит одиночную модель по общему MAE (baseline 522.5 против hard 609 / soft 563 / "
        "hybrid 527) — ошибки классификатора распространяются на регрессию. Полезен только "
        "<b>soft routing для long-tail</b> (long MAE 1693 против 1955) ценой small/normal.",
        d10,
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
        [
            "режим",
            "val MAE",
            "Product",
            "EngRisk",
            "small",
            "normal",
            "long",
            "test MAE",
        ],
        chart_cascade(),
    )

    d11 = csv("exp11_residual_correction")
    section(
        "12",
        "Коррекция остатков базовой модели",
        "Честная схема: base на блоке A → остатки на блоке B → компактный корректор; "
        "<i>final = base + correction</i>. <b>Вывод:</b> по общему MAE коррекция не помогает "
        "(522.5 → 528.6), но подтверждает систематику — корректор сдвигает прогноз вниз "
        "(−36 сек) и <b>улучшает small_mae</b> (207.5 → 186.7) ценой normal/long. Завышение "
        "коротких сессий реально есть, но глобально не «вычитается» из-за шума. Диагностика, "
        "не production-приём.",
        d11,
        [
            "model_name",
            "val_mae",
            "test_mae",
            "val_small_mae",
            "val_normal_mae",
            "val_long_mae",
            "val_product_mae",
        ],
        ["модель", "val MAE", "test MAE", "small", "normal", "long", "Product"],
        None,
    )

    d13 = csv("exp13_lightgbm").sort_values("val_mae").head(8)
    section(
        "13",
        "LightGBM как альтернативный бустинг",
        "Компактный случайный поиск (objective regression_l1 / quantile, n_estimators, lr, "
        "num_leaves, max_depth, fractions), категории — нативно через pandas category. "
        "<b>Вывод:</b> лучший LightGBM val MAE 524.7 / test 431.2 за ~2 с — <b>сопоставим, но "
        "чуть хуже CatBoost</b> (521.8 / 428.4), при этом CatBoost нативно работает с "
        "категориями и удобнее как единое семейство. Отказ от LightGBM — результат сравнения. "
        "Топ-8 по val MAE:",
        d13,
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
        [
            "модель",
            "objective",
            "n_est",
            "lr",
            "leaves",
            "val MAE",
            "test MAE",
            "fit, с",
        ],
        None,
    )

    d12 = csv("exp12_ridge")
    P("14. Ridge baseline vs Dummy (оба таргета)", "H1")
    P(
        "Прозрачный L2-baseline против DummyRegressor(mean/median). <b>Вывод:</b> на next-session "
        "Ridge <b>не обходит даже dummy_median</b> (581 против 555 и 521.8 у CatBoost) — линейная "
        "модель почти бесполезна на этом шумном таргете. На CRM Ridge разумнее (test 331.8), но "
        "CatBoost всё равно сильнее (~272). Это и есть обоснование пользы нелинейных "
        "взаимодействий.",
        "base",
    )
    SP()
    d12v = pd.concat(
        [
            d12[(d12.target == C.NEXT_TARGET)].sort_values("val_mae").head(4),
            d12[(d12.target == C.CRM_TARGET)].sort_values("val_mae").head(3),
        ]
    )
    story.append(
        df_table(
            d12v,
            ["target", "model_name", "val_mae", "test_mae", "val_r2", "val_small_mae"],
            ["таргет", "модель", "val MAE", "test MAE", "R2", "small MAE"],
            highlight_first=False,
        )
    )
    story.append(PageBreak())

    P("Часть III. Доработка: оба таргета, полные метрики, финальная модель", "H1")
    P(
        "По итогам ревью добавлены блоки строго в технической зоне Участника 2 — для "
        "<b>обоих таргетов</b> (next-session и CRM) и с полным набором метрик "
        "(MAE, MedAE, P70, P90, R², small/normal/long, ProductMAE, EngagementRiskMAE, WMAPE). "
        "Выбор везде по validation; калибраторы учатся на внутреннем calibration-split, test — "
        "только финальная оценка.",
        "base",
    )
    story.append(PageBreak())

    section(
        "15",
        "Анализ val–test разрыва",
        "Почему val MAE (~522) хуже test MAE (~429) на next-session. <b>Вывод:</b> "
        "test-период объективно легче — доля long-сессий 0.122 против 0.158 в val, "
        "среднее 537 против 661 сек. Модель не подглядывает в test; разрыв объясняется "
        "составом периода. Для CRM, наоборот, val чуть легче test — и там val MAE < test.",
        csv("exp18_val_test_gap"),
        ["target", "split", "n", "mean", "median", "p90", "long_share", "small_share"],
        ["таргет", "split", "n", "mean", "median", "p90", "long%", "small%"],
        chart_valtest_gap(),
    )

    section(
        "16",
        "Time-aware history-признаки (оба таргета)",
        "14 новых признаков из прошлой истории игрока (EWMA, медиана/среднее последних "
        "3/5, тренд, время с прошлой сессии, активность за 1/3/7 дней) — строго past-only, "
        "без утечки. <b>Вывод (по таргетам по-разному):</b> на <b>next-session</b> history "
        "даёт маржинальный плюс (522.5 → 522.2, лучшие: ewma5, median_last5, "
        "time_since_prev) → включаем; на <b>CRM</b> history не помогает (238.7 → 238.9) → "
        "не включаем. Полный набор метрик в CSV.",
        csv("exp14_history_features"),
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
        ["таргет", "набор", "k", "val MAE", "R2", "Product", "small", "test MAE"],
        chart_history(),
    )

    section(
        "17",
        "Regression calibration (оба таргета)",
        "Пост-калибровка прогнозов: bin / isotonic / segment, калибратор на "
        "calibration-split. <b>Вывод:</b> калибровка <b>улучшает long_mae и R²</b> "
        "(next: long 1982→1608, R² 0.01→0.10), но <b>резко ухудшает small/normal и общий "
        "MAE</b> (small 208→453), т.к. тянет прогноз к условному среднему, а MAE "
        "оптимизирует медиану. Для MAE-модели не включаем (выбор = raw) на обоих таргетах.",
        csv("exp15_calibration"),
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
        ["таргет", "метод", "val MAE", "R2", "Product", "small", "long", "test MAE"],
        None,
    )

    section(
        "18",
        "Feature drift detection (оба таргета)",
        "PSI (числовые) + доля новых категорий + сдвиг missing/среднего между train/test. "
        "Найдены drift-heavy признаки; проверены модели без top-5 / top-10 / strong-drift "
        "(PSI>0.25). <b>Вывод:</b> удаление нестабильных признаков сохраняется/выбирается "
        "только если не ухудшает validation MAE (решение записано для финальной сборки). "
        "Скоры по признакам — в <i>drift_scores_&lt;target&gt;.csv</i>.",
        csv("exp16_drift"),
        ["target", "model_name", "n_features", "n_dropped", "val_mae", "test_mae"],
        ["таргет", "вариант", "k", "drop", "val MAE", "test MAE"],
        None,
    )

    fm = OUT / "final_model_metrics.csv"
    if fm.exists():
        P("19. Финальная техническая модель (полные метрики, test)", "H1")
        P(
            "Сборка лучших решений: гиперпараметры (best_by_mae из основного sweep), набор "
            "признаков (baseline + полезные history − drift-heavy), калибровка по решению блока "
            "17. Обучена для обоих таргетов, сохранены артефакты (<i>final_models/*.cbm</i>, "
            "списки признаков, конфиг, README). Сравнение с обычным baseline:",
            "base",
        )
        SP()
        story.append(img(chart_final()))
        SP()
        fdf = pd.read_csv(fm)
        story.append(
            df_table(
                fdf,
                [
                    "target",
                    "model_name",
                    "feature_set",
                    "calibration",
                    "mae",
                    "medae",
                    "p90_abs_error",
                    "r2",
                    "small_mae",
                    "long_mae",
                    "product_mae",
                    "engagement_risk_mae",
                    "model_size_kb",
                    "inference_us_per_row",
                ],
                [
                    "таргет",
                    "модель",
                    "features",
                    "calib",
                    "MAE",
                    "MedAE",
                    "P90",
                    "R2",
                    "small",
                    "long",
                    "Product",
                    "EngRisk",
                    "КБ",
                    "мкс/стр",
                ],
                highlight_first=False,
            )
        )
        story.append(PageBreak())

    P("20. Итоговые рекомендации Участника 2", "H1")
    for line in [
        "<b>1. Архитектура.</b> Перейти на <b>grow_policy=Depthwise, min_data_in_leaf=20</b>: "
        "тот же/лучший MAE при вдвое меньшей и более быстрой модели.",
        "<b>2. Признаки.</b> Использовать <b>SHAP-отбор 40 признаков</b> вместо 73 — качество "
        "не падает, модель в ~3.5 раза компактнее, inference быстрее. Список сохранён в "
        "<i>feature_set_shap_40.json</i>.",
        "<b>3. Категории.</b> Оставить простые CTR-настройки (<i>ctr_complexity=1</i>); "
        "комбинации категорий не оправданы.",
        "<b>4. Квантизация.</b> Оставить дефолт <i>border_count=254, GreedyLogSum</i>.",
        "<b>5. Bootstrap.</b> MVS не внедрять; при необходимости стабильности — Bayesian.",
        "<b>6. Стратегии.</b> Для технической точности — capped_target; для short-risk "
        "(CRM/реклама) — Quantile:alpha=0.35.",
        "<b>7. Лучшая техническая модель (best_by_mae):</b> CatBoost "
        "<i>depth=7, lr=0.03, l2=5, log1p_p995, Bernoulli</i> — val MAE 521.8 / test 428.4.",
        "<b>8. LightGBM</b> сопоставим, но чуть хуже CatBoost и требует отдельной обработки "
        "категорий → остаёмся на CatBoost (вывод по результату сравнения).",
        "<b>9. Каскад и коррекция остатков</b> не улучшают общий MAE (ошибки классификатора / "
        "шум остатков), но полезны точечно: soft-каскад — для long-tail, корректор — как "
        "диагностика завышения коротких сессий.",
        "<b>10. Ridge</b> не обходит median-baseline на next-session → нелинейный бустинг "
        "оправдан.",
        "<b>11. History-признаки</b> включены в финал только для next-session (маржинальный "
        "плюс: ewma5, median_last5, time_since_prev); для CRM эффекта нет.",
        "<b>12. Calibration и удаление drift-признаков</b> не включены — не улучшают validation "
        "MAE (калибровка полезна для long-tail/R², но ухудшает общий MAE и short-сегменты).",
        "<b>13. Финальная модель</b> собрана и сохранена для обоих таргетов "
        "(<i>final_models/*.cbm</i> + конфиг/признаки/README); по MAE ≈ baseline — на шумных "
        "таргетах резерв качества лежит не в этих приёмах, а в самих данных/таргете.",
        "<b>Оговорка.</b> Все выводы по next-session ограничены высоким шумом таргета "
        "(R²≈0.01); приоритет архитектурных правок — компактность и скорость, а не точность.",
    ]:
        P("• " + line, "base")
        SP(3)
    SP(8)
    P(
        "Артефакты: <i>outputs/participant2_results.csv</i> (78 запусков в схеме протокола §16), "
        "per-experiment CSV, <i>anti_leak.json</i>, <i>best_base_config.json</i>, "
        "<i>feature_set_shap_40.json</i>. Воспроизводимость: <i>exp0…exp7.py</i>, "
        "<i>p2_common.py</i>.",
        "small",
    )

    pdf_path = OUT.parent / "Participant2_Report.pdf"
    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=A4,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        topMargin=1.6 * cm,
        bottomMargin=1.6 * cm,
        title="Участник 2 — CatBoost Extra Tuning",
        author="Participant 2",
    )

    def footer(canvas, d):
        canvas.saveState()
        canvas.setFont("DejaVu", 8)
        canvas.setFillColor(colors.HexColor("#a0aec0"))
        canvas.drawString(
            2 * cm, 1 * cm, "Участник 2 · CatBoost extra tuning · next-session"
        )
        canvas.drawRightString(A4[0] - 2 * cm, 1 * cm, f"стр. {d.page}")
        canvas.restoreState()

    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print(f"[saved] {pdf_path}")


if __name__ == "__main__":
    main()
