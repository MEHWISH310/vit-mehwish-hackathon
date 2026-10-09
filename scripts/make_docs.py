"""Regenerate every number and figure used in README.md and docs/presentation.pdf from a real run.

    python -m src.engine.pipeline       # first (writes outputs/signals.csv, eval_metrics.json)
    python scripts/make_docs.py         # -> docs/results.json, docs/architecture.png, docs/figures/, docs/presentation.pdf
"""
import json
import os
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

from src.engine.pipeline import OUT  # noqa: E402
from src.modules.rebalancer import RebalanceConfig, run_backtest  # noqa: E402
from src.modules.stress import PRESETS, SCENARIOS, load_portfolio, run_event_stress, run_stress  # noqa: E402
from src.universe import DATA, load_universe  # noqa: E402

DOCS = ROOT / "docs"
FIG = DOCS / "figures"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
POS, NEG, GREY, INK, INK2, GRID = "#2a78d6", "#d03b3b", "#898781", "#0b0b0b", "#52514e", "#e1e0d9"
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "axes.axisbelow": True, "axes.titleweight": "bold", "axes.titlesize": 12,
                     "axes.titlelocation": "left", "text.parse_math": False})


# ------------------------------------------------------------------ numbers
def compute() -> dict:
    signals = pd.read_csv(OUT / "signals.csv")
    prices = pd.read_csv(DATA / "prices.csv")
    eng = json.loads((OUT / "eval_metrics.json").read_text())
    r = {"engine": eng, "corpus": {
        "signals": len(signals), "by_source": signals.source_type.value_counts().to_dict(),
        "by_scope": signals.scope.value_counts().to_dict(),
        "by_event": signals.event_type.value_counts().to_dict(),
        "impact_gt7": int((signals.impact > 7).sum()),
        "impact_gt7_by_event": signals[signals.impact > 7].event_type.value_counts().to_dict(),
        "span": [signals.timestamp.min(), signals.timestamp.max()]}}

    cfg = RebalanceConfig()
    bt = run_backtest(signals, prices, cfg)
    w = bt.weights
    r["module_a"] = {"config": vars(cfg), "metrics": bt.metrics, "days": len(w),
                     "weight_min": float(w.min().min()), "weight_max": float(w.max().max()),
                     "max_turnover": float(bt.turnover.max()),
                     "final_weights": w.iloc[-1].round(4).sort_values(ascending=False).to_dict(),
                     "sensitivity": []}
    for tilt in [0.0, 0.75, 1.5, 3.0]:
        for hl in [12, 36, 96]:
            m = run_backtest(signals, prices, RebalanceConfig(tilt=tilt, half_life_h=hl)).metrics["sentiment_index"]
            r["module_a"]["sensitivity"].append({"tilt": tilt, "half_life_h": hl, **m})
    # placebo: same machinery, sentiment signs shuffled across signals -> should not beat the benchmark reliably
    rng = np.random.default_rng(0)
    placebo = []
    for _ in range(20):
        sh = signals.assign(sentiment=rng.permutation(signals.sentiment.values))
        placebo.append(run_backtest(sh, prices, cfg).metrics["sentiment_index"]["total_return"])
    r["module_a"]["placebo_shuffled_sentiment_total_return"] = {
        "mean": float(np.mean(placebo)), "min": float(np.min(placebo)), "max": float(np.max(placebo)), "runs": 20}

    port = load_portfolio()
    st = run_event_stress(signals, port)
    summ = st.summary()
    worst = st.triggers[int(summ.pnl_pct.idxmin())]
    brief = run_stress(port, PRESETS["Brief example: equities -10%, rates +200bp"])
    # naive valuation on market value instead of risk notional: derivatives' hedge almost vanishes
    naive = run_stress(port.assign(risk_notional=port.market_value), PRESETS["Brief example: equities -10%, rates +200bp"])
    by_ev = summ.groupby("event_type").agg(n=("pnl", "size"), mean_pct=("pnl_pct", "mean"),
                                            worst_pct=("pnl_pct", "min"), mean_pnl=("pnl", "mean"))
    r["module_b"] = {
        "portfolio_value": float(port.market_value.sum()),
        "portfolio_mix": port.asset_type.value_counts().to_dict(),
        "mv_by_type": port.groupby("asset_type").market_value.sum().round(1).to_dict(),
        "risk_notional_by_type": port.groupby("asset_type").risk_notional.sum().round(1).to_dict(),
        "high_impact_signals": int((signals.impact > 7).sum()),
        "triggers": len(summ), "suppressed": st.suppressed,
        "by_event": by_ev.round(5).to_dict(orient="index"),
        "worst": {**worst.summary(), "shock": vars(worst.shock),
                  "by_type": worst.by_asset_type.pnl.round(2).to_dict()},
        "brief_example": {"pnl": brief.pnl, "pnl_pct": brief.pnl_pct, "value_after": brief.value_after,
                          "by_type": brief.by_asset_type.pnl.round(2).to_dict(),
                          "rate_pnl": float(brief.assets.rate_pnl.sum()),
                          "equity_pnl": float(brief.assets.equity_pnl.sum())},
        "brief_example_naive_mv": {"pnl": naive.pnl, "pnl_pct": naive.pnl_pct,
                                   "derivative_pnl": float(naive.by_asset_type.pnl.get("Derivative", 0))},
        "scenarios": {k: {**vars(v.shock), "adverse_only": v.adverse_only, "rationale": v.rationale}
                      for k, v in SCENARIOS.items()},
    }
    return r, bt, st, worst, brief


# ------------------------------------------------------------------ architecture
def box(ax, x, y, w, h, title, body="", fc="#ffffff", ec="#c3c2b7", tc=INK):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=fc, ec=ec, lw=1.4))
    ax.text(x + w / 2, y + h - 0.2, title, ha="center", va="top", fontsize=11, fontweight="bold", color=tc)
    if body:
        ax.text(x + w / 2, y + h - 0.55, body, ha="center", va="top", fontsize=8.6, color=INK2, linespacing=1.45)


def arrow(ax, a, b, label="", color=GREY):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=14, color=color, lw=1.5,
                                 connectionstyle="arc3,rad=0"))
    if label:
        ax.text((a[0] + b[0]) / 2, (a[1] + b[1]) / 2 + 0.12, label, ha="center", fontsize=8, color=INK2,
                bbox=dict(fc="white", ec="none", pad=1))


def architecture(path):
    fig, ax = plt.subplots(figsize=(16.5, 7.6))
    ax.set_xlim(0, 16.5)
    ax.set_ylim(0, 7.6)
    ax.axis("off")
    ax.text(0.2, 7.45, "NLP Risk Engine: architecture and data flow", fontsize=16, fontweight="bold", color=INK, va="top")
    blue, orange, green, grey = "#e8f1fc", "#fdeee6", "#e6f6ef", "#f4f4f2"
    # sources
    box(ax, 0.2, 4.7, 2.7, 2.0, "Source 1: News", "data/news_articles.csv\n1,212 articles\n(id, timestamp, ticker?, body)")
    box(ax, 0.2, 2.2, 2.7, 2.0, "Source 2: Social", "data/social_posts.csv\n2,762 X-style posts\n(RT, @, #, $cashtags, emoji)")
    # ingestion + engine
    box(ax, 3.4, 2.2, 2.8, 4.5, "Ingestion", "src/engine/ingest.py\n\nunify both schemas\nclean RT / URL / @ / #\nentity linking\n(cashtag, name, alias)\n"
        "mask company names\nscope: company | market", fc=blue)
    box(ax, 6.7, 2.2, 3.1, 4.5, "NLP Risk Engine", "src/engine/pipeline.py\n\nSentiment -1..1\n70% finance-tuned VADER\n+ 30% TF-IDF ridge\n\n"
        "Event type, 8 labels\nTF-IDF + logistic reg.\n+ keyword fallback\n\nImpact 1..10: TF-IDF ridge\n5-fold out-of-fold scoring", fc=blue)
    # outputs
    box(ax, 10.3, 5.3, 2.6, 1.4, "Signals file", "outputs/signals.csv\noutputs/signals.jsonl")
    box(ax, 10.3, 3.55, 2.6, 1.4, "REST API (FastAPI)", "/signals  /sentiment/{t}\n/analyze  /health")
    box(ax, 10.3, 1.5, 2.6, 1.7, "SignalBus (pub/sub)", "subscribe(scope, min_impact,\nevent_types); replay in\ntime order")
    # consumers
    box(ax, 13.6, 4.9, 2.6, 1.8, "Module A", "Index rebalancer\n15 stocks; decay, tilt,\n[2%, 15%] caps, turnover cap", fc=orange)
    box(ax, 13.6, 2.6, 2.6, 1.8, "Module B", "Stress tester, 40-asset\nbanking book; scenario\nshocks x impact / 10", fc=green)
    box(ax, 13.6, 0.2, 2.6, 1.85, "Dashboard", "app.py (Streamlit)\nsignals + live scoring,\nweights / NAV, stress P&L")
    box(ax, 0.2, 0.2, 9.6, 1.2, "Synthetic data generator",
        "scripts/generate_data.py -> text, prices (data/prices.csv), portfolio (data/portfolio.csv);\n"
        "handwritten eval sets in data/eval/", fc=grey)
    arrow(ax, (2.9, 5.7), (3.4, 5.7))
    arrow(ax, (2.9, 3.2), (3.4, 3.2))
    arrow(ax, (6.2, 4.45), (6.7, 4.45))
    arrow(ax, (9.8, 5.6), (10.3, 6.0))
    arrow(ax, (9.8, 4.25), (10.3, 4.25))
    arrow(ax, (9.8, 2.9), (10.3, 2.35))
    arrow(ax, (12.9, 2.9), (13.6, 5.3), color=SERIES[1])
    arrow(ax, (12.9, 2.2), (13.6, 3.0), color=SERIES[2])
    ax.text(10.35, 1.2, "orange: sentiment -> A", fontsize=8.5, color="#b84a1c", ha="left")
    ax.text(10.35, 0.88, "green: event + impact > 7 -> B", fontsize=8.5, color="#0e7a52", ha="left")
    arrow(ax, (14.9, 2.6), (14.9, 2.05))
    ax.annotate("", xy=(16.2, 1.4), xytext=(16.2, 5.3),
                arrowprops=dict(arrowstyle="-|>", color=GREY, lw=1.5, connectionstyle="arc3,rad=-0.25"))
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ------------------------------------------------------------------ result figures
def fig_nav(bt, path):
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.plot(bt.nav.index, bt.nav.sentiment_index, color=SERIES[0], lw=2, label="Sentiment index")
    ax.plot(bt.nav.index, bt.nav.equal_weight, color=SERIES[1], lw=2, label="Equal-weight benchmark")
    for col, c in [("sentiment_index", SERIES[0]), ("equal_weight", SERIES[1])]:
        ax.annotate(f"{bt.nav[col].iloc[-1]:.1f}", (bt.nav.index[-1], bt.nav[col].iloc[-1]), xytext=(4, 0),
                    textcoords="offset points", va="center", fontsize=9, color=INK2)
    ax.set_title("NAV, start = 100 (65 trading days, synthetic prices)")
    ax.legend(frameon=False, loc="lower left", fontsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def fig_weights(bt, path):
    u = load_universe().set_index("ticker").sector
    sectors = list(dict.fromkeys(u[bt.weights.columns]))
    order = sorted(bt.weights.columns, key=lambda t: (sectors.index(u[t]), t))
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.stackplot(bt.weights.index, bt.weights[order].T.values,
                 colors=[SERIES[sectors.index(u[t])] for t in order], edgecolor="white", linewidth=0.6)
    ax.set_ylim(0, 1)
    ax.set_yticks([0, .25, .5, .75, 1], ["0%", "25%", "50%", "75%", "100%"])
    ax.set_title("Index weights over time (bands = stocks, colour = sector)")
    from matplotlib.patches import Patch
    ax.legend([Patch(color=SERIES[i]) for i in range(len(sectors))], sectors, frameon=False, fontsize=8, ncol=6,
              loc="upper center", bbox_to_anchor=(0.5, -0.1), handlelength=1, columnspacing=1)
    ax.grid(False)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def fig_stress(worst, brief, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, res, title in [(axes[0], worst, "Worst triggered event"), (axes[1], brief, "Brief example: equities -10%, rates +200bp")]:
        a = res.assets
        parts = [a.rate_pnl.sum(), a.spread_pnl.sum(), a.equity_pnl.sum()]
        labels = ["Before", "Rates", "Spreads", "Equity", "After"]
        level = res.value_before
        ax.bar(0, level, color=GREY, width=0.6)
        for i, p in enumerate(parts, 1):
            ax.bar(i, p, bottom=level, color=POS if p >= 0 else NEG, width=0.6)
            ax.text(i, level + max(p, 0) + 4, f"{p:+.1f}", ha="center", fontsize=9, color=INK2)
            level += p
        ax.bar(4, res.value_after, color=GREY, width=0.6)
        ax.text(0, res.value_before + 4, f"{res.value_before:,.1f}", ha="center", fontsize=9, color=INK2)
        ax.text(4, res.value_after + 4, f"{res.value_after:,.1f}\n({res.pnl_pct:+.2%})", ha="center", fontsize=9, color=INK2)
        lo = min(res.value_before, res.value_after) + min(0, min(parts))
        ax.set_ylim(lo - 60, res.value_before + 45)
        ax.set_xticks(range(5), labels)
        ax.set_title(title + " ($m)")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


# ------------------------------------------------------------------ deck
W, H = 13.333, 7.5


def slide(pdf, n, title, draw):
    fig = plt.figure(figsize=(W, H))
    fig.patch.set_facecolor("white")
    fig.text(0.045, 0.92, title, fontsize=24, fontweight="bold", color=INK, va="center")
    fig.add_artist(plt.Line2D([0.045, 0.955], [0.875, 0.875], color=SERIES[0], lw=2.5))
    fig.text(0.045, 0.03, "S&P Global & Crisil Campus Hackathon 2026  |  Mehwish, VIT  |  all data synthetic",
             fontsize=9, color=GREY)
    fig.text(0.955, 0.03, f"{n} / 7", fontsize=9, color=GREY, ha="right")
    draw(fig)
    pdf.savefig(fig)
    if os.environ.get("SLIDE_PNG_DIR"):  # optional per-slide previews, for checking layout
        fig.savefig(Path(os.environ["SLIDE_PNG_DIR"]) / f"slide{n}.png", dpi=80)
    plt.close(fig)


def bullets(fig, x, y, items, size=13.5, width=60, gap=0.012):
    for it in items:
        sub = it.startswith("  ")
        txt = textwrap.fill(it.strip(), width - (4 if sub else 0))
        fig.text(x + (0.02 if sub else 0), y, ("- " if sub else "• ") + txt, fontsize=size - (1.5 if sub else 0),
                 color=INK2 if sub else INK, va="top", linespacing=1.35)
        y -= (0.042 if sub else 0.048) * (txt.count("\n") + 1) * size / 13.5 + gap
    return y


def image(fig, path, rect):
    ax = fig.add_axes(rect)
    ax.imshow(plt.imread(path))
    ax.axis("off")


def stat(fig, x, y, value, label, color=INK):
    fig.text(x, y, value, fontsize=26, fontweight="bold", color=color, va="top")
    fig.text(x, y - 0.075, label, fontsize=10.5, color=INK2, va="top")


def deck(r, path):
    e, c, a, b = r["engine"], r["corpus"], r["module_a"], r["module_b"]
    ma, mb = a["metrics"]["sentiment_index"], a["metrics"]["equal_weight"]
    oof, ho = e["synthetic_out_of_fold"], e["handwritten_holdout_20"]
    with PdfPages(path) as pdf:
        def s1(fig):
            fig.text(0.045, 0.80, "Turning news & social chatter into structured risk signals", fontsize=17, color=INK2)
            fig.text(0.045, 0.73, "Candidate: Mehwish   |   College: VIT   |   Solo submission", fontsize=13, color=INK2)
            fig.text(0.045, 0.64, "Problem", fontsize=15, fontweight="bold", color=INK)
            bullets(fig, 0.045, 0.59, [
                "Market-moving information arrives as unstructured text (news, X posts) faster than analysts can read it.",
                "Risk and portfolio teams need it as numbers they can act on: how positive/negative, what kind of event, how severe.",
            ], size=13, width=62)
            fig.text(0.53, 0.64, "Approach", fontsize=15, fontweight="bold", color=INK)
            bullets(fig, 0.53, 0.59, [
                "One NLP Risk Engine: 2 sources in, one signal per text out: sentiment (-1..1), event type, impact (1..10).",
                "Signals published via file, REST API and an in-process pub/sub bus.",
                "Both downstream modules built on the bus:",
                "  A: sentiment-tilted index of 15 S&P 100 stocks",
                "  B: event-triggered stress test of a 40-asset wholesale banking book",
                "One Streamlit dashboard for the live demo.",
            ], size=13, width=58)
        slide(pdf, 1, "NLP Risk Engine + Index Rebalancer + Stress Tester", s1)

        slide(pdf, 2, "System design and data flow",
              lambda fig: image(fig, DOCS / "architecture.png", [0.04, 0.08, 0.92, 0.77]))

        def s3(fig):
            bullets(fig, 0.045, 0.82, [
                f"Ingestion: {c['by_source']['news']:,} news + {c['by_source']['social']:,} social texts; social cleaned (RT, URLs, @, #); "
                f"entity linking by cashtag/name/alias -> {c['by_scope']['company']:,} company + {c['by_scope']['market']:,} market-wide signals.",
                "Company names masked to 'company' so models learn language, not 'AAPL is usually positive'.",
                "Sentiment: finance-tuned VADER lexicon blended with TF-IDF ridge. Event: TF-IDF + logistic regression. "
                "Impact: regressor on text + source type.",
                "Every signal is scored out-of-fold (5-fold), so modules never consume memorised training rows.",
                "Tech: Python 3.13, pandas, scikit-learn, vaderSentiment, FastAPI, Streamlit, Plotly; 27 pytest tests.",
            ], size=12.5, width=60)
            tbl = [["Metric", "Synthetic (OOF)", "Holdout (20)"],
                   ["Sentiment Pearson r", f"{oof['sentiment']['pearson_r']:.3f}", f"{ho['sentiment_blend']['pearson_r']:.3f}"],
                   ["Sentiment 3-class acc", f"{oof['sentiment']['3class_acc']:.3f}", f"{ho['sentiment_blend']['3class_acc']:.3f}"],
                   ["Event accuracy", f"{oof['event_accuracy']:.3f}", f"{ho['event_accuracy']:.3f}"],
                   ["Impact MAE (1-10)", f"{oof['impact_mae']:.2f}", f"{ho['impact_mae']:.2f}"],
                   ["Lexicon-only 3-class acc", "-", f"{ho['sentiment_lexicon_only']['3class_acc']:.3f}"]]
            ax = fig.add_axes([0.56, 0.50, 0.40, 0.33])
            ax.axis("off")
            t = ax.table(cellText=tbl[1:], colLabels=tbl[0], loc="upper center", cellLoc="center", colWidths=[.4, .3, .3])
            t.auto_set_font_size(False)
            t.set_fontsize(11.5)
            t.scale(1, 2.0)
            for (i, j), cell in t.get_celld().items():
                cell.set_edgecolor(GRID)
                if i == 0:
                    cell.set_text_props(fontweight="bold", color=INK)
                    cell.set_facecolor("#f4f4f2")
            fig.text(0.56, 0.47, textwrap.fill(
                "Honest read: synthetic scores are high because test text shares the generator's templates. The 20 "
                "handwritten holdout texts (never tuned on) are the real generalisation check: event 0.90, sentiment "
                f"r {ho['sentiment_blend']['pearson_r']:.2f}. On them the plain lexicon (3-class "
                f"{ho['sentiment_lexicon_only']['3class_acc']:.2f}) beats the blend ({ho['sentiment_blend']['3class_acc']:.2f}): "
                "the ML part overfits template wording.", 62), fontsize=10.5, color=INK2, va="top", linespacing=1.4)
        slide(pdf, 3, "Implementation: the NLP Risk Engine", s3)

        def s4(fig):
            image(fig, FIG / "weights.png", [0.03, 0.40, 0.47, 0.46])
            image(fig, FIG / "nav.png", [0.51, 0.40, 0.47, 0.46])
            stat(fig, 0.05, 0.36, f"{ma['total_return']:+.1%}", f"index total return (EW {mb['total_return']:+.1%})", POS)
            stat(fig, 0.25, 0.36, f"{ma['sharpe']:.2f}", f"Sharpe (EW {mb['sharpe']:.2f})")
            stat(fig, 0.40, 0.36, f"{ma['max_drawdown']:.1%}", f"max drawdown (EW {mb['max_drawdown']:.1%})")
            stat(fig, 0.58, 0.36, f"{ma['avg_daily_turnover']:.1%}", "avg one-way turnover / day")
            pl = a["placebo_shuffled_sentiment_total_return"]
            bullets(fig, 0.05, 0.20, [
                f"Rule: decayed (36h) impact x source weighted sentiment -> weight = 1/15 x exp(1.5 x score), bounded "
                f"[2%, 15%], max 20% turnover/day; set at close t, earns t+1 (no look-ahead). Placebo with shuffled "
                f"sentiment: mean {pl['mean']:+.1%} over {pl['runs']} runs.",
                "Caveat: synthetic prices were generated from the same events as the text, so this outperformance is "
                "partly circular: it proves the plumbing works, not that the signal has alpha.",
            ], size=11.5, width=118, gap=0.004)
        slide(pdf, 4, "Results, Module A: sentiment-driven index rebalancing", s4)

        def s5(fig):
            image(fig, FIG / "stress.png", [0.03, 0.38, 0.62, 0.48])
            w = b["worst"]
            bullets(fig, 0.67, 0.83, [
                f"{b['high_impact_signals']} signals with impact > 7 -> {b['triggers']} distinct stress tests "
                f"(24h cooldown per event+ticker; positive idiosyncratic news skipped).",
                f"Worst: {w['event_type']} on {w['ticker']} (impact {w['impact']:.0f}): "
                f"{w['pnl_pct']:+.2%} = -${-w['pnl']:,.1f}m on a ${b['portfolio_value']:,.0f}m book.",
                f"Brief's example (-10% equity, +200bp): {b['brief_example']['pnl_pct']:+.2%}; swaps and puts "
                f"offset ${b['brief_example']['by_type'].get('Derivative', 0):+.1f}m.",
                f"Valuing derivatives on market value instead of risk notional would show only "
                f"${b['brief_example_naive_mv']['derivative_pnl']:+.1f}m of hedge: the naive approach mis-states risk.",
            ], size=11.5, width=42, gap=0.006)
            rows = sorted(b["by_event"].items(), key=lambda kv: kv[1]["mean_pct"])
            tbl = [[k, f"{v['n']}", f"{v['mean_pct']:+.2%}", f"{v['worst_pct']:+.2%}"] for k, v in rows]
            ax = fig.add_axes([0.06, 0.07, 0.56, 0.29])
            ax.axis("off")
            t = ax.table(cellText=tbl, colLabels=["Event type", "Triggers", "Mean P&L", "Worst P&L"], loc="upper center",
                         cellLoc="center", colWidths=[.4, .2, .2, .2])
            t.auto_set_font_size(False)
            t.set_fontsize(10)
            t.scale(1, 1.3)
            for (i, j), cell in t.get_celld().items():
                cell.set_edgecolor(GRID)
                if i == 0:
                    cell.set_text_props(fontweight="bold")
                    cell.set_facecolor("#f4f4f2")
        slide(pdf, 5, "Results, Module B: event-triggered stress testing", s5)

        def s6(fig):
            fig.text(0.045, 0.82, "Domain impact", fontsize=15, fontweight="bold", color=INK)
            bullets(fig, 0.045, 0.77, [
                "Portfolio managers: a transparent, rules-based tilt that reacts within one close to news flow, with turnover and concentration limits a risk committee can sign off.",
                "Risk / treasury desks: every high-impact headline becomes an immediate, explainable 'what does this do to our book' number, broken down by rates, credit and equity and by asset type and sector.",
                "One engine, many consumers: the pub/sub contract lets new modules (limits monitoring, alerts) subscribe without touching the NLP code.",
            ], size=12.5, width=55)
            fig.text(0.53, 0.82, "Challenges", fontsize=15, fontweight="bold", color=INK)
            bullets(fig, 0.53, 0.77, [
                "No labelled real-time feed: built a synthetic generator plus 55 handwritten texts to measure generalisation honestly.",
                "Entity linking & leakage: masking company names so models read the language.",
                "Look-ahead: signals after 16:00 roll to the next rebalance; tested explicitly.",
                f"Alert storms: {b['high_impact_signals']} raw hits -> {b['triggers']} triggers via strict threshold, cooldown and adverse-only rules.",
                "Derivatives: P&L on risk notional, not ~0 market value; sign tests for swaps, CDS, puts.",
            ], size=12.5, width=55)
        slide(pdf, 6, "Domain impact and challenges", s6)

        def s7(fig):
            fig.text(0.045, 0.82, "Limitations (stated plainly)", fontsize=15, fontweight="bold", color=INK)
            bullets(fig, 0.045, 0.77, [
                "All data is synthetic. Text, prices and portfolio come from scripts/generate_data.py.",
                "Prices were generated from the same events as the text, so the Module A backtest is partly circular.",
                "Engine generalisation is measured on only 20 handwritten holdout texts.",
                "Impact learned template intensity words: 'War breaks out in the Middle East; oil spikes...' scores only 5.7.",
                "Stress model is first-order (duration / beta): no convexity, vega, correlations, rating migration or default loss.",
                "Scenario sizes are expert judgement, not calibrated to history.",
            ], size=12.5, width=55)
            fig.text(0.53, 0.82, "Next steps", fontsize=15, fontweight="bold", color=INK)
            bullets(fig, 0.53, 0.77, [
                "Plug in real feeds (GDELT, NewsAPI, Kaggle stock tweets) behind the same ingest schema.",
                "Fine-tune a finance transformer (e.g. FinBERT) for sentiment and events; compare vs the lexicon baseline.",
                "Evaluate Module A on real prices (yfinance), with transaction costs and an event-study of signal decay.",
                "Calibrate shocks to historical episodes; add convexity, correlated factor shocks and credit migration.",
                "Replace the in-process bus with Kafka; stream signals to the dashboard live.",
            ], size=12.5, width=55)
        slide(pdf, 7, "Limitations and next steps", s7)


def main():
    DOCS.mkdir(exist_ok=True)
    FIG.mkdir(exist_ok=True)
    r, bt, st, worst, brief = compute()
    (DOCS / "results.json").write_text(json.dumps(r, indent=2, default=str))
    architecture(DOCS / "architecture.png")
    fig_nav(bt, FIG / "nav.png")
    fig_weights(bt, FIG / "weights.png")
    fig_stress(worst, brief, FIG / "stress.png")
    deck(r, DOCS / "presentation.pdf")
    print(json.dumps({k: r[k] for k in ["corpus", "module_a", "module_b"]}, indent=1, default=str)[:6000])


if __name__ == "__main__":
    main()
