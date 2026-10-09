"""Risk-signal dashboard: the NLP engine's output and both downstream modules in one place.

    python -m src.engine.pipeline     # once: trains the engine, writes outputs/signals.csv
    streamlit run app.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
import plotly.express as px  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from src.engine.pipeline import MODEL_PATH, OUT  # noqa: E402
from src.modules.rebalancer import RebalanceConfig, run_backtest, top_drivers  # noqa: E402
from src.modules.stress import PRESETS, SCENARIOS, Shock, load_portfolio, run_event_stress, run_stress  # noqa: E402
from src.universe import DATA, load_universe  # noqa: E402

# reference categorical palette (fixed order), diverging pair and neutral midpoint
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
POS, NEG, MID = "#2a78d6", "#d03b3b", "#f0efec"
DIVERGING = [[0, NEG], [0.5, MID], [1, POS]]
SIGNALS_PATH = OUT / "signals.csv"

st.set_page_config(page_title="NLP Risk Engine", page_icon=":bar_chart:", layout="wide")


def fig_style(fig, height=360):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=40, b=10), hovermode="closest",
                      legend=dict(orientation="h", yanchor="top", y=-0.12, x=0))
    return fig


@st.cache_data
def load_signals(mtime: float) -> pd.DataFrame:
    s = pd.read_csv(SIGNALS_PATH, parse_dates=["timestamp"])
    s["ticker"] = s["ticker"].fillna("")
    return s


@st.cache_data
def load_prices() -> pd.DataFrame:
    return pd.read_csv(DATA / "prices.csv", parse_dates=["date"])


@st.cache_data
def backtest(mtime: float, half_life_h, tilt, max_weight, max_turnover):
    cfg = RebalanceConfig(half_life_h=half_life_h, tilt=tilt, max_weight=max_weight, max_turnover=max_turnover)
    return run_backtest(load_signals(mtime), load_prices(), cfg), cfg


@st.cache_data
def stress_run(mtime: float, threshold, cooldown_h):
    st_ = run_event_stress(load_signals(mtime), load_portfolio(), threshold=threshold, cooldown_h=cooldown_h)
    return st_.triggers, st_.suppressed


@st.cache_resource
def load_engine():
    from src.engine.pipeline import RiskEngine
    return RiskEngine.load()


st.title("NLP Risk Engine: news & social text to risk signals")
if not SIGNALS_PATH.exists():
    st.error("No engine output found at `outputs/signals.csv`. Generate it first (takes a few seconds):")
    st.code("python -m src.engine.pipeline", language="bash")
    st.stop()

mtime = SIGNALS_PATH.stat().st_mtime
signals = load_signals(mtime)
universe = load_universe()

with st.sidebar:
    st.header("Module B: stress trigger")
    threshold = st.slider("Impact threshold (trigger when impact >)", 5.0, 9.5, 7.0, 0.5)
    cooldown_h = st.slider("Cooldown per (event, ticker), hours", 0, 72, 24, 6)
    st.header("Module A: rebalancer")
    tilt = st.slider("Sentiment tilt", 0.0, 4.0, 1.5, 0.25)
    max_weight = st.slider("Max weight per stock", 0.08, 0.30, 0.15, 0.01)
    max_turnover = st.slider("Max one-way turnover per day", 0.02, 1.0, 0.20, 0.02)
    half_life_h = st.slider("Sentiment half-life, hours", 6, 168, 36, 6)
    st.caption("All data is synthetic (see README). Prices were generated from the same events as the text, "
               "so backtest performance is optimistic by construction.")

tab_engine, tab_a, tab_b = st.tabs(["Engine signals", "Module A: index rebalancer", "Module B: stress test"])

# ---------------------------------------------------------------- Engine
with tab_engine:
    c = st.columns(4)
    c[0].metric("Signals", f"{len(signals):,}")
    n_src = signals.source_type.value_counts()
    c[1].metric("News / social", f"{n_src.get('news', 0):,} / {n_src.get('social', 0):,}")
    c[2].metric("Company / market scope", f"{(signals.scope == 'company').sum():,} / {(signals.scope == 'market').sum():,}")
    c[3].metric(f"High impact (> {threshold:g})", f"{(signals.impact > threshold).sum():,}")

    left, right = st.columns(2)
    counts = signals.event_type.value_counts().sort_values()
    fig = go.Figure(go.Bar(x=counts.values, y=counts.index, orientation="h", marker_color=SERIES[0],
                           hovertemplate="%{y}: %{x} signals<extra></extra>"))
    left.plotly_chart(fig_style(fig.update_layout(title="Signals by event type")), width="stretch")

    by_t = (signals[signals.scope == "company"].groupby("ticker")
            .agg(sentiment=("sentiment", "mean"), n=("sentiment", "size")).sort_values("sentiment"))
    fig = go.Figure(go.Bar(x=by_t.sentiment, y=by_t.index, orientation="h",
                           marker_color=[POS if v >= 0 else NEG for v in by_t.sentiment], customdata=by_t.n,
                           hovertemplate="%{y}: mean sentiment %{x:.3f} over %{customdata} signals<extra></extra>"))
    right.plotly_chart(fig_style(fig.update_layout(title="Mean sentiment by ticker (whole period)")),
                       width="stretch")

    st.subheader("Recent signals")
    f1, f2, f3 = st.columns(3)
    ev_f = f1.multiselect("Event type", sorted(signals.event_type.unique()))
    tk_f = f2.multiselect("Ticker", sorted(t for t in signals.ticker.unique() if t))
    min_imp = f3.slider("Min impact", 1.0, 10.0, 1.0, 0.5)
    view = signals[signals.impact >= min_imp]
    if ev_f:
        view = view[view.event_type.isin(ev_f)]
    if tk_f:
        view = view[view.ticker.isin(tk_f)]
    st.dataframe(view.sort_values("timestamp", ascending=False).head(200)
                 [["timestamp", "source_type", "ticker", "sentiment", "event_type", "impact", "event_confidence",
                   "headline"]], width="stretch", hide_index=True)

    st.subheader("Try it live")
    txt = st.text_area("Paste a headline or post", "Goldman Sachs warns of covenant breach as losses mount, a major shock for credit markets")
    src = st.radio("Source type", ["news", "social"], horizontal=True)
    if st.button("Analyze"):
        if not MODEL_PATH.exists():
            st.warning("No trained model at `models/engine.joblib`. Run `python -m src.engine.pipeline` first.")
        else:
            from src.engine.ingest import from_raw_text
            docs = from_raw_text(txt, src)
            if docs.empty:
                st.warning("Text is empty after cleaning.")
            else:
                out = load_engine().analyze(docs).iloc[0]
                r = st.columns(4)
                r[0].metric("Sentiment", f"{out.sentiment:+.2f}")
                r[1].metric("Event", out.event_type, f"confidence {out.event_confidence:.0%}", delta_color="off")
                r[2].metric("Impact (1-10)", f"{out.impact:.1f}")
                r[3].metric("Ticker", out.ticker or "market-wide")
                sc = SCENARIOS.get(out.event_type)
                fires = out.impact > threshold and sc is not None and (not sc.adverse_only or out.sentiment < 0)
                st.info(f"Module B would {'TRIGGER a ' + out.event_type + ' stress test' if fires else 'not trigger'}"
                        f" (impact {out.impact:.1f} vs threshold {threshold:g}).")

# ---------------------------------------------------------------- Module A
with tab_a:
    res, cfg = backtest(mtime, half_life_h, tilt, max_weight, max_turnover)
    m, b = res.metrics["sentiment_index"], res.metrics["equal_weight"]
    c = st.columns(5)
    c[0].metric("Total return", f"{m['total_return']:+.2%}", f"{m['total_return'] - b['total_return']:+.2%} vs EW")
    c[1].metric("Sharpe (ann.)", f"{m['sharpe']:.2f}", f"{m['sharpe'] - b['sharpe']:+.2f} vs EW")
    c[2].metric("Ann. volatility", f"{m['ann_vol']:.2%}", f"{m['ann_vol'] - b['ann_vol']:+.2%} vs EW",
                delta_color="inverse")
    c[3].metric("Max drawdown", f"{m['max_drawdown']:.2%}", f"{m['max_drawdown'] - b['max_drawdown']:+.2%} vs EW")
    c[4].metric("Avg daily turnover", f"{m['avg_daily_turnover']:.1%}")

    sector = universe.set_index("ticker").sector
    sectors = list(dict.fromkeys(sector[res.weights.columns]))
    order = sorted(res.weights.columns, key=lambda t: (sectors.index(sector[t]), t))
    fig = go.Figure()
    shown = set()
    for t in order:
        sec = sector[t]
        fig.add_trace(go.Scatter(x=res.weights.index, y=res.weights[t], name=sec, legendgroup=sec,
                                 showlegend=sec not in shown, stackgroup="w", mode="lines",
                                 line=dict(width=1, color="white"),
                                 fillcolor=SERIES[sectors.index(sec)],
                                 hovertemplate=f"{t} ({sec}): %{{y:.1%}}<extra></extra>"))
        shown.add(sec)
    fig.update_layout(title="Index weights over time (one band per stock, coloured by sector; hover for ticker)",
                      yaxis=dict(tickformat=".0%", range=[0, 1]))
    st.plotly_chart(fig_style(fig, 420), width="stretch")

    left, right = st.columns(2)
    nav = res.nav.rename(columns={"sentiment_index": "Sentiment index", "equal_weight": "Equal-weight benchmark"})
    fig = go.Figure([go.Scatter(x=nav.index, y=nav[col], name=col, mode="lines", line=dict(width=2, color=SERIES[i]),
                                hovertemplate="%{y:.2f}") for i, col in enumerate(nav.columns)])
    fig.update_layout(title="NAV (start = 100)", hovermode="x unified")
    left.plotly_chart(fig_style(fig), width="stretch")

    pick = right.multiselect("Compare individual weights", order, default=list(
        (res.weights.iloc[-1] - res.weights.iloc[0]).abs().sort_values().index[-3:]), max_selections=6)
    fig = go.Figure([go.Scatter(x=res.weights.index, y=res.weights[t], name=t, mode="lines",
                                line=dict(width=2, color=SERIES[i])) for i, t in enumerate(pick)])
    fig.add_hline(y=1 / len(order), line_dash="dot", line_color="gray", annotation_text="equal weight")
    fig.update_layout(title="Selected stock weights", yaxis_tickformat=".0%", hovermode="x unified")
    right.plotly_chart(fig_style(fig, 300), width="stretch")

    fig = go.Figure(go.Heatmap(z=res.scores[order].T.values, x=res.scores.index, y=order, colorscale=DIVERGING,
                               zmid=0, zmin=-1, zmax=1, colorbar=dict(title="score"),
                               hovertemplate="%{y} %{x|%b %d}: %{z:.2f}<extra></extra>"))
    fig.update_layout(title="Decayed sentiment score at each close (red negative, blue positive)")
    st.plotly_chart(fig_style(fig, 440), width="stretch")

    st.subheader("Latest weight changes and why")
    lb = st.slider("Compare latest close with N trading days earlier", 1, 20, 5)
    w, s = res.weights, res.scores
    d_now, d_then = w.index[-1], w.index[-1 - lb]
    at = d_now + pd.Timedelta(hours=16)
    rows = []
    for t in order:
        drv = top_drivers(signals, t, at, cfg, n=1)
        rows.append({"ticker": t, "sector": sector[t], "weight": w.at[d_now, t],
                     "change": w.at[d_now, t] - w.at[d_then, t], "vs equal weight": w.at[d_now, t] - 1 / len(order),
                     "score": s.at[d_now, t],
                     "main driver": "" if drv.empty else
                     f"[{drv.event_type.iloc[0]}, sent {drv.sentiment.iloc[0]:+.2f}, impact {drv.impact.iloc[0]:.1f}] "
                     f"{drv.headline.iloc[0]}"})
    tbl = pd.DataFrame(rows).sort_values("change", ascending=False)
    st.caption(f"{d_then:%Y-%m-%d} to {d_now:%Y-%m-%d}. Main driver = the signal with the largest decayed contribution.")
    st.dataframe(tbl, hide_index=True, width="stretch", column_config={
        "weight": st.column_config.NumberColumn(format="percent"),
        "change": st.column_config.NumberColumn(format="percent"),
        "vs equal weight": st.column_config.NumberColumn(format="percent"),
        "score": st.column_config.NumberColumn(format="%.2f")})

# ---------------------------------------------------------------- Module B
with tab_b:
    portfolio = load_portfolio()
    triggers, suppressed = stress_run(mtime, threshold, cooldown_h)
    summ = pd.DataFrame([t.summary() for t in triggers])
    c = st.columns(4)
    c[0].metric("Portfolio value", f"${portfolio.market_value.sum():,.1f}m")
    c[1].metric("Assets", f"{len(portfolio)}")
    c[2].metric("Stress tests triggered", f"{len(summ)}")
    c[3].metric("Worst trigger", f"{summ.pnl_pct.min():.2%}" if len(summ) else "n/a")
    st.caption("Portfolio: " + ", ".join(f"{v} {k}s" for k, v in portfolio.asset_type.value_counts().items())
               + ". Values in $m; P&L is computed on risk notional.")
    st.caption(f"Signals ignored: {suppressed['not_adverse']} not adverse (positive news on an idiosyncratic event), "
               f"{suppressed['cooldown']} inside cooldown, {suppressed['below_threshold']} at or below threshold.")

    if summ.empty:
        st.info("No signal exceeds the threshold. Lower it in the sidebar.")
    else:
        types = st.multiselect("Event types", sorted(summ.event_type.unique()), default=sorted(summ.event_type.unique()))
        idx = summ.index[summ.event_type.isin(types)]
        sub = summ.loc[idx]
        fig = go.Figure(go.Scatter(
            x=sub.timestamp, y=sub.event_type, mode="markers",
            marker=dict(size=8 + 60 * sub.pnl_pct.abs(), color=SERIES[0], opacity=0.7, line=dict(width=1, color="white")),
            customdata=sub[["ticker", "impact", "pnl_pct", "headline"]].values,
            hovertemplate="%{x|%b %d %H:%M} %{y} %{customdata[0]}<br>impact %{customdata[1]:.1f}, "
                          "P&L %{customdata[2]:.2%}<br>%{customdata[3]}<extra></extra>"))
        fig.update_layout(title="Triggered stress tests over time (marker size = portfolio loss)")
        st.plotly_chart(fig_style(fig, 320), width="stretch")

        if len(sub):
            labels = {i: f"{pd.Timestamp(summ.at[i, 'timestamp']):%Y-%m-%d %H:%M} | {summ.at[i, 'event_type']} "
                         f"{summ.at[i, 'ticker'] or '(market)'} | impact {summ.at[i, 'impact']:.1f} | "
                         f"{summ.at[i, 'pnl_pct']:+.2%}" for i in sub.index}
            default = list(sub.index).index(sub.pnl_pct.idxmin())
            sel = st.selectbox("Pick a triggered event (default: worst)", list(sub.index), index=default,
                               format_func=labels.get)
            res = triggers[sel]
            sig = res.signal
            st.markdown(f"**{sig['event_type']}** | {sig.get('ticker') or 'market-wide'} | impact **{sig['impact']:.1f}** | "
                        f"sentiment {sig['sentiment']:+.2f} | {sig['source_type']} | {pd.Timestamp(sig['timestamp'])}")
            st.markdown(f"> {sig['headline']}")
            sh = res.shock
            st.caption(f"Applied shock (scenario x impact/10): equity {sh.equity_pct:+.1f}%, rates {sh.rate_bp:+.0f}bp, "
                       f"spreads {sh.spread_bp:+.0f}bp" + (f"; {sh.ticker}: extra equity {sh.idio_equity_pct:+.1f}%, "
                                                           f"extra spread {sh.idio_spread_bp:+.0f}bp" if sh.ticker else "")
                       + f". Rationale: {SCENARIOS[res.scenario].rationale}.")

    def show_result(res, key):
        c = st.columns(3)
        c[0].metric("Value before", f"${res.value_before:,.1f}m")
        c[1].metric("Value after", f"${res.value_after:,.1f}m", f"{res.pnl:+,.1f}m ({res.pnl_pct:+.2%})")
        c[2].metric("Loss", f"${-res.pnl:,.1f}m" if res.pnl < 0 else ("none" if res.pnl == 0 else "none (net gain)"))
        left, right = st.columns(2)
        a = res.assets
        parts = [a.rate_pnl.sum(), a.spread_pnl.sum(), a.equity_pnl.sum()]
        fig = go.Figure(go.Waterfall(
            x=["Before", "Rates", "Credit spreads", "Equity", "After"],
            measure=["absolute", "relative", "relative", "relative", "total"],
            y=[res.value_before, *parts, 0], text=[f"{res.value_before:,.1f}", *[f"{p:+,.1f}" for p in parts],
                                                   f"{res.value_after:,.1f}"],
            increasing=dict(marker_color=POS), decreasing=dict(marker_color=NEG), totals=dict(marker_color="#898781"),
            connector=dict(line=dict(color="#c3c2b7", width=1))))
        lo = min(res.value_before, res.value_after) + min(0, min(parts))
        fig.update_layout(title="Portfolio value: before to after ($m)",
                          yaxis=dict(range=[lo * 0.97, max(res.value_before, res.value_after) * 1.01]))
        left.plotly_chart(fig_style(fig), width="stretch", key=f"wf{key}")

        bt = res.by_asset_type
        fig = go.Figure(go.Bar(x=bt.pnl, y=bt.index, orientation="h",
                               marker_color=[POS if v >= 0 else NEG for v in bt.pnl],
                               hovertemplate="%{y}: %{x:+,.2f}m<extra></extra>"))
        right.plotly_chart(fig_style(fig.update_layout(title="P&L by asset type ($m, hedges in blue)"), 300),
                           width="stretch", key=f"bt{key}")
        left, right = st.columns(2)
        bs = res.by_sector
        fig = go.Figure(go.Bar(x=bs.pnl, y=bs.index, orientation="h",
                               marker_color=[POS if v >= 0 else NEG for v in bs.pnl],
                               hovertemplate="%{y}: %{x:+,.2f}m<extra></extra>"))
        left.plotly_chart(fig_style(fig.update_layout(title="P&L by sector ($m)"), 340),
                          width="stretch", key=f"bs{key}")
        right.markdown("**Top 10 losing assets**")
        right.dataframe(res.top_losers(10).round(2), hide_index=True, width="stretch")

    if not summ.empty and len(sub):
        show_result(res, "event")

    st.divider()
    st.subheader("Custom shock (what-if)")
    preset = st.selectbox("Start from a preset", ["(none)", *PRESETS], index=1)
    p = PRESETS.get(preset, Shock())
    c = st.columns(4)
    eq = c[0].slider("Equity shock %", -40.0, 20.0, float(p.equity_pct), 1.0, key=f"eq{preset}")
    rt = c[1].slider("Rate shock (bp)", -300, 300, int(p.rate_bp), 10, key=f"rt{preset}")
    sp = c[2].slider("Credit spread shock (bp)", -100, 500, int(p.spread_bp), 10, key=f"sp{preset}")
    tk = c[3].selectbox("Idiosyncratic name (optional)", ["", *sorted(t for t in portfolio.linked_ticker.unique() if t)])
    idio = st.slider(f"Extra equity shock for {tk} %", -50.0, 0.0, -15.0, 1.0) if tk else 0.0
    show_result(run_stress(portfolio, Shock(eq, rt, sp, idio_equity_pct=idio,
                                            idio_spread_bp=-idio * 20 if tk else 0, ticker=tk)), "custom")
    st.caption("Simplified first-order model: P&L = -duration x dy x notional (rates), -spread duration x ds x rating "
               "multiplier x notional (credit), beta x equity move x notional (equity). No convexity, vega, correlations, "
               "rating migration or default losses. Idiosyncratic spread widening is set to 20bp per 1% of stock drop.")
