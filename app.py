"""Risk-signal dashboard: the NLP engine's output and both downstream modules in one place.

    python -m src.engine.pipeline     # once: trains the engine, writes outputs/signals.csv
    python -m src.engine.real         # optional: real-data mode (outputs/real_signals.csv)
    streamlit run app.py
"""
import html
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from src.engine.pipeline import MODEL_PATH, OUT  # noqa: E402
from src.engine.real import REAL, REAL_MODEL_PATH  # noqa: E402
from src.modules.rebalancer import RebalanceConfig, run_backtest, top_drivers  # noqa: E402
from src.modules.stress import PRESETS, SCENARIOS, Shock, load_portfolio, run_event_stress, run_stress  # noqa: E402
from src.universe import DATA, load_universe  # noqa: E402

from streamlit import config as st_config  # noqa: E402

# design tokens per theme: flat colours only (no gradients), shared by the CSS, Streamlit's theme and Plotly
THEMES = {
    "Light": dict(base="light", bg="#f4f5f7", surface="#ffffff", surface2="#f0f2f5", border="#e2e5ea",
                  text="#0f1419", muted="#566070", faint="#8b93a1", accent="#2563eb", accent_soft="#e8effd",
                  pos="#2563eb", pos_soft="#e8effd", neg="#d92d20", neg_soft="#fdecea", warn="#b86e00",
                  warn_soft="#fdf3e1", ok="#12805c", ok_soft="#e5f4ed", neutral_soft="#eef0f3", grid="#eceef2",
                  mid="#eeeeec", series=["#2563eb", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300",
                                         "#4a3aa7", "#e34948"]),
    "Dark": dict(base="dark", bg="#0d1117", surface="#151b23", surface2="#1c232d", border="#29313d",
                 text="#e6edf3", muted="#9ba6b4", faint="#6e7886", accent="#4f8ff7", accent_soft="#17294a",
                 pos="#4f8ff7", pos_soft="#17294a", neg="#f2645a", neg_soft="#3b1c1b", warn="#e5a23a",
                 warn_soft="#3a2b13", ok="#3fb984", ok_soft="#123222", neutral_soft="#222a35", grid="#222a35",
                 mid="#2a313c", series=["#4f8ff7", "#f08a4b", "#2fc28d", "#f2b632", "#f08cb4", "#5cbf5c",
                                        "#9d8cf2", "#f2645a"]),
}
PLURAL = {"Loan": "loans", "Bond": "bonds", "Derivative": "derivatives", "Equity": "equities"}
SYNTHETIC, REAL_MODE = "Synthetic (generated text + prices)", "Real (Google News headlines + Yahoo Finance prices)"
SOURCES = {  # signals file, prices file, trained engine, command that creates them
    SYNTHETIC: (OUT / "signals.csv", DATA / "prices.csv", MODEL_PATH, "python -m src.engine.pipeline"),
    REAL_MODE: (OUT / "real_signals.csv", REAL / "prices_yahoo.csv", REAL_MODEL_PATH, "python -m src.engine.real"),
}
EXAMPLES = {  # one-click demo inputs: (text, source type)
    "Credit shock": ("Goldman Sachs warns of covenant breach as losses mount, a major shock for credit markets", "news"),
    "Plain downgrade": ("Moody's downgrades Goldman Sachs to junk as trading losses mount", "news"),
    "Hawkish Fed": ("Fed signals surprise rate hike as inflation spikes to a 20-year high", "news"),
    "Ceasefire": ("Ceasefire in the Middle East lifts global risk appetite", "news"),
    "Bullish tweet": ("$NVDA smashes earnings, raises guidance \U0001F680", "social"),
}
DEFAULT_TEXT = EXAMPLES["Credit shock"][0]

st.set_page_config(page_title="NLP Risk Engine", page_icon=":bar_chart:", layout="wide")
# the top-bar toggle copies its value into a plain key: the theme rerun below would otherwise drop the widget's state
T = THEMES["Dark" if st.session_state.get("dark") else "Light"]
SERIES = T["series"]
POS, NEG, MID = T["pos"], T["neg"], T["mid"]
DIVERGING = [[0, NEG], [0.5, MID], [1, POS]]

# Streamlit sends its theme when a run starts, so switch the config and rerun once to restyle the native widgets
_native = {"base": T["base"], "primaryColor": T["accent"], "backgroundColor": T["bg"],
           "secondaryBackgroundColor": T["surface"], "textColor": T["text"], "borderColor": T["border"],
           "linkColor": T["accent"], "dataframeHeaderBackgroundColor": T["surface2"]}
if any(st_config.get_option(f"theme.{k}") != v for k, v in _native.items()):
    for k, v in _native.items():
        st_config.set_option(f"theme.{k}", v)
    st.rerun()

_vars = "".join(f"--{k.replace('_', '-')}:{v};" for k, v in T.items() if isinstance(v, str) and v.startswith("#"))
_svg = "data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'>{}</svg>"
SUN = _svg.format("<g fill='none' stroke='%23f59e0b' stroke-width='2.4' stroke-linecap='round'><circle cx='12' cy='12' "
                  "r='4'/><path d='M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4"
                  "M17.7 6.3l1.4-1.4'/></g>")
MOON = _svg.format("<path fill='%23fbbf24' d='M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z'/>")
TOGGLE = ".st-key-dark_mode label"  # the theme switch: track = first div, knob = the div inside it
st.markdown(f"""
<style>
:root {{{_vars} --neu: var(--faint);}}
.block-container {{padding-top: 4rem; max-width: 1400px;}}
[data-testid="stHeader"] {{background: transparent;}}
{TOGGLE} > div:first-child {{width: 52px !important; height: 28px !important; padding: 3px !important;
  box-sizing: border-box; background: var(--surface2) !important; box-shadow: inset 0 0 0 1px var(--border);}}
{TOGGLE} > div:first-child > div {{width: 22px !important; height: 22px !important; background: #fff url("{SUN}")
  center / 14px no-repeat !important; box-shadow: 0 1px 3px rgba(0,0,0,.25);}}
{TOGGLE}:has(input:checked) > div:first-child {{background: var(--accent) !important; box-shadow: none;}}
{TOGGLE}:has(input:checked) > div:first-child > div {{transform: translateX(24px) !important;
  background: #0d1117 url("{MOON}") center / 13px no-repeat !important;}}
.stApp {{background: var(--bg);}}
h2, h3 {{letter-spacing: -0.015em; color: var(--text);}}
h3 {{font-size: 1.2rem !important; font-weight: 650 !important; padding-bottom: .1rem !important;}}
[data-testid="stCaptionContainer"] p {{color: var(--muted);}}
[data-testid="stMetric"] {{background: var(--surface); border: 1px solid var(--border); border-radius: 12px;
  padding: 14px 16px;}}
[data-testid="stMetricLabel"] p {{color: var(--muted); font-size: .78rem; font-weight: 600; text-transform: uppercase;
  letter-spacing: .04em;}}
[data-testid="stMetricValue"] {{font-variant-numeric: tabular-nums; font-weight: 650; font-size: 1.75rem;}}
[data-testid="stTabs"] [role="tablist"] {{gap: 4px; border-bottom: 1px solid var(--border);}}
[data-testid="stTabs"] button {{padding: 8px 14px;}}
[data-testid="stTabs"] button p {{font-size: .95rem; font-weight: 600;}}
[data-testid="stTabs"] button[aria-selected="false"] p {{color: var(--muted);}}
[data-testid="stSidebar"] {{border-right: 1px solid var(--border);}}
[data-testid="stPlotlyChart"] {{background: var(--surface); border: 1px solid var(--border); border-radius: 12px;
  padding: 6px 4px 2px; overflow: hidden;}}
[data-testid="stExpander"] details {{background: var(--surface); border-color: var(--border); border-radius: 12px;}}
.topbar {{display: flex; align-items: center; gap: 10px; min-height: 40px;}}
.logo {{width: 30px; height: 30px; border-radius: 8px; background: var(--accent); color: #fff; display: flex;
  align-items: center; justify-content: center; font-weight: 800; font-size: .78rem;}}
.brand {{font-weight: 700; font-size: 1rem; color: var(--text);}}
.brand-sub {{color: var(--faint); font-size: .85rem;}}
.hero {{background: var(--surface); border: 1px solid var(--border); border-left: 5px solid var(--accent);
  border-radius: 14px; padding: 22px 26px 18px; margin: 6px 0 18px;}}
.hero .eyebrow {{font-size: .72rem; font-weight: 700; letter-spacing: .1em; text-transform: uppercase;
  color: var(--accent);}}
.hero h1 {{color: var(--text); font-size: 1.9rem; font-weight: 750; margin: 4px 0 6px; padding: 0;
  line-height: 1.15; letter-spacing: -0.02em;}}
.hero p {{color: var(--muted); margin: 0 0 14px; font-size: 1rem; max-width: 900px; line-height: 1.5;}}
.chip {{display: inline-flex; align-items: center; gap: 6px; padding: 3px 10px; border-radius: 999px;
  font-size: .78rem; font-weight: 600; margin: 0 6px 6px 0; background: var(--surface2); color: var(--muted);
  border: 1px solid var(--border);}}
.chip .dot {{width: 7px; height: 7px; border-radius: 50%; background: currentColor;}}
.chip.real {{background: var(--ok-soft); color: var(--ok); border-color: transparent;}}
.chip.syn {{background: var(--warn-soft); color: var(--warn); border-color: transparent;}}
.card {{background: var(--surface); border: 1px solid var(--border); border-radius: 14px; padding: 18px 20px;
  height: 100%; min-height: 330px;}}
.card .step {{display: flex; align-items: center; gap: 8px; font-size: .72rem; font-weight: 700; color: var(--accent);
  letter-spacing: .08em; text-transform: uppercase;}}
.card .num {{width: 22px; height: 22px; border-radius: 6px; background: var(--accent-soft); color: var(--accent);
  display: inline-flex; align-items: center; justify-content: center; font-size: .75rem; letter-spacing: 0;}}
.card h4 {{margin: 10px 0 8px; font-size: 1.08rem; font-weight: 650; color: var(--text); padding: 0;}}
.card .rule {{height: 1px; background: var(--border); margin: 12px 0;}}
.big {{font-size: 1.9rem; font-weight: 700; line-height: 1.1; margin: 4px 0; color: var(--text);
  font-variant-numeric: tabular-nums; letter-spacing: -0.02em;}}
.muted {{color: var(--muted); font-size: .9rem; line-height: 1.5;}}
.feed {{background: var(--surface); border: 1px solid var(--border); border-radius: 14px; padding: 2px 16px;}}
.feed-row {{display: flex; gap: 12px; align-items: center; padding: 11px 0; border-bottom: 1px solid var(--border);}}
.feed-row:last-child {{border-bottom: none;}}
.feed-time {{color: var(--faint); font-size: .78rem; min-width: 88px; font-variant-numeric: tabular-nums;}}
.feed-head {{flex: 1; font-size: .92rem; color: var(--text); line-height: 1.45;}}
.tag {{display: inline-block; padding: 2px 8px; border-radius: 6px; font-size: .72rem; font-weight: 600;
  background: var(--accent-soft); color: var(--accent); margin-right: 4px; white-space: nowrap;}}
.tag.tk {{background: var(--neutral-soft); color: var(--text);}}
.pill {{display: inline-block; min-width: 48px; text-align: center; padding: 2px 8px; border-radius: 999px;
  font-size: .76rem; font-weight: 700; font-variant-numeric: tabular-nums; white-space: nowrap;}}
.pill.pos {{background: var(--pos-soft); color: var(--pos);}}
.pill.neg {{background: var(--neg-soft); color: var(--neg);}}
.pill.warn {{background: var(--warn-soft); color: var(--warn);}}
.pill.neu {{background: var(--neutral-soft); color: var(--muted);}}
.signal {{background: var(--surface); border: 1px solid var(--border); border-radius: 14px; padding: 18px 22px;}}
.signal .headline {{font-size: 1.12rem; font-weight: 600; margin: 4px 0 10px; line-height: 1.45; color: var(--text);}}
.meter {{display: flex; gap: 3px; margin-top: 8px;}}
.meter span {{height: 10px; flex: 1; border-radius: 2px; background: var(--neutral-soft);}}
.note {{border: 1px solid var(--border); border-left: 4px solid var(--warn); background: var(--surface);
  padding: 10px 14px; border-radius: 8px; color: var(--muted); font-size: .92rem; margin: 6px 0 14px;}}
.note b {{color: var(--text);}}
.note.ok {{border-left-color: var(--accent);}}
</style>
""", unsafe_allow_html=True)


def fig_style(fig, height=360):
    axis = dict(gridcolor=T["grid"], linecolor=T["border"], zerolinecolor=T["border"], tickfont=dict(color=T["muted"]))
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=44, b=10), hovermode="closest",
                      template="plotly_dark" if T["base"] == "dark" else "plotly_white",
                      paper_bgcolor=T["surface"], plot_bgcolor=T["surface"],
                      font=dict(family="Inter, Source Sans Pro, sans-serif", color=T["text"]),
                      hoverlabel=dict(bgcolor=T["surface2"], bordercolor=T["border"], font=dict(color=T["text"])),
                      legend=dict(orientation="h", yanchor="top", y=-0.12, x=0, font=dict(color=T["muted"])))
    if fig.layout.title.text:  # styling an empty title makes Plotly print "undefined"
        fig.update_layout(title_font=dict(size=14, color=T["text"]))
    fig.update_xaxes(**axis)
    fig.update_yaxes(**axis)
    return fig


def esc(x) -> str:
    return html.escape(str(x))


def impact_cls(v: float) -> str:
    return "neg" if v > 7 else ("warn" if v > 5 else "neu")


def sent_cls(v: float) -> str:
    return "pos" if v > 0.2 else ("neg" if v < -0.2 else "neu")


def section(title: str, caption: str = ""):
    st.markdown(f"### {title}")
    if caption:
        st.caption(caption)


def note(text: str, ok: bool = False):
    st.markdown(f'<div class="note{" ok" if ok else ""}">{text}</div>', unsafe_allow_html=True)


@st.cache_data
def load_signals(path: str, mtime: float) -> pd.DataFrame:
    s = pd.read_csv(path, parse_dates=["timestamp"])
    s["ticker"] = s["ticker"].fillna("")
    return s


@st.cache_data
def load_prices(path: str) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["date"])


@st.cache_data
def backtest(sig_path: str, mtime: float, px_path: str, half_life_h, tilt, max_weight, max_turnover):
    cfg = RebalanceConfig(half_life_h=half_life_h, tilt=tilt, max_weight=max_weight, max_turnover=max_turnover)
    return run_backtest(load_signals(sig_path, mtime), load_prices(px_path), cfg), cfg


@st.cache_data
def stress_run(sig_path: str, mtime: float, threshold, cooldown_h, min_conf):
    st_ = run_event_stress(load_signals(sig_path, mtime), load_portfolio(), threshold=threshold, cooldown_h=cooldown_h,
                           min_confidence=min_conf)
    return st_.triggers, st_.suppressed


@st.cache_resource
def load_engine(path: str):
    from src.engine.pipeline import RiskEngine
    return RiskEngine.load(Path(path))


# ---------------------------------------------------------------- top bar (theme toggle) + sidebar + data
bar_l, bar_r = st.columns([5, 1], vertical_alignment="center")
bar_l.markdown('<div class="topbar"><div class="logo">NR</div><span class="brand">NLP Risk Engine</span>'
               '<span class="brand-sub">/ risk signals dashboard</span></div>', unsafe_allow_html=True)
with bar_r.container(horizontal_alignment="right"):
    st.session_state.dark_mode = st.session_state.get("dark", False)
    st.toggle("Dark mode", key="dark_mode", label_visibility="collapsed", width="content",
              on_change=lambda: st.session_state.update(dark=st.session_state.dark_mode))

with st.sidebar:
    st.markdown("## Controls")
    mode = st.radio("Data source", [SYNTHETIC, REAL_MODE])
SIGNALS_PATH, PRICES_PATH, ENGINE_PATH, MAKE_CMD = SOURCES[mode]
if not SIGNALS_PATH.exists():
    st.title("NLP Risk Engine")
    st.error(f"No engine output found at `outputs/{SIGNALS_PATH.name}`. Generate it first:")
    st.code(MAKE_CMD if mode == SYNTHETIC else "python -m src.engine.pipeline\n" + MAKE_CMD, language="bash")
    st.stop()

mtime = SIGNALS_PATH.stat().st_mtime
signals = load_signals(str(SIGNALS_PATH), mtime)
universe = load_universe()
is_real = mode == REAL_MODE
real_eval_path = OUT / "real_eval_metrics.json"
real_eval = json.loads(real_eval_path.read_text()) if real_eval_path.exists() else None

with st.sidebar:
    with st.expander("Module A: index rebalancer", expanded=True):
        tilt = st.slider("Sentiment tilt", 0.0, 4.0, 1.5, 0.25, help="0 = ignore sentiment (equal weight)")
        max_weight = st.slider("Max weight per stock", 0.08, 0.30, 0.15, 0.01)
        max_turnover = st.slider("Max one-way turnover per day", 0.02, 1.0, 0.20, 0.02)
        half_life_h = st.slider("Sentiment half-life, hours", 6, 168, 36, 6)
    with st.expander("Module B: stress trigger", expanded=True):
        threshold = st.slider("Impact threshold (trigger when impact >)", 5.0, 9.5, 7.0, 0.5)
        min_conf = st.slider("Min event-classifier confidence", 0.0, 0.95, 0.60, 0.05,
                             help="Below 0.60 the classifier is unsure and keyword rules decide; on real headlines "
                                  "those are mostly false alarms.")
        cooldown_h = st.slider("Cooldown per (event, ticker), hours", 0, 72, 24, 6)
    if is_real:
        st.caption("Real public data: Google News headlines (entering at the next day's close) and Yahoo Finance "
                   "closes. Sentiment retrained on real labelled text; event/impact models are synthetic-trained.")
    else:
        st.caption("All data is synthetic (see README). Prices were generated from the same events as the text, "
                   "so backtest performance is optimistic by construction.")

res_a, cfg = backtest(str(SIGNALS_PATH), mtime, str(PRICES_PATH), half_life_h, tilt, max_weight, max_turnover)
triggers, suppressed = stress_run(str(SIGNALS_PATH), mtime, threshold, cooldown_h, min_conf)
summ = pd.DataFrame([t.summary() for t in triggers])
portfolio = load_portfolio()
ma, mb = res_a.metrics["sentiment_index"], res_a.metrics["equal_weight"]
n_src = signals.source_type.value_counts()
span = f"{signals.timestamp.min():%d %b} to {signals.timestamp.max():%d %b %Y}"

# ---------------------------------------------------------------- hero
badge = ('<span class="chip real"><span class="dot"></span>Real data · Google News + Yahoo Finance</span>'
         if is_real else '<span class="chip syn"><span class="dot"></span>Synthetic data · generated corpus</span>')
st.markdown(f"""
<div class="hero">
  <div class="eyebrow">Text to risk signals</div>
  <h1>From headlines to portfolio action</h1>
  <p>Unstructured news &amp; social text in &rarr; sentiment, event type and impact out &rarr; an index that rebalances
  and a banking book that gets stress-tested.</p>
  {badge}<span class="chip">{len(signals):,} signals</span><span class="chip">{span}</span>
  <span class="chip">15 S&amp;P 100 stocks</span><span class="chip">40-asset banking book</span>
</div>""", unsafe_allow_html=True)

tab_home, tab_engine, tab_a, tab_b = st.tabs(["Overview", "Risk Engine", "Module A · Index rebalancer",
                                              "Module B · Stress test"])

# ---------------------------------------------------------------- Overview
with tab_home:
    acc = ""
    if real_eval:
        tw, pb = real_eval["twitter_test"]["real_trained_label"], real_eval["phrasebank_5fold"]["real_trained_label"]
        acc = (f"On held-out <b>real</b> text: {tw['accuracy']:.0%} accuracy on finance tweets, "
               f"{pb['accuracy']:.0%} on news sentences.")
    worst = summ.loc[(summ.pnl_pct * summ.sentiment.abs()).idxmin()] if len(summ) else None
    cards = [
        ("1 · Read", "Ingest two sources",
         f"{n_src.get('news', 0):,}", f"news {'headlines' if is_real else 'articles'}"
         + (f" + {n_src.get('social', 0):,} social posts" if n_src.get('social', 0) else ""),
         "Cleaned (RT, links, @, #), linked to a company by cashtag or name, or marked market-wide."),
        ("2 · Understand", "Score every text",
         f"{(signals.impact > threshold).sum():,}", f"high-impact signals (impact > {threshold:g})",
         "Each text gets sentiment (-1..1), one of 8 event types and an impact score (1-10). " + acc),
        ("3 · Act (Module A)", "Rebalance the index",
         f"{ma['total_return']:+.1%}", f"vs equal weight {mb['total_return']:+.1%}",
         "Good news raises a stock's weight, bad news lowers it, within 2-15% and a daily turnover cap."),
        ("3 · Act (Module B)", "Stress-test the book",
         f"{len(summ)}", "stress tests triggered",
         (f"Most adverse: {esc(worst.headline[:90])} &rarr; <b>{worst.pnl_pct:+.2%}</b>" if worst is not None
          else "No trigger at the current settings.")),
    ]
    cols = st.columns(4)
    for col, (step, title, big, sub, body) in zip(cols, cards):
        col.markdown(f"""<div class="card"><div class="step">{step}</div><h4>{title}</h4>
            <div class="big">{big}</div><div class="muted">{sub}</div><div class="rule"></div>
            <div class="muted">{body}</div></div>""", unsafe_allow_html=True)

    st.write("")
    left, right = st.columns([3, 2])
    with left:
        section("Live risk feed", "The most recent confident, high-impact signals: what the modules act on.")
        feed = (signals[(signals.impact > threshold) & (signals.event_confidence >= min_conf)]
                .sort_values("timestamp", ascending=False).head(8))
        rows = []
        for r in feed.itertuples():
            rows.append(f"""<div class="feed-row"><div class="feed-time">{r.timestamp:%d %b %H:%M}</div>
              <div class="feed-head"><span class="tag">{esc(r.event_type)}</span>
              {f'<span class="tag tk">{esc(r.ticker)}</span>' if r.ticker else ''}{esc(r.headline[:120])}</div>
              <span class="pill {sent_cls(r.sentiment)}">{r.sentiment:+.2f}</span>
              <span class="pill {impact_cls(r.impact)}">{r.impact:.1f}</span></div>""")
        st.markdown(f'<div class="feed">{"".join(rows)}</div>', unsafe_allow_html=True)
        st.caption("Pills: sentiment (blue positive, red negative) and impact (amber above 5, red above 7).")
    with right:
        section("Signals by event type")
        counts = signals.event_type.value_counts().sort_values()
        fig = go.Figure(go.Bar(x=counts.values, y=counts.index, orientation="h", marker_color=SERIES[0],
                               hovertemplate="%{y}: %{x} signals<extra></extra>"))
        st.plotly_chart(fig_style(fig, 380).update_layout(margin=dict(t=10)), width="stretch")

    with st.expander("System design: how the pieces connect"):
        arch = ROOT / "docs" / "architecture.png"
        if arch.exists():
            st.image(str(arch), width="stretch")

# ---------------------------------------------------------------- Risk Engine
with tab_engine:
    section("Try it live", "Pick an example or type any headline or post; the trained engine scores it instantly.")
    if "txt" not in st.session_state:
        st.session_state.txt, st.session_state.src = DEFAULT_TEXT, "news"

    def _use_example():
        pick = st.session_state.get("example")
        if pick:
            st.session_state.txt, st.session_state.src = EXAMPLES[pick]

    st.pills("Examples", list(EXAMPLES), key="example", on_change=_use_example)
    c1, c2 = st.columns([5, 1])
    txt = c1.text_area("Headline or post", key="txt", height=90)
    src = c2.radio("Source type", ["news", "social"], key="src")
    if c2.button("Analyze", type="primary", width="stretch"):
        if not ENGINE_PATH.exists():
            st.warning(f"No trained model at `models/{ENGINE_PATH.name}`. Run `{MAKE_CMD}` first.")
        else:
            from src.engine.ingest import from_raw_text
            docs = from_raw_text(txt, src)
            if docs.empty:
                st.warning("Text is empty after cleaning.")
            else:
                out = load_engine(str(ENGINE_PATH)).analyze(docs).iloc[0]
                segs = "".join(f'<span style="background:var(--{impact_cls(out.impact)})"></span>' if i < round(out.impact)
                               else '<span></span>' for i in range(10))
                st.markdown(f"""<div class="signal"><div class="headline">{esc(txt)}</div>
                    <span class="tag">{esc(out.event_type)} · {out.event_confidence:.0%} confident</span>
                    <span class="tag tk">{esc(out.ticker or 'market-wide')}</span>
                    <span class="tag tk">{esc(src)}</span>
                    <div style="display:flex;gap:28px;margin-top:12px;flex-wrap:wrap">
                      <div><div class="muted">Sentiment</div><div class="big" style="color:var(--{sent_cls(out.sentiment)})">
                        {out.sentiment:+.2f}</div></div>
                      <div style="min-width:220px"><div class="muted">Impact {out.impact:.1f} / 10</div>
                        <div class="meter">{segs}</div></div>
                    </div></div>""", unsafe_allow_html=True)
                sc = SCENARIOS.get(out.event_type)
                fires = (out.impact > threshold and out.event_confidence >= min_conf and sc is not None
                         and (not sc.adverse_only or out.sentiment < 0))
                why = (f"impact {out.impact:.1f} vs threshold {threshold:g}, confidence {out.event_confidence:.0%}, "
                       f"sentiment {out.sentiment:+.2f}")
                if fires:
                    st.success(f"Module B would TRIGGER a {out.event_type} stress test ({why}).")
                else:
                    st.info(f"Module B would not trigger ({why}).")

    st.divider()
    c = st.columns(4)
    c[0].metric("Signals", f"{len(signals):,}")
    c[1].metric("News / social", f"{n_src.get('news', 0):,} / {n_src.get('social', 0):,}")
    c[2].metric("Company / market scope", f"{(signals.scope == 'company').sum():,} / {(signals.scope == 'market').sum():,}")
    c[3].metric(f"High impact (> {threshold:g})", f"{(signals.impact > threshold).sum():,}")

    if real_eval:
        section("How accurate is the sentiment on REAL text?",
                f"Held-out real data: {real_eval['twitter_test']['n']:,} finance tweets (official test split) and "
                f"{real_eval['phrasebank_5fold']['n']:,} Financial PhraseBank news sentences (5-fold CV).")
        names = {"majority_class_baseline": "Always 'neutral' (baseline)", "lexicon_only": "Lexicon only",
                 "synthetic_trained_engine": "Engine trained on synthetic text",
                 "real_trained_label": "Retrained on real labelled text"}
        labels = list(names.values())
        fig = go.Figure()
        for i, (key, name) in enumerate([("twitter_test", "Real finance tweets"), ("phrasebank_5fold", "Real news sentences")]):
            vals = [real_eval[key][k]["macro_f1"] for k in names]
            fig.add_trace(go.Bar(y=labels, x=vals, name=name, orientation="h", marker_color=SERIES[i],
                                 text=[f"{v:.2f}" for v in vals], textposition="outside",
                                 hovertemplate=f"{name}<br>%{{y}}: macro-F1 %{{x:.3f}}<extra></extra>"))
        fig.update_layout(barmode="group", xaxis=dict(range=[0, 1], title="macro-F1 (3 classes)"), bargap=0.3)
        st.plotly_chart(fig_style(fig, 330).update_layout(margin=dict(t=10)), width="stretch")
        note("Trained only on synthetic text, the engine was no better than always guessing 'neutral' on real wording. "
             "Real-data mode retrains sentiment on real labelled text; synthetic mode keeps the original engine.", ok=True)

    left, right = st.columns(2)
    with left:
        section("Mean sentiment by stock", "Average over the whole period, company-level signals only.")
        by_t = (signals[signals.scope == "company"].groupby("ticker")
                .agg(sentiment=("sentiment", "mean"), n=("sentiment", "size")).sort_values("sentiment"))
        fig = go.Figure(go.Bar(x=by_t.sentiment, y=by_t.index, orientation="h",
                               marker_color=[POS if v >= 0 else NEG for v in by_t.sentiment], customdata=by_t.n,
                               hovertemplate="%{y}: mean sentiment %{x:.3f} over %{customdata} signals<extra></extra>"))
        st.plotly_chart(fig_style(fig, 420).update_layout(margin=dict(t=10)), width="stretch")
    with right:
        section("Signal explorer")
        f1, f2 = st.columns(2)
        ev_f = f1.multiselect("Event type", sorted(signals.event_type.unique()))
        tk_f = f2.multiselect("Ticker", sorted(t for t in signals.ticker.unique() if t))
        min_imp = st.slider("Min impact", 1.0, 10.0, 1.0, 0.5)
        view = signals[signals.impact >= min_imp]
        if ev_f:
            view = view[view.event_type.isin(ev_f)]
        if tk_f:
            view = view[view.ticker.isin(tk_f)]
        st.dataframe(view.sort_values("timestamp", ascending=False).head(200)
                     [["timestamp", "ticker", "sentiment", "event_type", "impact", "headline"]],
                     width="stretch", hide_index=True, height=300, column_config={
                         "sentiment": st.column_config.ProgressColumn(min_value=-1, max_value=1, format="%.2f"),
                         "impact": st.column_config.ProgressColumn(min_value=1, max_value=10, format="%.1f")})

# ---------------------------------------------------------------- Module A
with tab_a:
    section("Tactical index rebalancing",
            "Subscribes to company sentiment. Good news raises a stock's weight, bad news lowers it; weights set at "
            "today's close earn tomorrow's return (no look-ahead).")
    if is_real:
        note(f"<b>Honest result on real data:</b> the tilt returns {ma['total_return']:+.2%} vs {mb['total_return']:+.2%} "
             "for equal weight, within the range of randomly shuffled sentiment. Public headlines get priced in fast.")
    else:
        note("<b>Read with care:</b> synthetic prices were generated from the same events as the text, so this "
             "outperformance is partly circular. Switch the data source to <b>Real</b> for the honest test.")
    res = res_a
    c = st.columns(5)
    c[0].metric("Total return", f"{ma['total_return']:+.2%}", f"{ma['total_return'] - mb['total_return']:+.2%} vs EW")
    c[1].metric("Sharpe (ann.)", f"{ma['sharpe']:.2f}", f"{ma['sharpe'] - mb['sharpe']:+.2f} vs EW")
    c[2].metric("Ann. volatility", f"{ma['ann_vol']:.2%}", f"{ma['ann_vol'] - mb['ann_vol']:+.2%} vs EW",
                delta_color="inverse")
    c[3].metric("Max drawdown", f"{ma['max_drawdown']:.2%}", f"{ma['max_drawdown'] - mb['max_drawdown']:+.2%} vs EW")
    c[4].metric("Avg daily turnover", f"{ma['avg_daily_turnover']:.1%}")

    sector = universe.set_index("ticker").sector
    sectors = list(dict.fromkeys(sector[res.weights.columns]))
    order = sorted(res.weights.columns, key=lambda t: (sectors.index(sector[t]), t))
    left, right = st.columns([3, 2])
    with left:
        fig = go.Figure()
        shown = set()
        for t in order:
            sec = sector[t]
            fig.add_trace(go.Scatter(x=res.weights.index, y=res.weights[t], name=sec, legendgroup=sec,
                                     showlegend=sec not in shown, stackgroup="w", mode="lines",
                                     line=dict(width=1, color=T["surface"]), fillcolor=SERIES[sectors.index(sec)],
                                     hovertemplate=f"{t} ({sec}): %{{y:.1%}}<extra></extra>"))
            shown.add(sec)
        fig.update_layout(title="Index weights over time (one band per stock; hover for ticker)",
                          yaxis=dict(tickformat=".0%", range=[0, 1]))
        st.plotly_chart(fig_style(fig, 420), width="stretch")
    with right:
        nav = res.nav.rename(columns={"sentiment_index": "Sentiment index", "equal_weight": "Equal-weight benchmark"})
        fig = go.Figure([go.Scatter(x=nav.index, y=nav[col], name=col, mode="lines", line=dict(width=2, color=SERIES[i]),
                                    hovertemplate="%{y:.2f}") for i, col in enumerate(nav.columns)])
        fig.update_layout(title="NAV (start = 100)", hovermode="x unified")
        st.plotly_chart(fig_style(fig, 420), width="stretch")

    section("Latest weight changes and why", "Main driver = the signal with the largest decayed contribution.")
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
    st.caption(f"{d_then:%Y-%m-%d} to {d_now:%Y-%m-%d}")
    st.dataframe(tbl, hide_index=True, width="stretch", column_config={
        "weight": st.column_config.NumberColumn(format="percent"),
        "change": st.column_config.NumberColumn(format="percent"),
        "vs equal weight": st.column_config.NumberColumn(format="percent"),
        "score": st.column_config.ProgressColumn(min_value=-1, max_value=1, format="%.2f")})

    left, right = st.columns([3, 2])
    with left:
        fig = go.Figure(go.Heatmap(z=res.scores[order].T.values, x=res.scores.index, y=order, colorscale=DIVERGING,
                                   zmid=0, zmin=-1, zmax=1, colorbar=dict(title="score"),
                                   hovertemplate="%{y} %{x|%b %d}: %{z:.2f}<extra></extra>"))
        fig.update_layout(title="Decayed sentiment score at each close (red negative, blue positive)")
        st.plotly_chart(fig_style(fig, 440), width="stretch")
    with right:
        pick = st.multiselect("Compare individual weights", order, default=list(
            (res.weights.iloc[-1] - res.weights.iloc[0]).abs().sort_values().index[-3:]), max_selections=6)
        fig = go.Figure([go.Scatter(x=res.weights.index, y=res.weights[t], name=t, mode="lines",
                                    line=dict(width=2, color=SERIES[i])) for i, t in enumerate(pick)])
        fig.add_hline(y=1 / len(order), line_dash="dot", line_color=T["faint"], annotation_text="equal weight",
                      annotation_font_color=T["muted"])
        fig.update_layout(title="Selected stock weights", yaxis_tickformat=".0%", hovermode="x unified")
        st.plotly_chart(fig_style(fig, 360), width="stretch")

# ---------------------------------------------------------------- Module B
with tab_b:
    section("Strategic stress testing",
            "Subscribes to event type + impact. A confident, adverse event with impact above the threshold applies "
            "that event's shock (scaled by impact/10) to a synthetic wholesale-banking book.")
    c = st.columns(4)
    c[0].metric("Portfolio value", f"${portfolio.market_value.sum():,.1f}m")
    c[1].metric("Assets", f"{len(portfolio)}")
    c[2].metric("Stress tests triggered", f"{len(summ)}")
    c[3].metric("Worst trigger", f"{summ.pnl_pct.min():.2%}" if len(summ) else "n/a")
    st.caption("Portfolio: " + ", ".join(f"{v} {PLURAL.get(k, k)}" for k, v in portfolio.asset_type.value_counts().items())
               + ". Values in $m; P&L is computed on risk notional. "
               f"Signals ignored: {suppressed['low_confidence']} unsure event label, {suppressed['not_adverse']} not "
               f"adverse, {suppressed['cooldown']} inside cooldown, {suppressed['below_threshold']} at or below threshold.")

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
            increasing=dict(marker_color=POS), decreasing=dict(marker_color=NEG), totals=dict(marker_color=T["faint"]),
            connector=dict(line=dict(color=T["border"], width=1))))
        lo = min(res.value_before, res.value_after) + min(0, min(parts))
        fig.update_layout(title="Portfolio value: before to after ($m)",
                          yaxis=dict(range=[lo * 0.97, max(res.value_before, res.value_after) * 1.01]))
        left.plotly_chart(fig_style(fig), width="stretch", key=f"wf{key}")

        bt = res.by_asset_type
        fig = go.Figure(go.Bar(x=bt.pnl, y=bt.index, orientation="h",
                               marker_color=[POS if v >= 0 else NEG for v in bt.pnl],
                               hovertemplate="%{y}: %{x:+,.2f}m<extra></extra>"))
        right.plotly_chart(fig_style(fig.update_layout(title="P&L by asset type ($m, hedges in blue)")),
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

    if summ.empty:
        st.info("No signal passes the trigger rules. Lower the threshold or confidence in the sidebar.")
    else:
        types = st.multiselect("Event types", sorted(summ.event_type.unique()), default=sorted(summ.event_type.unique()))
        sub = summ[summ.event_type.isin(types)]
        fig = go.Figure(go.Scatter(
            x=sub.timestamp, y=sub.event_type, mode="markers",
            marker=dict(size=8 + 60 * sub.pnl_pct.abs(), color=SERIES[0], opacity=0.7, line=dict(width=1, color=T["surface"])),
            customdata=sub[["ticker", "impact", "pnl_pct", "headline"]].values,
            hovertemplate="%{x|%b %d %H:%M} %{y} %{customdata[0]}<br>impact %{customdata[1]:.1f}, "
                          "P&L %{customdata[2]:.2%}<br>%{customdata[3]}<extra></extra>"))
        fig.update_layout(title="Triggered stress tests over time (marker size = portfolio loss; hover for headline)")
        st.plotly_chart(fig_style(fig, 300), width="stretch")

        if len(sub):
            labels = {i: f"{pd.Timestamp(summ.at[i, 'timestamp']):%Y-%m-%d %H:%M} | {summ.at[i, 'event_type']} "
                         f"{summ.at[i, 'ticker'] or '(market)'} | impact {summ.at[i, 'impact']:.1f} | "
                         f"{summ.at[i, 'pnl_pct']:+.2%}" for i in sub.index}
            # many triggers tie on loss (the shock depends only on impact), so rank by loss x how negative the news is
            default = list(sub.index).index((sub.pnl_pct * sub.sentiment.abs()).idxmin())
            sel = st.selectbox("Pick a triggered event (default: most adverse = loss x negativity)", list(sub.index),
                               index=default, format_func=labels.get)
            res = triggers[sel]
            sig, sh = res.signal, res.shock
            shocks = [f"equity {sh.equity_pct:+.1f}%", f"rates {sh.rate_bp:+.0f}bp", f"spreads {sh.spread_bp:+.0f}bp"]
            if sh.ticker:
                shocks += [f"{sh.ticker} equity {sh.idio_equity_pct:+.1f}%", f"{sh.ticker} spread {sh.idio_spread_bp:+.0f}bp"]
            st.markdown(f"""<div class="signal"><div class="muted">{pd.Timestamp(sig['timestamp']):%d %b %Y %H:%M} ·
                {esc(sig['source_type'])}</div><div class="headline">{esc(sig['headline'])}</div>
                <span class="tag">{esc(sig['event_type'])}</span>
                <span class="tag tk">{esc(sig.get('ticker') or 'market-wide')}</span>
                <span class="pill {sent_cls(sig['sentiment'])}">sentiment {sig['sentiment']:+.2f}</span>
                <span class="pill {impact_cls(sig['impact'])}">impact {sig['impact']:.1f}</span>
                <div class="muted" style="margin-top:10px">Applied shock (scenario x impact/10): {' · '.join(shocks)}.
                {esc(SCENARIOS[res.scenario].rationale)}.</div></div>""", unsafe_allow_html=True)
            st.write("")
            show_result(res, "event")

    st.divider()
    section("Custom shock (what-if)", "Apply any scenario to the book, e.g. the brief's own example.")
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
