"""Module A: tactical index rebalancer driven by the engine's Sentiment Score.

    res = run_backtest(signals, prices, RebalanceConfig())
    res.weights      # date x ticker, weights set at that day's close
    res.nav          # sentiment index vs equal-weight benchmark (start = 100)
    res.metrics      # total return, ann. vol, Sharpe, max drawdown, avg daily turnover

How a weight is set
  1. Every company signal adds  w = impact * source_weight  (news 1.0, social 0.5) to a per-stock
     running sum; old evidence decays exponentially (half-life 36h).
  2. score = sum(w * sentiment) / (sum(w) + prior_strength)  -> in [-1, 1]; thin coverage shrinks to 0.
  3. target = base * exp(tilt * score), clipped to [min_weight, max_weight] and renormalised.
     Positive sentiment raises the weight, negative lowers it.
  4. Trade from the current (drifted) weights toward the target, at most `max_turnover` one-way per day.

Timing: rebalance at the 16:00 close of day t using only signals stamped <= 16:00 on day t; those
weights earn day t+1's close-to-close return (no look-ahead). Between rebalances weights drift with prices.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.engine.aggregate import SOURCE_WEIGHT
from src.engine.bus import SignalBus

CLOSE = pd.Timedelta(hours=16)
TRADING_DAYS = 252


@dataclass
class RebalanceConfig:
    half_life_h: float = 36.0
    prior_strength: float = 3.0
    tilt: float = 1.5
    min_weight: float = 0.02
    max_weight: float = 0.15
    max_turnover: float = 0.20  # one-way, per rebalance

    def __post_init__(self):
        if not 0 <= self.min_weight < self.max_weight <= 1:
            raise ValueError("need 0 <= min_weight < max_weight <= 1")


def one_way_turnover(old: np.ndarray, new: np.ndarray) -> float:
    return float(0.5 * np.abs(np.asarray(new) - np.asarray(old)).sum())


def bound_weights(w: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Clip to [lo, hi] and renormalise to 1, redistributing only across names still inside the box."""
    w = np.asarray(w, float)
    n = len(w)
    if n * lo > 1 + 1e-12 or n * hi < 1 - 1e-12:
        raise ValueError(f"bounds [{lo}, {hi}] infeasible for {n} names")
    w = w / w.sum()
    for _ in range(100):
        c = np.clip(w, lo, hi)
        gap = 1 - c.sum()
        if abs(gap) < 1e-12:
            return c
        free = (c < hi) if gap > 0 else (c > lo)
        c[free] += gap * c[free] / c[free].sum()
        w = c
    return np.clip(w, lo, hi) / np.clip(w, lo, hi).sum()


class SentimentRebalancer:
    """Bus subscriber: `bus.subscribe(rb.on_signal, scope="company", name="Module A")`."""

    def __init__(self, tickers, cfg: RebalanceConfig = None):
        self.cfg = cfg or RebalanceConfig()
        self.tickers = list(tickers)
        self._idx = {t: i for i, t in enumerate(self.tickers)}
        n = len(self.tickers)
        self.base = np.full(n, 1.0 / n)
        self.weights = self.base.copy()
        self._num = np.zeros(n)
        self._den = np.zeros(n)
        self._t = None  # time the running sums were last decayed to
        self.n_signals = 0

    def _decay_to(self, t: pd.Timestamp):
        if self._t is not None and t > self._t:
            hours = (t - self._t) / pd.Timedelta(hours=1)
            f = 0.5 ** (hours / self.cfg.half_life_h)
            self._num *= f
            self._den *= f
        if self._t is None or t > self._t:
            self._t = t

    def on_signal(self, s: dict):
        i = self._idx.get(s.get("ticker"))
        if i is None:
            return
        self._decay_to(pd.Timestamp(s["timestamp"]))
        w = float(s["impact"]) * SOURCE_WEIGHT.get(s.get("source_type"), 0.5)
        self._num[i] += w * float(s["sentiment"])
        self._den[i] += w
        self.n_signals += 1

    def scores(self, at: pd.Timestamp) -> np.ndarray:
        self._decay_to(pd.Timestamp(at))
        return self._num / (self._den + self.cfg.prior_strength)

    def target_weights(self, scores: np.ndarray) -> np.ndarray:
        raw = self.base * np.exp(self.cfg.tilt * np.asarray(scores))
        return bound_weights(raw, self.cfg.min_weight, self.cfg.max_weight)

    def drift(self, returns: np.ndarray):
        """Let weights float with one period of returns (no trading)."""
        w = self.weights * (1 + np.asarray(returns))
        self.weights = w / w.sum()

    def rebalance(self, at: pd.Timestamp) -> dict:
        """Move from the current weights toward the sentiment target, respecting bounds and the turnover cap.
        If price drift pushed a name outside [min, max], that forced fix is done first (bounds win over the cap)."""
        sc = self.scores(at)
        target = self.target_weights(sc)
        start = self.weights
        inside = bound_weights(start, self.cfg.min_weight, self.cfg.max_weight)
        budget = max(self.cfg.max_turnover - one_way_turnover(start, inside), 0.0)
        need = one_way_turnover(inside, target)
        alpha = 1.0 if need <= budget else budget / need
        new = inside + alpha * (target - inside)  # convex mix of two in-bounds weight vectors -> in bounds
        self.weights = new / new.sum()
        return {"scores": sc, "target": target, "weights": self.weights.copy(),
                "turnover": one_way_turnover(start, self.weights)}


@dataclass
class BacktestResult:
    weights: pd.DataFrame  # date x ticker, set at close
    targets: pd.DataFrame
    scores: pd.DataFrame
    turnover: pd.Series
    nav: pd.DataFrame  # columns: sentiment_index, equal_weight
    metrics: dict = field(default_factory=dict)


def perf_metrics(nav: pd.Series, turnover: pd.Series = None) -> dict:
    r = nav.pct_change().dropna()
    vol = float(r.std() * np.sqrt(TRADING_DAYS))
    m = {"total_return": float(nav.iloc[-1] / nav.iloc[0] - 1),
         "ann_vol": vol,
         "sharpe": float(r.mean() * TRADING_DAYS / vol) if vol > 0 else 0.0,
         "max_drawdown": float((nav / nav.cummax() - 1).min())}
    if turnover is not None:
        m["avg_daily_turnover"] = float(turnover.iloc[1:].mean())  # skip the day-0 build from equal weight
    return m


def run_backtest(signals: pd.DataFrame, prices: pd.DataFrame, cfg: RebalanceConfig = None) -> BacktestResult:
    """Stream signals through a SignalBus in time order, rebalancing at each close.

    signals: engine output (timestamp, ticker, scope, sentiment, impact, source_type, ...)
    prices:  long format (date, ticker, close)"""
    cfg = cfg or RebalanceConfig()
    px = prices.assign(date=pd.to_datetime(prices["date"])).pivot(index="date", columns="ticker", values="close")
    px = px.sort_index().dropna(axis=1, how="any")
    rets = px.pct_change().fillna(0.0)
    tickers = list(px.columns)

    rb = SentimentRebalancer(tickers, cfg)
    bus = SignalBus()
    bus.subscribe(rb.on_signal, scope="company", name="Module A")

    sig = signals.copy()
    sig["timestamp"] = pd.to_datetime(sig["timestamp"])
    sig["ticker"] = sig["ticker"].fillna("")
    recs = sig.sort_values("timestamp").to_dict("records")
    k = 0

    rows_w, rows_t, rows_s, turn = {}, {}, {}, {}
    nav, bench = [100.0], [100.0]
    ew = np.full(len(tickers), 1.0 / len(tickers))
    for j, d in enumerate(px.index):
        r = rets.loc[d, tickers].to_numpy()
        if j > 0:  # yesterday's close weights earn today's return, then drift
            nav.append(nav[-1] * (1 + float(rb.weights @ r)))
            bench.append(bench[-1] * (1 + float(ew @ r)))
            rb.drift(r)
        close = d + CLOSE
        while k < len(recs) and recs[k]["timestamp"] <= close:
            bus.publish(recs[k])
            k += 1
        out = rb.rebalance(close)
        rows_w[d], rows_t[d], rows_s[d], turn[d] = out["weights"], out["target"], out["scores"], out["turnover"]

    frame = lambda rows: pd.DataFrame.from_dict(rows, orient="index", columns=tickers).rename_axis("date")
    navdf = pd.DataFrame({"sentiment_index": nav, "equal_weight": bench}, index=px.index)
    turnover = pd.Series(turn, name="turnover").rename_axis("date")
    metrics = {"sentiment_index": perf_metrics(navdf.sentiment_index, turnover),
               "equal_weight": perf_metrics(navdf.equal_weight),
               "signals_consumed": rb.n_signals}
    return BacktestResult(frame(rows_w), frame(rows_t), frame(rows_s), turnover, navdf, metrics)


def top_drivers(signals: pd.DataFrame, ticker: str, at: pd.Timestamp, cfg: RebalanceConfig = None,
                n: int = 3) -> pd.DataFrame:
    """The signals contributing most (decayed impact x source weight x |sentiment|) to `ticker`'s score at `at`.
    Used by the dashboard to explain *why* a weight moved."""
    cfg = cfg or RebalanceConfig()
    s = signals[(signals["ticker"] == ticker) & (signals["scope"] == "company")].copy()
    s["timestamp"] = pd.to_datetime(s["timestamp"])
    s = s[s["timestamp"] <= at]
    if s.empty:
        return s
    age_h = (at - s["timestamp"]) / pd.Timedelta(hours=1)
    s["contribution"] = (s["impact"] * s["source_type"].map(SOURCE_WEIGHT).fillna(0.5)
                         * s["sentiment"] * 0.5 ** (age_h / cfg.half_life_h))
    return s.reindex(s["contribution"].abs().sort_values(ascending=False).index).head(n)
