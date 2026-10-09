"""Module B: event-triggered stress test of a synthetic wholesale-banking portfolio.

    st = StressTester(load_portfolio())
    bus.subscribe(st.on_signal, min_impact=7, name="Module B")
    bus.replay(signals)
    st.triggers          # one StressResult per distinct high-impact event
    run_stress(portfolio, Shock(equity_pct=-10, rate_bp=200))   # any custom shock, run by hand

Trigger rule: impact > threshold (strictly, default 7) AND the event type has a scenario. Idiosyncratic
scenarios (Earnings, M&A, Regulatory/Legal, Product Launch) only fire on negative sentiment, since we stress
the adverse case. A 24h cooldown per (event_type, ticker) collapses a burst of articles/posts about the
same story into one trigger. Every shock is scaled by severity = impact / 10.

Valuation is a deliberately SIMPLIFIED FIRST-ORDER model. Per asset, on risk_notional (not market value:
a swap or option has ~0 MV but large exposure):
    rate P&L   = -rate_duration   * dy * risk_notional
    spread P&L = -spread_duration * ds * rating_multiplier * risk_notional
    equity P&L =  equity_beta * equity_shock * risk_notional   (+ idiosyncratic shock on the linked name)
Signs fall out of the data: pay-fixed swaps have negative rate duration (gain when rates rise), CDS
protection has negative spread duration (gains when spreads widen), index puts have beta -1.
NOT modelled: convexity/gamma, vega, cross-asset correlations, rating migration or default losses,
funding/liquidity effects, second-round effects. Treat outputs as an indicative first-order sensitivity.
"""
from dataclasses import dataclass, field, replace
from typing import Optional

import pandas as pd

from src.universe import DATA

PORTFOLIO_PATH = DATA / "portfolio.csv"

# worse credit quality -> spreads widen more for the same market move
RATING_MULTIPLIER = {"AA": 0.6, "A+": 0.7, "A": 0.8, "BBB+": 0.9, "BBB": 1.0, "NR": 1.0, "BB": 1.8, "B": 2.5}


@dataclass(frozen=True)
class Shock:
    equity_pct: float = 0.0  # market-wide equity move, %
    rate_bp: float = 0.0  # parallel shift of the rate curve, bp
    spread_bp: float = 0.0  # credit spread widening, bp
    idio_equity_pct: float = 0.0  # extra move for the linked ticker's stock, %
    idio_spread_bp: float = 0.0  # extra widening for the linked ticker's bonds, bp
    ticker: str = ""  # linked ticker for the idiosyncratic part ('' = none)

    def scaled(self, k: float) -> "Shock":
        return replace(self, equity_pct=self.equity_pct * k, rate_bp=self.rate_bp * k,
                       spread_bp=self.spread_bp * k, idio_equity_pct=self.idio_equity_pct * k,
                       idio_spread_bp=self.idio_spread_bp * k)


@dataclass(frozen=True)
class Scenario:
    shock: Shock
    adverse_only: bool  # only fire when the signal's sentiment is negative
    rationale: str


# Full-severity (impact = 10) shocks; scaled by impact / 10 at trigger time.
SCENARIOS = {
    "Geopolitical": Scenario(Shock(-12, -40, 100), False,
                             "Risk-off: equities sell off, flight to quality pulls yields down, credit widens"),
    "Macroeconomic": Scenario(Shock(-8, 100, 60), False,
                              "Inflation / hawkish surprise: rates up, equities down, spreads wider"),
    "Credit Event": Scenario(Shock(-6, -10, 150, idio_equity_pct=-15, idio_spread_bp=300), False,
                             "Downgrade/default: contagion in credit, name-specific gap lower"),
    "Regulatory/Legal": Scenario(Shock(-2, 0, 15, idio_equity_pct=-12, idio_spread_bp=120), True,
                                 "Fine / probe / ruling: mostly hits the named company"),
    "Earnings": Scenario(Shock(-1, 0, 5, idio_equity_pct=-10, idio_spread_bp=40), True,
                         "Earnings miss / guidance cut: name-specific repricing"),
    "Merger/Acquisition": Scenario(Shock(0, 0, 5, idio_equity_pct=-8, idio_spread_bp=80), True,
                                   "Deal stress: leverage-funded acquisition or broken deal"),
    "Product Launch": Scenario(Shock(-0.5, 0, 0, idio_equity_pct=-7, idio_spread_bp=25), True,
                               "Failed launch / recall: name-specific"),
}

# manual what-if presets for the dashboard (not scaled by severity)
PRESETS = {
    "Brief example: equities -10%, rates +200bp": Shock(-10, 200, 0),
    "2008-style credit crunch": Shock(-35, -150, 400),
    "2022-style rate shock": Shock(-18, 250, 80),
    "Covid-style crash (Mar 2020)": Shock(-30, -100, 250),
}


def load_portfolio(path=PORTFOLIO_PATH) -> pd.DataFrame:
    p = pd.read_csv(path)
    p["linked_ticker"] = p["linked_ticker"].fillna("")
    return p


@dataclass
class StressResult:
    shock: Shock
    assets: pd.DataFrame  # one row per asset with rate/spread/equity/total P&L
    signal: Optional[dict] = None
    scenario: str = "Custom"

    @property
    def value_before(self) -> float:
        return float(self.assets["market_value"].sum())

    @property
    def pnl(self) -> float:
        return float(self.assets["pnl"].sum())

    @property
    def value_after(self) -> float:
        return self.value_before + self.pnl

    @property
    def pnl_pct(self) -> float:
        return self.pnl / self.value_before

    def by(self, col: str) -> pd.DataFrame:
        cols = ["market_value", "rate_pnl", "spread_pnl", "equity_pnl", "pnl"]
        return self.assets.groupby(col)[cols].sum().sort_values("pnl")

    @property
    def by_asset_type(self) -> pd.DataFrame:
        return self.by("asset_type")

    @property
    def by_sector(self) -> pd.DataFrame:
        return self.by("sector")

    def top_losers(self, n: int = 10) -> pd.DataFrame:
        cols = ["asset_id", "asset_type", "instrument", "counterparty", "linked_ticker", "rating",
                "market_value", "risk_notional", "pnl"]
        return self.assets.nsmallest(n, "pnl")[cols]

    def summary(self) -> dict:
        s = self.signal or {}
        return {"scenario": self.scenario, "timestamp": s.get("timestamp"), "event_type": s.get("event_type"),
                "ticker": s.get("ticker", ""), "impact": s.get("impact"), "sentiment": s.get("sentiment"),
                "headline": s.get("headline"), "value_before": self.value_before, "value_after": self.value_after,
                "pnl": self.pnl, "pnl_pct": self.pnl_pct}


def run_stress(portfolio: pd.DataFrame, shock: Shock, signal: dict = None, scenario: str = "Custom") -> StressResult:
    """Apply one shock to every asset (first-order duration/beta model; see module docstring)."""
    a = portfolio.copy()
    a["linked_ticker"] = a["linked_ticker"].fillna("")
    linked = (a["linked_ticker"] == shock.ticker) & (shock.ticker != "")
    mult = a["rating"].map(RATING_MULTIPLIER).fillna(1.0)
    dy = shock.rate_bp / 1e4
    ds = (shock.spread_bp + linked * shock.idio_spread_bp) / 1e4
    rn = a["risk_notional"]
    a["rate_pnl"] = -a["rate_duration"] * dy * rn
    a["spread_pnl"] = -a["spread_duration"] * ds * mult * rn
    # idiosyncratic equity move hits the stock itself (assets with equity exposure to the linked name)
    idio = linked * (a["equity_beta"] != 0) * shock.idio_equity_pct / 100
    a["equity_pnl"] = (a["equity_beta"] * shock.equity_pct / 100 + idio) * rn
    a["pnl"] = a["rate_pnl"] + a["spread_pnl"] + a["equity_pnl"]
    return StressResult(shock, a, signal, scenario)


class StressTester:
    """Bus subscriber: `bus.subscribe(st.on_signal, min_impact=7, name="Module B")`."""

    def __init__(self, portfolio: pd.DataFrame, scenarios: dict = None, threshold: float = 7.0,
                 cooldown: pd.Timedelta = pd.Timedelta(hours=24)):
        self.portfolio = portfolio
        self.scenarios = scenarios or SCENARIOS
        self.threshold = threshold
        self.cooldown = cooldown
        self.triggers: list[StressResult] = []
        self.suppressed = {"below_threshold": 0, "no_scenario": 0, "not_adverse": 0, "cooldown": 0}
        self._last: dict = {}

    def check(self, s: dict) -> Optional[str]:
        """None if the signal should trigger, otherwise the reason it does not."""
        if not s["impact"] > self.threshold:
            return "below_threshold"
        sc = self.scenarios.get(s["event_type"])
        if sc is None:
            return "no_scenario"
        if sc.adverse_only and not s["sentiment"] < 0:
            return "not_adverse"
        key = (s["event_type"], s.get("ticker") or "")
        last = self._last.get(key)
        if last is not None and pd.Timestamp(s["timestamp"]) - last < self.cooldown:
            return "cooldown"
        return None

    def on_signal(self, s: dict):
        why = self.check(s)
        if why:
            self.suppressed[why] += 1
            return
        ticker = s.get("ticker") or ""
        self._last[(s["event_type"], ticker)] = pd.Timestamp(s["timestamp"])
        shock = replace(self.scenarios[s["event_type"]].shock.scaled(s["impact"] / 10), ticker=ticker)
        self.triggers.append(run_stress(self.portfolio, shock, s, s["event_type"]))

    def summary(self) -> pd.DataFrame:
        return pd.DataFrame([t.summary() for t in self.triggers])


def run_event_stress(signals: pd.DataFrame, portfolio: pd.DataFrame = None, threshold: float = 7.0,
                     cooldown_h: float = 24) -> StressTester:
    """Replay engine signals through a SignalBus into a StressTester."""
    from src.engine.bus import SignalBus

    st = StressTester(load_portfolio() if portfolio is None else portfolio, threshold=threshold,
                      cooldown=pd.Timedelta(hours=cooldown_h))
    bus = SignalBus()
    bus.subscribe(st.on_signal, min_impact=threshold, name="Module B")
    sig = signals.copy()
    sig["ticker"] = sig["ticker"].fillna("")
    bus.replay(sig)
    return st
