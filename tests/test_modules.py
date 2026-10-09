from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.modules.rebalancer import (RebalanceConfig, SentimentRebalancer, bound_weights, one_way_turnover,
                                    run_backtest)
from src.modules.stress import SCENARIOS, Shock, StressTester, load_portfolio, run_event_stress, run_stress

TICKERS = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "GGG", "HHH", "III", "JJJ"]


def _sig(ts, ticker, sentiment, impact=8.0, source_type="news", event_type="Earnings", scope="company"):
    return dict(timestamp=pd.Timestamp(ts), ticker=ticker, scope=scope, sentiment=sentiment, impact=impact,
                source_type=source_type, event_type=event_type, headline=f"{ticker} test")


def _flat_prices(days=6, tickers=TICKERS):
    dates = pd.bdate_range("2026-01-05", periods=days)
    return pd.DataFrame([dict(date=d, ticker=t, close=100.0) for d in dates for t in tickers])


# ---------------- Module A ----------------

def test_bound_weights_sum_to_one_and_respect_box():
    w = bound_weights(np.array([50, 1, 1, 1, 1, 1, 1, 1, 1, 1.0]), 0.02, 0.15)
    assert w.sum() == pytest.approx(1) and w.max() <= 0.15 + 1e-12 and w.min() >= 0.02 - 1e-12


def test_positive_sentiment_raises_weight_negative_lowers():
    rb = SentimentRebalancer(TICKERS, RebalanceConfig(max_turnover=1.0))
    rb.on_signal(_sig("2026-01-05 10:00", "AAA", +0.9))
    rb.on_signal(_sig("2026-01-05 11:00", "BBB", -0.9))
    out = rb.rebalance(pd.Timestamp("2026-01-05 16:00"))
    eq = 1 / len(TICKERS)
    assert out["weights"][0] > eq > out["weights"][1]
    assert out["scores"][0] > 0 > out["scores"][1]


def test_scores_decay_and_shrink_thin_coverage():
    cfg = RebalanceConfig()
    rb = SentimentRebalancer(TICKERS, cfg)
    rb.on_signal(_sig("2026-01-05 10:00", "AAA", 1.0, impact=1, source_type="social"))
    s0 = rb.scores(pd.Timestamp("2026-01-05 10:00"))[0]
    assert 0 < s0 < 0.2  # one weak social post barely moves the score
    rb2 = SentimentRebalancer(TICKERS, cfg)
    rb2.on_signal(_sig("2026-01-05 10:00", "AAA", 1.0))
    a = rb2.scores(pd.Timestamp("2026-01-05 10:00"))[0]
    b = rb2.scores(pd.Timestamp("2026-01-05 10:00") + pd.Timedelta(hours=cfg.half_life_h * 4))[0]
    assert a > b > 0


def test_backtest_weights_bounds_and_turnover_cap():
    rng = np.random.default_rng(0)
    sig = [_sig(pd.Timestamp("2026-01-05 07:00") + pd.Timedelta(hours=int(h)), rng.choice(TICKERS),
                float(rng.uniform(-1, 1)), impact=float(rng.uniform(1, 10)))
           for h in rng.integers(0, 24 * 8, 300)]
    cfg = RebalanceConfig(max_turnover=0.05)
    res = run_backtest(pd.DataFrame(sig), _flat_prices(8), cfg)
    w = res.weights
    assert np.allclose(w.sum(axis=1), 1)
    assert (w.to_numpy() >= cfg.min_weight - 1e-9).all() and (w.to_numpy() <= cfg.max_weight + 1e-9).all()
    assert (res.turnover <= cfg.max_turnover + 1e-9).all()
    assert (res.turnover > 0).any()


def test_backtest_no_look_ahead():
    """A huge positive signal at 10:00 on day 3 must not affect weights before day 3's close,
    and an after-close (18:00) signal waits for the next day's rebalance."""
    dates = pd.bdate_range("2026-01-05", periods=5)
    base = pd.DataFrame([_sig(dates[0] + pd.Timedelta(hours=9), "CCC", 0.1, impact=1)])
    news = pd.concat([base, pd.DataFrame([_sig(dates[2] + pd.Timedelta(hours=10), "AAA", 1.0, impact=10)])])
    late = pd.concat([base, pd.DataFrame([_sig(dates[2] + pd.Timedelta(hours=18), "AAA", 1.0, impact=10)])])
    px = _flat_prices(5)
    w0 = run_backtest(base, px).weights["AAA"]
    w1 = run_backtest(news, px).weights["AAA"]
    w2 = run_backtest(late, px).weights["AAA"]
    assert (w1.iloc[:2] == w0.iloc[:2]).all() and w1.iloc[2] > w0.iloc[2]
    assert (w2.iloc[:3] == w0.iloc[:3]).all() and w2.iloc[3] > w0.iloc[3]


def test_backtest_weights_earn_next_day_return():
    """Day-t weights earn day t+1's return: a jump in AAA on day 2 is earned with day-1 weights."""
    dates = pd.bdate_range("2026-01-05", periods=3)
    px = _flat_prices(3)
    px.loc[(px.ticker == "AAA") & (px.date == dates[2]), "close"] = 110.0
    sig = pd.DataFrame([_sig(dates[1] + pd.Timedelta(hours=9), "AAA", 1.0, impact=10)])
    res = run_backtest(sig, px, RebalanceConfig(max_turnover=1.0))
    expected = 1 + res.weights["AAA"].iloc[1] * 0.10
    assert res.nav.sentiment_index.iloc[2] / res.nav.sentiment_index.iloc[1] == pytest.approx(expected)
    assert res.nav.sentiment_index.iloc[1] == pytest.approx(100.0)


def test_turnover_helper():
    assert one_way_turnover([0.5, 0.5], [0.6, 0.4]) == pytest.approx(0.1)


# ---------------- Module B ----------------

@pytest.fixture(scope="module")
def portfolio():
    return load_portfolio()


def _pnl(res, instrument):
    return res.assets.loc[res.assets.instrument == instrument, "pnl"].sum()


def test_portfolio_has_mixed_asset_types(portfolio):
    assert {"Loan", "Bond", "Derivative", "Equity"} <= set(portfolio.asset_type)


def test_zero_shock_zero_pnl(portfolio):
    r = run_stress(portfolio, Shock())
    assert r.pnl == 0 and r.value_after == pytest.approx(r.value_before)


def test_rates_up_hurts_bonds_helps_pay_fixed_swaps(portfolio):
    r = run_stress(portfolio, Shock(rate_bp=100))
    assert (r.assets.loc[r.assets.asset_type == "Bond", "pnl"] < 0).all()
    assert _pnl(r, "Pay-Fixed Interest Rate Swap") > 0


def test_spread_widening_helps_cds_protection(portfolio):
    r = run_stress(portfolio, Shock(spread_bp=100))
    assert _pnl(r, "CDS Protection Bought") > 0
    assert r.by_asset_type.loc["Loan", "pnl"] < 0


def test_equity_drop_helps_puts_hurts_stocks(portfolio):
    r = run_stress(portfolio, Shock(equity_pct=-10))
    assert _pnl(r, "Equity Index Put Option") > 0 and _pnl(r, "Common Stock") < 0


def test_rating_multiplier_and_idiosyncratic_shock(portfolio):
    r = run_stress(portfolio, Shock(idio_equity_pct=-20, idio_spread_bp=100, ticker="GS"))
    hit = r.assets[r.assets.pnl != 0]
    assert set(hit.linked_ticker) == {"GS"} and set(hit.asset_type) == {"Bond", "Equity"}
    gs_eq = r.assets[(r.assets.linked_ticker == "GS") & (r.assets.asset_type == "Equity")]
    assert gs_eq.pnl.iloc[0] == pytest.approx(-0.20 * gs_eq.risk_notional.iloc[0])
    # same spread move costs a B loan more per unit of spread risk than an A loan
    r2 = run_stress(portfolio, Shock(spread_bp=100)).assets
    per_unit = -r2.pnl / (r2.spread_duration * r2.risk_notional)
    assert per_unit[r2.rating == "B"].iloc[0] > per_unit[r2.rating == "A"].iloc[0]


def test_trigger_only_when_impact_strictly_above_threshold(portfolio):
    st = StressTester(portfolio)
    st.on_signal(_sig("2026-01-05 10:00", "", -0.8, impact=7.0, event_type="Geopolitical", scope="market"))
    assert not st.triggers and st.suppressed["below_threshold"] == 1
    st.on_signal(_sig("2026-01-05 11:00", "", -0.8, impact=7.1, event_type="Geopolitical", scope="market"))
    assert len(st.triggers) == 1
    res = st.triggers[0]
    assert res.shock.equity_pct == pytest.approx(SCENARIOS["Geopolitical"].shock.equity_pct * 0.71)
    assert res.pnl < 0 and res.signal["impact"] == 7.1


def test_trigger_cooldown_and_adverse_only(portfolio):
    st = StressTester(portfolio)
    st.on_signal(_sig("2026-01-05 10:00", "GS", -0.9, impact=9, event_type="Credit Event"))
    st.on_signal(_sig("2026-01-05 20:00", "GS", -0.9, impact=9, event_type="Credit Event"))  # same story
    st.on_signal(_sig("2026-01-06 11:00", "GS", -0.9, impact=9, event_type="Credit Event"))  # >24h later
    st.on_signal(_sig("2026-01-05 12:00", "AAPL", +0.9, impact=9, event_type="Earnings"))  # beat: no stress
    st.on_signal(_sig("2026-01-05 12:00", "AAPL", -0.9, impact=9, event_type="Other"))  # no scenario
    assert len(st.triggers) == 2
    assert st.suppressed == {"below_threshold": 0, "no_scenario": 1, "not_adverse": 1, "cooldown": 1}


def test_event_stress_over_bus(portfolio):
    sig = pd.DataFrame([_sig("2026-01-05 10:00", "", -0.8, impact=9, event_type="Macroeconomic", scope="market"),
                        _sig("2026-01-05 11:00", "", -0.8, impact=5, event_type="Macroeconomic", scope="market")])
    st = run_event_stress(sig, portfolio)
    assert len(st.triggers) == 1 and len(st.summary()) == 1


# ---------------- Dashboard ----------------

APP = str(Path(__file__).resolve().parents[1] / "app.py")


def test_dashboard_runs_end_to_end():
    from src.engine.pipeline import MODEL_PATH, OUT, main
    if not (MODEL_PATH.exists() and (OUT / "signals.csv").exists()):
        main()
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(APP, default_timeout=180).run()
    assert not at.exception, at.exception
    assert len(at.tabs) == 3 and not at.error
    at.button[0].click().run()  # "Analyze" in the try-it-live box scores text with the saved model
    assert not at.exception, at.exception
    assert any(m.label == "Sentiment" for m in at.metric)


def test_dashboard_explains_missing_signals(monkeypatch, tmp_path):
    import src.engine.pipeline as pl
    monkeypatch.setattr(pl, "OUT", tmp_path)  # app reads OUT at import time
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    assert "python -m src.engine.pipeline" in at.code[0].value
