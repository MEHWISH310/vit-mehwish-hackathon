import numpy as np
import pandas as pd
import pytest

from src.modules.rebalancer import (RebalanceConfig, SentimentRebalancer, bound_weights, one_way_turnover,
                                    run_backtest)

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
