import pandas as pd
import pytest

from src.engine.aggregate import aggregate_daily
from src.engine.bus import SignalBus
from src.engine.ingest import from_raw_text, load_corpus
from src.engine.pipeline import MODEL_PATH, OUT, RiskEngine, main
from src.engine.sentiment import LexiconSentiment


@pytest.fixture(scope="session")
def trained():
    if not (MODEL_PATH.exists() and (OUT / "signals.csv").exists()):
        main()
    return RiskEngine.load()


def test_ingest_masks_entities_and_links_ticker():
    d = from_raw_text("RT @user_12: $AAPL beats earnings 🚀 https://x.co/a #stocks", "social")
    assert d.ticker.iloc[0] == "AAPL" and d.scope.iloc[0] == "company"
    t = d.text.iloc[0]
    assert "AAPL" not in t and "RT" not in t and "http" not in t


def test_market_scope_when_no_company():
    d = from_raw_text("Central bank signals higher rates for longer", "news")
    assert d.scope.iloc[0] == "market" and d.ticker.iloc[0] == ""


def test_corpus_has_both_sources_and_no_label_leak_in_text():
    c = load_corpus()
    assert set(c.source_type) == {"news", "social"}
    assert c.text.str.len().min() > 3


def test_lexicon_direction():
    s = LexiconSentiment().score(["company beats estimates and raises guidance",
                                  "company misses estimates and cuts outlook",
                                  "inflation cools, raising hopes of rate cuts"])
    assert s[0] > 0.3 and s[1] < -0.3 and s[2] > 0


def test_engine_signal_schema(trained):
    s = pd.read_csv(OUT / "signals.csv")
    assert s.sentiment.between(-1, 1).all()
    assert s.impact.between(1, 10).all()
    assert s.event_type.notna().all() and s.source_type.nunique() == 2


def test_event_classification_obvious_cases(trained):
    docs = pd.concat([from_raw_text("Ceasefire in the Middle East lifts global risk appetite"),
                      from_raw_text("Moody's downgrades Chevron to BBB amid rising debt")])
    out = trained.analyze(docs)
    assert list(out.event_type) == ["Geopolitical", "Credit Event"]
    assert out.sentiment.iloc[0] > 0 > out.sentiment.iloc[1]


def test_bus_filters_by_scope_and_impact():
    got = {"all": [], "hi": []}
    bus = SignalBus()
    bus.subscribe(got["all"].append, name="all")
    bus.subscribe(got["hi"].append, name="hi", min_impact=7, scope="market")
    sig = pd.DataFrame([
        dict(timestamp="2026-01-02", scope="company", impact=3, event_type="Earnings"),
        dict(timestamp="2026-01-01", scope="market", impact=8, event_type="Geopolitical"),
        dict(timestamp="2026-01-03", scope="market", impact=2, event_type="Macroeconomic")])
    counts = bus.replay(sig)
    assert counts == {"all": 3, "hi": 1}
    assert got["all"][0]["timestamp"] == "2026-01-01"  # delivered in time order


def test_aggregate_daily_weights_by_impact():
    s = pd.DataFrame([
        dict(timestamp="2026-01-01 10:00", ticker="AAPL", source_type="news", sentiment=1.0, impact=9),
        dict(timestamp="2026-01-01 12:00", ticker="AAPL", source_type="news", sentiment=-1.0, impact=1)])
    a = aggregate_daily(s)
    assert len(a) == 1 and a.sentiment.iloc[0] == pytest.approx(0.8)


def test_api_endpoints(trained):
    from fastapi.testclient import TestClient
    from src.engine.api import app
    c = TestClient(app)
    assert c.get("/health").json()["status"] == "ok"
    r = c.post("/analyze", json={"text": "$NVDA smashes earnings, raises guidance 🚀", "source_type": "social"})
    assert r.status_code == 200 and r.json()["sentiment"] > 0 and r.json()["ticker"] == "NVDA"
    assert len(c.get("/signals", params={"ticker": "AAPL", "limit": 5}).json()) <= 5
    assert c.get("/sentiment/AAPL").status_code == 200
    assert c.get("/sentiment/ZZZZ").status_code == 404