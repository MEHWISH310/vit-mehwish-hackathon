from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.engine.real import REAL_MODEL_PATH, RealSentimentModel, load_real_headlines, to_class

APP = str(Path(__file__).resolve().parents[1] / "app.py")


def test_real_sentiment_model_learns_direction():
    pos = ["profit rises sharply, beats estimates", "shares jump on record sales", "upgrade lifts stock",
           "strong growth, raises guidance", "revenue surges to a record"] * 4
    neg = ["profit falls, misses estimates", "shares slump on weak sales", "downgrade hits stock",
           "losses widen, cuts guidance", "revenue plunges amid weak demand"] * 4
    neu = ["company to hold annual meeting", "board meets on tuesday", "company announces date of report",
           "annual meeting scheduled in may", "company publishes report date"] * 4
    m = RealSentimentModel().fit(pos + neg + neu, ["positive"] * 20 + ["negative"] * 20 + ["neutral"] * 20)
    s = m.predict(["record profit beats estimates", "sales plunge, misses estimates"])
    assert s[0] > 0.2 > -0.2 > s[1]
    assert np.abs(m.predict(pos + neg)).max() <= 1
    assert list(to_class([0.5, 0.0, -0.5])) == ["positive", "neutral", "negative"]


def test_real_headlines_timing_and_entity_filter(tmp_path):
    f = tmp_path / "h.csv"
    pd.DataFrame([
        dict(id="G1", published="2026-08-03 14:00", query_ticker="AAPL", publisher="X", headline="Apple beats estimates", url=""),
        dict(id="G2", published="2026-08-03 07:00", query_ticker="AAPL", publisher="X", headline="Fruit prices rise in July", url=""),
        dict(id="G3", published="2026-08-03 07:00", query_ticker="", publisher="X", headline="Fed holds rates steady", url=""),
        dict(id="G4", published="2026-08-04 07:00", query_ticker="", publisher="X", headline="Goldman Sachs sees oil at $85", url=""),
    ]).to_csv(f, index=False)
    d = load_real_headlines(f).set_index("source_id")
    assert "G2" not in d.index  # company query, but no universe company named -> dropped
    assert d.loc["G1", "ticker"] == "AAPL" and d.loc["G3", "scope"] == "market" and d.loc["G4", "ticker"] == "GS"
    # conservative timing: known at 09:00 the next day, i.e. first usable at the next close
    assert d.loc["G1", "timestamp"] == pd.Timestamp("2026-08-04 09:00")


@pytest.fixture(scope="module")
def real_outputs():
    from src.engine.pipeline import MODEL_PATH, OUT
    from src.engine.pipeline import main as synthetic_main
    from src.engine.real import main as real_main
    if not MODEL_PATH.exists():
        synthetic_main()
    if not ((OUT / "real_signals.csv").exists() and REAL_MODEL_PATH.exists()):
        real_main()
    return OUT


def test_real_signals_schema_and_beats_synthetic_on_real_text(real_outputs):
    import json
    s = pd.read_csv(real_outputs / "real_signals.csv")
    assert len(s) > 500 and s.sentiment.between(-1, 1).all() and s.impact.between(1, 10).all()
    assert {"company", "market"} <= set(s.scope)
    ev = json.loads((real_outputs / "real_eval_metrics.json").read_text())
    for k in ["twitter_test", "phrasebank_5fold"]:
        assert ev[k]["real_trained_label"]["macro_f1"] > ev[k]["synthetic_trained_engine"]["macro_f1"]


def test_dashboard_real_mode(real_outputs):
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(APP, default_timeout=180).run()
    at.sidebar.radio[0].set_value(at.sidebar.radio[0].options[1]).run()
    assert not at.exception, at.exception
    assert len(at.tabs) == 3 and not at.error
    at.button[0].click().run()
    assert not at.exception, at.exception
