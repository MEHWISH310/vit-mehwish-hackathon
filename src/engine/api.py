"""Simple REST API exposing the engine's structured signals to downstream applications.

    uvicorn src.engine.api:app --reload        # then open http://127.0.0.1:8000/docs
"""
from typing import Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

from src.engine.aggregate import aggregate_daily
from src.engine.ingest import from_raw_text
from src.engine.pipeline import MODEL_PATH, OUT, RiskEngine

app = FastAPI(title="NLP Risk Signal API", version="1.0",
              description="Sentiment score, event classification and impact score from news & social text.")
_cache: dict = {}


def _signals() -> pd.DataFrame:
    if "signals" not in _cache:
        f = OUT / "signals.csv"
        if not f.exists():
            raise HTTPException(503, "No signals yet. Run: python -m src.engine.pipeline")
        _cache["signals"] = pd.read_csv(f, parse_dates=["timestamp"]).fillna({"ticker": ""})
    return _cache["signals"]


def _engine() -> RiskEngine:
    if "engine" not in _cache:
        if not MODEL_PATH.exists():
            raise HTTPException(503, "No trained model. Run: python -m src.engine.pipeline")
        _cache["engine"] = RiskEngine.load()
    return _cache["engine"]


class AnalyzeIn(BaseModel):
    text: str
    source_type: str = "news"  # 'news' | 'social'


@app.get("/health")
def health():
    return {"status": "ok", "signals_loaded": len(_signals()) if (OUT / "signals.csv").exists() else 0}


@app.get("/signals")
def get_signals(ticker: Optional[str] = None, event_type: Optional[str] = None, scope: Optional[str] = None,
                min_impact: float = 0.0, since: Optional[str] = None, limit: int = Query(100, le=2000)):
    s = _signals()
    if ticker:
        s = s[s.ticker == ticker.upper()]
    if event_type:
        s = s[s.event_type == event_type]
    if scope:
        s = s[s.scope == scope]
    if since:
        s = s[s.timestamp >= pd.Timestamp(since)]
    s = s[s.impact >= min_impact].sort_values("timestamp", ascending=False).head(limit)
    return s.astype({"timestamp": str}).to_dict("records")


@app.get("/sentiment/{ticker}")
def ticker_sentiment(ticker: str, days: int = 5):
    d = aggregate_daily(_signals())
    d = d[d.ticker == ticker.upper()].sort_values("date").tail(days)
    if d.empty:
        raise HTTPException(404, f"No signals for {ticker}")
    return d.astype({"date": str}).to_dict("records")


@app.post("/analyze")
def analyze(body: AnalyzeIn):
    """Score any piece of text on the fly."""
    docs = from_raw_text(body.text, body.source_type)
    if docs.empty:
        raise HTTPException(422, "Text is empty after cleaning")
    sig = _engine().analyze(docs).iloc[0].to_dict()
    sig["timestamp"] = str(sig["timestamp"])
    return sig