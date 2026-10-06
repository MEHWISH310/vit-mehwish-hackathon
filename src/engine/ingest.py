"""Data ingestion: two heterogeneous sources -> one clean, unified schema.

Sources
  1. financial news articles   (data/news_articles.csv)
  2. social posts (X-style)    (data/social_posts.csv)

Unified columns
  source_id, source_type ('news'|'social'), timestamp, ticker, scope ('company'|'market'),
  raw_text, text (entity-masked, cleaned), and optional gt_* labels (training/eval only).
"""
import re
from pathlib import Path

import pandas as pd

from src.universe import DATA, find_ticker, mask_entities

RT_PREFIX = re.compile(r"^\s*RT\s+@\w+:\s*", re.IGNORECASE)
URL = re.compile(r"https?://\S+")
MENTION = re.compile(r"@\w+")
HASHTAG = re.compile(r"#(\w+)")
GT = ["gt_sentiment", "gt_event", "gt_impact"]


def clean_text(raw: str, source_type: str) -> str:
    t = str(raw)
    if source_type == "social":
        t = RT_PREFIX.sub("", t)
        t = URL.sub(" ", t)
        t = MENTION.sub(" ", t)
        t = HASHTAG.sub(r"\1", t)  # keep the word, drop the '#'
    return re.sub(r"\s+", " ", mask_entities(t)).strip()


def _finish(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["raw_text"] = df["raw_text"].fillna("").astype(str)
    # entity linking: trust the feed's ticker if present, else detect from text
    given = df["ticker"].fillna("").astype(str).str.strip()
    detected = df["raw_text"].map(lambda s: find_ticker(s) or "")
    df["ticker"] = given.where(given != "", detected)
    df["scope"] = df["ticker"].map(lambda t: "company" if t else "market")
    df["text"] = [clean_text(r, s) for r, s in zip(df["raw_text"], df["source_type"])]
    df = df[(df["text"].str.len() > 3) & df["timestamp"].notna()]  # drop empty / undated rows
    df = df.drop_duplicates(subset="source_id")
    keep = ["source_id", "source_type", "timestamp", "ticker", "scope", "raw_text", "text"]
    return df[keep + [c for c in GT if c in df.columns]].sort_values("timestamp").reset_index(drop=True)


def load_news(path: Path = DATA / "news_articles.csv") -> pd.DataFrame:
    d = pd.read_csv(path)
    d = d.rename(columns={"id": "source_id", "body": "raw_text"})
    d["source_type"] = "news"
    return _finish(d)


def load_social(path: Path = DATA / "social_posts.csv") -> pd.DataFrame:
    d = pd.read_csv(path)
    d = d.rename(columns={"id": "source_id", "text": "raw_text"})
    d["source_type"] = "social"
    return _finish(d)


def load_corpus(data_dir: Path = DATA) -> pd.DataFrame:
    """Both sources merged into one time-ordered stream."""
    c = pd.concat([load_news(data_dir / "news_articles.csv"), load_social(data_dir / "social_posts.csv")])
    return c.sort_values("timestamp").reset_index(drop=True)


def from_raw_text(text: str, source_type: str = "news") -> pd.DataFrame:
    """Wrap one ad-hoc string (used by the API's /analyze endpoint)."""
    d = pd.DataFrame([dict(source_id="adhoc", source_type=source_type, timestamp=pd.Timestamp.now(),
                           ticker="", raw_text=text)])
    return _finish(d)