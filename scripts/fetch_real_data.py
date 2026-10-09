"""Download the REAL (public) data used alongside the synthetic corpus. Output goes to data/real/ and is committed,
so the project runs offline; re-run this only to refresh it.

    python scripts/fetch_real_data.py              # everything (~5-10 min, mostly polite pauses between requests)
    python scripts/fetch_real_data.py --only prices

Sources (all free, no API key)
  prices     Yahoo Finance daily closes via yfinance, the 15 index stocks, same window as the synthetic data
  headlines  Google News RSS search: real, dated headlines per company + market-wide topics.
             (GDELT's DOC API was the first choice but rate-limited this network with HTTP 429; Google News
             aggregates the same publishers: Reuters, Bloomberg, CNBC, Yahoo Finance, ...)
  labelled   Financial PhraseBank v1.0 (Malo et al. 2014, CC BY-NC-SA 3.0): 4,846 news sentences labelled by
             finance professionals; and twitter-financial-news-sentiment (zeroshot, MIT): real finance tweets
             labelled bearish / bullish / neutral. Used only to EVALUATE the engine on real text.
"""
import argparse
import email.utils
import html
import io
import re
import sys
import time
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.universe import load_universe  # noqa: E402

REAL = ROOT / "data" / "real"
START, END = "2026-07-06", "2026-10-02"  # same window as the synthetic data
UA = {"User-Agent": "Mozilla/5.0 (student hackathon research)"}

# how people actually search for each company in news
QUERY_NAME = {"AAPL": "Apple", "MSFT": "Microsoft", "NVDA": "Nvidia", "GOOGL": "Alphabet OR Google",
              "META": "Meta Platforms OR Facebook", "AMZN": "Amazon", "WMT": "Walmart", "KO": "Coca-Cola",
              "JPM": "JPMorgan", "BAC": "Bank of America", "GS": "Goldman Sachs", "XOM": "Exxon",
              "CVX": "Chevron", "JNJ": "Johnson & Johnson", "PFE": "Pfizer"}
MARKET_TOPICS = ['"Federal Reserve" OR "interest rates" OR "rate cut" OR "rate hike"',
                 'inflation OR recession OR "jobs report" OR GDP',
                 'tariffs OR "trade war" OR sanctions',
                 'war OR conflict OR geopolitical OR ceasefire',
                 '"credit rating" OR downgrade OR default OR "bond market"',
                 '"oil prices" OR OPEC']


def get(url: str, tries: int = 4) -> bytes:
    for k in range(tries):
        try:
            return urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60).read()
        except Exception as e:  # noqa: BLE001 - network: back off and retry
            if k == tries - 1:
                raise
            print(f"  retry after {e}", flush=True)
            time.sleep(5 * (k + 1))


def fetch_prices():
    import yfinance as yf
    tickers = list(load_universe().ticker)
    end = (pd.Timestamp(END) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    px = yf.download(tickers, start=START, end=end, auto_adjust=True, progress=False)["Close"]
    px = px.dropna(how="all").round(2)
    long = px.reset_index(names="date").melt(id_vars="date", var_name="ticker", value_name="close").dropna()
    long["date"] = pd.to_datetime(long["date"]).dt.strftime("%Y-%m-%d")
    long.sort_values(["date", "ticker"]).to_csv(REAL / "prices_yahoo.csv", index=False)
    print(f"prices: {len(long)} rows, {long.date.min()} -> {long.date.max()}, {long.ticker.nunique()} tickers")


def _rss(query: str, after: pd.Timestamp, before: pd.Timestamp):
    q = f"{query} after:{after:%Y-%m-%d} before:{before:%Y-%m-%d}"
    url = "https://news.google.com/rss/search?" + urllib.parse.urlencode(
        {"q": q, "hl": "en-US", "gl": "US", "ceid": "US:en"})
    x = get(url).decode("utf-8", "replace")
    for it in re.findall(r"<item>(.*?)</item>", x, re.S):
        title = html.unescape(re.search(r"<title>(.*?)</title>", it, re.S).group(1)).strip()
        src = re.search(r"<source[^>]*>(.*?)</source>", it, re.S)
        date = re.search(r"<pubDate>(.*?)</pubDate>", it)
        link = re.search(r"<link>(.*?)</link>", it)
        yield {"title": title, "publisher": html.unescape(src.group(1)) if src else "",
               "published": email.utils.parsedate_to_datetime(date.group(1)).strftime("%Y-%m-%d %H:%M") if date else "",
               "url": link.group(1) if link else ""}


def fetch_headlines(market_only: bool = False):
    """market_only: re-fetch just the market-wide topics and keep the existing company headlines."""
    weeks = pd.date_range(START, END, freq="7D")
    jobs = [] if market_only else [(t, f'"{q}"' if " OR " not in q else f"({q})") for t, q in QUERY_NAME.items()]
    jobs += [("", f"({q})") for q in MARKET_TOPICS]
    rows = []
    for n, (ticker, q) in enumerate(jobs, 1):
        got = 0
        for w in weeks:
            for r in _rss(q, w, w + pd.Timedelta(days=7)):
                rows.append({"query_ticker": ticker, "query": q, **r})
                got += 1
            time.sleep(1.5)  # be polite
        print(f"[{n}/{len(jobs)}] {ticker or 'market'} {q}: {got}", flush=True)
    d = pd.DataFrame(rows)
    if market_only:
        old = pd.read_csv(REAL / "news_headlines.csv").rename(columns={"headline": "title"})
        d = pd.concat([old[old.query_ticker.notna()].drop(columns="id"), d])
    # Google appends " - Publisher" to titles; keep the headline itself
    d["headline"] = [re.sub(r"\s+-\s+" + re.escape(p) + r"\s*$", "", t) if p else t for t, p in zip(d.title, d.publisher)]
    d = d[(d.published >= START) & (d.published <= END + " 23:59")]
    d = d.drop_duplicates(subset="headline").sort_values("published").reset_index(drop=True)
    d.insert(0, "id", [f"G{i:05d}" for i in range(len(d))])
    # URLs are Google redirect links (~300 chars each) and would triple the file size; publisher + headline + date
    # identify each item, and re-running this script reproduces the set
    d[["id", "published", "query_ticker", "publisher", "headline"]].to_csv(REAL / "news_headlines.csv", index=False)
    print(f"headlines: {len(d)} unique, {d.published.min()} -> {d.published.max()}")


def fetch_labelled():
    z = zipfile.ZipFile(io.BytesIO(get(
        "https://huggingface.co/datasets/takala/financial_phrasebank/resolve/main/data/FinancialPhraseBank-v1.0.zip")))
    name = next(n for n in z.namelist() if n.endswith("Sentences_AllAgree.txt"))
    rows = []
    for line in z.read(name).decode("latin-1").splitlines():
        if "@" in line:
            text, label = line.rsplit("@", 1)
            rows.append({"text": text.strip(), "label": label.strip()})
    pb = pd.DataFrame(rows)
    pb.to_csv(REAL / "phrasebank_allagree.csv", index=False)
    print(f"phrasebank (100% annotator agreement): {len(pb)} sentences {pb.label.value_counts().to_dict()}")

    for split in ["train", "valid"]:  # official split: train is used for training, valid only for testing
        tw = pd.read_csv(io.BytesIO(get(
            f"https://huggingface.co/datasets/zeroshot/twitter-financial-news-sentiment/resolve/main/sent_{split}.csv")))
        tw["label"] = tw["label"].map({0: "negative", 1: "positive", 2: "neutral"})
        tw.to_csv(REAL / f"twitter_fin_sentiment_{split}.csv", index=False)
        print(f"twitter ({split} split): {len(tw)} tweets {tw.label.value_counts().to_dict()}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["prices", "headlines", "headlines-market", "labelled"])
    a = ap.parse_args()
    REAL.mkdir(parents=True, exist_ok=True)
    for name, fn in [("prices", fetch_prices), ("labelled", fetch_labelled), ("headlines", fetch_headlines)]:
        if a.only in (None, name):
            fn()
    if a.only == "headlines-market":
        fetch_headlines(market_only=True)
