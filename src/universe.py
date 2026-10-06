"""Stock universe + entity linking (which company does a piece of text talk about?)."""
import re
from functools import lru_cache
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

# extra names people use in text, besides the official short name in universe.csv
ALIASES = {
    "GOOGL": ["Alphabet", "Google"],
    "META": ["Meta Platforms", "Meta", "Facebook"],
    "JPM": ["JPMorgan Chase", "JPMorgan"],
    "BAC": ["Bank of America"],
    "GS": ["Goldman Sachs", "Goldman"],
    "JNJ": ["Johnson & Johnson", "J&J"],
    "XOM": ["Exxon Mobil", "Exxon"],
    "KO": ["Coca-Cola", "Coca Cola"],
}
CASHTAG = re.compile(r"\$([A-Za-z]{1,5})\b")


@lru_cache(maxsize=1)
def load_universe() -> pd.DataFrame:
    return pd.read_csv(DATA / "universe.csv")


@lru_cache(maxsize=1)
def _name_patterns():
    u = load_universe()
    pats = []
    for _, r in u.iterrows():
        names = {r["name"], *ALIASES.get(r["ticker"], [])}
        for n in names:
            pats.append((r["ticker"], n))
    pats.sort(key=lambda x: -len(x[1]))  # longest first: "Meta Platforms" before "Meta"
    return [(t, re.compile(r"\b" + re.escape(n) + r"\b", re.IGNORECASE)) for t, n in pats]


def find_ticker(text: str):
    """Return the first universe ticker mentioned in `text` (cashtag first, then company name)."""
    valid = set(load_universe().ticker)
    for m in CASHTAG.finditer(text):
        if m.group(1).upper() in valid:
            return m.group(1).upper()
    for tk, pat in _name_patterns():
        if pat.search(text):
            return tk
    return None


def mask_entities(text: str) -> str:
    """Replace company names / cashtags with the neutral word 'company'.

    Stops the models from memorising 'AAPL is usually positive' instead of
    reading the language of the sentence."""
    valid = set(load_universe().ticker)
    text = CASHTAG.sub(lambda m: "company" if m.group(1).upper() in valid else m.group(0), text)
    for _, pat in _name_patterns():
        text = pat.sub("company", text)
    return text