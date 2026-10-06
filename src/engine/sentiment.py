"""Sentiment score in [-1, 1] = blend of
   (a) a finance-tuned lexicon model (VADER + finance vocabulary): works with zero training data
   (b) a TF-IDF + Ridge regressor trained on labelled text: learns corpus-specific phrasing
The blend is more robust to unseen wording than either alone (see outputs/eval_metrics.json)."""
import re

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Finance-specific valence (VADER's default lexicon is built for general English).
FIN_LEXICON = {
    # positive
    "beats": 2.0, "beat": 1.8, "tops": 2.0, "raises": 1.5, "raised": 1.5, "record": 1.5, "upgrade": 2.5,
    "upgrades": 2.5, "upgraded": 2.5, "rally": 2.5, "surge": 2.5, "surges": 2.5, "jump": 2.0, "jumps": 2.0,
    "bullish": 2.5, "approval": 2.0, "approved": 2.0, "clearance": 1.8, "synergies": 1.5, "ceasefire": 2.0,
    "accord": 1.5, "peace": 2.0, "cools": 1.2, "cooling": 1.2, "slows": 0.8, "easing": 1.5, "stronger": 1.8,
    "strong": 1.8, "outperform": 2.0, "dividend": 1.0, "refinances": 1.0, "wins": 2.0, "win": 1.8,
    "swell": 1.5, "ripping": 2.0, "printing": 1.5, "solid": 1.5, "crushing": 2.0, "moon": 2.0, "ratecut": 2.0,
    "landmark": 1.5, "boosts": 1.8, "boost": 1.5, "lifts": 1.5, "lifting": 1.5,
    # negative
    "misses": -2.0, "missed": -2.0, "miss": -1.8, "downgrade": -2.5, "downgrades": -2.5, "downgraded": -2.5,
    "cuts": -1.8, "cut": -1.5, "slashes": -2.0, "recall": -2.2, "recalls": -2.2, "pulls": -1.5, "fined": -2.2,
    "fine": -0.5, "probe": -2.0, "investigation": -2.0, "antitrust": -1.5, "default": -2.5, "defaults": -2.5,
    "breach": -2.5, "plunge": -3.0, "tumble": -2.5, "tumbles": -2.5, "selloff": -2.5, "slide": -1.8,
    "sliding": -1.8, "bearish": -2.5, "recession": -2.5, "slowdown": -2.0, "contracts": -1.5, "sanctions": -2.0,
    "conflict": -2.0, "clashes": -2.0, "war": -2.5, "escalating": -2.0, "tariffs": -1.5, "delays": -1.5,
    "delay": -1.5, "delaying": -1.5, "shrink": -1.5, "shrinking": -1.5, "weak": -1.8, "softer": -1.5,
    "widen": -1.5, "widening": -1.5, "worries": -1.8, "concerns": -1.5, "hammered": -2.5, "bleed": -2.5,
    "ouch": -1.5, "overpays": -1.8, "overpaying": -1.8, "loses": -1.8, "scraps": -1.5, "walks": -1.0,
    "disappoints": -2.2, "disappoint": -2.0, "soured": -2.2, "dilution": -1.5, "downside": -1.5,
    "ratehike": -2.0, "elevated": -0.8, "unemployment": -1.0, "spiking": -1.2,
    "ugly": -1.8, "yikes": -1.8, "provisions": -0.8, "warns": -1.8, "warn": -1.5, "warning": -1.5,
    "lawsuit": -1.2, "shock": -1.8, "chaos": -2.0, "trade-war": -2.0, "pulled": -1.5, "negative": -1.5,
    # emoji (padded with spaces before scoring)
    "🚀": 2.0, "📈": 1.5, "🟢": 1.2, "✅": 1.0, "🎉": 1.5, "🔥": 1.0,
    "📉": -1.8, "😬": -1.2, "⚠️": -1.2, "😰": -1.8, "🚨": -1.5,
}
_EMOJI = re.compile("([" + "".join(k for k in FIN_LEXICON if not k.isascii() and len(k) == 1) + "])")


def _prep(text: str) -> str:
    t = re.sub(r"\brate cuts?\b", "ratecut", text, flags=re.IGNORECASE)  # 'rate cut' is good news,
    t = re.sub(r"\brate hikes?\b", "ratehike", t, flags=re.IGNORECASE)  # but 'cuts outlook' is bad
    t = t.replace("⚠️", " ⚠️ ")
    return _EMOJI.sub(r" \1 ", t)


class LexiconSentiment:
    def __init__(self):
        self.an = SentimentIntensityAnalyzer()
        self.an.lexicon.update(FIN_LEXICON)

    def score(self, texts) -> np.ndarray:
        return np.array([self.an.polarity_scores(_prep(t))["compound"] for t in texts])


class SentimentModel:
    def __init__(self, w_lexicon: float = 0.7):
        self.w = w_lexicon
        self.lex = LexiconSentiment()
        self.ml = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
                                Ridge(alpha=1.0))

    def fit(self, texts, y):
        self.ml.fit(list(texts), np.asarray(y, dtype=float))
        return self

    def components(self, texts):
        texts = list(texts)
        return self.lex.score(texts), np.clip(self.ml.predict(texts), -1, 1)

    def predict(self, texts) -> np.ndarray:
        lex, ml = self.components(texts)
        return np.clip(self.w * lex + (1 - self.w) * ml, -1, 1)