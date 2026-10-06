"""Event classification (what happened?) and impact scoring (how much will it matter, 1-10?)."""
import re

import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge

EVENT_LABELS = ["Earnings", "Credit Event", "Merger/Acquisition", "Product Launch",
                "Regulatory/Legal", "Geopolitical", "Macroeconomic", "Other"]

# Rule-based safety net, used only when the ML classifier is unsure.
RULES = {
    "Earnings": r"earnings|profit|revenue|guidance|quarter|\beps\b|margin|sales|forecast|outlook|volumes|dividend",
    "Credit Event": r"downgrad|upgrad|rating|default|covenant|spread|bond|debt|loan|provision|credit|reserves|liquidity",
    "Merger/Acquisition": r"acqui|merger|\bmerge|takeover|buyout|\bdeal\b|\bbought\b|\bbuy\b|stake",
    "Product Launch": r"launch|unveil|introduc|recall|product|release|rollout|shelves|\bgpu\b|\bchip\b|\bdrug\b",
    "Regulatory/Legal": r"regulat|\bfine[ds]?\b|probe|investigat|lawsuit|court|antitrust|approval|\bfda\b|clearance|ruling|appeal",
    "Geopolitical": r"\bwar\b|conflict|sanction|ceasefire|tariff|trade (deal|talks)|military|invasion|peace|clash|naval|fighting",
    "Macroeconomic": r"inflation|\bfed\b|central bank|\bgdp\b|recession|unemployment|rate (cut|hike)|rates|\bjobs\b|\bcpi\b",
}


class EventClassifier:
    def __init__(self, min_conf: float = 0.60):
        self.min_conf = min_conf
        self.vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
        self.clf = LogisticRegression(C=8.0, max_iter=2000)

    def fit(self, texts, labels):
        self.clf.fit(self.vec.fit_transform(list(texts)), list(labels))
        return self

    def predict(self, texts):
        texts = list(texts)
        proba = self.clf.predict_proba(self.vec.transform(texts))
        idx = proba.argmax(1)
        labels = self.clf.classes_[idx].astype(object)
        conf = proba.max(1)
        # keyword rules take over when the model is unsure, or says "Other" although
        # explicit event vocabulary is present ("Other" = no event detected)
        for i in np.where((conf < self.min_conf) | (labels == "Other"))[0]:
            hits = {lab: len(re.findall(pat, texts[i].lower())) for lab, pat in RULES.items()}
            best = max(hits, key=hits.get)
            if hits[best] > 0:
                labels[i] = best
        return labels, conf


class ImpactModel:
    """Predicts market-impact severity on a 1-10 scale from the text + source type."""

    def __init__(self):
        self.vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
        self.reg = Ridge(alpha=1.0)

    def _x(self, texts, source_types, fit=False):
        m = self.vec.fit_transform(texts) if fit else self.vec.transform(texts)
        social = csr_matrix((np.asarray(source_types) == "social").astype(float).reshape(-1, 1))
        return hstack([m, social]).tocsr()

    def fit(self, texts, source_types, y):
        self.reg.fit(self._x(list(texts), source_types, fit=True), np.asarray(y, dtype=float))
        return self

    def predict(self, texts, source_types):
        return np.clip(self.reg.predict(self._x(list(texts), source_types)), 1, 10)