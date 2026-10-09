"""Real-data mode: the same engine applied to REAL public text and prices (data/real/, see scripts/fetch_real_data.py).

    python -m src.engine.real       # after python -m src.engine.pipeline

What changes vs the synthetic pipeline
  * Sentiment is retrained on real labelled financial text (Financial PhraseBank + finance tweets), because the
    synthetic-trained model does not generalise to real wording (measured below, reported in README).
  * Event type and impact keep the synthetic-trained models (no real labels exist for them).
  * Input is real, dated news headlines (Google News RSS) instead of generated articles.

Outputs
  outputs/real_signals.csv          same schema as outputs/signals.csv, consumable by Module A / Module B
  outputs/real_eval_metrics.json    synthetic-trained vs real-trained sentiment on held-out real text
  models/engine_real.joblib
"""
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline, make_union

from src.engine.ingest import _finish, clean_text
from src.engine.pipeline import MODEL_PATH, OUT, RiskEngine
from src.engine.sentiment import LexiconSentiment
from src.universe import DATA, ROOT

REAL = DATA / "real"
REAL_MODEL_PATH = ROOT / "models" / "engine_real.joblib"
LABELS = ["negative", "neutral", "positive"]


class RealSentimentModel:
    """3-class classifier (word + character n-gram TF-IDF, logistic regression) trained on real labelled text.
    score = w_lexicon * lexicon + (1 - w_lexicon) * (P(positive) - P(negative))  in [-1, 1].
    w_lexicon is picked by cross-validation on the TRAINING data only."""

    def __init__(self, w_lexicon: float = 0.0):
        self.w = w_lexicon
        self.lex = LexiconSentiment()
        feats = make_union(TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
                           TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, sublinear_tf=True))
        self.clf = make_pipeline(feats, LogisticRegression(C=4.0, max_iter=3000, class_weight="balanced"))

    def fit(self, texts, labels):
        self.clf.fit(list(texts), list(labels))
        return self

    def predict_label(self, texts) -> np.ndarray:
        return self.clf.predict(list(texts))

    def predict(self, texts) -> np.ndarray:
        texts = list(texts)
        p = self.clf.predict_proba(texts)
        c = list(self.clf.classes_)
        ml = p[:, c.index("positive")] - p[:, c.index("negative")]
        return np.clip(self.w * self.lex.score(texts) + (1 - self.w) * ml, -1, 1) if self.w else ml


def to_class(score, thr: float = 0.2):
    score = np.asarray(score)
    return np.where(score > thr, "positive", np.where(score < -thr, "negative", "neutral"))


def load_labelled():
    """(twitter train, twitter test, phrasebank) with cleaned text in column x."""
    tw_tr = pd.read_csv(REAL / "twitter_fin_sentiment_train.csv")
    tw_te = pd.read_csv(REAL / "twitter_fin_sentiment_valid.csv")
    pb = pd.read_csv(REAL / "phrasebank_allagree.csv")
    for d, src in [(tw_tr, "social"), (tw_te, "social"), (pb, "news")]:
        d["x"] = [clean_text(t, src) for t in d.text]
    return tw_tr, tw_te, pb


def _m(y, pred) -> dict:
    return {"accuracy": round(float(accuracy_score(y, pred)), 3),
            "macro_f1": round(float(f1_score(y, pred, average="macro")), 3)}


def pick_lexicon_weight(tw_tr, grid=(0.0, 0.15, 0.3, 0.5), k: int = 3) -> tuple[float, dict]:
    """Choose the lexicon blend weight by k-fold CV on the twitter TRAIN split (macro-F1 of the thresholded score)."""
    res = {}
    for w in grid:
        pred = np.empty(len(tw_tr), dtype=object)
        for tr, te in StratifiedKFold(k, shuffle=True, random_state=0).split(tw_tr.x, tw_tr.label):
            m = RealSentimentModel(w).fit(tw_tr.x.iloc[tr], tw_tr.label.iloc[tr])
            pred[te] = to_class(m.predict(tw_tr.x.iloc[te]))
        res[w] = round(float(f1_score(tw_tr.label, pred, average="macro")), 4)
    return max(res, key=res.get), res


def evaluate_real(synthetic_engine: RiskEngine, w: float, tw_tr, tw_te, pb) -> dict:
    """Held-out real-text evaluation: twitter (official train/test split) and PhraseBank (5-fold CV)."""
    out = {}
    # 1) twitter: train on train split, test on the held-out validation split
    syn = synthetic_engine.sentiment
    m = RealSentimentModel(w).fit(tw_tr.x, tw_tr.label)
    lex = syn.lex.score(tw_te.x)
    out["twitter_test"] = {
        "n": len(tw_te), "majority_class_baseline": _m(tw_te.label, ["neutral"] * len(tw_te)),
        "lexicon_only": _m(tw_te.label, to_class(lex)),
        "synthetic_trained_engine": _m(tw_te.label, to_class(syn.predict(tw_te.x))),
        "real_trained_score": _m(tw_te.label, to_class(m.predict(tw_te.x))),
        "real_trained_label": _m(tw_te.label, m.predict_label(tw_te.x))}
    # 2) phrasebank: 5-fold CV, each fold's model trained on twitter train + the other 4 PhraseBank folds
    sc, lab = np.zeros(len(pb)), np.empty(len(pb), dtype=object)
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(pb.x, pb.label):
        mm = RealSentimentModel(w).fit(pd.concat([tw_tr.x, pb.x.iloc[tr]]), pd.concat([tw_tr.label, pb.label.iloc[tr]]))
        sc[te], lab[te] = mm.predict(pb.x.iloc[te]), mm.predict_label(pb.x.iloc[te])
    out["phrasebank_5fold"] = {
        "n": len(pb), "majority_class_baseline": _m(pb.label, ["neutral"] * len(pb)),
        "lexicon_only": _m(pb.label, to_class(syn.lex.score(pb.x))),
        "synthetic_trained_engine": _m(pb.label, to_class(syn.predict(pb.x))),
        "real_trained_score": _m(pb.label, to_class(sc)),
        "real_trained_label": _m(pb.label, lab)}
    return out


def load_real_headlines(path=REAL / "news_headlines.csv") -> pd.DataFrame:
    """Real headlines -> the engine's unified schema.

    Timing is deliberately conservative: Google News often stamps only the publication DATE, so a headline is
    treated as known at 09:00 the next day, i.e. it can first move weights at the next trading day's close.
    Headlines fetched by a company query must actually name a universe company, else they are dropped (e.g.
    an 'Apple' query returning fruit-price stories)."""
    d = pd.read_csv(path)
    raw = d.rename(columns={"id": "source_id", "headline": "raw_text"})
    raw["timestamp"] = pd.to_datetime(raw["published"]).dt.normalize() + pd.Timedelta(days=1, hours=9)
    raw["source_type"] = "news"
    raw["ticker"] = ""  # let entity linking decide, exactly as for any other feed
    docs = _finish(raw[["source_id", "source_type", "timestamp", "ticker", "raw_text"]])
    q = d.set_index("id")["query_ticker"].fillna("")
    from_company_query = docs.source_id.map(q).ne("")
    return docs[~(from_company_query & (docs.scope == "market"))].reset_index(drop=True)


def main():
    if not MODEL_PATH.exists():
        raise SystemExit("Run `python -m src.engine.pipeline` first (trains the event and impact models).")
    syn = RiskEngine.load()
    tw_tr, tw_te, pb = load_labelled()
    w, cv = pick_lexicon_weight(tw_tr)
    metrics = {"lexicon_weight_cv_on_twitter_train": {"chosen": w, "macro_f1_by_weight": cv},
               **evaluate_real(syn, w, tw_tr, tw_te, pb)}

    real = RiskEngine()
    real.event, real.impact = syn.event, syn.impact  # no real labels for these: keep synthetic-trained models
    real.sentiment = RealSentimentModel(w).fit(pd.concat([tw_tr.x, pb.x]), pd.concat([tw_tr.label, pb.label]))
    REAL_MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(real, REAL_MODEL_PATH)

    docs = load_real_headlines()
    sig = real.analyze(docs).sort_values("timestamp").reset_index(drop=True)
    sig.to_csv(OUT / "real_signals.csv", index=False)
    metrics["real_headlines"] = {"n_signals": len(sig), "by_scope": sig.scope.value_counts().to_dict(),
                                 "by_event": sig.event_type.value_counts().to_dict(),
                                 "impact_gt7": int((sig.impact > 7).sum())}
    (OUT / "real_eval_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
    print(f"wrote {len(sig)} real signals -> outputs/real_signals.csv; model -> models/engine_real.joblib")


if __name__ == "__main__":
    from src.engine.real import main as _main  # real module path so the pickled model loads from anywhere

    _main()
