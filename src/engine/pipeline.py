"""The AI/NLP Risk Engine: text in -> structured risk signals out.

    python -m src.engine.pipeline        # train, score the whole corpus, write outputs/, evaluate

Signal schema (one row per article/post)
  signal_id, source_id, source_type, timestamp, ticker, scope,
  sentiment (-1..1), event_type, impact (1..10), event_confidence, headline
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error
from sklearn.model_selection import KFold

from src.engine.events import EVENT_LABELS, EventClassifier, ImpactModel
from src.engine.ingest import from_raw_text, load_corpus, clean_text
from src.engine.sentiment import SentimentModel
from src.universe import ROOT

OUT = ROOT / "outputs"
MODEL_PATH = ROOT / "models" / "engine.joblib"
EVAL_PATH = ROOT / "data" / "eval" / "handwritten_eval.csv"  # dev set: used while tuning
HOLDOUT_PATH = ROOT / "data" / "eval" / "handwritten_holdout.csv"  # written before tuning; never tuned on
SIGNAL_COLS = ["signal_id", "source_id", "source_type", "timestamp", "ticker", "scope",
               "sentiment", "event_type", "impact", "event_confidence", "headline"]


class RiskEngine:
    def __init__(self):
        self.sentiment = SentimentModel()
        self.event = EventClassifier()
        self.impact = ImpactModel()

    def fit(self, docs: pd.DataFrame):
        self.sentiment.fit(docs.text, docs.gt_sentiment)
        self.event.fit(docs.text, docs.gt_event)
        self.impact.fit(docs.text, docs.source_type, docs.gt_impact)
        return self

    def analyze(self, docs: pd.DataFrame) -> pd.DataFrame:
        s = self.sentiment.predict(docs.text)
        ev, conf = self.event.predict(docs.text)
        imp = self.impact.predict(docs.text, docs.source_type)
        out = pd.DataFrame({
            "signal_id": ["S" + str(i) for i in docs.source_id],
            "source_id": docs.source_id.values, "source_type": docs.source_type.values,
            "timestamp": docs.timestamp.values, "ticker": docs.ticker.values, "scope": docs.scope.values,
            "sentiment": np.round(s, 3), "event_type": ev, "impact": np.round(imp, 1),
            "event_confidence": np.round(conf, 3),
            "headline": docs.raw_text.str.slice(0, 140).values,
        })
        return out[SIGNAL_COLS]

    def save(self, path: Path = MODEL_PATH):
        path.parent.mkdir(exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: Path = MODEL_PATH) -> "RiskEngine":
        return joblib.load(path)


def out_of_fold_signals(docs: pd.DataFrame, k: int = 5, seed: int = 0) -> pd.DataFrame:
    """Score every document with a model that never saw it (k-fold), so the signals the
    downstream modules consume are not memorised training rows."""
    parts = []
    for tr, te in KFold(k, shuffle=True, random_state=seed).split(docs):
        parts.append(RiskEngine().fit(docs.iloc[tr]).analyze(docs.iloc[te]))
    return pd.concat(parts).sort_values("timestamp").reset_index(drop=True)


def _sent_metrics(pred, true):
    pred, true = np.asarray(pred), np.asarray(true)
    cls = lambda x: np.where(x > 0.2, "pos", np.where(x < -0.2, "neg", "neu"))
    return {"mae": round(float(mean_absolute_error(true, pred)), 3),
            "pearson_r": round(float(np.corrcoef(pred, true)[0, 1]), 3),
            "3class_acc": round(float(accuracy_score(cls(true), cls(pred))), 3)}


def evaluate(engine_factory_docs: pd.DataFrame, oof: pd.DataFrame) -> dict:
    """(1) out-of-fold accuracy on the synthetic corpus; (2) accuracy on 35 independent hand-written
    texts whose wording is NOT drawn from the generator's templates (the honest generalisation test)."""
    docs = engine_factory_docs.sort_values("timestamp").reset_index(drop=True)
    m = docs.set_index("source_id").join(oof.set_index("source_id"), rsuffix="_p")
    res = {"synthetic_out_of_fold": {
        "sentiment": _sent_metrics(m.sentiment, m.gt_sentiment),
        "event_accuracy": round(float(accuracy_score(m.gt_event, m.event_type)), 3),
        "event_macro_f1": round(float(f1_score(m.gt_event, m.event_type, average="macro")), 3),
        "impact_mae": round(float(mean_absolute_error(m.gt_impact, m.impact)), 2)}}

    eng = RiskEngine().fit(docs)
    for name, path in [("handwritten_dev_35", EVAL_PATH), ("handwritten_holdout_20", HOLDOUT_PATH)]:
        hw = pd.read_csv(path)
        hw["raw_text"] = hw.text
        hw["text"] = [clean_text(t, s) for t, s in zip(hw.text, hw.source_type)]
        lex, ml = eng.sentiment.components(hw.text)
        sig = eng.analyze(hw.assign(source_id=[f"H{i}" for i in range(len(hw))], ticker="", scope="market",
                                    timestamp=pd.Timestamp.now()))
        res[name] = {
            "sentiment_lexicon_only": _sent_metrics(lex, hw.gt_sentiment),
            "sentiment_ml_only": _sent_metrics(ml, hw.gt_sentiment),
            "sentiment_blend": _sent_metrics(sig.sentiment, hw.gt_sentiment),
            "event_accuracy": round(float(accuracy_score(hw.gt_event, sig.event_type)), 3),
            "impact_mae": round(float(mean_absolute_error(hw.gt_impact, sig.impact)), 2)}
    return res, eng


def main():
    docs = load_corpus()
    print(f"ingested {len(docs)} docs: {docs.source_type.value_counts().to_dict()} "
          f"| company-scope {int((docs.scope == 'company').sum())}, market-scope {int((docs.scope == 'market').sum())}")
    signals = out_of_fold_signals(docs)
    metrics, final_engine = evaluate(docs, signals)
    OUT.mkdir(exist_ok=True)
    signals.to_csv(OUT / "signals.csv", index=False)
    with open(OUT / "signals.jsonl", "w") as f:  # streaming-friendly output for downstream consumers
        for r in signals.astype({"timestamp": str}).to_dict("records"):
            f.write(json.dumps(r) + "\n")
    (OUT / "eval_metrics.json").write_text(json.dumps(metrics, indent=2))
    final_engine.save()
    print(json.dumps(metrics, indent=2))
    print(f"wrote {len(signals)} signals -> outputs/signals.csv, outputs/signals.jsonl; model -> {MODEL_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    # import under the real module path so the saved model unpickles from anywhere (api, tests)
    from src.engine.pipeline import main as _main

    _main()