"""Benchmark: FinBERT (pre-trained finance transformer) vs the engine's real-trained sentiment model.

    pip install torch transformers     # optional, only for this script (~1 GB incl. the model download)
    python scripts/eval_finbert.py     # ~5-10 min on CPU -> docs/finbert_results.json

Same held-out real text and the same cleaning as `python -m src.engine.real`:
  * twitter_test   2,388 finance tweets, official held-out split. A FAIR test: FinBERT never saw these tweets,
                   and our model is trained only on the separate train split.
  * phrasebank     2,264 Financial PhraseBank sentences (all annotators agree). NOT a fair test for FinBERT:
                   ProsusAI/finbert was fine-tuned on Financial PhraseBank, so most of these sentences are in its
                   training data. Reported for completeness and labelled as such.

FinBERT is used off the shelf (zero-shot on tweets); fine-tuning it on the tweet train split would need a GPU.
Speed is measured on the same CPU for both models (texts per second, inference only).
The result is written to docs/ (committed), so the README, deck and dashboard do not need torch installed.
"""
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.engine.real import RealSentimentModel, _m, load_labelled, pick_lexicon_weight  # noqa: E402
from src.universe import ROOT  # noqa: E402

MODEL = "ProsusAI/finbert"
OUT_PATH = ROOT / "docs" / "finbert_results.json"


def finbert_predict(texts, batch: int = 32) -> tuple[np.ndarray, float, float]:
    """Labels (negative / neutral / positive), texts per second on CPU, and model size in MB (fp32 weights)."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL).eval()
    id2label = {i: l.lower() for i, l in model.config.id2label.items()}
    texts, labels = list(texts), []
    t0 = time.perf_counter()
    with torch.no_grad():
        for i in range(0, len(texts), batch):
            enc = tok(texts[i:i + batch], padding=True, truncation=True, max_length=128, return_tensors="pt")
            labels += [id2label[j] for j in model(**enc).logits.argmax(-1).tolist()]
            print(f"\r  FinBERT {min(i + batch, len(texts))}/{len(texts)}", end="", flush=True)
    print()
    speed = len(texts) / (time.perf_counter() - t0)
    return np.array(labels), speed, sum(p.numel() for p in model.parameters()) * 4 / 1e6


def ours_speed(model: RealSentimentModel, texts) -> float:
    t0 = time.perf_counter()
    model.predict_label(list(texts))
    return len(texts) / (time.perf_counter() - t0)


def main():
    tw_tr, tw_te, pb = load_labelled()
    w, _ = pick_lexicon_weight(tw_tr)
    ours = RealSentimentModel(w).fit(tw_tr.x, tw_tr.label)

    print(f"FinBERT on {len(tw_te):,} held-out tweets ...")
    fb_tw, fb_speed, fb_mb = finbert_predict(tw_te.x)
    print(f"FinBERT on {len(pb):,} PhraseBank sentences ...")
    fb_pb, _, _ = finbert_predict(pb.x)

    res = {
        "model": MODEL,
        "twitter_test": {"n": len(tw_te), "fair_test": True,
                         "finbert_zero_shot": _m(tw_te.label, fb_tw),
                         "engine_real_trained": _m(tw_te.label, ours.predict_label(tw_te.x))},
        "phrasebank": {"n": len(pb), "fair_test": False,
                       "note": "ProsusAI/finbert was fine-tuned on Financial PhraseBank: in-sample for FinBERT",
                       "finbert": _m(pb.label, fb_pb)},
        "speed_texts_per_sec_cpu": {"finbert": round(fb_speed, 1), "engine_real_trained": round(ours_speed(ours, tw_te.x), 1)},
        "model_size_mb": {"finbert": round(fb_mb), "engine_real_trained": round(len(pickle.dumps(ours)) / 1e6, 1)},
    }
    s = res["speed_texts_per_sec_cpu"]
    res["speedup_engine_vs_finbert"] = round(s["engine_real_trained"] / s["finbert"], 1)
    OUT_PATH.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    print(f"wrote {OUT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
