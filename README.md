# NLP Risk Engine: Sentiment Index Rebalancer & Event-Driven Stress Tester - S&P Global & Crisil Campus Hackathon

**Candidate Name:** Mehwish
**College Email ID:** [your_id@vitstudent.ac.in]
**College / Campus:** VIT
**Demo Video Link:** [YouTube (unlisted) link - to be added]
**Slide Deck Link (if hosted externally):** not hosted externally; the deck is in the repo at [`docs/presentation.pdf`](docs/presentation.pdf)

## 1. Project Overview / Problem Statement & Approach

Market-moving information arrives as unstructured text: news articles, analyst notes and social-media chatter. Portfolio and risk teams cannot read it all, and cannot act on it until it is turned into numbers. The case study asks for a unified **AI/NLP Risk Engine** that ingests text from at least two sources and outputs, per company or event, a **sentiment score** (-1..1), an **event classification** and an **impact score** (1..10), exposed to downstream applications. At least one downstream module must be built on top of it.

The engine ingests **financial news** and **X/Twitter-style posts** into one schema. It cleans social noise (RT, URLs, @mentions, hashtags), links each text to a company (cashtag, name or alias) or marks it market-wide, and masks company names so the models learn language rather than tickers. It then scores every text with three models: a finance-tuned lexicon blended with an ML regressor for sentiment, a TF-IDF + logistic-regression event classifier with a keyword fallback, and an impact regressor. Signals are published three ways: a file (`outputs/signals.csv` / `.jsonl`), a REST API (FastAPI) and an in-process publish/subscribe **SignalBus**.

**Both** downstream modules subscribe to that bus. **Module A** (tactical) keeps a time-decayed, impact-weighted sentiment score per stock and tilts a 15-stock S&P 100 index toward positive sentiment and away from negative sentiment, with weight caps and a turnover limit. **Module B** (strategic) listens for high-impact events (impact > 7) and runs a scenario stress test on a 40-asset synthetic wholesale-banking portfolio of loans, bonds, derivatives and equity, showing value before and after. A Streamlit dashboard shows all of it live.

## 2. Architecture & Tech Stack

![Architecture](docs/architecture.png)

**Data flow:** `data/news_articles.csv` + `data/social_posts.csv` -> `src/engine/ingest.py` (unify, clean, entity-link, mask) -> `src/engine/pipeline.py` (`RiskEngine`: sentiment, event, impact; 5-fold out-of-fold scoring) -> `outputs/signals.csv|jsonl`, `src/engine/api.py`, `src/engine/bus.py` -> `src/modules/rebalancer.py` (Module A), `src/modules/stress.py` (Module B) -> `app.py` (dashboard).

| Component | Path | What it does |
|---|---|---|
| Ingestion | `src/engine/ingest.py`, `src/universe.py` | 2 sources -> one schema; social cleaning; entity linking; company-name masking; `company` vs `market` scope |
| Sentiment | `src/engine/sentiment.py` | 70% VADER extended with 132 finance terms and emoji + 30% TF-IDF ridge regressor, clipped to [-1, 1] |
| Event + impact | `src/engine/events.py` | 8 labels (Earnings, Credit Event, Merger/Acquisition, Product Launch, Regulatory/Legal, Geopolitical, Macroeconomic, Other): TF-IDF + logistic regression, keyword rules when unsure; impact 1-10 via TF-IDF ridge + source type |
| Signal output | `src/engine/pipeline.py`, `src/engine/api.py`, `src/engine/bus.py` | CSV/JSONL files; REST `/signals`, `/sentiment/{ticker}`, `/analyze`, `/health`; pub/sub `subscribe(callback, scope=, min_impact=, event_types=)` |
| Module A | `src/modules/rebalancer.py` | `SentimentRebalancer`, `RebalanceConfig`, `run_backtest` |
| Module B | `src/modules/stress.py` | `StressTester`, `SCENARIOS`, `run_stress` (any custom shock), `run_event_stress` |
| Dashboard | `app.py` | Streamlit + Plotly, 3 tabs |
| Docs | `scripts/make_docs.py` | regenerates `docs/results.json`, figures, `architecture.png`, `presentation.pdf` from a real run |

**Tech stack:** Python 3.13, pandas, NumPy, scikit-learn, vaderSentiment, FastAPI + Uvicorn, Streamlit, Plotly, matplotlib, pytest. Everything runs locally on CPU; no external API keys are needed.

### Module A: tactical index rebalancing (subscribes to Sentiment Score)
- Mock index: 15 S&P 100 stocks across 6 sectors (`data/universe.csv`), starting at equal weight (1/15).
- Each company signal adds weight `w = impact x source_weight` (news 1.0, social 0.5). Evidence decays with a **36h half-life**. Score = `sum(w x sentiment) / (sum(w) + 3)`; the prior of 3 shrinks thinly covered names toward 0.
- Target weight = `1/15 x exp(1.5 x score)`, clipped to **[2%, 15%]** and renormalised. **Positive sentiment raises the weight; negative lowers it.**
- Trades move toward the target by at most **20% one-way turnover per day**. If price drift pushed a name outside the bounds, that fix happens first.
- **No look-ahead:** the index rebalances at the 16:00 close of day *t* using only signals stamped at or before 16:00. Those weights earn day *t+1*'s return, and weights drift with prices between rebalances. Both rules are unit-tested.
- Dashboard: stacked-area weights over time, sentiment-score heatmap, NAV vs the equal-weight benchmark, a table of latest weight changes with the headline that drove each, and metric cards. All parameters are adjustable in the sidebar.

### Module B: strategic stress testing (subscribes to Event Classification + Impact Score)
- Portfolio (`data/portfolio.csv`, USD m): 14 loans, 12 bonds, 8 derivatives (pay-fixed swaps, index puts, CDS protection bought) and 6 equities. Market value is $2,097.5m. Derivatives have $15.3m of market value but **$901.9m of risk notional**.
- **Trigger:** `impact > 7` (strictly) and the event type has a scenario. Idiosyncratic scenarios fire only on negative sentiment, because we stress the adverse case. A **24h cooldown per (event type, ticker)** collapses a burst of stories about the same event into one test.
- **Scenarios** at full severity (impact 10), scaled by `severity = impact / 10`:

| Event type | Equity | Rates | Credit spreads | Linked name (extra) | Fires on |
|---|---|---|---|---|---|
| Geopolitical | -12% | -40bp (flight to quality) | +100bp | - | any high-impact |
| Macroeconomic | -8% | +100bp | +60bp | - | any high-impact |
| Credit Event | -6% | -10bp | +150bp | equity -15%, spread +300bp | any high-impact |
| Regulatory/Legal | -2% | 0 | +15bp | equity -12%, spread +120bp | negative sentiment |
| Earnings | -1% | 0 | +5bp | equity -10%, spread +40bp | negative sentiment |
| Merger/Acquisition | 0% | 0 | +5bp | equity -8%, spread +80bp | negative sentiment |
| Product Launch | -0.5% | 0 | 0 | equity -7%, spread +25bp | negative sentiment |

- Custom presets in the dashboard, not scaled by severity: the brief's own example (**equities -10%, rates +200bp**), a 2008-style credit crunch, a 2022-style rate shock, and a Covid-style crash. Free sliders cover equity, rates, spreads and an optional idiosyncratic name.
- **Valuation:** a simplified first-order model on **risk notional**, not market value:
  - rate P&L = `-rate_duration x dy x risk_notional`
  - spread P&L = `-spread_duration x ds x rating_multiplier x risk_notional`, with multipliers AA 0.6, A+ 0.7, A 0.8, BBB+ 0.9, BBB/NR 1.0, BB 1.8, B 2.5
  - equity P&L = `equity_beta x equity_shock x risk_notional`, plus the idiosyncratic shock on the linked ticker
  - Pay-fixed swaps (negative rate duration) gain when rates rise, CDS protection (negative spread duration) gains when spreads widen, and puts (beta -1) gain in a sell-off. All three signs are unit-tested.
- Dashboard: timeline of triggered tests, pick any trigger -> headline and impact, a before/after waterfall split into rates, spreads and equity, P&L by asset type and by sector, and the top-10 losing assets. A custom-shock panel supports live what-ifs.

## 3. Dataset Used

**All data in this repository is synthetic.** It was generated by [`scripts/generate_data.py`](scripts/generate_data.py) (fixed seed 42, fully reproducible) and lives in `data/`. No proprietary, client or confidential data is used. Company names and tickers are real public S&P 100 constituents, because the brief asks for an S&P 100 index, but **every headline, post, price and position is fabricated**.

| File | Rows | Content |
|---|---|---|
| `data/news_articles.csv` | 1,212 | Template-generated news articles (id, timestamp, optional ticker, body) |
| `data/social_posts.csv` | 2,762 | X-style posts with RT prefixes, @mentions, #hashtags, $cashtags, emoji, slang |
| `data/universe.csv` | 15 | Mock index: ticker, name, sector, beta |
| `data/prices.csv` | 975 | Daily closes, 15 tickers x 65 business days (2026-07-06 to 2026-10-02) |
| `data/portfolio.csv` | 40 | Wholesale-banking book: asset type, rating, notional, market value, risk notional, rate/spread duration, equity beta |
| `data/eval/handwritten_eval.csv` | 35 | Hand-written dev texts with labels (used while tuning) |
| `data/eval/handwritten_holdout.csv` | 20 | Hand-written holdout texts, written before tuning and never tuned on |

**Assumptions and caveats**
- The text generator draws "events" (type, polarity, impact) and renders them through templates. `gt_*` columns hold those ground-truth labels and are used only for training and evaluation, never at inference time.
- **Prices were generated from the same events as the text**: a market factor plus idiosyncratic noise plus event shocks applied the next day. Any backtest on them is therefore partly circular. It shows the pipeline works end to end, **not** that the signal would earn money on real markets.
- Portfolio values are in USD millions. Derivatives carry tiny market values but large risk notionals, which is why stress P&L is computed on risk notional.
- Public sources the design is ready for, not used here: GDELT, NewsAPI, Kaggle financial-news and stock-tweet datasets, yfinance.

## 4. Quickstart & Installation

Runtime: Python 3.13 (tested with 3.13.7) on Windows 11. Pure Python, so it should also run on macOS and Linux.

```bash
git clone https://github.com/MEHWISH310/vit-mehwish-hackathon.git
cd vit-mehwish-hackathon
pip install -r requirements.txt
python -m src.engine.pipeline      # ~5-10 s: trains the engine, writes outputs/signals.csv(.jsonl), eval_metrics.json, models/engine.joblib
python -m pytest -q tests          # 27 tests
streamlit run app.py               # dashboard at http://localhost:8501
```

Optional:
```bash
uvicorn src.engine.api:app --reload   # REST API, docs at http://127.0.0.1:8000/docs
python scripts/make_docs.py           # regenerate docs/results.json, figures, architecture.png, presentation.pdf
python scripts/generate_data.py       # regenerate the synthetic data in data/ (already committed)
```

`outputs/` and `models/` are generated and git-ignored. The dashboard tells you to run the pipeline if they are missing.

## 5. Key Results & Domain Impact

Every number below comes from `python -m src.engine.pipeline` (`outputs/eval_metrics.json`) and `python scripts/make_docs.py` (`docs/results.json`).

### Engine
3,974 signals: 1,212 news + 2,762 social; 3,547 company-scope and 427 market-wide; 369 with impact > 7.

| Metric | Synthetic corpus, 5-fold out-of-fold | Handwritten holdout (20 texts, never tuned on) |
|---|---|---|
| Sentiment Pearson r | 0.908 | 0.878 |
| Sentiment 3-class accuracy | 0.899 | 0.750 |
| Event accuracy | 0.969 (macro-F1 0.974) | 0.900 |
| Impact MAE (1-10 scale) | 0.85 | 1.18 |

**Honest read.** The synthetic scores are high because test texts share the generator's templates. The 20 handwritten holdout texts are the real generalisation check. On them the plain lexicon component (3-class 0.80, r 0.884) slightly **beats** the lexicon+ML blend (0.75, r 0.878), so the ML part partly learns template wording. On the 35-text dev set the blend wins (0.914 vs 0.857), but that set was used while tuning. Twenty texts is a small sample.

### Module A: sentiment index vs equal weight (65 trading days, synthetic prices)

| | Sentiment index | Equal-weight benchmark |
|---|---|---|
| Total return | **+3.43%** | -6.78% |
| Annualised volatility | 13.36% | 12.88% |
| Sharpe (annualised, rf = 0) | **1.06** | -2.08 |
| Max drawdown | -9.31% | -14.45% |
| Avg daily one-way turnover | 11.1% | - |

Weights stayed within [2.0%, 15.0%] every day, and turnover never exceeded the 20% cap. A **placebo** run, which shuffles sentiment across signals (20 runs), averages **-6.89%** (range -8.96% to -5.28%), about the same as the benchmark. So the tilt's edge comes from the sentiment signal, not from the mechanics. Sensitivity: tilt 0 reproduces the benchmark exactly. Total return rises with tilt (0.75 -> -1.6%, 1.5 -> +3.4%, 3.0 -> +7.0% at a 36h half-life) and with a shorter half-life (12h -> +5.3% at tilt 1.5), at the cost of higher turnover. **This outperformance is partly circular**, because prices were generated from the same events as the text. Read it as proof the plumbing works, not as evidence of alpha.

![Module A dashboard](docs/img/dashboard_module_a.png)

### Module B: event-triggered stress tests (portfolio $2,097.5m)
- 369 signals with impact > 7 -> **248 distinct stress tests**. Of the rest, 77 were positive news on an idiosyncratic event (no adverse stress) and 44 fell inside the 24h cooldown.

| Event type | Triggers | Mean P&L | Worst P&L |
|---|---|---|---|
| Credit Event | 54 | -4.52% | -5.73% |
| Macroeconomic | 23 | -3.23% | -3.77% |
| Geopolitical | 23 | -2.61% | -3.17% |
| Regulatory/Legal | 42 | -0.58% | -0.81% |
| Earnings | 48 | -0.23% | -0.42% |
| Merger/Acquisition | 20 | -0.22% | -0.38% |
| Product Launch | 38 | -0.05% | -0.18% |

- **Worst trigger:** *"Goldman Sachs warns of covenant breach as cash flow tightens..."*, a Credit Event with impact 10 (2026-10-01 18:05). Value went from $2,097.5m to $1,977.3m (**-$120.2m, -5.73%**): loans -$63.2m, bonds -$57.9m, equity -$12.6m, partly offset by derivatives at +$13.6m (CDS protection gains as spreads widen).
- **The brief's example (equities -10%, rates +200bp):** -$70.5m (-3.36%). Rates cost -$62.0m and equity -$8.6m. The pay-fixed swaps and index puts offset **+$77.8m**.
- **Why risk notional matters:** valuing derivatives on market value instead would show only +$1.5m of hedge and double the reported loss to -7.00%. A naive approach materially mis-states the book's risk.

![Module B dashboard](docs/img/dashboard_module_b.png)

### Domain impact
- **Portfolio managers** get a transparent, rules-based tilt that reacts to news within one close, with concentration and turnover limits a risk committee can sign off on.
- **Risk and treasury desks** get, for every high-impact headline, an immediate and explainable answer to "what does this do to our book", split into rates, credit and equity and by asset type and sector, plus what-if shocks during a meeting.
- **One engine, many consumers:** the pub/sub contract lets new modules (limit monitoring, alerting) subscribe without touching the NLP code. In production the in-process bus would be swapped for Kafka behind the same interface.

### Limitations and next steps
- All data is synthetic, and the Module A backtest is partly circular (see Dataset). Next step: real feeds (GDELT, NewsAPI, Kaggle tweets) and real prices (yfinance), with transaction costs and an event study of how quickly signals decay.
- Engine generalisation is measured on only 20 handwritten texts. Next step: a larger labelled set and a fine-tuned finance transformer (e.g. FinBERT) compared against the lexicon baseline.
- The impact model learned the generator's intensity words ("major shock", "sharply"), so it under-rates plainly worded real-world events. For example, *"War breaks out in the Middle East; oil spikes and markets sell off sharply"* scores 5.7 and would not trigger Module B. Next step: train impact on real price reactions (an event study) instead of synthetic labels.
- The stress model is **first-order** (duration/beta): no convexity or gamma, vega, cross-asset correlations, rating migration, default losses or liquidity effects. Scenario sizes are expert judgement, not calibrated to history. Next step: calibrate to historical episodes and add correlated factor shocks.

## Repository structure

```
vit-mehwish-hackathon/
├── README.md, LICENSE (MIT), requirements.txt
├── app.py                     Streamlit dashboard
├── src/engine/                ingestion, sentiment, events/impact, aggregation, pipeline, API, SignalBus
├── src/modules/               rebalancer.py (Module A), stress.py (Module B)
├── data/                      all synthetic input data (+ data/eval handwritten sets)
├── scripts/                   generate_data.py, make_docs.py
├── tests/                     test_engine.py, test_modules.py
└── docs/                      presentation.pdf, architecture.png, demo_script.md, results.json, figures/, img/
```

## License
MIT, see [LICENSE](LICENSE).
