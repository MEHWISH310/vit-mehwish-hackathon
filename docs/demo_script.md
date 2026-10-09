# Live demo run-of-show (target 4:40, hard limit 5:00)

The problem statement caps the live demo at 5 minutes, so this script is timed to 4:40 with 20 s of slack. The same flow works for the recorded video: the guidelines suggest about 10 minutes, but a tight 5-minute recording satisfies both.

**Before you start (not on the clock)**
- Run `pip install -r requirements.txt`, `python -m src.engine.pipeline` and `python -m src.engine.real` once.
- Open `streamlit run app.py` in a browser at 100% zoom, on the **Overview** tab, sidebar at defaults (data source = Synthetic).
- Click through all three tabs in **both** data sources once, so every view is cached and instant.
- Have a terminal open in the repo root with the font enlarged. Close notifications. Have `docs/presentation.pdf` open in another window in case you need a slide.

| Time | Screen | What to do | What to say |
|---|---|---|---|
| 0:00-0:25 | Dashboard **Overview** tab | Point at the hero banner and the four story cards (Read, Understand, Act A, Act B). | "Market-moving information arrives as text. I built one NLP Risk Engine that turns each news article or post into a sentiment score, an event type and an impact score, plus two modules that act on them: a sentiment-tilted index and an event-triggered stress test of a banking book. I built it on synthetic data, then tested it on real data, and the real data changed my conclusions." |
| 0:25-0:45 | Terminal | Run `python -m pytest -q tests` | "32 tests: no look-ahead, weight limits, hedge signs, trigger rules and the dashboard itself." |
| 0:45-1:30 | **Risk Engine** (Synthetic) | Click **Analyze** on the pre-filled Goldman headline (signal card: Credit Event, impact 7.8, green "would TRIGGER"). Click the **Plain downgrade** example, then **Analyze** (impact 5.9, "would not trigger"). Scroll to **How accurate is the sentiment on REAL text?** | "Two sources go into one schema; every text becomes a structured signal. This table is the honest part: trained only on synthetic text, sentiment was no better than guessing 'neutral' on real tweets and news. Retrained on real labelled data, it reaches 0.83 accuracy on held-out tweets and 0.89 on news." |
| 1:30-2:20 | **Module A** (Synthetic) | Point at the cards (+3.43% vs -6.78%) and the stacked weights. Drag **Sentiment tilt** to 0: NAV collapses onto the benchmark. Set it back to 1.5. | "Positive news raises a weight and negative news lowers it, within 2-15%, with at most 20% turnover a day and no look-ahead. On synthetic data it wins, but those prices were built from the same events, so it's circular." |
| 2:20-3:00 | Sidebar: **Data source -> Real**, still on Module A (the banner badge turns green: REAL DATA) | Switch the source. Read the cards (+7.04% vs +7.86%). Scroll to **Latest weight changes and why** and read a real headline driver. | "Same code, now on 22,000 real Google News headlines and real Yahoo Finance prices. Here the tilt doesn't beat equal weight; it's within the range of random tilts. Public headlines get priced in fast. I'm reporting that, not tuning it away." |
| 3:00-4:05 | **Module B** (Real) | Point at "51 stress tests" and the ignored-signals caption (458 unsure). The selector opens on "US economy grows sluggish 1.5%... inflation tops Fed target": show the waterfall and P&L by asset type. Filter event types to *Geopolitical* and pick a US-Iran sanctions or US-Canada tariff headline. | "On real news, a test fires only if impact is above 7, the event label is confident and the news is adverse. That removed 458 false alarms, like a lifestyle story about a 'default setting'. What's left are real events: the Iran war, sanctions, the trade war, Fed hikes. This one costs 57.7 million, or 2.75%, of a 2.1 billion book, and the hedges claw back 34 million." |
| 4:05-4:40 | Module B, **Custom shock** | It opens on the brief's "equities -10%, rates +200bp" (-3.36%). Drag **Credit spread shock** to +200bp and show the loss grow. | "The brief's own scenario: minus 70 million, with swaps and puts offsetting 78 million. Value them on market value instead of notional and that hedge disappears. The model is first-order; calibrating to history and adding correlations is next." |
| 4:40 | - | Stop. | "Code, data, deck and this script are in the repo. Thank you." |

**If something breaks:**
- The dashboard says an output file is missing: run the command it shows (`python -m src.engine.pipeline` or `python -m src.engine.real`) and refresh.
- The first load of a view is slow: it's caching the backtest and stress runs. That's why you click through everything before recording.
