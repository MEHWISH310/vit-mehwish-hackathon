"""
Synthetic data generator for the NLP Risk Engine hackathon project.

Creates (all in ./data, fully reproducible via SEED):
  universe.csv        15 S&P 100 stocks used as the mock index
  news_articles.csv   synthetic financial news (source 1 of the engine)
  social_posts.csv    synthetic X/Twitter-style posts (source 2 of the engine)
  prices.csv          daily closing prices (synthetic, event-driven)
  portfolio.csv       synthetic wholesale-banking portfolio (loans/bonds/derivatives/equity)

IMPORTANT: everything here is SYNTHETIC. Company names are real public tickers
(as the problem statement asks for S&P 100 stocks) but every headline, tweet,
price and position is fabricated for simulation. No real news or client data.

Columns prefixed `gt_` are ground-truth labels used ONLY to train/evaluate the
NLP models; the engine never reads them at inference time.

Run:  python scripts/generate_data.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RNG = np.random.default_rng(SEED)

START, END = "2026-07-06", "2026-10-02"  # ~63 business days
DAYS = pd.bdate_range(START, END)

# ticker, name, sector, equity beta, illustrative start price
UNIVERSE = [
    ("AAPL", "Apple", "Technology", 1.20, 235),
    ("MSFT", "Microsoft", "Technology", 1.10, 510),
    ("NVDA", "NVIDIA", "Technology", 1.70, 140),
    ("GOOGL", "Alphabet", "Communication", 1.15, 190),
    ("META", "Meta Platforms", "Communication", 1.30, 700),
    ("AMZN", "Amazon", "Consumer", 1.25, 215),
    ("WMT", "Walmart", "Consumer", 0.55, 95),
    ("KO", "Coca-Cola", "Consumer", 0.50, 70),
    ("JPM", "JPMorgan Chase", "Financials", 1.05, 280),
    ("BAC", "Bank of America", "Financials", 1.25, 48),
    ("GS", "Goldman Sachs", "Financials", 1.35, 700),
    ("XOM", "Exxon Mobil", "Energy", 0.90, 110),
    ("CVX", "Chevron", "Energy", 0.95, 155),
    ("JNJ", "Johnson & Johnson", "Healthcare", 0.55, 160),
    ("PFE", "Pfizer", "Healthcare", 0.60, 26),
]
U = pd.DataFrame(UNIVERSE, columns=["ticker", "name", "sector", "beta", "start_price"])
SECTOR_OF = dict(zip(U.ticker, U.sector))

PRODUCTS = {
    "Technology": ["AI chip", "cloud platform", "software suite", "developer toolkit"],
    "Communication": ["AI assistant", "ad platform", "streaming service"],
    "Consumer": ["delivery service", "private-label line", "beverage line"],
    "Financials": ["digital banking app", "wealth platform", "trading platform"],
    "Energy": ["LNG project", "offshore field", "renewables unit"],
    "Healthcare": ["oncology drug", "vaccine", "diagnostic test"],
}
AGENCIES = ["Moody's-style agency", "a major rating agency", "S&P-style agency"]
RATINGS = ["AA", "A+", "A", "A-", "BBB+", "BBB", "BBB-", "BB+"]
REGIONS = ["the Middle East", "Eastern Europe", "the South China Sea", "the Red Sea", "North Africa"]
SOURCES = ["MarketDesk Wire", "FinTimes Daily", "Capital Ledger", "Global Markets Brief"]

# ---------------------------------------------------------------------------
# News templates: event -> polarity -> [(template, base_impact)]
# {co} company, {pct} percent, {amt} billions, {rating}, {agency}, {product}, {region}, {sector}
# ---------------------------------------------------------------------------
NEWS = {
    "Earnings": {
        "pos": [("{co} beats quarterly earnings estimates as revenue climbs {pct}%", 5),
                ("{co} raises full-year guidance after a strong quarter", 5),
                ("{co} posts record profit, with earnings per share ahead of consensus", 6)],
        "neg": [("{co} misses earnings expectations as margins shrink", 5),
                ("{co} cuts full-year outlook citing weak demand", 6),
                ("{co} reports revenue decline of {pct}% year over year", 5)],
        "neu": [("{co} to report quarterly results next week", 2),
                ("{co} earnings come in line with analyst expectations", 2)],
    },
    "Credit Event": {
        "pos": [("{agency} upgrades {co} to {rating} on improving leverage", 5),
                ("{co} refinances debt at lower rates and extends maturities", 4)],
        "neg": [("{agency} downgrades {co} to {rating} amid rising debt", 7),
                ("{co} warns of covenant breach as cash flow tightens", 8),
                ("{co} bond spreads widen on liquidity concerns", 6),
                ("{co} raises loan loss provisions by {pct}% as defaults rise", 6)],
        "neu": [("{agency} affirms {co} at {rating} with a stable outlook", 2)],
    },
    "Merger/Acquisition": {
        "pos": [("{co} to acquire a rival in a ${amt}bn deal expected to boost margins", 5),
                ("{co} completes acquisition ahead of schedule, synergies seen", 4)],
        "neg": [("{co} faces antitrust challenge over proposed ${amt}bn merger", 6),
                ("{co} walks away from ${amt}bn takeover talks", 4),
                ("{co} overpays for acquisition, analysts warn of dilution", 5)],
        "neu": [("{co} confirms it is in talks over a possible acquisition", 3)],
    },
    "Product Launch": {
        "pos": [("{co} unveils new {product}, early orders exceed expectations", 4),
                ("{co} launches {product} to strong reviews", 3)],
        "neg": [("{co} delays {product} launch after quality issues", 4),
                ("{co} recalls {product} over safety defect", 6)],
        "neu": [("{co} announces date for its next {product} event", 2)],
    },
    "Regulatory/Legal": {
        "pos": [("{co} wins regulatory approval for its {product}", 4),
                ("{co} settles lawsuit on favourable terms", 3)],
        "neg": [("{co} hit with ${amt}00m fine by regulators", 6),
                ("{co} faces new government investigation into its practices", 6),
                ("{co} loses court ruling, appeal expected", 5)],
        "neu": [("Regulators open consultation on rules affecting the {sector} sector", 3)],
    },
}
MARKET_NEWS = {  # market-wide (no single ticker)
    "Geopolitical": {
        "pos": [("Ceasefire in {region} lifts global risk appetite", 5),
                ("Trade agreement signed with partners in {region} cheers markets", 4)],
        "neg": [("Escalating conflict in {region} rattles global markets", 8),
                ("New sanctions disrupt energy supply chains linked to {region}", 7),
                ("Tariffs on imports from {region} spark fears of a trade war", 7)],
        "neu": [("World leaders meet in {region} for talks", 2)],
    },
    "Macroeconomic": {
        "pos": [("Inflation cools to {pct}%, raising hopes of rate cuts", 5),
                ("Strong jobs report boosts growth outlook", 4),
                ("Central bank holds rates and signals easing ahead", 5)],
        "neg": [("Inflation surprises to the upside at {pct}%, rate hike fears grow", 7),
                ("Central bank signals higher rates for longer", 6),
                ("Recession fears mount as unemployment rises", 8),
                ("GDP contracts by {pct}% in surprise slowdown", 8)],
        "neu": [("Central bank minutes due for release on Wednesday", 2)],
    },
}
NEG_MODS = [("", 0), (", sending shares lower", 1), (", triggering a sharp selloff", 2),
            (" in what analysts call a major shock", 3)]
POS_MODS = [("", 0), (", lifting shares", 1), (", sparking a strong rally", 2),
            (" in a landmark result", 3)]
FILLER = {
    "pos": ["Investors responded positively, with analysts raising price targets.",
            "Traders cited improving fundamentals and strong momentum."],
    "neg": ["Investors reacted cautiously, citing uncertainty over near-term outlook.",
            "Analysts flagged downside risks and trimmed their estimates."],
    "neu": ["The development was largely expected by market participants.",
            "Analysts said it was too early to assess any lasting impact."],
}

# ---------------------------------------------------------------------------
# Social templates
# ---------------------------------------------------------------------------
TW_EVENT = {  # (event, pol) -> templates ; {t} = cashtag
    ("Earnings", "pos"): ["${t} earnings beat!! EPS way above estimates 🚀", "${t} smashing guidance, loading up 📈"],
    ("Earnings", "neg"): ["${t} missed earnings and cut guide... brutal 📉", "${t} margins shrinking, earnings were weak"],
    ("Credit Event", "pos"): ["${t} credit upgrade, balance sheet looking healthier ✅"],
    ("Credit Event", "neg"): ["${t} downgraded, spreads widening ⚠️ #credit", "${t} debt worries growing, bonds selling off"],
    ("Merger/Acquisition", "pos"): ["${t} acquisition could be a game changer 👀🔥"],
    ("Merger/Acquisition", "neg"): ["${t} merger facing antitrust heat, deal at risk", "${t} overpaying for this acquisition imo"],
    ("Product Launch", "pos"): ["${t} new launch looks 🔥 early demand is huge"],
    ("Product Launch", "neg"): ["${t} product recall, yikes 😬", "${t} delaying the launch again, not good"],
    ("Regulatory/Legal", "pos"): ["${t} wins approval 🎉 big relief"],
    ("Regulatory/Legal", "neg"): ["${t} fined again, regulators not messing around", "${t} under investigation, this could get ugly"],
}
TW_GENERIC = {
    "pos": ["${t} looking strong today 🚀", "bought more ${t} on the dip, feeling good 📈",
            "${t} crushing it, bullish", "${t} breakout incoming?? 🔥"],
    "neg": ["${t} getting hammered rn 📉", "sold my ${t}, not liking this",
            "${t} looks bad 😬 bearish", "ugh ${t} again... down bad"],
    "neu": ["watching ${t} today", "${t} flat, nothing happening", "anyone have thoughts on ${t}?",
            "${t} volume is low today"],
}
TW_MARKET = {
    ("Geopolitical", "neg"): ["war escalating, markets gonna bleed 😰 #markets", "sanctions + tariffs = chaos for stocks 📉 #geopolitics"],
    ("Geopolitical", "pos"): ["ceasefire!! risk on 🟢 #markets", "trade deal signed, finally good news 🎉"],
    ("Macroeconomic", "neg"): ["inflation print hot, fed won't cut 😬 #Fed", "recession signals flashing red 🚨 #macro"],
    ("Macroeconomic", "pos"): ["cpi cooled, rate cuts soon? 📈 #Fed", "jobs data strong, soft landing vibes ✅ #macro"],
}
EMOJI = ["🔥", "📊", "💬", "👀", "🤔"]


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def pick(lst):
    return lst[int(RNG.integers(len(lst)))]


def fmt(tpl, co="", sector=""):
    return tpl.format(
        co=co, sector=sector, pct=round(float(RNG.uniform(2, 14)), 1),
        amt=int(RNG.integers(2, 40)), rating=pick(RATINGS), agency=pick(AGENCIES),
        product=pick(PRODUCTS.get(sector, sum(PRODUCTS.values(), []))), region=pick(REGIONS),
    )


def label_from(pol, impact):
    """Ground-truth sentiment score in [-1, 1] tied to impact."""
    if pol == "neu":
        return float(np.clip(RNG.normal(0, 0.06), -0.15, 0.15))
    mag = float(np.clip(0.25 + 0.07 * impact + RNG.normal(0, 0.06), 0.18, 1.0))
    return mag if pol == "pos" else -mag


def sample_pol(mood, p_neu=0.15):
    p_pos = (1 - p_neu) * sigmoid(1.4 * mood)
    r = RNG.random()
    return "pos" if r < p_pos else ("neg" if r < p_pos + (1 - p_neu) * (1 - sigmoid(1.4 * mood)) else "neu")


def ts(day):
    return day + pd.Timedelta(hours=int(RNG.integers(6, 21)), minutes=int(RNG.integers(0, 60)))


# ---------------------------------------------------------------------------
# Latent sentiment "mood" per stock (AR(1)) + market mood -> makes trends
# ---------------------------------------------------------------------------
n_d, n_s = len(DAYS), len(U)
mood = np.zeros((n_d, n_s))
mmood = np.zeros(n_d)
for t in range(1, n_d):
    mood[t] = 0.85 * mood[t - 1] + RNG.normal(0, 0.6, n_s)
    mmood[t] = 0.9 * mmood[t - 1] + RNG.normal(0, 0.5)

# ---------------------------------------------------------------------------
# News
# ---------------------------------------------------------------------------
news, events_for_prices = [], []
ev_names = list(NEWS)
ev_w = np.array([0.30, 0.14, 0.14, 0.22, 0.20])


def news_row(day, ticker, event, pol, tpl_pair, co="", sector=""):
    tpl, base = tpl_pair
    mod, bonus = pick(NEG_MODS if pol == "neg" else POS_MODS if pol == "pos" else [("", 0)])
    impact = int(np.clip(base + bonus + RNG.integers(-1, 2), 1, 10)) if pol != "neu" else int(RNG.integers(1, 4))
    head = fmt(tpl, co, sector) + mod
    t = ts(day)
    row = dict(timestamp=t, source=pick(SOURCES), ticker=ticker, headline=head,
               body=head + ". " + pick(FILLER[pol]),
               gt_sentiment=round(label_from(pol, impact), 3), gt_event=event, gt_impact=impact)
    events_for_prices.append((day, ticker, event, pol, impact, True))
    return row


for ti, day in enumerate(DAYS):
    for si, r in U.iterrows():
        for _ in range(RNG.poisson(1.06)):
            ev = ev_names[RNG.choice(len(ev_names), p=ev_w)]
            pol = sample_pol(mood[ti, si])
            news.append(news_row(day, r.ticker, ev, pol, pick(NEWS[ev][pol]), r["name"], r.sector))
    for _ in range(RNG.poisson(2.3)):
        ev = pick(list(MARKET_NEWS))
        pol = sample_pol(mmood[ti])
        news.append(news_row(day, "", ev, pol, pick(MARKET_NEWS[ev][pol])))

news = pd.DataFrame(news).sort_values("timestamp").reset_index(drop=True)
news.insert(0, "id", [f"N{i:05d}" for i in range(len(news))])

# ---------------------------------------------------------------------------
# Social posts
# ---------------------------------------------------------------------------
posts = []


def decorate(text):
    if RNG.random() < 0.30:
        text += " " + pick(["#stocks", "#investing", "#trading", "#markets"])
    if RNG.random() < 0.20:
        text += " " + pick(EMOJI)
    if RNG.random() < 0.12:
        text = text.upper()
    if RNG.random() < 0.10:
        text = f"RT @user_{RNG.integers(1000, 9999)}: " + text
    return text


for ti, day in enumerate(DAYS):
    for si, r in U.iterrows():
        for _ in range(RNG.poisson(2.5)):
            pol = sample_pol(mood[ti, si], p_neu=0.25)
            if RNG.random() < 0.45 and pol != "neu":
                ev = pick([e for (e, p) in TW_EVENT if p == pol])
                text, event = pick(TW_EVENT[(ev, pol)]), ev
                impact = int(np.clip(RNG.integers(3, 8), 1, 10))
            else:
                text, event = pick(TW_GENERIC[pol]), "Other"
                impact = int(RNG.integers(1, 4)) if pol == "neu" else int(RNG.integers(2, 5))
            posts.append(dict(timestamp=ts(day), platform="X-sim", user=f"user_{RNG.integers(1000, 9999)}",
                              ticker=r.ticker, text=decorate(text.format(t=r.ticker)),
                              gt_sentiment=round(label_from(pol, impact), 3), gt_event=event, gt_impact=impact))
    for _ in range(RNG.poisson(4)):
        ev = pick(list(MARKET_NEWS))
        pol = sample_pol(mmood[ti], p_neu=0.0)
        impact = int(RNG.integers(4, 9))
        posts.append(dict(timestamp=ts(day), platform="X-sim", user=f"user_{RNG.integers(1000, 9999)}",
                          ticker="", text=decorate(pick(TW_MARKET[(ev, pol)])),
                          gt_sentiment=round(label_from(pol, impact), 3), gt_event=ev, gt_impact=impact))

posts = pd.DataFrame(posts).sort_values("timestamp").reset_index(drop=True)
posts.insert(0, "id", [f"T{i:05d}" for i in range(len(posts))])

# ---------------------------------------------------------------------------
# Prices: market factor + idiosyncratic noise + event shocks (applied next day)
# NOTE: because shocks come from the same synthetic events, sentiment is
# genuinely predictive here BY CONSTRUCTION - state this as a limitation.
# ---------------------------------------------------------------------------
day_idx = {d: i for i, d in enumerate(DAYS)}
mkt_shock = np.zeros(n_d)
stk_shock = np.zeros((n_d, n_s))
tick_idx = {t: i for i, t in enumerate(U.ticker)}
energy = [i for i, t in enumerate(U.ticker) if SECTOR_OF[t] == "Energy"]
for day, tk, ev, pol, imp, _ in events_for_prices:
    if pol == "neu":
        continue
    sign = 1 if pol == "pos" else -1
    j = min(day_idx[pd.Timestamp(day).normalize()] + 1, n_d - 1)
    if tk:
        stk_shock[j, tick_idx[tk]] += sign * imp * 0.0009
    else:
        mkt_shock[j] += sign * imp * 0.0003
        if ev == "Geopolitical" and pol == "neg":  # supply-risk premium helps energy
            stk_shock[j, energy] += imp * 0.0005

mkt_ret = RNG.normal(0.0012, 0.006, n_d) + mkt_shock
ret = U.beta.values * mkt_ret[:, None] + RNG.normal(0, 0.011, (n_d, n_s)) + stk_shock
ret = np.clip(ret, -0.15, 0.15)
close = U.start_price.values * np.cumprod(1 + ret, axis=0)
prices = pd.DataFrame(close, index=DAYS, columns=U.ticker).round(2)
prices = prices.reset_index(names="date").melt(id_vars="date", var_name="ticker", value_name="close")
prices = prices.sort_values(["date", "ticker"]).reset_index(drop=True)

# ---------------------------------------------------------------------------
# Synthetic wholesale-banking portfolio (USD millions)
# risk_notional = amount that rate / spread / equity sensitivities apply to
# ---------------------------------------------------------------------------
BORROWERS = ["Aster Industrial", "Brightwave Energy", "Cobalt Logistics", "Delta Retail Group",
             "Evergreen Foods", "Falcon Aerospace", "Granite Mining", "Harbor Shipping",
             "Ionic Semiconductors", "Juniper Healthcare", "Keystone Utilities", "Lumen Telecom",
             "Meridian Real Estate", "Northstar Steel"]
SECTORS = ["Energy", "Financials", "Technology", "Healthcare", "Consumer", "Industrials", "Real Estate"]
LOAN_RATINGS = ["A", "BBB", "BBB", "BB", "BB", "B"]
REGION = ["US", "US", "EU", "Asia"]
pf, k = [], 0


def add(**kw):
    global k
    k += 1
    pf.append(dict(asset_id=f"A{k:03d}", **kw))


for i in range(14):  # loans
    n = round(float(RNG.uniform(20, 150)), 1)
    add(asset_type="Loan", instrument="Corporate Term Loan", counterparty=BORROWERS[i], linked_ticker="",
        sector=pick(SECTORS), region=pick(REGION), rating=pick(LOAN_RATINGS), notional=n,
        market_value=round(n * RNG.uniform(0.96, 1.0), 1), risk_notional=0,
        rate_duration=round(float(RNG.uniform(0.5, 3)), 2), spread_duration=round(float(RNG.uniform(2, 5)), 2),
        equity_beta=0.0)
bond_tk = RNG.choice(U.ticker, 12, replace=False)
for tk in bond_tk:  # bonds
    n = round(float(RNG.uniform(20, 120)), 1)
    d = round(float(RNG.uniform(3, 9)), 2)
    add(asset_type="Bond", instrument="Corporate Bond", counterparty=U.set_index("ticker").loc[tk, "name"],
        linked_ticker=tk, sector=SECTOR_OF[tk], region="US", rating=pick(["AA", "A+", "A", "BBB+", "BBB"]),
        notional=n, market_value=round(n * RNG.uniform(0.93, 1.04), 1), risk_notional=0,
        rate_duration=d, spread_duration=round(d * 0.95, 2), equity_beta=0.0)
for _ in range(3):  # pay-fixed swaps: gain when rates rise
    n = round(float(RNG.uniform(100, 300)), 1)
    add(asset_type="Derivative", instrument="Pay-Fixed Interest Rate Swap", counterparty="Dealer Bank",
        linked_ticker="", sector="Financials", region="US", rating="A", notional=n,
        market_value=round(n * RNG.uniform(0.0, 0.02), 1), risk_notional=n,
        rate_duration=-round(float(RNG.uniform(4, 8)), 2), spread_duration=0.0, equity_beta=0.0)
for _ in range(2):  # equity index puts: gain when equities fall
    n = round(float(RNG.uniform(40, 90)), 1)
    add(asset_type="Derivative", instrument="Equity Index Put Option", counterparty="Exchange",
        linked_ticker="", sector="Index", region="US", rating="NR", notional=n,
        market_value=round(float(RNG.uniform(1, 4)), 1), risk_notional=n,
        rate_duration=0.0, spread_duration=0.0, equity_beta=-1.0)
for _ in range(3):  # CDS protection bought: gain when spreads widen
    n = round(float(RNG.uniform(30, 80)), 1)
    add(asset_type="Derivative", instrument="CDS Protection Bought", counterparty="Dealer Bank",
        linked_ticker="", sector="Credit Index", region="US", rating="NR", notional=n,
        market_value=round(float(RNG.uniform(0.2, 1.5)), 1), risk_notional=n,
        rate_duration=0.0, spread_duration=-round(float(RNG.uniform(3, 5)), 2), equity_beta=0.0)
for tk in RNG.choice(U.ticker, 6, replace=False):  # equity holdings
    add(asset_type="Equity", instrument="Common Stock", counterparty=U.set_index("ticker").loc[tk, "name"],
        linked_ticker=tk, sector=SECTOR_OF[tk], region="US", rating="NR", notional=0,
        market_value=round(float(RNG.uniform(10, 60)), 1), risk_notional=0,
        rate_duration=0.0, spread_duration=0.0, equity_beta=float(U.set_index("ticker").loc[tk, "beta"]))
portfolio = pd.DataFrame(pf)
# funded assets: sensitivities apply to market value
funded = portfolio.asset_type != "Derivative"
portfolio.loc[funded, "risk_notional"] = portfolio.loc[funded, "market_value"]

# ---------------------------------------------------------------------------
# Write
# ---------------------------------------------------------------------------
DATA.mkdir(exist_ok=True)
U.to_csv(DATA / "universe.csv", index=False)
news.to_csv(DATA / "news_articles.csv", index=False)
posts.to_csv(DATA / "social_posts.csv", index=False)
prices.to_csv(DATA / "prices.csv", index=False)
portfolio.to_csv(DATA / "portfolio.csv", index=False)

print(f"universe   : {len(U)} stocks")
print(f"news       : {len(news)} rows | events {news.gt_event.value_counts().to_dict()}")
print(f"social     : {len(posts)} rows")
print(f"prices     : {len(prices)} rows, {DAYS[0].date()} -> {DAYS[-1].date()}")
print(f"portfolio  : {len(portfolio)} positions, {portfolio.asset_type.value_counts().to_dict()}, "
      f"MV ${portfolio.market_value.sum():,.0f}m")