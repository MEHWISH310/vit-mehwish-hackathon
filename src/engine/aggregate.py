"""Roll individual signals up into one sentiment number per ticker per day."""
import pandas as pd

SOURCE_WEIGHT = {"news": 1.0, "social": 0.5}  # news is more reliable than social chatter


def aggregate_daily(signals: pd.DataFrame) -> pd.DataFrame:
    """Impact- and source-weighted mean sentiment per (date, ticker).
    ticker == '' rows are market-wide signals (Geopolitical / Macro with no single company)."""
    s = signals.copy()
    s["ticker"] = s["ticker"].fillna("")
    s["date"] = pd.to_datetime(s["timestamp"]).dt.normalize()
    s["w"] = s["impact"] * s["source_type"].map(SOURCE_WEIGHT).fillna(0.5)
    s["ws"] = s["w"] * s["sentiment"]
    g = (s.groupby(["date", "ticker"])
          .agg(ws=("ws", "sum"), w=("w", "sum"), n=("w", "size"), max_impact=("impact", "max"))
          .reset_index())
    g["sentiment"] = (g["ws"] / g["w"]).round(4)
    return g[["date", "ticker", "sentiment", "n", "max_impact"]]