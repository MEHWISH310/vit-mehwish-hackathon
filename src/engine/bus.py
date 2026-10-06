"""A tiny publish/subscribe bus: how downstream modules 'subscribe' to the engine's signals.

    bus = SignalBus()
    bus.subscribe(rebalancer.on_signal, scope="company", name="Module A")
    bus.subscribe(stress_tester.on_signal, min_impact=7, name="Module B")
    bus.replay(signals_df)          # streams signals in timestamp order, like a live feed

In production this would sit on Kafka / a message queue; the interface (subscribe + callback)
stays the same, which is the point of the design."""
from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional

import pandas as pd


@dataclass
class _Sub:
    callback: Callable[[dict], None]
    name: str
    scope: Optional[str] = None
    min_impact: float = 0.0
    event_types: Optional[set] = None
    delivered: int = field(default=0)

    def wants(self, s: dict) -> bool:
        return ((self.scope is None or s["scope"] == self.scope)
                and s["impact"] >= self.min_impact
                and (self.event_types is None or s["event_type"] in self.event_types))


class SignalBus:
    def __init__(self):
        self._subs: list[_Sub] = []

    def subscribe(self, callback, *, name: str = "", scope: Optional[str] = None,
                  min_impact: float = 0.0, event_types: Optional[Iterable[str]] = None):
        self._subs.append(_Sub(callback, name or getattr(callback, "__name__", "sub"), scope, min_impact,
                               set(event_types) if event_types else None))

    def publish(self, signal: dict) -> int:
        n = 0
        for sub in self._subs:
            if sub.wants(signal):
                sub.callback(signal)
                sub.delivered += 1
                n += 1
        return n

    def replay(self, signals: pd.DataFrame) -> dict:
        """Publish every signal in time order. Returns {subscriber name: signals delivered}."""
        for rec in signals.sort_values("timestamp").to_dict("records"):
            self.publish(rec)
        return {s.name: s.delivered for s in self._subs}