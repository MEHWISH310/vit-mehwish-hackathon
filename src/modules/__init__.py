"""Downstream applications that consume the Risk Engine's signals through the SignalBus.

    Module A  rebalancer.py  tactical sentiment-tilted index (subscribes to Sentiment Score)
    Module B  stress.py      event-triggered portfolio stress test (subscribes to Event + Impact)
"""
