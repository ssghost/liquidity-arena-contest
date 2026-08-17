import asyncio

from agent.engine import DualTrackTradingEngine
from agent.executor import OrderExecutor


async def run_async_tests():
    print("\n1. Testing OrderExecutor (Mock Mode)...")
    executor = OrderExecutor(mock_mode=True)
    order_res = await executor.place_order(
        symbol="BINANCE_PERP_BTC_USDT",
        side="BUY",
        price=60000.0,
        quantity=0.01,
    )
    assert order_res["status"] == "FILLED", "Mock order should be FILLED immediately."
    order_id = order_res["order_id"]

    cancel_res = await executor.cancel_order(order_id)
    assert cancel_res is True, "Order cancellation failed."
    assert executor.active_orders[order_id]["status"] == "CANCELLED", "Order status mismatch."
    print("OrderExecutor placement and cancellation checks passed.")

    print("\n2. Testing DualTrackTradingEngine Intelligence Injection...")
    engine = DualTrackTradingEngine(symbol="BINANCE_PERP_BTC_USDT", mock_execution=True)

    test_news = "Regulatory body issues warning on speculative perpetual trading."
    engine.update_intelligence(test_news, source_type="news")

    assert engine.latest_nlp_intel is not None, "Failed to update NLP intelligence."
    assert engine.latest_nlp_intel["event_category"] == "Macro_Policy", "Event category mismatch."
    print("DualTrackTradingEngine intelligence ingestion passed.")

    print("\n3. Testing Fast Path L2 Snapshot Processing...")
    mock_snapshot = {
        "symbol": "BINANCE_PERP_BTC_USDT",
        "data": {
            "bids": [[60000.0, 10.0], [59990.0, 5.0]],
            "asks": [[60010.0, 1.0], [60020.0, 1.0]],
        },
    }

    await engine._process_orderbook_snapshot(mock_snapshot)
    assert len(engine.executor.active_orders) > 0, "Fast path should trigger an order on strong imbalance."
    print("Fast path snapshot ingestion and decision routing passed.")

def main():
    asyncio.run(run_async_tests())

if __name__ == "__main__":
    main()