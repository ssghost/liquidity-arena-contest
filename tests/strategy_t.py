import json
import os
from pathlib import Path

from strategy.backtester import Backtester
from strategy.manager import RiskManager
from strategy.adapter import StrategyAdapter
from interface.logger import ReasoningLogger


def run_tests() -> None:
    print("\n1. Testing RiskManager...")
    rm = RiskManager(initial_balance=10000.0, max_leverage=2.0, nav_drawdown_limit=0.80)
    assert rm.calculate_nav() == 1.0, "Initial NAV must be 1.0."

    passed, reason, meta = rm.evaluate_order("BINANCE_PERP_BTC_USDT", "BUY_OPEN", 60000.0, 0.01)
    assert passed is True, f"Normal order should pass: {reason}"

    passed_excess, reason_excess, meta_excess = rm.evaluate_order(
        "BINANCE_PERP_BTC_USDT", "BUY_OPEN", 60000.0, 1.0
    )
    assert passed_excess is False, "Excessive leverage order should be blocked."
    print("RiskManager leverage constraints and NAV evaluations passed.")

    print("\n2. Testing StrategyAdapter...")
    logger = ReasoningLogger(log_file="logs/test_phase2_reasoning.jsonl")
    adapter = StrategyAdapter(risk_manager=rm, reasoning_logger=logger, obi_threshold=0.20)

    mock_bids = [[60000.0, 5.0], [59990.0, 3.0]]
    mock_asks = [[60010.0, 1.0], [60020.0, 1.0]]

    features = adapter.calculate_microstructure_features(mock_bids, mock_asks)
    assert features["obi"] > 0.20, "OBI calculation mismatch."

    mock_nlp = {"trade_action_filter": "ALLOW_ALL", "directional_bias": 0.5, "impact_level": "MEDIUM"}
    decision = adapter.generate_decision("BINANCE_PERP_BTC_USDT", mock_bids, mock_asks, mock_nlp)
    assert decision["decision"]["action"] == "BUY_OPEN", "Strategy should trigger BUY_OPEN on bullish signal."
    print("StrategyAdapter microstructure calculation and signal synthesis passed.")

    print("\n3. Testing Backtester offline replay...")
    temp_data_file = "logs/mock_market_data.jsonl"
    os.makedirs(os.path.dirname(temp_data_file), exist_ok=True)

    try:
        with open(temp_data_file, "w", encoding="utf-8") as f:
            f.write(json.dumps({
                "symbol": "BINANCE_PERP_BTC_USDT",
                "data": {"bids": [[60000.0, 5.0], [59990.0, 3.0]], "asks": [[60010.0, 1.0], [60020.0, 1.0]]}
            }) + "\n")
            f.write(json.dumps({
                "symbol": "BINANCE_PERP_BTC_USDT",
                "data": {"bids": [[60100.0, 1.0], [60090.0, 1.0]], "asks": [[60110.0, 6.0], [60120.0, 4.0]]}
            }) + "\n")
            f.write(json.dumps({
                "symbol": "BINANCE_PERP_BTC_USDT",
                "data": {"bids": [[60050.0, 5.0], [60040.0, 2.0]], "asks": [[60060.0, 1.0], [60070.0, 1.0]]}
            }) + "\n")

        backtester = Backtester(
            data_path=temp_data_file,
            initial_balance=10000.0,
            fee_rate=0.0005,
            slippage_bps=0.5,
        )
        results = backtester.run()

        required_fields = [
            "initial_balance",
            "final_nav",
            "total_return_pct",
            "max_drawdown_pct",
            "sharpe_ratio",
            "total_orders",
            "closed_trades",
            "win_rate_pct",
            "profit_factor",
            "total_fees_paid",
        ]
        for field in required_fields:
            assert field in results, f"Backtest results missing field: {field}"

        assert results["total_orders"] > 0, "Backtest should execute simulated orders."
        print(f"Backtest execution passed. Metrics: {results}")
    finally:
        if os.path.exists(temp_data_file):
            os.remove(temp_data_file)

if __name__ == "__main__":
    run_tests()