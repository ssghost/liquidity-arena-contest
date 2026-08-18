import json
import os
from interface.parser import NLPParser
from interface.logger import ReasoningLogger
from interface.limiter import TokenLimiter


def run_tests() -> None:
    print("\n1. Testing TokenLimiter...")
    limiter = TokenLimiter(daily_budget_usd=10.0)
    assert limiter.can_make_request(
        estimated_tokens=1000
    ), "Token budget check failed."
    limiter.record_usage(prompt_tokens=500, completion_tokens=200)
    assert limiter.usage_state["input_tokens"] == 500, "Input token count mismatch."
    assert (
        limiter.usage_state["output_tokens"] == 200
    ), "Output token count mismatch."
    print("TokenLimiter tracking and budget checks passed.")

    print("\n2. Testing NLPParser...")
    raw_news = "Federal Reserve announces unexpected interest rate cut. Crypto market surges."
    messages = NLPParser.build_messages(raw_news, source_type="news")
    assert len(messages) == 2, "Failed to construct prompt message payload."

    mock_llm_response = json.dumps({
        "timestamp": "2026-08-16T10:00:00Z",
        "target_asset": "BTC",
        "event_category": "Macro_Policy",
        "impact_level": "HIGH",
        "volatility_bias": "EXPANSION",
        "directional_bias": 0.85,
        "trade_action_filter": "ALLOW_ALL",
        "event_summary": "Fed cuts rates, creating bullish liquidity expansion.",
        "confidence": 0.9,
    })

    parsed_result = NLPParser.parse_response(mock_llm_response)
    assert parsed_result is not None, "NLP parser returned None."
    assert parsed_result["target_asset"] == "BTC", "Target asset parsing mismatch."
    assert parsed_result["directional_bias"] == 0.85, "Directional bias parsing mismatch."
    print("NLPParser schema parsing and validation passed.")

    print("\n3. Testing TokenLimiter cache mechanism...")
    limiter.store_cache(raw_news, parsed_result)
    cached_data = limiter.get_cached_result(raw_news)
    assert cached_data is not None, "Failed to retrieve cached intelligence."
    assert (
        cached_data["event_category"] == "Macro_Policy"
    ), "Cached data mismatch."
    print("TokenLimiter caching operations passed.")

    print("\n4. Testing ReasoningLogger...")
    logger = ReasoningLogger(log_file="logs/test_reasoning_log.jsonl")
    mock_market_data = {
        "mid_price": 60500.0,
        "obi": 0.35,
        "funding_rate": 0.0001,
        "volatility": 0.02,
        "signal_state": "BULLISH_MOMENTUM",
    }
    mock_risk = {"leverage": 1.5, "nav": 1.0, "passed": True}

    log_entry = logger.log_decision(
        symbol="BINANCE_PERP_BTC_USDT",
        market_data=mock_market_data,
        nlp_intelligence=parsed_result,
        logical_deduction="Macro rate cut bias aligned with positive orderbook imbalance.",
        action="BUY_OPEN",
        action_params={"price": 60500.0, "quantity": 0.001},
        risk_evaluation=mock_risk,
    )

    assert os.path.exists("logs/test_reasoning_log.jsonl"), "Reasoning log file was not created."
    assert (
        log_entry["decision"]["action"] == "BUY_OPEN"
    ), "Logged trade action mismatch."
    print("ReasoningLogger entry generation passed.")

if __name__ == "__main__":
    run_tests()