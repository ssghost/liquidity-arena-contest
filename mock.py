import asyncio
import json
import logging
import os
import urllib.request
from dotenv import load_dotenv

load_dotenv()

from agent.engine import DualTrackTradingEngine
from interface.parser import NLPParser

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("LiveMockTest")


def query_llm_live(news_text: str) -> str:
    api_base = os.getenv("LLM_API_BASE")
    api_key = os.getenv("LLM_API_KEY")
    model = os.getenv("LLM_MODEL")

    messages = NLPParser.build_messages(raw_text=news_text, source_type="news")
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.0,
    }
    data_bytes = json.dumps(payload).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    req = urllib.request.Request(f"{api_base}/chat/completions", data=data_bytes, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=30) as response:
        res_json = json.loads(response.read().decode("utf-8"))
        return res_json["choices"][0]["message"]["content"]


async def simulate_market_feed(engine: DualTrackTradingEngine) -> None:
    await asyncio.sleep(1.0)  
    
    sample_snapshot = {
        "symbol": "BINANCE_PERP_BTC_USDT",
        "data": {
            "bids": [[68500.0, 5.0], [68490.0, 3.2], [68480.0, 4.0]],
            "asks": [[68505.0, 0.8], [68510.0, 1.1], [68520.0, 0.9]],
        }
    }
    
    logger.info("Feeding L2 orderbook snapshot into engine...")
    await engine._process_orderbook_snapshot(sample_snapshot)


async def run_full_system_test() -> None:
    engine = DualTrackTradingEngine(
        symbol="BINANCE_PERP_BTC_USDT",
        initial_balance=10000.0,
        mock_execution=True,
        log_file="logs/test_live_reasoning.jsonl",
    )

    raw_news = "U.S. SEC approves multiple Bitcoin and Ethereum staking ETFs, triggering immediate institutional inflows."
    logger.info(f"Querying live LLM with news: {raw_news}")
    llm_output = query_llm_live(raw_news)
    logger.info(f"Live LLM Raw Output received: {llm_output}")

    engine.update_intelligence(raw_news=raw_news, mock_llm_json=llm_output)

    engine_task = asyncio.create_task(engine.start())
    feed_task = asyncio.create_task(simulate_market_feed(engine))

    await asyncio.sleep(3.0)

    engine.stop()
    feed_task.cancel()
    engine_task.cancel()
    try:
        await asyncio.gather(engine_task, feed_task, return_exceptions=True)
    except Exception:
        pass

    assert engine.latest_nlp_intel is not None, "NLP intelligence must be loaded into engine"
    logger.info(f"Engine State Verified. Active Intel Bias: {engine.latest_nlp_intel.get('directional_bias')}")

if __name__ == "__main__":
    try:
        asyncio.run(run_full_system_test())
    except KeyboardInterrupt:
        pass