import asyncio
import json
import logging
from typing import Any, Dict, Optional
import websockets

from interface.limiter import TokenLimiter
from interface.logger import ReasoningLogger
from interface.parser import NLPParser
from agent.executor import OrderExecutor
from strategy.manager import RiskManager
from strategy.adapter import StrategyAdapter

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("DualTrackEngine")

class DualTrackTradingEngine:
    def __init__(
        self,
        ws_url: str = "wss://market.liquiditytech.example.com/ws",
        symbol: str = "BINANCE_PERP_BTC_USDT",
        initial_balance: float = 10000.0,
        mock_execution: bool = True,
        log_file: str = "logs/live_reasoning.jsonl",
    ):
        self.ws_url = ws_url
        self.symbol = symbol
        self.is_running = False

        self.reasoning_logger = ReasoningLogger(log_file=log_file)
        self.risk_manager = RiskManager(
            initial_balance=initial_balance,
            max_leverage=2.0,
            nav_drawdown_limit=0.80,
        )
        self.token_limiter = TokenLimiter(daily_budget_usd=10.0)
        self.strategy_adapter = StrategyAdapter(
            risk_manager=self.risk_manager,
            reasoning_logger=self.reasoning_logger,
            obi_threshold=0.25,
            default_order_size=0.01,
        )
        self.executor = OrderExecutor(mock_mode=mock_execution)

        self.latest_nlp_intel: Optional[Dict[str, Any]] = None

    async def start(self) -> None:
        self.is_running = True
        logger.info("Starting Dual-Track Trading Engine...")
        await asyncio.gather(
            self._fast_path_market_loop(),
            self._slow_path_intelligence_loop(),
        )

    def stop(self) -> None:
        self.is_running = False
        logger.info("Stopping Dual-Track Trading Engine...")

    async def _fast_path_market_loop(self) -> None:
        logger.info("Fast Path (L2 Order Book Stream) initialized.")
        while self.is_running:
            try:
                if "example.com" in self.ws_url:
                    await asyncio.sleep(1.0)
                    continue

                async with websockets.connect(self.ws_url) as ws:
                    logger.info(f"Connected to Market WebSocket: {self.ws_url}")
                    async for message in ws:
                        if not self.is_running:
                            break
                        data = json.loads(message)
                        await self._process_orderbook_snapshot(data)
            except Exception as e:
                logger.error(f"Fast Path WebSocket connection error: {e}")
                await asyncio.sleep(2.0)

    async def _process_orderbook_snapshot(self, event_data: Dict[str, Any]) -> None:
        data = event_data.get("data", {})
        bids = data.get("bids", [])
        asks = data.get("asks", [])
        symbol = event_data.get("symbol", self.symbol)

        if not bids or not asks:
            return

        decision_record = self.strategy_adapter.generate_decision(
            symbol=symbol,
            bids=bids,
            asks=asks,
            nlp_intel=self.latest_nlp_intel,
        )

        decision = decision_record.get("decision", {})
        action = decision.get("action")
        params = decision.get("order_params", {})

        if action in ["BUY_OPEN", "SELL_OPEN"] and params:
            side = "BUY" if action == "BUY_OPEN" else "SELL"
            await self.executor.place_order(
                symbol=symbol,
                side=side,
                price=params["price"],
                quantity=params["quantity"],
            )
        elif action == "CLOSE_POSITION":
            logger.warning(f"Executing emergency position closure for {symbol}")

    async def _slow_path_intelligence_loop(self) -> None:
        logger.info("Slow Path (NLP News / Sentiment Stream) initialized.")
        while self.is_running:
            try:
                await asyncio.sleep(10.0)
            except Exception as e:
                logger.error(f"Slow Path error: {e}")
                await asyncio.sleep(5.0)

    def update_intelligence(
        self,
        raw_news: str,
        source_type: str = "news",
        mock_llm_json: Optional[str] = None,
    ) -> None:
        cached = self.token_limiter.get_cached_result(raw_news)
        if cached:
            self.latest_nlp_intel = cached
            logger.info("NLP intelligence updated from cache.")
            return

        messages = NLPParser.build_messages(
            raw_text=raw_news, source_type=source_type
        )
        estimated_tokens = sum(len(m.get("content", "")) // 4 for m in messages)

        if not self.token_limiter.can_make_request(
            estimated_tokens=estimated_tokens
        ):
            logger.warning("Token budget exceeded. Skipping NLP update.")
            return

        if mock_llm_json is None:
            mock_llm_json = json.dumps({
                "timestamp": "2026-08-17T00:00:00Z",
                "target_asset": self.symbol.split("_")[-2]
                if "_" in self.symbol
                else "BTC",
                "event_category": "Macro_Policy",
                "impact_level": "LOW",
                "volatility_bias": "STABLE",
                "directional_bias": 0.0,
                "trade_action_filter": "ALLOW_ALL",
                "event_summary": "Routine macro context processed.",
                "confidence": 0.8,
            })

        parsed = NLPParser.parse_response(mock_llm_json)
        if parsed:
            self.token_limiter.record_usage(
                prompt_tokens=estimated_tokens, completion_tokens=80
            )
            self.token_limiter.store_cache(raw_news, parsed)
            self.latest_nlp_intel = parsed
            logger.info(
                f"NLP intelligence successfully parsed and cached: {parsed.get('event_category')}"
            )