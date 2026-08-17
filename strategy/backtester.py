import json
import math
import os
from typing import Any, Dict, List
from interface.logger import ReasoningLogger
from strategy.manager import RiskManager
from strategy.adapter import StrategyAdapter


class Backtester:
    def __init__(
        self,
        data_path: str,
        initial_balance: float = 10000.0,
        obi_threshold: float = 0.25,
        order_size: float = 0.01,
        log_file: str = "logs/backtest_reasoning.jsonl",
    ):
        self.data_path = data_path
        self.initial_balance = initial_balance
        self.risk_manager = RiskManager(
            initial_balance=initial_balance,
            max_leverage=2.0,
            nav_drawdown_limit=0.80,
        )
        self.logger = ReasoningLogger(log_file=log_file)
        self.strategy = StrategyAdapter(
            risk_manager=self.risk_manager,
            reasoning_logger=self.logger,
            obi_threshold=obi_threshold,
            default_order_size=order_size,
        )

        self.nav_history: List[float] = [1.0]
        self.trade_history: List[Dict[str, Any]] = []

    def run(self) -> Dict[str, Any]:
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Market data file not found: {self.data_path}")

        with open(self.data_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except Exception:
                    continue

                data = event.get("data", {})
                bids = data.get("bids", [])
                asks = data.get("asks", [])
                symbol = event.get("symbol", "BINANCE_PERP_BTC_USDT")

                if not bids or not asks:
                    continue

                decision_record = self.strategy.generate_decision(
                    symbol=symbol,
                    bids=bids,
                    asks=asks,
                    nlp_intel=None,  
                )

                action = decision_record["decision"]["action"]
                params = decision_record["decision"]["order_params"]
                if action in ["BUY_OPEN", "SELL_OPEN"] and params:
                    self._execute_simulated_order(symbol, action, params["price"], params["quantity"])

                current_nav = self.risk_manager.calculate_nav()
                self.nav_history.append(current_nav)

        return self.calculate_metrics()

    def _execute_simulated_order(self, symbol: str, action: str, price: float, quantity: float) -> None:
        side_multiplier = 1.0 if action == "BUY_OPEN" else -1.0
        if symbol not in self.risk_manager.positions:
            self.risk_manager.positions[symbol] = {
                "quantity": side_multiplier * quantity,
                "entry_price": price,
                "current_price": price,
            }
        else:
            pos = self.risk_manager.positions[symbol]
            new_qty = pos["quantity"] + side_multiplier * quantity
            if new_qty != 0:
                pos["entry_price"] = (
                    pos["quantity"] * pos["entry_price"] + (side_multiplier * quantity) * price
                ) / new_qty
            pos["quantity"] = new_qty
            pos["current_price"] = price

        self.trade_history.append({
            "symbol": symbol,
            "action": action,
            "price": price,
            "quantity": quantity,
        })

    def calculate_metrics(self) -> Dict[str, Any]:
        if not self.nav_history:
            return {"total_return": 0.0, "max_drawdown": 0.0, "sharpe_ratio": 0.0}

        final_nav = self.nav_history[-1]
        total_return = (final_nav - 1.0) * 100.0

        peak = self.nav_history[0]
        max_drawdown = 0.0
        for nav in self.nav_history:
            if nav > peak:
                peak = nav
            drawdown = (peak - nav) / peak if peak > 0 else 0.0
            if drawdown > max_drawdown:
                max_drawdown = drawdown

        periodic_returns = [
            (self.nav_history[i] - self.nav_history[i - 1]) / self.nav_history[i - 1]
            for i in range(1, len(self.nav_history))
            if self.nav_history[i - 1] > 0
        ]

        if len(periodic_returns) > 1:
            mean_ret = sum(periodic_returns) / len(periodic_returns)
            variance = sum((r - mean_ret) ** 2 for r in periodic_returns) / (len(periodic_returns) - 1)
            std_dev = math.sqrt(variance)
            sharpe_ratio = (mean_ret / std_dev) * math.sqrt(365 * 24 * 60) if std_dev > 0 else 0.0
        else:
            sharpe_ratio = 0.0

        return {
            "initial_balance": self.initial_balance,
            "final_nav": round(final_nav, 4),
            "total_return_pct": round(total_return, 2),
            "max_drawdown_pct": round(max_drawdown * 100.0, 2),
            "sharpe_ratio": round(sharpe_ratio, 2),
            "total_trades": len(self.trade_history),
        }