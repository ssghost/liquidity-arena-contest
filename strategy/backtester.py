import json
import math
import os
from typing import Any, Dict, List, Optional
from interface.logger import ReasoningLogger
from interface.parser import NLPParser
from strategy.adapter import StrategyAdapter
from strategy.manager import RiskManager

class Backtester:
    def __init__(
        self,
        data_path: str = "data/orderbook.jsonl",
        news_path: str = "data/news_feed.jsonl",
        initial_balance: float = 10000.0,
        obi_threshold: float = 0.30,
        order_size: float = 0.01,
        log_file: str = "logs/backtest_reasoning.jsonl",
        fee_rate: float = 0.0005,
        slippage_bps: float = 0.5,
    ):
        self.data_path = data_path
        self.news_path = news_path
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.log_file = log_file
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
        self.fee_rate = fee_rate
        self.slippage_bps = slippage_bps
        self.closed_trades: List[Dict[str, Any]] = []

    def _load_news_timeline(self, sim_start_time: float) -> List[Dict[str, Any]]:
        if not os.path.exists(self.news_path):
            return []

        raw_news_list: List[Dict[str, Any]] = []
        with open(self.news_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                    parsed_intel = NLPParser.heuristic_parse(item)
                    raw_news_list.append(parsed_intel)
                except Exception:
                    continue

        if not raw_news_list:
            return []

        raw_news_list.sort(key=lambda x: x.get("timestamp", 0))
        min_news_ts = raw_news_list[0].get("timestamp", 0)
        
        timeline: List[Dict[str, Any]] = []
        for idx, news in enumerate(raw_news_list):
            item = dict(news)
            offset = (news.get("timestamp", 0) - min_news_ts)
            if offset < 0 or offset > 172800:
                offset = (idx / max(1, len(raw_news_list))) * 165600.0  
            item["sim_timestamp"] = sim_start_time + offset
            timeline.append(item)

        timeline.sort(key=lambda x: x["sim_timestamp"])
        return timeline

    def run(
        self,
        verbose: bool = True,
        save_summary: bool = True,
        summary_path: str = "logs/backtest_summary.json",
    ) -> Dict[str, Any]:
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Market data file not found: {self.data_path}")

        if verbose:
            print("Starting Event-Driven Backtest with NLP Intelligence Feed")

        nlp_timeline: List[Dict[str, Any]] = []
        current_nlp_intel: Optional[Dict[str, Any]] = None
        nlp_idx = 0
        sim_start_time = None

        with open(self.data_path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except Exception:
                    continue

                if event.get("event") == "subscribe":
                    continue

                arg = event.get("arg", {})
                data = event.get("data", {})

                raw_bids = data.get("Bids") or data.get("bids", [])
                raw_asks = data.get("Asks") or data.get("asks", [])
                symbol = arg.get("sym") or event.get("symbol", "BINANCE_PERP_BTC_USDT")

                if not raw_bids or not raw_asks:
                    continue

                try:
                    bids = [[float(b[0]), float(b[1])] for b in raw_bids]
                    asks = [[float(a[0]), float(a[1])] for a in raw_asks]
                except (ValueError, IndexError):
                    continue

                current_time = event.get("timestamp") or event.get("time") or (line_no * 0.3)
                if sim_start_time is None:
                    sim_start_time = current_time
                    nlp_timeline = self._load_news_timeline(sim_start_time)
                    if verbose and nlp_timeline:
                        print(f"Loaded {len(nlp_timeline)} NLP intelligence events for dual-track backtest.")

                while nlp_idx < len(nlp_timeline) and current_time >= nlp_timeline[nlp_idx]["sim_timestamp"]:
                    current_nlp_intel = nlp_timeline[nlp_idx]
                    nlp_idx += 1

                mid_price = (bids[0][0] + asks[0][0]) / 2.0
                if symbol in self.risk_manager.positions:
                    self.risk_manager.positions[symbol]["current_price"] = mid_price

                decision_record = self.strategy.generate_decision(
                    symbol=symbol,
                    bids=bids,
                    asks=asks,
                    nlp_intel=current_nlp_intel,
                )

                action = decision_record["decision"]["action"]
                params = decision_record["decision"]["order_params"]

                if action in ["BUY_OPEN", "SELL_OPEN"] and params:
                    self._execute_simulated_order(symbol, action, params["price"], params["quantity"])
                elif action == "CLOSE_POSITION" and params:
                    close_action = "SELL_CLOSE" if params.get("side") == "SELL" else "BUY_CLOSE"
                    self._execute_simulated_order(symbol, close_action, params["price"], params["quantity"])

                unrealized_pnl = sum(
                    (pos["current_price"] - pos["entry_price"]) * pos["quantity"]
                    for pos in self.risk_manager.positions.values()
                )
                current_nav = (self.balance + unrealized_pnl) / self.initial_balance
                self.nav_history.append(current_nav)

        metrics = self.calculate_metrics()

        if verbose:
            print("BACKTEST PERFORMANCE SUMMARY")
            print(f"Final NAV         : {metrics['final_nav']:.4f}")
            print(f"Total Return      : {metrics['total_return_pct']:+.2f}%")
            print(f"Max Drawdown      : {metrics['max_drawdown_pct']:.2f}%")
            print(f"Sharpe Ratio      : {metrics['sharpe_ratio']:.2f}")
            print(f"Win Rate          : {metrics['win_rate_pct']:.2f}% ({metrics['closed_trades']} closed trades)")
            print(f"Profit Factor     : {metrics['profit_factor']:.2f}")
            print(f"Total Orders Exec : {metrics['total_orders']}")
            print(f"Total Fees Paid   : ${metrics['total_fees_paid']:.4f}")
            print(f"Reasoning Log     : {self.log_file}")

        if save_summary:
            os.makedirs(os.path.dirname(summary_path), exist_ok=True)
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2)
            if verbose:
                print(f"Summary metrics saved to: {summary_path}")

        return metrics

    def _execute_simulated_order(self, symbol: str, action: str, price: float, quantity: float) -> None:
        is_buy = action in ["BUY_OPEN", "BUY_CLOSE"]
        side_multiplier = 1.0 if is_buy else -1.0
        slippage_factor = (1.0 + self.slippage_bps * 0.0001) if is_buy else (1.0 - self.slippage_bps * 0.0001)
        exec_price = price * slippage_factor
        fee = exec_price * quantity * self.fee_rate

        realized_pnl = 0.0

        if symbol not in self.risk_manager.positions:
            self.risk_manager.positions[symbol] = {
                "quantity": side_multiplier * quantity,
                "entry_price": exec_price,
                "current_price": exec_price,
            }
            self.balance -= fee
        else:
            pos = self.risk_manager.positions[symbol]
            current_qty = pos["quantity"]

            if (current_qty > 0 and side_multiplier < 0) or (current_qty < 0 and side_multiplier > 0):
                closed_qty = min(abs(current_qty), quantity)
                if current_qty > 0:
                    realized_pnl = (exec_price - pos["entry_price"]) * closed_qty - fee
                else:
                    realized_pnl = (pos["entry_price"] - exec_price) * closed_qty - fee

                self.balance += realized_pnl
                self.closed_trades.append({
                    "symbol": symbol,
                    "closed_qty": closed_qty,
                    "entry_price": pos["entry_price"],
                    "exit_price": exec_price,
                    "pnl": realized_pnl,
                    "fee": fee,
                })
            else:
                self.balance -= fee

            new_qty = current_qty + side_multiplier * quantity
            if new_qty != 0:
                if (current_qty >= 0 and side_multiplier > 0) or (current_qty <= 0 and side_multiplier < 0):
                    pos["entry_price"] = (
                        abs(current_qty) * pos["entry_price"] + quantity * exec_price
                    ) / abs(new_qty)
            else:
                pos["entry_price"] = 0.0

            pos["quantity"] = new_qty
            pos["current_price"] = exec_price

        self.trade_history.append({
            "symbol": symbol,
            "action": action,
            "price": exec_price,
            "quantity": quantity,
            "fee": fee,
            "realized_pnl": realized_pnl,
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

        sample_step = 200
        sampled_nav = self.nav_history[::sample_step]
        if len(sampled_nav) < 2:
            sampled_nav = self.nav_history

        periodic_returns = [
            (sampled_nav[i] - sampled_nav[i - 1]) / sampled_nav[i - 1]
            for i in range(1, len(sampled_nav))
            if sampled_nav[i - 1] > 0
        ]

        if len(periodic_returns) > 1:
            mean_ret = sum(periodic_returns) / len(periodic_returns)
            variance = sum((r - mean_ret) ** 2 for r in periodic_returns) / (len(periodic_returns) - 1)
            std_dev = math.sqrt(variance)
            sharpe_ratio = (mean_ret / std_dev) * math.sqrt(365 * 24 * 60) if std_dev > 1e-9 else 0.0
        else:
            sharpe_ratio = 0.0

        pnls = [t["pnl"] for t in self.closed_trades]
        winning_trades = [p for p in pnls if p > 0]
        losing_trades = [p for p in pnls if p < 0]

        total_closed = len(pnls)
        win_rate_pct = (len(winning_trades) / total_closed * 100.0) if total_closed > 0 else 0.0
        total_gain = sum(winning_trades)
        total_loss = abs(sum(losing_trades))
        profit_factor = (total_gain / total_loss) if total_loss > 0 else (999.0 if total_gain > 0 else 0.0)
        total_fees = sum(t.get("fee", 0.0) for t in self.trade_history)

        return {
            "initial_balance": self.initial_balance,
            "final_nav": round(final_nav, 4),
            "total_return_pct": round(total_return, 2),
            "max_drawdown_pct": round(max_drawdown * 100.0, 2),
            "sharpe_ratio": round(sharpe_ratio, 2),
            "total_orders": len(self.trade_history),
            "closed_trades": total_closed,
            "win_rate_pct": round(win_rate_pct, 2),
            "profit_factor": round(profit_factor, 2),
            "total_fees_paid": round(total_fees, 4),
        }

if __name__ == "__main__":
    Backtester().run()