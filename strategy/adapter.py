import time
from typing import Any, Dict, List, Optional
from interface.logger import ReasoningLogger
from strategy.manager import RiskManager


class StrategyAdapter:
    def __init__(
        self,
        risk_manager: RiskManager,
        reasoning_logger: Optional[ReasoningLogger] = None,
        obi_threshold: float = 0.30,
        default_order_size: float = 0.01,
        tp_bps: float = 110.0,             
        sl_bps: float = 35.0,              
        max_holding_ticks: int = 6000,     
        max_inventory_qty: float = 0.01,   
        cooldown_ticks: int = 300,         
        ema_alpha: float = 0.08,          
        trend_alpha: float = 0.00033,     
        max_spread_bps: float = 2.5,      
    ):
        self.risk_manager = risk_manager
        self.logger = reasoning_logger
        self.obi_threshold = obi_threshold
        self.default_order_size = default_order_size
        self.tp_bps = tp_bps
        self.sl_bps = sl_bps
        self.max_holding_ticks = max_holding_ticks
        self.max_inventory_qty = max_inventory_qty
        self.cooldown_ticks = cooldown_ticks
        self.ema_alpha = ema_alpha
        self.trend_alpha = trend_alpha
        self.max_spread_bps = max_spread_bps

        self.position_age: Dict[str, int] = {}
        self.cooldown_counter: Dict[str, int] = {}
        self.ema_obi_state: Dict[str, float] = {}
        self.trend_price_state: Dict[str, float] = {}

    def calculate_microstructure_features(
        self, bids: List[List[float]], asks: List[List[float]], symbol: str = "BINANCE_PERP_BTC_USDT"
    ) -> Dict[str, float]:
        if not bids or not asks:
            return {
                "obi": 0.0,
                "ema_obi": 0.0,
                "spread": 0.0,
                "spread_bps": 0.0,
                "mid_price": 0.0,
                "microprice": 0.0,
                "trend_price": 0.0,
                "trend_momentum_bps": 0.0,
                "best_bid": 0.0,
                "best_ask": 0.0,
            }

        best_bid_p, best_bid_q = bids[0][0], bids[0][1]
        best_ask_p, best_ask_q = asks[0][0], asks[0][1]

        total_bid_depth = sum(b[1] for b in bids[:5])
        total_ask_depth = sum(a[1] for a in asks[:5])
        depth_sum = total_bid_depth + total_ask_depth

        obi = (total_bid_depth - total_ask_depth) / depth_sum if depth_sum > 0 else 0.0
        spread = best_ask_p - best_bid_p
        mid_price = (best_bid_p + best_ask_p) / 2.0
        spread_bps = (spread / mid_price) * 10000.0 if mid_price > 0 else 0.0

        prev_ema_obi = self.ema_obi_state.get(symbol, obi)
        ema_obi = self.ema_alpha * obi + (1.0 - self.ema_alpha) * prev_ema_obi
        self.ema_obi_state[symbol] = ema_obi

        prev_trend = self.trend_price_state.get(symbol, mid_price)
        trend_price = self.trend_alpha * mid_price + (1.0 - self.trend_alpha) * prev_trend
        self.trend_price_state[symbol] = trend_price
        trend_momentum_bps = ((mid_price - trend_price) / trend_price) * 10000.0 if trend_price > 0 else 0.0

        top_qty_sum = best_bid_q + best_ask_q
        microprice = (
            (best_bid_p * best_ask_q + best_ask_p * best_bid_q) / top_qty_sum
            if top_qty_sum > 0
            else mid_price
        )

        return {
            "obi": obi,
            "ema_obi": ema_obi,
            "spread": spread,
            "spread_bps": spread_bps,
            "mid_price": mid_price,
            "microprice": microprice,
            "trend_price": trend_price,
            "trend_momentum_bps": trend_momentum_bps,
            "best_bid": best_bid_p,
            "best_ask": best_ask_p,
        }

    def generate_decision(
        self,
        symbol: str,
        bids: List[List[float]],
        asks: List[List[float]],
        nlp_intel: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        features = self.calculate_microstructure_features(bids, asks, symbol=symbol)
        mid_price = features["mid_price"]
        ema_obi = features["ema_obi"]
        trend_price = features["trend_price"]
        spread_bps = features["spread_bps"]

        pos = self.risk_manager.positions.get(
            symbol, {"quantity": 0.0, "entry_price": 0.0, "current_price": mid_price}
        )
        current_qty = pos["quantity"]
        entry_price = pos["entry_price"]

        current_age = self.position_age.get(symbol, 0)
        current_cooldown = self.cooldown_counter.get(symbol, 0)

        if current_qty != 0:
            current_age += 1
            self.position_age[symbol] = current_age
        else:
            self.position_age[symbol] = 0

        if current_cooldown > 0:
            self.cooldown_counter[symbol] = current_cooldown - 1

        action = "HOLD"
        reason = "Market within neutral parameters."
        order_params: Dict[str, Any] = {}

        if current_qty > 0 and entry_price > 0:
            pnl_bps = ((mid_price - entry_price) / entry_price) * 10000.0

            if pnl_bps >= self.tp_bps:
                action = "CLOSE_POSITION"
                reason = f"Take-Profit triggered (+{pnl_bps:.2f} bps)."
                order_params = {"price": features["best_bid"], "quantity": abs(current_qty), "side": "SELL"}
            elif pnl_bps <= -self.sl_bps:
                action = "CLOSE_POSITION"
                reason = f"Stop-Loss triggered ({pnl_bps:.2f} bps, SL threshold: -{self.sl_bps:.2f} bps)."
                order_params = {"price": features["best_bid"], "quantity": abs(current_qty), "side": "SELL"}
            elif current_age >= self.max_holding_ticks:
                action = "CLOSE_POSITION"
                reason = f"Time Stop triggered ({current_age} ticks, PnL: {pnl_bps:.2f} bps)."
                order_params = {"price": features["best_bid"], "quantity": abs(current_qty), "side": "SELL"}

        if (
            action == "HOLD"
            and abs(current_qty) < self.max_inventory_qty
            and self.cooldown_counter.get(symbol, 0) == 0
            and spread_bps <= self.max_spread_bps
        ):
            nlp_filter = nlp_intel.get("trade_action_filter", "ALLOW_ALL") if nlp_intel else "ALLOW_ALL"
            directional_bias = nlp_intel.get("directional_bias", 0.0) if nlp_intel else 0.0

            if (
                mid_price > trend_price
                and ema_obi > self.obi_threshold
                and nlp_filter in ["ALLOW_ALL", "ALLOW_BUY_ONLY"]
            ):
                if directional_bias >= -0.2:
                    passed, r_reason, _ = self.risk_manager.evaluate_order(
                        symbol, "BUY_OPEN", features["best_bid"], self.default_order_size
                    )
                    if passed:
                        action = "BUY_OPEN"
                        reason = (
                            f"Trend 15m Bullish (Price > EMA) with OBI support ({ema_obi:.2f}). "
                            f"{r_reason}"
                        )
                        order_params = {"price": features["best_bid"], "quantity": self.default_order_size}

        if action == "CLOSE_POSITION":
            self.cooldown_counter[symbol] = self.cooldown_ticks
            self.position_age[symbol] = 0

        decision_record = {
            "timestamp": time.time(),
            "symbol": symbol,
            "microstructure": features,
            "decision": {
                "action": action,
                "reason": reason,
                "order_params": order_params,
            },
        }

        if self.logger:
            self.logger.log_decision(
                symbol=symbol,
                market_data=features,
                nlp_intelligence=nlp_intel,
                logical_deduction=reason,
                action=action,
                action_params=order_params,
                risk_evaluation={"nav": self.risk_manager.calculate_nav()},
            )

        return decision_record