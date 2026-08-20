import time
from typing import Any, Dict, List, Optional
from interface.logger import ReasoningLogger
from strategy.manager import RiskManager


class StrategyAdapter:
    def __init__(
        self,
        risk_manager: RiskManager,
        reasoning_logger: Optional[ReasoningLogger] = None,
        obi_threshold: float = 0.85,
        default_order_size: float = 0.01,
        # >>>>>>>> MODIFICATION START: 追蹤止損、時間衰減與動量過濾參數 <<<<<<<<
        tp_bps: float = 35.0,              # 止盈 35 bps (0.35%)
        sl_bps: float = 14.0,              # 初始止損 14 bps (0.14%)
        trailing_trigger_bps: float = 15.0,# 觸發保本止損門檻 (+15 bps)
        trailing_sl_bps: float = 3.0,      # 保本止損鎖定利潤 (+3 bps)
        max_holding_ticks: int = 200,      # 最大持倉快照數 (~60 秒，時間止損)
        max_inventory_qty: float = 0.01,   # 最大持倉約束
        cooldown_ticks: int = 180,         # 平倉後冷卻 180 快照 (~55 秒)
        min_holding_ticks: int = 40,       # 最小持倉 40 快照 (~12 秒)
        ema_alpha: float = 0.08,           # EMA 平滑係數
        max_spread_bps: float = 2.5,       # 最大允許點差 (2.5 bps)
        # >>>>>>>> MODIFICATION END <<<<<<<<
    ):
        self.risk_manager = risk_manager
        self.logger = reasoning_logger
        self.obi_threshold = obi_threshold
        self.default_order_size = default_order_size
        self.tp_bps = tp_bps
        self.sl_bps = sl_bps
        self.trailing_trigger_bps = trailing_trigger_bps
        self.trailing_sl_bps = trailing_sl_bps
        self.max_holding_ticks = max_holding_ticks
        self.max_inventory_qty = max_inventory_qty
        self.cooldown_ticks = cooldown_ticks
        self.min_holding_ticks = min_holding_ticks
        self.ema_alpha = ema_alpha
        self.max_spread_bps = max_spread_bps

        self.position_age: Dict[str, int] = {}
        self.cooldown_counter: Dict[str, int] = {}
        self.ema_obi_state: Dict[str, float] = {}
        self.last_mid_price: Dict[str, float] = {}
        self.max_favorable_pnl: Dict[str, float] = {}

    def calculate_microstructure_features(
        self, bids: List[List[float]], asks: List[List[float]], symbol: str = "BINANCE_PERP_BTC_USDT"
    ) -> Dict[str, float]:
        if not bids or not asks:
            return {"obi": 0.0, "ema_obi": 0.0, "spread": 0.0, "spread_bps": 0.0, "mid_price": 0.0, "microprice": 0.0}

        best_bid_p, best_bid_q = bids[0][0], bids[0][1]
        best_ask_p, best_ask_q = asks[0][0], asks[0][1]

        total_bid_depth = sum(b[1] for b in bids[:5])
        total_ask_depth = sum(a[1] for a in asks[:5])
        depth_sum = total_bid_depth + total_ask_depth

        obi = (total_bid_depth - total_ask_depth) / depth_sum if depth_sum > 0 else 0.0
        spread = best_ask_p - best_bid_p
        mid_price = (best_bid_p + best_ask_p) / 2.0
        spread_bps = (spread / mid_price) * 10000.0 if mid_price > 0 else 0.0

        prev_ema = self.ema_obi_state.get(symbol, obi)
        ema_obi = self.ema_alpha * obi + (1.0 - self.ema_alpha) * prev_ema
        self.ema_obi_state[symbol] = ema_obi

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
        microprice = features["microprice"]
        ema_obi = features["ema_obi"]
        spread_bps = features["spread_bps"]

        pos = self.risk_manager.positions.get(
            symbol, {"quantity": 0.0, "entry_price": 0.0, "current_price": mid_price}
        )
        current_qty = pos["quantity"]
        entry_price = pos["entry_price"]

        current_age = self.position_age.get(symbol, 0)
        current_cooldown = self.cooldown_counter.get(symbol, 0)
        prev_mid = self.last_mid_price.get(symbol, mid_price)

        if current_qty != 0:
            current_age += 1
            self.position_age[symbol] = current_age
        else:
            self.position_age[symbol] = 0
            self.max_favorable_pnl[symbol] = 0.0

        if current_cooldown > 0:
            self.cooldown_counter[symbol] = current_cooldown - 1

        action = "HOLD"
        reason = "Market within neutral parameters."
        order_params: Dict[str, Any] = {}

        # >>>>>>>> MODIFICATION START: 動態保本、時間止損與持倉退出 <<<<<<<<
        if current_qty > 0 and entry_price > 0:
            pnl_bps = ((mid_price - entry_price) / entry_price) * 10000.0
            max_pnl = max(self.max_favorable_pnl.get(symbol, 0.0), pnl_bps)
            self.max_favorable_pnl[symbol] = max_pnl

            # 動態止損線：若浮盈曾達 trailing_trigger_bps，止損線上調至 trailing_sl_bps
            effective_sl = -self.sl_bps if max_pnl < self.trailing_trigger_bps else self.trailing_sl_bps

            if pnl_bps >= self.tp_bps:
                action = "CLOSE_POSITION"
                reason = f"Take-Profit hit (+{pnl_bps:.2f} bps)."
                order_params = {"price": features["best_bid"], "quantity": abs(current_qty), "side": "SELL"}
            elif pnl_bps <= effective_sl:
                action = "CLOSE_POSITION"
                reason = f"Stop-Loss hit ({pnl_bps:.2f} bps, SL: {effective_sl:.2f} bps)."
                order_params = {"price": features["best_bid"], "quantity": abs(current_qty), "side": "SELL"}
            elif current_age >= self.max_holding_ticks:
                action = "CLOSE_POSITION"
                reason = f"Time Stop reached ({current_age} ticks, PnL: {pnl_bps:.2f} bps)."
                order_params = {"price": features["best_bid"], "quantity": abs(current_qty), "side": "SELL"}
            elif current_age >= self.min_holding_ticks and ema_obi < -self.obi_threshold * 0.8:
                action = "CLOSE_POSITION"
                reason = f"EMA OBI reversal exit (EMA_OBI: {ema_obi:.3f})."
                order_params = {"price": features["best_bid"], "quantity": abs(current_qty), "side": "SELL"}

        elif current_qty < 0 and entry_price > 0:
            pnl_bps = ((entry_price - mid_price) / entry_price) * 10000.0
            max_pnl = max(self.max_favorable_pnl.get(symbol, 0.0), pnl_bps)
            self.max_favorable_pnl[symbol] = max_pnl

            effective_sl = -self.sl_bps if max_pnl < self.trailing_trigger_bps else self.trailing_sl_bps

            if pnl_bps >= self.tp_bps:
                action = "CLOSE_POSITION"
                reason = f"Take-Profit hit (+{pnl_bps:.2f} bps)."
                order_params = {"price": features["best_ask"], "quantity": abs(current_qty), "side": "BUY"}
            elif pnl_bps <= effective_sl:
                action = "CLOSE_POSITION"
                reason = f"Stop-Loss hit ({pnl_bps:.2f} bps, SL: {effective_sl:.2f} bps)."
                order_params = {"price": features["best_ask"], "quantity": abs(current_qty), "side": "BUY"}
            elif current_age >= self.max_holding_ticks:
                action = "CLOSE_POSITION"
                reason = f"Time Stop reached ({current_age} ticks, PnL: {pnl_bps:.2f} bps)."
                order_params = {"price": features["best_ask"], "quantity": abs(current_qty), "side": "BUY"}
            elif current_age >= self.min_holding_ticks and ema_obi > self.obi_threshold * 0.8:
                action = "CLOSE_POSITION"
                reason = f"EMA OBI reversal exit (EMA_OBI: {ema_obi:.3f})."
                order_params = {"price": features["best_ask"], "quantity": abs(current_qty), "side": "BUY"}
        # >>>>>>>> MODIFICATION END <<<<<<<<

        # >>>>>>>> MODIFICATION START: 開倉動量與極端失衡過濾 <<<<<<<<
        if (
            action == "HOLD"
            and abs(current_qty) < self.max_inventory_qty
            and self.cooldown_counter.get(symbol, 0) == 0
            and spread_bps <= self.max_spread_bps
        ):
            nlp_filter = nlp_intel.get("trade_action_filter", "ALLOW_ALL") if nlp_intel else "ALLOW_ALL"
            directional_bias = nlp_intel.get("directional_bias", 0.0) if nlp_intel else 0.0

            # 開多條件：EMA_OBI 強大、Microprice 支撐、且價格未在下跌中
            if (
                ema_obi > self.obi_threshold
                and microprice > mid_price
                and mid_price >= prev_mid
                and nlp_filter in ["ALLOW_ALL", "ALLOW_BUY_ONLY"]
            ):
                if directional_bias >= -0.2:
                    passed, r_reason, _ = self.risk_manager.evaluate_order(
                        symbol, "BUY_OPEN", features["best_bid"], self.default_order_size
                    )
                    if passed:
                        action = "BUY_OPEN"
                        reason = f"High-conviction bid flow (EMA_OBI: {ema_obi:.3f}). {r_reason}"
                        order_params = {"price": features["best_bid"], "quantity": self.default_order_size}

            # 開空條件：EMA_OBI 壓制、Microprice 偏空、且價格未在上漲中
            elif (
                ema_obi < -self.obi_threshold
                and microprice < mid_price
                and mid_price <= prev_mid
                and nlp_filter in ["ALLOW_ALL", "ALLOW_SELL_ONLY"]
            ):
                if directional_bias <= 0.2:
                    passed, r_reason, _ = self.risk_manager.evaluate_order(
                        symbol, "SELL_OPEN", features["best_ask"], self.default_order_size
                    )
                    if passed:
                        action = "SELL_OPEN"
                        reason = f"High-conviction ask flow (EMA_OBI: {ema_obi:.3f}). {r_reason}"
                        order_params = {"price": features["best_ask"], "quantity": self.default_order_size}
        # >>>>>>>> MODIFICATION END <<<<<<<<

        if action == "CLOSE_POSITION":
            self.cooldown_counter[symbol] = self.cooldown_ticks
            self.position_age[symbol] = 0

        self.last_mid_price[symbol] = mid_price

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