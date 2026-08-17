from typing import Any, Dict, List, Optional, Tuple
from interface.logger import ReasoningLogger
from risk.manager import RiskManager


class StrategyAdapter:
    def __init__(
        self,
        risk_manager: RiskManager,
        reasoning_logger: ReasoningLogger,
        obi_threshold: float = 0.25,
        default_order_size: float = 0.01,
    ):
        self.risk_manager = risk_manager
        self.reasoning_logger = reasoning_logger
        self.obi_threshold = obi_threshold
        self.default_order_size = default_order_size

    @staticmethod
    def calculate_microstructure_features(
        bids: List[List[float]], asks: List[List[float]]
    ) -> Dict[str, float]:
        if not bids or not asks:
            return {"mid_price": 0.0, "micro_price": 0.0, "spread": 0.0, "obi": 0.0}

        best_bid_p, best_bid_q = float(bids[0][0]), float(bids[0][1])
        best_ask_p, best_ask_q = float(asks[0][0]), float(asks[0][1])

        mid_price = (best_bid_p + best_ask_p) / 2.0
        spread = best_ask_p - best_bid_p

        bid_vol_top5 = sum(float(b[1]) for b in bids[:5])
        ask_vol_top5 = sum(float(a[1]) for a in asks[:5])
        total_vol = bid_vol_top5 + ask_vol_top5

        obi = (bid_vol_top5 - ask_vol_top5) / total_vol if total_vol > 0 else 0.0
        micro_price = (
            (best_bid_p * best_ask_q + best_ask_p * best_bid_q) / (best_bid_q + best_ask_q)
            if (best_bid_q + best_ask_q) > 0
            else mid_price
        )

        return {
            "mid_price": round(mid_price, 4),
            "micro_price": round(micro_price, 4),
            "spread": round(spread, 4),
            "obi": round(obi, 4),
        }

    def generate_decision(
        self,
        symbol: str,
        bids: List[List[float]],
        asks: List[List[float]],
        nlp_intel: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        features = self.calculate_microstructure_features(bids, asks)
        mid_price = features["mid_price"]
        obi = features["obi"]

        if mid_price <= 0:
            return {"action": "HOLD", "reason": "Invalid market depth"}

        self.risk_manager.update_market_price(symbol, mid_price)

        if obi > self.obi_threshold:
            quant_signal = "BULLISH"
        elif obi < -self.obi_threshold:
            quant_signal = "BEARISH"
        else:
            quant_signal = "NEUTRAL"

        features["signal_state"] = quant_signal

        action_filter = nlp_intel.get("trade_action_filter", "ALLOW_ALL") if nlp_intel else "ALLOW_ALL"
        directional_bias = nlp_intel.get("directional_bias", 0.0) if nlp_intel else 0.0

        if action_filter in ["PAUSE_TRADING", "CLOSE_POSITION"]:
            proposed_action = "CLOSE_POSITION" if action_filter == "CLOSE_POSITION" else "HOLD"
            deduction = f"NLP filter enforced: {action_filter}."
        elif quant_signal == "BULLISH" and directional_bias >= -0.2:
            proposed_action = "BUY_OPEN"
            deduction = f"Positive OBI ({obi:.4f}) aligned with non-bearish NLP bias ({directional_bias:.2f})."
        elif quant_signal == "BEARISH" and directional_bias <= 0.2:
            proposed_action = "SELL_OPEN"
            deduction = f"Negative OBI ({obi:.4f}) aligned with non-bullish NLP bias ({directional_bias:.2f})."
        else:
            proposed_action = "HOLD"
            deduction = f"Neutral state or signal conflict (OBI: {obi:.4f}, NLP Bias: {directional_bias:.2f})."

        order_params = {"price": mid_price, "quantity": self.default_order_size} if proposed_action != "HOLD" else {}
        passed, risk_reason, risk_meta = self.risk_manager.evaluate_order(
            symbol=symbol,
            action=proposed_action,
            price=mid_price,
            quantity=self.default_order_size,
        )

        final_action = proposed_action if passed else "HOLD"
        if not passed:
            deduction += f" Blocked by Risk Manager: {risk_reason}"

        log_record = self.reasoning_logger.log_decision(
            symbol=symbol,
            market_data=features,
            nlp_intelligence=nlp_intel,
            logical_deduction=deduction,
            action=final_action,
            action_params=order_params if final_action != "HOLD" else {},
            risk_evaluation=risk_meta,
        )

        return log_record