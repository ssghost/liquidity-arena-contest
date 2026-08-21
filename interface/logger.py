import json
import os
import time
from typing import Any, Dict, Optional


class ReasoningLogger:
    def __init__(self, log_file: str = "logs/reasoning_log.jsonl"):
        self.log_file = log_file
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)

    def log_decision(
        self,
        symbol: str,
        market_data: Dict[str, Any],
        nlp_intelligence: Optional[Dict[str, Any]],
        logical_deduction: str,
        action: str,
        action_params: Dict[str, Any],
        risk_evaluation: Dict[str, Any],
    ) -> Dict[str, Any]:
        timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        decision_id = f"dec_{int(time.time() * 1000)}"

        obi = market_data.get("obi")
        if obi is None:
            obi = market_data.get("orderbook_imbalance_ratio", 0.0)

        quant_signal = market_data.get("signal_state")
        if not quant_signal:
            quant_signal = "BULLISH" if obi > 0.15 else ("BEARISH" if obi < -0.15 else "NEUTRAL")

        current_nav = risk_evaluation.get("nav", 1.0)
        current_leverage = risk_evaluation.get("leverage", 0.06)
        risk_passed = risk_evaluation.get("passed", (current_nav >= 0.80 and current_leverage <= 2.0))

        confidence = (
            nlp_intelligence.get("confidence")
            if (nlp_intelligence and nlp_intelligence.get("confidence") is not None)
            else (nlp_intelligence.get("confidence_score", 0.5) if nlp_intelligence else round(min(0.95, 0.5 + abs(obi)), 2))
        )

        entry = {
            "decision_id": decision_id,
            "timestamp": timestamp,
            "symbol": symbol,
            
            "inputs": {
                "market_microstructure": {
                    "mid_price": market_data.get("mid_price"),
                    "orderbook_imbalance_ratio": round(obi, 4) if isinstance(obi, float) else obi,
                    "spread_bps": round(market_data.get("spread_bps", 0.0), 2),
                    "trend_momentum_bps": round(market_data.get("trend_momentum_bps", 0.0), 2),
                    "funding_rate": market_data.get("funding_rate"),
                    "recent_volatility": market_data.get("volatility"),
                },
                "intelligence_feed": nlp_intelligence or {},
            },
            
            "analysis": {
                "impact_level": (
                    nlp_intelligence.get("impact_level", "NONE")
                    if nlp_intelligence
                    else "NONE"
                ),
                "directional_bias": (
                    nlp_intelligence.get("directional_bias", 0.0)
                    if nlp_intelligence
                    else 0.0
                ),
                "action_filter": (
                    nlp_intelligence.get("trade_action_filter", "ALLOW_ALL")
                    if nlp_intelligence
                    else "ALLOW_ALL"
                ),
                "quantitative_signal": quant_signal,
            },
            
            "reasoning": {
                "deduction_chain": logical_deduction,
                "confidence_score": confidence,
            },
        
            "decision": {
                "action": action,
                "order_params": action_params,
                "risk_verification": {
                    "estimated_leverage": round(current_leverage, 2),
                    "current_nav": round(current_nav, 4),
                    "risk_check_passed": risk_passed,
                },
            },
        }

        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        return entry