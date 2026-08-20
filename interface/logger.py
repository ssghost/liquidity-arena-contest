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

        entry = {
            "decision_id": decision_id,
            "timestamp": timestamp,
            "symbol": symbol,
            # Layer 1: Information Inputs (Microstructure + Intelligence)
            "inputs": {
                "market_microstructure": {
                    "mid_price": market_data.get("mid_price"),
                    "orderbook_imbalance_ratio": market_data.get("obi"),
                    "funding_rate": market_data.get("funding_rate"),
                    "recent_volatility": market_data.get("volatility"),
                },
                "intelligence_feed": nlp_intelligence or {},
            },
            # Layer 2: Sentiment & Anomaly Analysis
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
                "quantitative_signal": market_data.get("signal_state", "NEUTRAL"),
            },
            # Layer 3: Logical Deduction & Cross-Validation
            "reasoning": {
                "deduction_chain": logical_deduction,
                "confidence_score": (
                    nlp_intelligence.get("confidence", 0.5)
                    if nlp_intelligence
                    else 0.5
                ),
            },
            # Layer 4: Hard Risk Check & Trading Decision
            "decision": {
                "action": action,  
                "order_params": action_params,
                "risk_verification": {
                    "estimated_leverage": risk_evaluation.get(
                        "leverage", 1.0
                    ),  
                    "current_nav": risk_evaluation.get(
                        "nav", 1.0
                    ),  
                    "risk_check_passed": risk_evaluation.get(
                        "passed", True
                    ),
                },
            },
        }

        # Append to JSONL file
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        return entry