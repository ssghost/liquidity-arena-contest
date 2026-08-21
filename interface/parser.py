import json
import re
from typing import Any, Dict, List, Optional


class NLPParser:
    SYSTEM_PROMPT = """You are an institutional quantitative trading intelligence engine. Analyze the provided news, policy update, or social post to assess market impact and risk filters for crypto perpetuals.

Output strictly valid JSON matching the following schema:
{
  "timestamp": "<ISO 8601 or original timestamp>",
  "target_asset": "<BTC / ETH / ALL / OTHER>",
  "event_category": "<Macro_Policy / Regulatory / Security_Exploit / Market_Structure / Technical_Anomaly / General_News>",
  "impact_level": "<HIGH / MEDIUM / LOW / NONE>",
  "volatility_bias": "<EXPANSION / CONTRACTION / NEUTRAL>",
  "directional_bias": <Float between -1.0 and 1.0; -1.0 extreme bearish, 1.0 extreme bullish, 0.0 neutral>,
  "trade_action_filter": "<ALLOW_ALL / ALLOW_BUY_ONLY / ALLOW_SELL_ONLY / REDUCE_SIZE / PAUSE_TRADING / CLOSE_POSITION>",
  "event_summary": "<One-sentence factual summary>",
  "confidence": <Float between 0.0 and 1.0>
}
Output only the raw JSON string without any explanation or Markdown formatting."""

    BULLISH_KEYWORDS = [
        "etf inflow", "inflows", "rate cut", "dovish", "adoption",
        "surge", "rally", "breakout", "approval", "accumulate", "bullish"
    ]
    BEARISH_KEYWORDS = [
        "sec lawsuit", "outflows", "rate hike", "hawkish", "hack",
        "exploit", "liquidation", "ban", "crackdown", "sell-off", "bearish"
    ]

    @staticmethod
    def build_messages(
        raw_text: str, source_type: str = "news"
    ) -> List[Dict[str, str]]:
        user_content = f"Source Type: {source_type}\nRaw Content:\n{raw_text}"
        return [
            {"role": "system", "content": NLPParser.SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

    @staticmethod
    def parse_response(llm_output: str) -> Optional[Dict[str, Any]]:
        if not llm_output or not isinstance(llm_output, str):
            return None

        try:
            cleaned_text = re.sub(
                r"^```json\s*|\s*```$", "", llm_output.strip(), flags=re.MULTILINE
            ).strip()
            data = json.loads(cleaned_text)

            if not isinstance(data, dict):
                return None

            required_keys = [
                "target_asset",
                "event_category",
                "impact_level",
                "volatility_bias",
                "directional_bias",
                "trade_action_filter",
                "event_summary",
                "confidence",
            ]
            for key in required_keys:
                if key not in data:
                    return None

            data["directional_bias"] = max(
                -1.0, min(1.0, float(data["directional_bias"]))
            )
            data["confidence"] = max(0.0, min(1.0, float(data["confidence"])))

            valid_filters = [
                "ALLOW_ALL",
                "ALLOW_BUY_ONLY",
                "ALLOW_SELL_ONLY",
                "REDUCE_SIZE",
                "PAUSE_TRADING",
                "CLOSE_POSITION",
            ]
            if data["trade_action_filter"] not in valid_filters:
                data["trade_action_filter"] = "ALLOW_ALL"

            return data
        except (json.JSONDecodeError, ValueError, TypeError):
            return None

    @classmethod
    def heuristic_parse(cls, news_record: Dict[str, Any]) -> Dict[str, Any]:
        headline = str(news_record.get("headline", "")).lower()
        content = str(news_record.get("content", "")).lower()
        text = f"{headline} {content}"

        bull_hits = sum(1 for kw in cls.BULLISH_KEYWORDS if kw in text)
        bear_hits = sum(1 for kw in cls.BEARISH_KEYWORDS if kw in text)

        if bull_hits > bear_hits:
            directional_bias = min(1.0, 0.3 + 0.15 * (bull_hits - bear_hits))
            impact_level = "HIGH" if (bull_hits - bear_hits) >= 2 else "MEDIUM"
            trade_action_filter = "ALLOW_BUY_ONLY"
            confidence = min(0.95, 0.65 + 0.1 * bull_hits)
            summary = f"Bullish catalyst detected from {news_record.get('source', 'NEWS')}."
        elif bear_hits > bull_hits:
            directional_bias = max(-1.0, -0.3 - 0.15 * (bear_hits - bull_hits))
            impact_level = "HIGH" if (bear_hits - bull_hits) >= 2 else "MEDIUM"
            trade_action_filter = "CLOSE_POSITION" if bear_hits >= 2 else "ALLOW_SELL_ONLY"
            confidence = min(0.95, 0.65 + 0.1 * bear_hits)
            summary = f"Bearish risk detected from {news_record.get('source', 'NEWS')}."
        else:
            directional_bias = 0.0
            impact_level = "LOW"
            trade_action_filter = "ALLOW_ALL"
            confidence = 0.50
            summary = f"Neutral market update from {news_record.get('source', 'NEWS')}."

        return {
            "timestamp": news_record.get("timestamp"),
            "target_asset": "BTC",
            "event_category": "Macro_Policy" if "fed" in text or "sec" in text else "General_News",
            "impact_level": impact_level,
            "volatility_bias": "EXPANSION" if (bull_hits + bear_hits) > 0 else "NEUTRAL",
            "directional_bias": round(directional_bias, 2),
            "trade_action_filter": trade_action_filter,
            "event_summary": summary,
            "confidence": round(confidence, 2),
            "source": news_record.get("source", "FEED"),
            "headline": news_record.get("headline", ""),
        }