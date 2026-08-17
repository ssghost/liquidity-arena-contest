import json
import re
from typing import Any, Dict, List, Optional


class NLPParser:
    """Unstructured market intelligence parser for event classification,

    anomaly detection, and quantitative trading filter signals.
    """

    SYSTEM_PROMPT = """You are an institutional quantitative trading intelligence engine. Analyze the provided news, policy update, or social post to assess market impact and risk filters for crypto perpetuals.

Output strictly valid JSON matching the following schema:
{
  "timestamp": "<ISO 8601 or original timestamp>",
  "target_asset": "<BTC / ETH / ALL / OTHER>",
  "event_category": "<Macro_Policy / Regulatory / Security_Exploit / Market_Structure / Technical_Anomaly / General_News>",
  "impact_level": "<HIGH / MEDIUM / LOW / NONE>",
  "volatility_bias": "<EXPANSION / CONTRACTION / NEUTRAL>",
  "directional_bias": <Float between -1.0 and 1.0; -1.0 extreme bearish, 1.0 extreme bullish, 0.0 neutral>,
  "trade_action_filter": "<ALLOW_ALL / REDUCE_SIZE / PAUSE_TRADING / CLOSE_POSITION>",
  "event_summary": "<One-sentence factual summary>",
  "confidence": <Float between 0.0 and 1.0>
}
Output only the raw JSON string without any explanation or Markdown formatting."""

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

            # Numeric normalization
            data["directional_bias"] = max(
                -1.0, min(1.0, float(data["directional_bias"]))
            )
            data["confidence"] = max(0.0, min(1.0, float(data["confidence"])))

            # Validate categorical fields
            if data["trade_action_filter"] not in [
                "ALLOW_ALL",
                "REDUCE_SIZE",
                "PAUSE_TRADING",
                "CLOSE_POSITION",
            ]:
                data["trade_action_filter"] = "ALLOW_ALL"

            return data
        except (json.JSONDecodeError, ValueError, TypeError):
            return None