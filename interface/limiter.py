import hashlib
import json
import os
import time
from typing import Any, Dict, Optional

class TokenLimiter:
    def __init__(
        self,
        daily_budget_usd: float = 10.0,
        cost_per_1k_input_tokens: float = 0.0015,
        cost_per_1k_output_tokens: float = 0.0020,
        state_file: str = "logs/token_usage.json",
        cache_file: str = "logs/nlp_cache.json",
    ):
        self.daily_budget_usd = daily_budget_usd
        self.cost_per_1k_in = cost_per_1k_input_tokens
        self.cost_per_1k_out = cost_per_1k_output_tokens
        self.state_file = state_file
        self.cache_file = cache_file

        os.makedirs(os.path.dirname(self.state_file), exist_ok=True)
        self.usage_state = self._load_state()
        self.cache = self._load_cache()

    def _get_current_day(self) -> str:
        return time.strftime("%Y-%m-%d", time.gmtime())

    def _load_state(self) -> Dict[str, Any]:
        current_day = self._get_current_day()
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if data.get("date") == current_day:
                        return data
            except Exception:
                pass
        return {
            "date": current_day,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_cost_usd": 0.0,
        }

    def _save_state(self) -> None:
        with open(self.state_file, "w", encoding="utf-8") as f:
            json.dump(self.usage_state, f, indent=2)

    def _load_cache(self) -> Dict[str, Any]:
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {}

    def _save_cache(self) -> None:
        with open(self.cache_file, "w", encoding="utf-8") as f:
            json.dump(self.cache, f, ensure_ascii=False, indent=2)

    def _hash_content(self, text: str) -> str:
        return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()

    def get_cached_result(self, raw_text: str) -> Optional[Dict[str, Any]]:
        key = self._hash_content(raw_text)
        return self.cache.get(key)

    def store_cache(self, raw_text: str, parsed_result: Dict[str, Any]) -> None:
        key = self._hash_content(raw_text)
        self.cache[key] = parsed_result
        self._save_cache()

    def can_make_request(self, estimated_tokens: int = 1000) -> bool:
        current_day = self._get_current_day()
        if self.usage_state.get("date") != current_day:
            self.usage_state = {
                "date": current_day,
                "input_tokens": 0,
                "output_tokens": 0,
                "total_cost_usd": 0.0,
            }
            self._save_state()

        estimated_cost = (estimated_tokens / 1000.0) * max(
            self.cost_per_1k_in, self.cost_per_1k_out
        )
        return (
            self.usage_state["total_cost_usd"] + estimated_cost
        ) <= self.daily_budget_usd * 0.95

    def record_usage(self, prompt_tokens: int, completion_tokens: int) -> None:
        current_day = self._get_current_day()
        if self.usage_state.get("date") != current_day:
            self.usage_state = {
                "date": current_day,
                "input_tokens": 0,
                "output_tokens": 0,
                "total_cost_usd": 0.0,
            }

        cost = (prompt_tokens / 1000.0) * self.cost_per_1k_in + (
            completion_tokens / 1000.0
        ) * self.cost_per_1k_out

        self.usage_state["input_tokens"] += prompt_tokens
        self.usage_state["output_tokens"] += completion_tokens
        self.usage_state["total_cost_usd"] += cost
        self._save_state()