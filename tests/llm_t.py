import json
import os
import time
import urllib.request
from dotenv import load_dotenv

from interface.limiter import TokenLimiter
from interface.parser import NLPParser

load_dotenv()

def query_llm_endpoint(messages: list) -> str:
    api_base = os.getenv("LLM_API_BASE").rstrip("/")
    api_key = os.getenv("LLM_API_KEY")
    model = os.getenv("LLM_MODEL")

    url = f"{api_base}/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.0,
    }
    data_bytes = json.dumps(payload).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    req = urllib.request.Request(url, data=data_bytes, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=30) as response:
        res_json = json.loads(response.read().decode("utf-8"))
        return res_json["choices"][0]["message"]["content"]


def process_intelligence(
    news_text: str, limiter: TokenLimiter
) -> dict:
    cached = limiter.get_cached_result(news_text)
    if cached:
        return cached

    if not limiter.can_make_request(estimated_tokens=500):
        return {"error": "Daily token budget exceeded"}

    messages = NLPParser.build_messages(news_text, source_type="news")
    raw_response = query_llm_endpoint(messages)

    intel = NLPParser.parse_response(raw_response)
    if not intel:
        intel = {"raw_output": raw_response, "parse_error": True}

    limiter.record_usage(prompt_tokens=250, completion_tokens=80)
    limiter.store_cache(news_text, intel)
    return intel


def run_nlp_test() -> None:
    limiter = TokenLimiter(
        daily_budget_usd=10.0,
        state_file="logs/test_token_usage.json",
        cache_file="logs/test_nlp_cache.json",
    )

    test_headline = (
        "U.S. SEC approves multiple Bitcoin and Ethereum staking ETFs, "
        "triggering immediate institutional inflows across major exchanges."
    )

    print(f"\n[Test News Headline]:\n{test_headline}\n")
    print("Sending prompt to LLM endpoint...")

    start_t = time.time()
    intel_1 = process_intelligence(test_headline, limiter)
    elapsed_1 = time.time() - start_t

    print(f"\nFirst Call (Model Inference, elapsed: {elapsed_1:.2f}s)")
    print(f"Target Asset:        {intel_1.get('target_asset')}")
    print(f"Event Category:      {intel_1.get('event_category')}")
    print(f"Impact Level:        {intel_1.get('impact_level')}")
    print(f"Volatility Bias:     {intel_1.get('volatility_bias')}")
    print(f"Directional Bias:    {intel_1.get('directional_bias')}")
    print(f"Trade Action Filter: {intel_1.get('trade_action_filter')}")
    print(f"Confidence:          {intel_1.get('confidence')}")

    required_fields = [
        "target_asset",
        "event_category",
        "impact_level",
        "volatility_bias",
        "directional_bias",
        "trade_action_filter",
        "confidence",
    ]
    for field in required_fields:
        assert field in intel_1, f"Missing required field: {field}"

    start_t = time.time()
    intel_2 = process_intelligence(test_headline, limiter)
    elapsed_2 = time.time() - start_t

    print(f"\nSecond Call (Cache Hit, elapsed: {elapsed_2:.4f}s)")
    assert intel_1 == intel_2, "Cached output must match original output."
    assert elapsed_2 < 0.05, f"Expected instant cache hit (<0.05s), got {elapsed_2:.4f}s"
    print("Cache hit verified (latency < 0.05s, zero token wasted).")

if __name__ == "__main__":
    run_nlp_test()