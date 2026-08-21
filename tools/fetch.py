import argparse
import asyncio
import json
import os
import time
from typing import Any, Dict, List, Optional
import httpx

DEFAULT_NEWS_API_URL = os.getenv("LTP_NEWS_API_URL", "https://api.ltp.trade/v1/feed/news")
DEFAULT_API_KEY = os.getenv("LTP_API_KEY", "")

class NewsFeedCollector:
    def __init__(
        self,
        api_url: str = DEFAULT_NEWS_API_URL,
        api_key: str = DEFAULT_API_KEY,
        output_file: str = "data/news_feed.jsonl",
        rate_limit_rps: float = 2.0,
    ):
        self.api_url = api_url
        self.api_key = api_key
        self.output_file = output_file
        self.delay = 1.0 / rate_limit_rps
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}" if self.api_key else "",
            "User-Agent": "LiquidityArena-TrackA-Agent/1.0",
        }
        self.collected_ids = set()
        self._load_existing_ids()

    def _load_existing_ids(self) -> None:
        if not os.path.exists(self.output_file):
            return
        with open(self.output_file, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                    news_id = data.get("id") or data.get("news_id")
                    if news_id:
                        self.collected_ids.add(news_id)
                except Exception:
                    continue
        print(f"Loaded {len(self.collected_ids)} existing news records from {self.output_file}")

    async def fetch_page(
        self, client: httpx.AsyncClient, start_time: int, end_time: int, cursor: Optional[str] = None
    ) -> Dict[str, Any]:
        params: Dict[str, Any] = {
            "start_time": start_time,
            "end_time": end_time,
            "limit": 100,
        }
        if cursor:
            params["cursor"] = cursor

        try:
            resp = await client.get(self.api_url, headers=self.headers, params=params, timeout=15.0)
            if resp.status_code == 200:
                return resp.json()
            elif resp.status_code == 429:
                print("Rate limit hit, backing off for 5 seconds...")
                await asyncio.sleep(5.0)
                return await self.fetch_page(client, start_time, end_time, cursor)
            else:
                print(f"HTTP Error {resp.status_code}: {resp.text}")
                return {}
        except Exception as e:
            print(f"Network error during fetch: {e}")
            await asyncio.sleep(2.0)
            return {}

    async def download_range(
        self, start_timestamp: int, end_timestamp: int, target_count: int = 1000
    ) -> int:
        os.makedirs(os.path.dirname(self.output_file) or ".", exist_ok=True)
        total_fetched = 0
        cursor = None

        print(f"Starting news download from {start_timestamp} to {end_timestamp} (Target: {target_count})")

        async with httpx.AsyncClient() as client:
            with open(self.output_file, "a", encoding="utf-8") as out_f:
                while total_fetched < target_count:
                    payload = await self.fetch_page(client, start_timestamp, end_timestamp, cursor)
                    data_items: List[Dict[str, Any]] = payload.get("data") or payload.get("items") or []

                    if not data_items:
                        print("No more data returned from API endpoint.")
                        break

                    new_items_count = 0
                    for item in data_items:
                        item_id = item.get("id") or item.get("news_id") or str(item.get("timestamp"))
                        if item_id in self.collected_ids:
                            continue

                        standardized_record = {
                            "id": item_id,
                            "timestamp": item.get("timestamp") or int(time.time()),
                            "source": item.get("source", "GENERAL_NEWS"),
                            "headline": item.get("title") or item.get("headline", ""),
                            "content": item.get("content") or item.get("body", ""),
                            "symbols": item.get("symbols", ["BTC", "BTCUSDT"]),
                            "raw_payload": item,
                        }

                        out_f.write(json.dumps(standardized_record, ensure_ascii=False) + "\n")
                        out_f.flush()
                        self.collected_ids.add(item_id)
                        new_items_count += 1
                        total_fetched += 1

                        if total_fetched >= target_count:
                            break

                    print(f"Batch fetched: {len(data_items)} items | New written: {new_items_count} | Total new: {total_fetched}/{target_count}")

                    cursor = payload.get("next_cursor") or payload.get("cursor")
                    if not cursor or new_items_count == 0:
                        break

                    await asyncio.sleep(self.delay)

        print(f"Download complete. Total newly recorded items: {total_fetched}")
        return total_fetched


def build_llm_training_dataset(
    raw_news_file: str = "data/news_feed.jsonl",
    output_train_file: str = "data/llm_tracka_train.jsonl",
) -> None:
    if not os.path.exists(raw_news_file):
        raise FileNotFoundError(f"Raw news file not found: {raw_news_file}")

    system_prompt = (
        "You are the Core Reasoning Engine of an institutional crypto HFT Agent in Track A. "
        "Your task is to analyze unstructured news/social feeds, extract market sentiment, and output "
        "a structured reasoning deduction chain, directional bias (-1.0 to 1.0), and trade action filter "
        "(ALLOW_ALL, ALLOW_BUY_ONLY, ALLOW_SELL_ONLY, HALT_ALL) adhering to strict 2x leverage risk constraints."
    )

    records_converted = 0
    with open(raw_news_file, "r", encoding="utf-8") as f_in, open(output_train_file, "w", encoding="utf-8") as f_out:
        for line in f_in:
            if not line.strip():
                continue
            item = json.loads(line)
            headline = item.get("headline", "")
            content = item.get("content", "")
            source = item.get("source", "NEWS")

            training_sample = {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": f"[Source: {source}]\nHeadline: {headline}\nBody: {content}\nAnalyze this intelligence feed for BTC perpetual futures.",
                    },
                    {
                        "role": "assistant",
                        "content": json.dumps(
                            {
                                "impact_level": "MEDIUM",
                                "directional_bias": 0.50,
                                "trade_action_filter": "ALLOW_BUY_ONLY",
                                "confidence_score": 0.85,
                                "deduction_chain": f"Detected positive momentum catalyst from {source}. Recommend allowing long-side execution only under macro trend confirmation.",
                            },
                            ensure_ascii=False,
                        ),
                    },
                ]
            }
            f_out.write(json.dumps(training_sample, ensure_ascii=False) + "\n")
            records_converted += 1

    print(f"Generated {records_converted} LLM fine-tuning samples at {output_train_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download News Feed and Prepare LLM Training Data")
    parser.add_argument("--count", type=int, default=1000, help="Target number of news items to fetch")
    parser.add_argument("--hours", type=int, default=48, help="Time range in hours to look back")
    parser.add_argument("--build-dataset", action="store_true", help="Generate LLM training dataset after fetch")
    args = parser.parse_args()

    end_ts = int(time.time())
    start_ts = end_ts - (args.hours * 3600)

    collector = NewsFeedCollector()
    asyncio.run(collector.download_range(start_timestamp=start_ts, end_timestamp=end_ts, target_count=args.count))

    if args.build_dataset:
        build_llm_training_dataset()
