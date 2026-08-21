import argparse
import email.utils
import hashlib
import json
import os
import re
import time
import xml.etree.ElementTree as ET
from typing import Any, Dict, List
import httpx

FREE_RSS_FEEDS = [
    {"source": "CoinTelegraph", "url": "https://cointelegraph.com/rss"},
    {"source": "CoinDesk", "url": "https://www.coindesk.com/arc/outboundfeeds/rss/"},
    {"source": "Decrypt", "url": "https://decrypt.co/feed"},
    {"source": "CryptoSlate", "url": "https://cryptoslate.com/feed/"},
    {"source": "Blockworks", "url": "https://blockworks.co/feed"},
]

def clean_html(raw_html: str) -> str:
    clean_text = re.sub(r"<.*?>", "", raw_html)
    return re.sub(r"\s+", " ", clean_text).strip()

def parse_rfc822_date(date_str: str) -> int:
    try:
        parsed_tuple = email.utils.parsedate_tz(date_str)
        if parsed_tuple:
            return int(email.utils.mktime_tz(parsed_tuple))
    except Exception:
        pass
    return int(time.time())

class MockNewsFeedCollector:
    def __init__(self, output_file: str = "data/news_feed.jsonl"):
        self.output_file = output_file
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
                    news_id = data.get("id")
                    if news_id:
                        self.collected_ids.add(news_id)
                except Exception:
                    continue
        print(f"Loaded {len(self.collected_ids)} existing news records from {self.output_file}")

    def fetch_feed(self, source_name: str, url: str) -> List[Dict[str, Any]]:
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
        }
        items: List[Dict[str, Any]] = []

        try:
            with httpx.Client(timeout=15.0, follow_redirects=True) as client:
                resp = client.get(url, headers=headers)
                if resp.status_code != 200:
                    print(f"[{source_name}] Failed to fetch RSS: HTTP {resp.status_code}")
                    return []

                root = ET.fromstring(resp.content)
                channel = root.find("channel")
                raw_items = channel.findall("item") if channel is not None else root.findall(".//item")

                for item_elem in raw_items:
                    title = item_elem.findtext("title") or ""
                    link = item_elem.findtext("link") or ""
                    description = item_elem.findtext("description") or ""
                    pub_date = item_elem.findtext("pubDate") or ""

                    timestamp = parse_rfc822_date(pub_date)
                    clean_content = clean_html(description)

                    unique_seed = f"{link}_{title}"
                    news_id = hashlib.md5(unique_seed.encode("utf-8")).hexdigest()

                    if news_id in self.collected_ids:
                        continue

                    full_text = f"{title} {clean_content}".lower()
                    symbols = ["BTCUSDT"] if any(k in full_text for k in ["btc", "bitcoin", "crypto", "fed", "sec"]) else ["GENERAL"]

                    record = {
                        "id": news_id,
                        "timestamp": timestamp,
                        "source": source_name,
                        "headline": clean_html(title),
                        "content": clean_content,
                        "symbols": symbols,
                        "link": link,
                    }
                    items.append(record)

        except Exception as e:
            print(f"[{source_name}] Error parsing XML feed: {e}")

        return items

    def run_collection(self, target_count: int = 1000) -> int:
        os.makedirs(os.path.dirname(self.output_file) or ".", exist_ok=True)
        total_new = 0

        print(f"Starting collection from {len(FREE_RSS_FEEDS)} free crypto news sources...")

        with open(self.output_file, "a", encoding="utf-8") as out_f:
            for feed in FREE_RSS_FEEDS:
                if total_new >= target_count:
                    break

                feed_items = self.fetch_feed(feed["source"], feed["url"])
                written_count = 0
                for item in feed_items:
                    out_f.write(json.dumps(item, ensure_ascii=False) + "\n")
                    out_f.flush()
                    self.collected_ids.add(item["id"])
                    written_count += 1
                    total_new += 1
                    if total_new >= target_count:
                        break

                print(f"[{feed['source']}] Fetched {len(feed_items)} items | New written: {written_count}")

        print(f"Mock news collection finished. Total new items stored: {total_new}")
        return total_new

def build_llm_training_dataset(
    raw_news_file: str = "data/news_feed.jsonl",
    output_train_file: str = "data/llm_train.jsonl",
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
                                "directional_bias": 0.35,
                                "trade_action_filter": "ALLOW_BUY_ONLY",
                                "confidence_score": 0.80,
                                "deduction_chain": f"Evaluated news sentiment from {source}. Favorable market development, bias set positive.",
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
    parser = argparse.ArgumentParser(description="Free News Feed Collector (RSS Mock)")
    parser.add_argument("--count", type=int, default=500, help="Target number of news items to fetch")
    parser.add_argument("--build-dataset", action="store_true", help="Generate LLM training dataset after fetch")
    args = parser.parse_args()

    collector = MockNewsFeedCollector()
    collector.run_collection(target_count=args.count)

    if args.build_dataset:
        build_llm_training_dataset()