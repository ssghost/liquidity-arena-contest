import json
import os
import sys
from pathlib import Path

def validate_dataset(file_path: str, dataset_type: str = "orderbook") -> None:
    if not os.path.exists(file_path):
        print(f"[Error] File not found: {file_path}")
        return

    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
    print(f"\nValidating {dataset_type} dataset: {file_path} ({file_size_mb:.2f} MB)...")

    total_lines = 0
    valid_records = 0
    corrupted_lines = 0
    symbols = set()
    first_timestamp = None
    last_timestamp = None

    with open(file_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            total_lines += 1
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                if dataset_type == "orderbook":
                    data = record.get("data", {})
                    bids = data.get("bids", [])
                    asks = data.get("asks", [])
                    sym = record.get("symbol") or record.get("sym") or data.get("symbol")
                    ts = record.get("timestamp") or record.get("time") or record.get("ts")

                    if bids and asks:
                        valid_records += 1
                    else:
                        corrupted_lines += 1
                else:  
                    trades = record.get("data", [])
                    sym = record.get("symbol") or record.get("sym")
                    ts = record.get("timestamp") or record.get("time") or record.get("ts")
                    if trades:
                        valid_records += 1
                    else:
                        corrupted_lines += 1

                if sym:
                    symbols.add(sym)
                if ts:
                    if first_timestamp is None:
                        first_timestamp = ts
                    last_timestamp = ts

            except Exception:
                corrupted_lines += 1

            if total_lines % 200000 == 0:
                print(f"Processed {total_lines:,} lines...")

    print(f"\n{dataset_type.upper()} Dataset Summary")
    print(f"Total Lines:       {total_lines:,}")
    print(f"Valid Records:     {valid_records:,}")
    print(f"Corrupted / Empty: {corrupted_lines:,}")
    print(f"Detected Symbols:  {list(symbols)}")
    print(f"Start Timestamp:   {first_timestamp}")
    print(f"End Timestamp:     {last_timestamp}")

if __name__ == "__main__":
    validate_dataset("data/orderbook.jsonl", dataset_type="orderbook")
    validate_dataset("data/tick.jsonl", dataset_type="tick")