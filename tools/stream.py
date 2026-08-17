import asyncio
import json
import os
import time
import websockets

URL = "wss://mds-uat.liquiditytech.com/marketdata/v2/public?binary=false"
SYMBOL = "BINANCE_PERP_BTC_USDT"

async def stream_data(hours: float) -> None:
    end_time = time.time() + hours * 3600
    os.makedirs("data", exist_ok=True)
    print("Streaming data into data folder...")

    with open("data/orderbook.jsonl", "w", encoding="utf-8") as f_ob, open(
        "data/tick.jsonl", "w", encoding="utf-8"
    ) as f_tick:
        while time.time() < end_time:
            try:
                async with websockets.connect(URL, ping_interval=None) as ws:
                    sub_payload = {
                        "event": "subscribe",
                        "arg": [
                            {"channel": "ORDER_BOOK", "sym": SYMBOL},
                            {"channel": "TRADE", "sym": SYMBOL},
                        ],
                    }
                    await ws.send(json.dumps(sub_payload))
                    last_ping_time = time.time()

                    async for msg in ws:
                        now = time.time()
                        if now >= end_time:
                            break
                        if now - last_ping_time >= 20:
                            ping_payload = {"ping": int(now * 1000)}
                            await ws.send(json.dumps(ping_payload))
                            last_ping_time = now

                        data = json.loads(msg)
                        if not isinstance(data, dict):
                            continue

                        channel = str(
                            data.get("channel", "")
                            or data.get("topic", "")
                            or data.get("stream", "")
                        )

                        if not channel:
                            arg_data = data.get("arg")
                            if isinstance(arg_data, dict):
                                channel = str(arg_data.get("channel", ""))
                            elif isinstance(arg_data, list) and len(arg_data) > 0 and isinstance(arg_data[0], dict):
                                channel = str(arg_data[0].get("channel", ""))

                        channel_upper = channel.upper()

                        if "ORDER_BOOK" in channel_upper:
                            f_ob.write(msg + "\n")
                            f_ob.flush()
                        elif "TRADE" in channel_upper or "TICK" in channel_upper:
                            f_tick.write(msg + "\n")
                            f_tick.flush()

            except Exception:
                if time.time() < end_time:
                    await asyncio.sleep(2)

    print("Streaming is finished...")

if __name__ == "__main__":
    asyncio.run(stream_data(hours=48.0))
