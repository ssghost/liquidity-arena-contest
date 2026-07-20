import asyncio, gzip, json, time, websockets

URL = "wss://mds-uat.liquiditytech.com/marketdata/v2/public"

async def stream_data(hours: float) -> None:
    end_time = time.time() + hours * 3600
    print("Streaming data into data folder...")

    with open("data/orderbook.jsonl", "a", encoding="utf-8") as f_ob, open(
        "data/tick.jsonl", "a", encoding="utf-8"
    ) as f_tick:
        while time.time() < end_time:
            try:
                async with websockets.connect(
                    URL, ping_interval=20, ping_timeout=20
                ) as ws:
                    sub_payload = {
                        "op": "subscribe",
                        "args": ["orderbook", "tick"],
                    }
                    await ws.send(json.dumps(sub_payload))

                    async for msg in ws:
                        if time.time() >= end_time:
                            break

                        if isinstance(msg, bytes):
                            try:
                                msg = gzip.decompress(msg).decode("utf-8")
                            except Exception:
                                msg = msg.decode("utf-8", errors="ignore")

                        data = json.loads(msg)
                        channel = str(
                            data.get("channel", "")
                            or data.get("topic", "")
                            or data.get("stream", "")
                        )

                        if "orderbook" in channel:
                            f_ob.write(msg + "\n")
                            f_ob.flush()
                        else:
                            f_tick.write(msg + "\n")
                            f_tick.flush()
            except Exception:
                if time.time() < end_time:
                    await asyncio.sleep(2)

if __name__ == "__main__":
    asyncio.run(stream_data(hours=2.0))
