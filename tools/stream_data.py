import asyncio, json, time, websockets

URL = "wss://mds-uat.liquiditytech.com/marketdata/v2/public"


async def stream_data(hours: float) -> None:
    end_time = time.time() + hours * 3600
    async with websockets.connect(URL) as ws:
        sub_payload = {"op": "subscribe", "args": ["orderbook", "tick"]}
        await ws.send(json.dumps(sub_payload))

        with open("data/orderbook.jsonl", "a", encoding="utf-8") as f_ob, open(
            "data/tick.jsonl", "a", encoding="utf-8"
        ) as f_tick:
            while time.time() < end_time:
                try:
                    remaining = max(0.1, end_time - time.time())
                    msg = await asyncio.wait_for(ws.recv(), timeout=remaining)
                except asyncio.TimeoutError:
                    break

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


if __name__ == "__main__":
    asyncio.run(stream_data(hours=2.0))