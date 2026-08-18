import asyncio
import json
import logging
import os
import ssl
import time
import websockets

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("DataStreamer")

URL = "wss://mds-uat.liquiditytech.com/marketdata/v2/public?binary=false"
SYMBOL = "BINANCE_PERP_BTC_USDT"


async def stream_data(hours: float) -> None:
    end_time = time.time() + hours * 3600
    os.makedirs("data", exist_ok=True)
    logger.info(f"Starting data stream for {hours} hours into data/ ...")

    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE

    ob_count = 0
    tick_count = 0

    with open("data/orderbook.jsonl", "a", encoding="utf-8") as f_ob, open(
        "data/tick.jsonl", "a", encoding="utf-8"
    ) as f_tick:
        while time.time() < end_time:
            try:
                logger.info(f"Connecting to WebSocket: {URL}")
                async with websockets.connect(
                    URL, ssl=ssl_context, ping_interval=None
                ) as ws:
                    logger.info("Sending subscription...")
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
                            elif (
                                isinstance(arg_data, list)
                                and len(arg_data) > 0
                                and isinstance(arg_data[0], dict)
                            ):
                                channel = str(arg_data[0].get("channel", ""))

                        channel_upper = channel.upper()

                        if "ORDER_BOOK" in channel_upper:
                            f_ob.write(msg + "\n")
                            f_ob.flush()
                            ob_count += 1
                        elif "TRADE" in channel_upper or "TICK" in channel_upper:
                            f_tick.write(msg + "\n")
                            f_tick.flush()
                            tick_count += 1

                        if (ob_count + tick_count) % 500 == 0 and (ob_count + tick_count) > 0:
                            logger.info(f"Streamed {ob_count} OB snapshots, {tick_count} Trade ticks.")

            except Exception as e:
                logger.error(f"WebSocket Stream Exception: {type(e).__name__}: {e}")
                if time.time() < end_time:
                    logger.info("Reconnecting in 2 seconds...")
                    await asyncio.sleep(2)

    logger.info(f"Streaming finished. Total OB: {ob_count}, Total Ticks: {tick_count}")

if __name__ == "__main__":
    asyncio.run(stream_data(hours=48.0))