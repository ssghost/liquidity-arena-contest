import asyncio
import logging
from typing import Any, Dict, Optional
import aiohttp

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("OrderExecutor")


class OrderExecutor:
    def __init__(
        self,
        api_base_url: str = "https://api.liquiditytech.example.com",
        api_key: Optional[str] = None,
        mock_mode: bool = True,
    ):
        self.api_base_url = api_base_url
        self.api_key = api_key
        self.mock_mode = mock_mode
        self.active_orders: Dict[str, Dict[str, Any]] = {}

    async def place_order(
        self,
        symbol: str,
        side: str,
        price: float,
        quantity: float,
        order_type: str = "LIMIT",
    ) -> Dict[str, Any]:
        order_id = f"ord_{int(asyncio.get_event_loop().time() * 1000)}_{side.lower()}"

        order_payload = {
            "order_id": order_id,
            "symbol": symbol,
            "side": side.upper(), 
            "price": price,
            "quantity": quantity,
            "type": order_type,
            "status": "PENDING",
        }

        if self.mock_mode:
            order_payload["status"] = "FILLED"
            order_payload["filled_price"] = price
            order_payload["filled_quantity"] = quantity
            self.active_orders[order_id] = order_payload
            logger.info(
                f"[MOCK EXECUTION] Order {order_id} {side} {quantity} {symbol} @ {price} FILLED"
            )
            return order_payload

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}" if self.api_key else "",
        }
        endpoint = f"{self.api_base_url}/v1/order"

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    endpoint, json=order_payload, headers=headers, timeout=5.0
                ) as resp:
                    if resp.status == 200:
                        res_json = await resp.json()
                        order_payload["status"] = res_json.get(
                            "status", "SUBMITTED"
                        )
                        self.active_orders[order_id] = order_payload
                        return res_json
                    else:
                        err_text = await resp.text()
                        logger.error(
                            f"Order placement failed ({resp.status}): {err_text}"
                        )
                        order_payload["status"] = "REJECTED"
                        return {"error": err_text, "status_code": resp.status}
        except Exception as e:
            logger.error(f"Network exception placing order {order_id}: {e}")
            order_payload["status"] = "FAILED"
            return {"error": str(e), "status": "FAILED"}

    async def cancel_order(self, order_id: str) -> bool:
        if self.mock_mode:
            if order_id in self.active_orders:
                self.active_orders[order_id]["status"] = "CANCELLED"
                logger.info(f"[MOCK EXECUTION] Order {order_id} CANCELLED")
                return True
            return False

        headers = {
            "Authorization": f"Bearer {self.api_key}" if self.api_key else ""
        }
        endpoint = f"{self.api_base_url}/v1/order/{order_id}"

        try:
            async with aiohttp.ClientSession() as session:
                async with session.delete(
                    endpoint, headers=headers, timeout=5.0
                ) as resp:
                    return resp.status == 200
        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            return False