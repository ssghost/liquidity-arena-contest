from typing import Any, Dict, Tuple

class RiskManager:
    def __init__(
        self,
        initial_balance: float = 10000.0,
        max_leverage: float = 2.0,
        nav_drawdown_limit: float = 0.80,
        max_single_order_ratio: float = 0.30,
    ):
        self.initial_balance = initial_balance
        self.current_cash = initial_balance
        self.max_leverage = max_leverage
        self.nav_drawdown_limit = nav_drawdown_limit
        self.max_single_order_ratio = max_single_order_ratio

        self.positions: Dict[str, Dict[str, float]] = {}

    def update_market_price(self, symbol: str, current_price: float) -> None:
        if symbol in self.positions:
            self.positions[symbol]["current_price"] = current_price

    def calculate_nav(self) -> float:
        unrealized_pnl = 0.0
        for pos in self.positions.values():
            qty = pos["quantity"]
            entry = pos["entry_price"]
            curr = pos["current_price"]
            unrealized_pnl += qty * (curr - entry)

        total_equity = self.current_cash + unrealized_pnl
        return total_equity / self.initial_balance if self.initial_balance > 0 else 0.0

    def calculate_current_leverage(self) -> float:
        total_nominal_exposure = sum(
            abs(pos["quantity"] * pos["current_price"]) for pos in self.positions.values()
        )
        current_equity = self.calculate_nav() * self.initial_balance
        if current_equity <= 0:
            return float("inf")
        return total_nominal_exposure / current_equity

    def evaluate_order(
        self,
        symbol: str,
        action: str,
        price: float,
        quantity: float,
    ) -> Tuple[bool, str, Dict[str, Any]]:
        current_nav = self.calculate_nav()

        if current_nav < self.nav_drawdown_limit:
            if action in ["BUY_OPEN", "SELL_OPEN"]:
                return False, f"NAV Breached Limit ({current_nav:.4f} < {self.nav_drawdown_limit}). New entries halted.", {
                    "nav": current_nav,
                    "leverage": self.calculate_current_leverage(),
                    "passed": False,
                }

        if action in ["BUY_CLOSE", "SELL_CLOSE", "CLOSE_POSITION"]:
            return True, "Risk check passed (de-risking).", {
                "nav": current_nav,
                "leverage": self.calculate_current_leverage(),
                "passed": True,
            }

        order_nominal_value = abs(price * quantity)
        max_order_val = (current_nav * self.initial_balance) * self.max_single_order_ratio
        if order_nominal_value > max_order_val:
            return False, f"Order size exceeds max single order threshold ({order_nominal_value:.2f} > {max_order_val:.2f}).", {
                "nav": current_nav,
                "leverage": self.calculate_current_leverage(),
                "passed": False,
            }

        projected_nominal_exposure = sum(
            abs(pos["quantity"] * pos["current_price"]) for pos in self.positions.values()
        ) + order_nominal_value
        current_equity = current_nav * self.initial_balance
        projected_leverage = projected_nominal_exposure / current_equity if current_equity > 0 else float("inf")

        if projected_leverage > self.max_leverage:
            return False, f"Projected leverage exceeds limit ({projected_leverage:.2f}x > {self.max_leverage:.2f}x).", {
                "nav": current_nav,
                "leverage": projected_leverage,
                "passed": False,
            }

        return True, "Risk check passed.", {
            "nav": current_nav,
            "leverage": projected_leverage,
            "passed": True,
        }