import itertools
import json
import math
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

def load_orderbook_data(data_path: str = "data/orderbook.jsonl") -> Tuple[np.ndarray, np.ndarray]:
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Market data file not found: {data_path}")

    prices: List[float] = []
    obis: List[float] = []

    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                evt = json.loads(line)
                if evt.get("event") == "subscribe":
                    continue
                d = evt.get("data", {})
                bids = d.get("Bids") or d.get("bids", [])
                asks = d.get("Asks") or d.get("asks", [])
                if not bids or not asks:
                    continue
                bp, ap = float(bids[0][0]), float(asks[0][0])
                bq = sum(float(b[1]) for b in bids[:5])
                aq = sum(float(a[1]) for a in asks[:5])
                obi = (bq - aq) / (bq + aq) if (bq + aq) > 0 else 0.0

                prices.append((bp + ap) / 2.0)
                obis.append(obi)
            except Exception:
                continue

    return np.array(prices, dtype=np.float64), np.array(obis, dtype=np.float64)


def precompute_indicators(prices: np.ndarray, obis: np.ndarray) -> Dict[str, np.ndarray]:
    n = len(prices)
    ema_15m = np.zeros(n, dtype=np.float64)
    ema_30m = np.zeros(n, dtype=np.float64)
    ema_obi = np.zeros(n, dtype=np.float64)

    ema_15m[0] = prices[0]
    ema_30m[0] = prices[0]
    ema_obi[0] = obis[0]

    for t in range(1, n):
        ema_15m[t] = 0.00033 * prices[t] + (1.0 - 0.00033) * ema_15m[t - 1]
        ema_30m[t] = 0.00017 * prices[t] + (1.0 - 0.00017) * ema_30m[t - 1]
        ema_obi[t] = 0.08 * obis[t] + (1.0 - 0.08) * ema_obi[t - 1]

    return {
        "ema_15m": ema_15m,
        "ema_30m": ema_30m,
        "ema_obi": ema_obi,
    }


def evaluate_configuration(
    prices: np.ndarray,
    ema_obi: np.ndarray,
    trend_ema: np.ndarray,
    obi_th: float,
    tp_bps: float,
    sl_bps: float,
    max_hold: int,
    cooldown: int,
    initial_balance: float = 10000.0,
    order_size: float = 0.01,
    fee_rate: float = 0.0005,
    slippage_bps: float = 0.5,
) -> Optional[Dict[str, Any]]:
    n = len(prices)
    pos = 0
    entry_price = 0.0
    hold_ticks = 0
    cd = 0

    balance = initial_balance
    nav_series: List[float] = [1.0]
    closed_pnls: List[float] = []
    total_fees = 0.0

    for t in range(n):
        if cd > 0:
            cd -= 1

        current_price = prices[t]

        if pos == 1:
            hold_ticks += 1
            pnl_bps = ((current_price - entry_price) / entry_price) * 10000.0

            is_close = False
            if pnl_bps >= tp_bps or pnl_bps <= -sl_bps or hold_ticks >= max_hold:
                is_close = True

            if is_close:
                fee_cost = current_price * order_size * fee_rate * 2.0
                slippage_cost = current_price * order_size * (slippage_bps * 0.0001) * 2.0
                total_trade_cost = fee_cost + slippage_cost

                realized_dollar = (current_price - entry_price) * order_size - total_trade_cost
                balance += realized_dollar
                closed_pnls.append(realized_dollar)
                total_fees += total_trade_cost

                pos = 0
                entry_price = 0.0
                hold_ticks = 0
                cd = cooldown
        else:
            if cd == 0 and current_price > trend_ema[t] and ema_obi[t] > obi_th:
                pos = 1
                entry_price = current_price
                hold_ticks = 0

        if t % 200 == 0:
            unrealized_dollar = (current_price - entry_price) * order_size if pos == 1 else 0.0
            nav_series.append((balance + unrealized_dollar) / initial_balance)

    total_closed = len(closed_pnls)
    if total_closed == 0:
        return None

    final_nav = nav_series[-1]
    total_return_pct = (final_nav - 1.0) * 100.0

    peak = nav_series[0]
    max_mdd = 0.0
    for nav in nav_series:
        if nav > peak:
            peak = nav
        dd = (peak - nav) / peak if peak > 0 else 0.0
        if dd > max_mdd:
            max_mdd = dd

    periodic_rets = [
        (nav_series[i] - nav_series[i - 1]) / nav_series[i - 1]
        for i in range(1, len(nav_series))
        if nav_series[i - 1] > 0
    ]
    if len(periodic_rets) > 1:
        mean_ret = float(np.mean(periodic_rets))
        std_ret = float(np.std(periodic_rets))
        sharpe = (mean_ret / std_ret) * math.sqrt(365 * 24 * 60) if std_ret > 1e-9 else 0.0
    else:
        sharpe = 0.0

    wins = [p for p in closed_pnls if p > 0]
    win_rate = (len(wins) / total_closed) * 100.0
    profit_factor = (
        (sum(wins) / abs(sum(p for p in closed_pnls if p < 0)))
        if any(p < 0 for p in closed_pnls)
        else 99.0
    )

    return {
        "obi_th": obi_th,
        "tp_bps": tp_bps,
        "sl_bps": sl_bps,
        "max_hold": max_hold,
        "cooldown": cooldown,
        "total_trades": total_closed,
        "win_rate": round(win_rate, 2),
        "profit_factor": round(profit_factor, 2),
        "total_return_pct": round(total_return_pct, 2),
        "max_mdd_pct": round(max_mdd * 100.0, 2),
        "sharpe": round(sharpe, 2),
        "fees": round(total_fees, 4),
    }


def run_optimization(
    data_path: str = "data/orderbook.jsonl",
    output_report: str = "logs/optimization_results.json",
) -> List[Dict[str, Any]]:
    print(f"Loading data from {data_path}...")
    prices, obis = load_orderbook_data(data_path)
    print(f"Loaded {len(prices)} snapshots. Precomputing trend and OBI features...")
    indicators = precompute_indicators(prices, obis)

    trend_configs = [("15m", indicators["ema_15m"]), ("30m", indicators["ema_30m"])]
    obi_grid = [0.30, 0.45, 0.60]
    tp_grid = [70.0, 90.0, 110.0, 130.0]
    sl_grid = [25.0, 35.0, 45.0]
    hold_grid = [3000, 6000]
    cooldown_grid = [150, 300]

    combos = list(itertools.product(trend_configs, obi_grid, tp_grid, sl_grid, hold_grid, cooldown_grid))
    total_runs = len(combos)
    print(f"Running grid search over {total_runs} parameter configurations...\n")

    results: List[Dict[str, Any]] = []
    for (trend_name, trend_arr), obi, tp, sl, hold, cd in combos:
        res = evaluate_configuration(
            prices=prices,
            ema_obi=indicators["ema_obi"],
            trend_ema=trend_arr,
            obi_th=obi,
            tp_bps=tp,
            sl_bps=sl,
            max_hold=hold,
            cooldown=cd,
        )
        if res and res["total_return_pct"] > 0:
            res["trend"] = trend_name
            results.append(res)

    results.sort(key=lambda x: x["sharpe"], reverse=True)

    print("TOP 10 PARAMETER CONFIGURATIONS (Ranked by Sharpe Ratio)")
    print(f"{'Rank':<5} | {'Trend':<5} | {'OBI':<5} | {'TP(bps)':<8} | {'SL(bps)':<8} | {'Hold':<6} | {'CD':<5} | {'Trades':<7} | {'WinRate':<8} | {'Return(%)':<10} | {'MDD(%)':<8} | {'Sharpe':<8}")

    for rank, r in enumerate(results[:10], 1):
        print(
            f"#{rank:<4} | "
            f"{r['trend']:<5} | "
            f"{r['obi_th']:<5.2f} | "
            f"{r['tp_bps']:<8.1f} | "
            f"{r['sl_bps']:<8.1f} | "
            f"{r['max_hold']:<6} | "
            f"{r['cooldown']:<5} | "
            f"{r['total_trades']:<7} | "
            f"{r['win_rate']:>7.2f}% | "
            f"{r['total_return_pct']:>9.2f}% | "
            f"{r['max_mdd_pct']:>7.2f}% | "
            f"{r['sharpe']:>8.2f}"
        )

    print("=" * 115)
    print(f"Positive Configurations Found: {len(results)} / {total_runs}")

    os.makedirs(os.path.dirname(output_report), exist_ok=True)
    with open(output_report, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nOptimization report saved to: {output_report}")

    return results

if __name__ == "__main__":
    run_optimization()