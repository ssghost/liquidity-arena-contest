import itertools
import json
import os
from typing import Any, Dict, List

from strategy.backtester import Backtester


def run_grid_search(
    data_path: str = "data/orderbook.jsonl",
    initial_balance: float = 10000.0,
    # >>>>>>>> MODIFICATION START: 掃描極端強信號區間 <<<<<<<<
    obi_thresholds: List[float] = [0.80, 0.85, 0.88, 0.90, 0.92],
    order_sizes: List[float] = [0.005, 0.01],
    # >>>>>>>> MODIFICATION END <<<<<<<<
    slippage_bps: float = 0.5,
    fee_rate: float = 0.0005,
    output_report: str = "logs/grid_search_results.json",
) -> List[Dict[str, Any]]:
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Market data file not found: {data_path}")

    param_combinations = list(itertools.product(obi_thresholds, order_sizes))
    total_runs = len(param_combinations)
    print(f"Starting Grid Search: {total_runs} Parameter Configurations")
    print(f"Data Source: {data_path}\n")

    results: List[Dict[str, Any]] = []

    for idx, (obi, size) in enumerate(param_combinations, 1):
        print(f"[{idx:02d}/{total_runs:02d}] Testing obi={obi:.2f}, size={size} ...", end=" ", flush=True)

        backtester = Backtester(
            data_path=data_path,
            initial_balance=initial_balance,
            obi_threshold=obi,
            order_size=size,
            fee_rate=fee_rate,
            slippage_bps=slippage_bps,
            log_file=f"logs/grid_obi{obi:.2f}_sz{size}.jsonl",
        )

        metrics = backtester.run()
        record = {
            "obi_threshold": obi,
            "order_size": size,
            "final_nav": metrics["final_nav"],
            "total_return_pct": metrics["total_return_pct"],
            "max_drawdown_pct": metrics["max_drawdown_pct"],
            "sharpe_ratio": metrics["sharpe_ratio"],
            "win_rate_pct": metrics["win_rate_pct"],
            "profit_factor": metrics["profit_factor"],
            "total_orders": metrics["total_orders"],
            "closed_trades": metrics["closed_trades"],
            "total_fees_paid": metrics["total_fees_paid"],
        }
        results.append(record)
        print(f"Done -> Sharpe: {record['sharpe_ratio']:>5.2f} | Return: {record['total_return_pct']:>6.2f}% | MDD: {record['max_drawdown_pct']:>5.2f}% | Trades: {record['closed_trades']}")

    results.sort(key=lambda x: x["sharpe_ratio"], reverse=True)

    print("\n" + "=" * 90)
    print("TOP 5 PARAMETER CONFIGURATIONS (Ranked by Sharpe Ratio)")
    print("=" * 90)
    for rank, res in enumerate(results[:5], 1):
        print(
            f"Rank #{rank} | OBI: {res['obi_threshold']:.2f} | Size: {res['order_size']} | "
            f"Sharpe: {res['sharpe_ratio']:>6.2f} | Return: {res['total_return_pct']:>6.2f}% | "
            f"MDD: {res['max_drawdown_pct']:>5.2f}% | WinRate: {res['win_rate_pct']:>5.2f}% | "
            f"Trades: {res['closed_trades']} | Fees: ${res['total_fees_paid']:.2f}"
        )
    print("=" * 90)

    os.makedirs(os.path.dirname(output_report), exist_ok=True)
    with open(output_report, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nOptimization report saved to: {output_report}")

    return results


if __name__ == "__main__":
    run_grid_search()