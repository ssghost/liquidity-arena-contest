import json
import os
from typing import Any, Dict, List
import matplotlib.pyplot as plt

def load_backtest_data(
    log_file: str = "logs/backtest_reasoning.jsonl",
    summary_file: str = "logs/backtest_summary.json",
) -> Dict[str, Any]:
    if not os.path.exists(log_file):
        raise FileNotFoundError(f"Log file not found: {log_file}")

    nav_series: List[float] = []
    timestamps: List[str] = []
    trades: List[Dict[str, Any]] = []

    with open(log_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except Exception:
                continue

            risk_info = (
                record.get("risk_evaluation")
                or record.get("decision", {}).get("risk_verification")
                or record.get("decision", {}).get("risk_evaluation")
                or {}
            )
            nav = risk_info.get("nav") or risk_info.get("current_nav") or 1.0

            nav_series.append(nav)
            timestamps.append(record.get("timestamp", ""))

            decision_obj = record.get("decision", {})
            action = record.get("action") or decision_obj.get("action")
            
            if action in ["BUY_OPEN", "SELL_OPEN", "CLOSE_POSITION"]:
                order_params = record.get("action_params") or decision_obj.get("order_params") or {}
                price = order_params.get("price")
                trades.append({
                    "index": len(nav_series) - 1,
                    "action": action,
                    "price": price,
                    "nav": nav,
                })

    summary = {}
    if os.path.exists(summary_file):
        with open(summary_file, "r", encoding="utf-8") as f:
            summary = json.load(f)

    return {
        "nav_series": nav_series,
        "timestamps": timestamps,
        "trades": trades,
        "summary": summary,
    }

def plot_performance(
    data: Dict[str, Any], output_chart: str = "images/backtest_performance.png"
) -> None:
    nav_series = data["nav_series"]
    trades = data["trades"]
    summary = data["summary"]

    if not nav_series:
        print("No NAV data available to plot.")
        return

    drawdowns = []
    peak = nav_series[0]
    for nav in nav_series:
        if nav > peak:
            peak = nav
        dd = (peak - nav) / peak if peak > 0 else 0.0
        drawdowns.append(-dd * 100.0)

    step = max(1, len(nav_series) // 2000)
    sampled_indices = list(range(0, len(nav_series), step))
    sampled_nav = [nav_series[i] for i in sampled_indices]
    sampled_dd = [drawdowns[i] for i in sampled_indices]

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(12, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )

    ax1.plot(sampled_indices, sampled_nav, label="Strategy NAV", color="#1f77b4", linewidth=1.5)
    ax1.axhline(1.0, color="gray", linestyle="--", alpha=0.7, label="Baseline (1.0)")
    ax1.axhline(0.8, color="red", linestyle=":", alpha=0.8, label="Liquidation Line (0.8)")

    buy_trades = [t for t in trades if t["action"] == "BUY_OPEN"]
    close_trades = [t for t in trades if t["action"] == "CLOSE_POSITION"]

    if buy_trades:
        ax1.scatter(
            [t["index"] for t in buy_trades],
            [t["nav"] for t in buy_trades],
            marker="^",
            color="green",
            s=40,
            label=f"BUY_OPEN ({len(buy_trades)})",
            zorder=5,
        )

    if close_trades:
        ax1.scatter(
            [t["index"] for t in close_trades],
            [t["nav"] for t in close_trades],
            marker="v",
            color="orange",
            s=40,
            label=f"CLOSE_POSITION ({len(close_trades)})",
            zorder=5,
        )

    net_pnl = summary.get("net_pnl_usd", 0.0)
    ret_pct = summary.get("total_return_pct", 0.0)
    sharpe = summary.get("sharpe_ratio", 0.0)
    max_dd = summary.get("max_drawdown_pct", 0.0)
    win_rate = summary.get("win_rate_pct", 0.0)
    closed_trades_count = summary.get("closed_trades", len(close_trades))

    title_text = (
        f"Strategy Performance | Net PnL: {net_pnl:+.2f} USD ({ret_pct:+.2f}%) | "
        f"Sharpe: {sharpe:.2f} | MaxDD: {max_dd:.2f}%\n"
        f"Win Rate: {win_rate:.2f}% | Closed Trades: {closed_trades_count} | Profit Factor: {summary.get('profit_factor', 0.0):.2f}"
    )
    ax1.set_title(title_text, fontsize=11, fontweight="bold")

    ax1.set_ylabel("Net Asset Value (NAV)")
    ax1.legend(loc="upper left")
    ax1.grid(True, linestyle="--", alpha=0.5)

    ax2.fill_between(sampled_indices, sampled_dd, 0, color="#d62728", alpha=0.3)
    ax2.plot(sampled_indices, sampled_dd, color="#d62728", linewidth=1.0)
    ax2.set_title("Underwater Drawdown (%)", fontsize=10)
    ax2.set_xlabel("Decision Step (Snapshot Index)")
    ax2.set_ylabel("Drawdown %")
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_chart) or ".", exist_ok=True)
    plt.savefig(output_chart, dpi=200)
    plt.close()

    print(f"Performance chart generated: {output_chart}")

if __name__ == "__main__":
    data = load_backtest_data()
    plot_performance(data)
