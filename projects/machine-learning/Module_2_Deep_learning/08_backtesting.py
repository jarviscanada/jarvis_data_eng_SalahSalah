"""
Demo 08: Backtesting Framework with Business Impact Metrics
=============================================================
Module 2 - Introduction to Deep Learning
ML Engineer Training Curriculum

Demonstrates:
- Running a realistic backtest with transaction costs
- Computing business metrics (dollar P&L, Sharpe, max drawdown)
- Plotting equity curves and drawdown charts
- Generating a monthly returns table
- Comparing strategy vs buy-and-hold benchmark

This is the framework you will adapt for Capstone Step 6.

Usage:
    python demos/08_backtesting.py
"""

import numpy as np
import pandas as pd
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


# ======================================================================
# BACKTESTING ENGINE
# ======================================================================

class SimpleBacktester:
    """
    A straightforward backtesting engine for daily signal-based strategies.

    Workflow:
        1. Provide daily predictions and actual returns for the test period.
        2. The backtester simulates trading: long when prediction > 0,
           flat (cash) when prediction <= 0.
        3. Transaction costs are subtracted on each day the position changes.
        4. All results are reported net of costs.

    This is NOT a production backtesting engine. It assumes:
        - Trades execute at the close price (no slippage beyond the cost parameter).
        - No leverage, no margin, no shorting (long/flat only).
        - Single asset at a time.

    For production backtesting, use libraries like Backtrader, Zipline,
    or QuantConnect.
    """

    def __init__(
        self,
        predictions: np.ndarray,
        actual_returns: np.ndarray,
        dates: pd.DatetimeIndex = None,
        transaction_cost_bps: float = 10.0,
        initial_capital: float = 1_000_000.0,
    ):
        """
        Args:
            predictions: Model's predicted returns (or any signal).
                         Positive = go long, non-positive = stay flat.
            actual_returns: Realized daily returns (log or simple).
            dates: Date index for the test period.
            transaction_cost_bps: Round-trip cost in basis points (10 = 0.10%).
            initial_capital: Starting capital in dollars.
        """
        assert len(predictions) == len(actual_returns), (
            f"Length mismatch: predictions={len(predictions)}, "
            f"actual_returns={len(actual_returns)}"
        )

        self.predictions = np.array(predictions)
        self.actual_returns = np.array(actual_returns)
        self.dates = dates
        self.cost_per_trade = transaction_cost_bps / 10_000  # Convert bps to decimal
        self.initial_capital = initial_capital
        self.results = None

    def run(self) -> pd.DataFrame:
        """Run the backtest and return a DataFrame of daily results."""
        n = len(self.predictions)

        # Position: 1 = long, 0 = flat (cash)
        positions = (self.predictions > 0).astype(float)

        # Detect trades (position changes)
        trades = np.zeros(n)
        trades[0] = positions[0]  # First day: entering a position is a trade
        trades[1:] = np.abs(np.diff(positions))

        # Strategy returns: actual return when long, 0 when flat
        gross_returns = positions * self.actual_returns

        # Subtract transaction costs on trade days
        costs = trades * self.cost_per_trade
        net_returns = gross_returns - costs

        # Cumulative returns
        cumulative_gross = np.cumprod(1 + gross_returns)
        cumulative_net = np.cumprod(1 + net_returns)

        # Benchmark: buy and hold (always long)
        cumulative_benchmark = np.cumprod(1 + self.actual_returns)

        # Build results DataFrame
        results = pd.DataFrame({
            "prediction": self.predictions,
            "actual_return": self.actual_returns,
            "position": positions,
            "trade": trades,
            "gross_return": gross_returns,
            "cost": costs,
            "net_return": net_returns,
            "cumulative_gross": cumulative_gross,
            "cumulative_net": cumulative_net,
            "cumulative_benchmark": cumulative_benchmark,
        })

        if self.dates is not None:
            results.index = self.dates[:n]

        self.results = results
        return results

    def compute_metrics(self) -> dict:
        """Compute business-relevant performance metrics."""
        if self.results is None:
            self.run()

        r = self.results
        net = r["net_return"].values
        bench = r["actual_return"].values
        n_days = len(net)

        # Annualization factor (252 trading days)
        ann = 252

        # Strategy metrics (net of costs)
        total_return = r["cumulative_net"].iloc[-1] - 1
        ann_return = (1 + total_return) ** (ann / n_days) - 1
        ann_vol = net.std() * np.sqrt(ann)
        sharpe = ann_return / ann_vol if ann_vol > 0 else 0

        # Maximum drawdown
        cum = r["cumulative_net"].values
        running_max = np.maximum.accumulate(cum)
        drawdowns = (cum - running_max) / running_max
        max_drawdown = drawdowns.min()
        max_dd_end = np.argmin(drawdowns)
        max_dd_start = np.argmax(cum[:max_dd_end + 1]) if max_dd_end > 0 else 0

        # Win rate
        trading_days = r[r["position"] == 1]
        if len(trading_days) > 0:
            win_rate = (trading_days["net_return"] > 0).mean()
        else:
            win_rate = 0

        # Benchmark metrics
        bench_total = r["cumulative_benchmark"].iloc[-1] - 1
        bench_ann = (1 + bench_total) ** (ann / n_days) - 1
        bench_vol = bench.std() * np.sqrt(ann)
        bench_sharpe = bench_ann / bench_vol if bench_vol > 0 else 0

        # Dollar P&L
        final_value = self.initial_capital * r["cumulative_net"].iloc[-1]
        dollar_pnl = final_value - self.initial_capital

        # Trade count
        n_trades = int(r["trade"].sum())

        # Total costs
        total_costs = r["cost"].sum() * self.initial_capital

        metrics = {
            "Test Period": f"{n_days} trading days",
            "Starting Capital": f"${self.initial_capital:,.0f}",
            "Final Portfolio Value": f"${final_value:,.0f}",
            "Dollar P&L": f"${dollar_pnl:,.0f}",
            "Total Return (net)": f"{total_return:.2%}",
            "Annualized Return (net)": f"{ann_return:.2%}",
            "Annualized Volatility": f"{ann_vol:.2%}",
            "Sharpe Ratio": f"{sharpe:.2f}",
            "Max Drawdown": f"{max_drawdown:.2%}",
            "Win Rate": f"{win_rate:.2%}",
            "Number of Trades": n_trades,
            "Total Transaction Costs": f"${total_costs:,.0f}",
            "Benchmark Return": f"{bench_total:.2%}",
            "Benchmark Ann. Return": f"{bench_ann:.2%}",
            "Benchmark Sharpe": f"{bench_sharpe:.2f}",
            "Alpha (vs Benchmark)": f"{ann_return - bench_ann:.2%}",
        }

        return metrics

    def monthly_returns(self) -> pd.DataFrame:
        """Compute monthly returns table."""
        if self.results is None:
            self.run()

        if self.results.index.dtype == "int64":
            print("Warning: no date index. Monthly returns require dates.")
            return pd.DataFrame()

        monthly = self.results["net_return"].resample("ME").apply(
            lambda x: (1 + x).prod() - 1
        )
        monthly_df = monthly.to_frame("Strategy")

        bench_monthly = self.results["actual_return"].resample("ME").apply(
            lambda x: (1 + x).prod() - 1
        )
        monthly_df["Benchmark"] = bench_monthly
        monthly_df["Excess"] = monthly_df["Strategy"] - monthly_df["Benchmark"]

        return monthly_df

    def plot_equity_curve(self, save_path: str = None):
        """Plot cumulative P&L: strategy vs benchmark."""
        if self.results is None:
            self.run()

        r = self.results
        fig, axes = plt.subplots(2, 1, figsize=(12, 8), height_ratios=[3, 1])

        # Top: equity curve
        ax1 = axes[0]
        capital = self.initial_capital
        ax1.plot(
            r.index if hasattr(r.index, "date") else range(len(r)),
            r["cumulative_net"] * capital,
            label="Strategy (net of costs)",
            color="#1565C0",
            linewidth=1.5,
        )
        ax1.plot(
            r.index if hasattr(r.index, "date") else range(len(r)),
            r["cumulative_benchmark"] * capital,
            label="Buy & Hold Benchmark",
            color="#999999",
            linewidth=1,
            linestyle="--",
        )
        ax1.axhline(y=capital, color="#CCCCCC", linewidth=0.5)
        ax1.set_title("Strategy Equity Curve vs Benchmark", fontsize=14, fontweight="bold")
        ax1.set_ylabel("Portfolio Value ($)")
        ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f"${x:,.0f}"))
        ax1.legend(loc="upper left")
        ax1.grid(True, alpha=0.3)

        # Bottom: drawdown
        ax2 = axes[1]
        cum = r["cumulative_net"].values
        running_max = np.maximum.accumulate(cum)
        drawdowns = (cum - running_max) / running_max * 100
        ax2.fill_between(
            r.index if hasattr(r.index, "date") else range(len(r)),
            drawdowns,
            0,
            color="#EF5350",
            alpha=0.4,
        )
        ax2.set_title("Drawdown (%)", fontsize=12)
        ax2.set_ylabel("Drawdown %")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Equity curve saved to: {save_path}")
        plt.close(fig)


# ======================================================================
# MAIN DEMO
# ======================================================================

def main():
    np.random.seed(42)

    print("=" * 60)
    print("BACKTESTING FRAMEWORK DEMO")
    print("=" * 60)

    # -- Generate synthetic test data --
    # Simulate 500 trading days (~2 years) of test data
    n_days = 500
    dates = pd.bdate_range(start="2022-01-03", periods=n_days)

    # Actual market returns: random walk with slight upward drift
    actual_returns = np.random.randn(n_days) * 0.012 + 0.0003

    # Model predictions: correlated with actual returns + noise
    # This simulates a model with ~53% directional accuracy
    signal_strength = 0.15
    noise = np.random.randn(n_days) * 0.015
    predictions = signal_strength * actual_returns + noise

    direction_acc = np.mean((predictions > 0) == (actual_returns > 0))
    print(f"\nSimulated model directional accuracy: {direction_acc:.1%}")
    print(f"Test period: {dates[0].date()} to {dates[-1].date()} ({n_days} trading days)")

    # -- Run backtest --
    print("\n" + "-" * 60)
    print("RUNNING BACKTEST")
    print("-" * 60)

    backtester = SimpleBacktester(
        predictions=predictions,
        actual_returns=actual_returns,
        dates=dates,
        transaction_cost_bps=10.0,
        initial_capital=1_000_000.0,
    )

    results = backtester.run()

    # -- Print metrics --
    print("\n" + "-" * 60)
    print("PERFORMANCE METRICS")
    print("-" * 60)

    metrics = backtester.compute_metrics()
    for key, value in metrics.items():
        print(f"  {key:30s}  {value}")

    # -- Monthly returns --
    print("\n" + "-" * 60)
    print("MONTHLY RETURNS")
    print("-" * 60)

    monthly = backtester.monthly_returns()
    if not monthly.empty:
        monthly_display = monthly.copy()
        for col in monthly_display.columns:
            monthly_display[col] = monthly_display[col].apply(lambda x: f"{x:.2%}")
        print(monthly_display.to_string())

    # -- Plot equity curve --
    os.makedirs("reports", exist_ok=True)
    backtester.plot_equity_curve(save_path="reports/equity_curve.png")

    # -- Summary for executive report --
    print("\n" + "=" * 60)
    print("EXECUTIVE SUMMARY (copy this into your report)")
    print("=" * 60)
    print(f"""
STRATEGY OVERVIEW
The deep learning model generates daily trading signals for S&P 500
stocks. When the model predicts a positive next-day return, the strategy
goes long. When the prediction is non-positive, the strategy moves to
cash. All results are net of 10 basis point transaction costs.

KEY RESULTS (Test Period: {dates[0].date()} to {dates[-1].date()})
  Starting Capital:       {metrics['Starting Capital']}
  Final Portfolio Value:  {metrics['Final Portfolio Value']}
  Dollar P&L:             {metrics['Dollar P&L']}
  Annualized Return:      {metrics['Annualized Return (net)']}
  Sharpe Ratio:           {metrics['Sharpe Ratio']}
  Max Drawdown:           {metrics['Max Drawdown']}

VERSUS BENCHMARK (Buy & Hold)
  Benchmark Return:       {metrics['Benchmark Return']}
  Alpha:                  {metrics['Alpha (vs Benchmark)']}

OPERATIONAL METRICS
  Number of Trades:       {metrics['Number of Trades']}
  Transaction Costs:      {metrics['Total Transaction Costs']}
  Win Rate:               {metrics['Win Rate']}

RECOMMENDATION
[Based on your actual results, write one of:]
  DEPLOY   - Strategy shows consistent alpha with acceptable risk.
  ITERATE  - Promising signal but needs refinement (reduce drawdown,
             improve Sharpe, test on more assets).
  ABANDON  - Model does not add value over buy-and-hold after costs.
""")

    print("=" * 60)
    print("DEMO COMPLETE")
    print("=" * 60)
    print("\nFiles created:")
    print("  reports/equity_curve.png - Equity curve and drawdown chart")
    print("\nNext steps for your capstone:")
    print("  1. Replace synthetic predictions with your actual model output")
    print("  2. Replace synthetic returns with real stock returns from the test period")
    print("  3. Run the backtest and paste the metrics into your executive summary")
    print("  4. Include the equity curve chart in your final report")


if __name__ == "__main__":
    main()
