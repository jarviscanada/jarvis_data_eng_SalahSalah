# Backtest Report: Deep Learning Equity Signal

**Strategy:** [Model Name] - Daily Long/Flat Signal on S&P 500 Stocks
**Test Period:** [Start Date] to [End Date] ([X] trading days)
**Capital:** $1,000,000 starting
**Transaction Costs:** 10 basis points per trade (round-trip)

---

## 1. Model Comparison Summary

| Model | Params | MSE | MAE | Dir Acc | Sharpe | Max DD | Train Time |
|-------|--------|-----|-----|---------|--------|--------|------------|
| Linear Regression (baseline) | [X] | [X] | [X] | [X%] | [X.XX] | [X%] | [Xs] |
| Feedforward NN (128-64-32) | [X] | [X] | [X] | [X%] | [X.XX] | [X%] | [Xm] |
| Feedforward NN (256-128-64) | [X] | [X] | [X] | [X%] | [X.XX] | [X%] | [Xm] |
| LSTM (64 hidden, 60-day) | [X] | [X] | [X] | [X%] | [X.XX] | [X%] | [Xm] |
| LSTM (64 hidden, 120-day) | [X] | [X] | [X] | [X%] | [X.XX] | [X%] | [Xm] |
| 1D CNN | [X] | [X] | [X] | [X%] | [X.XX] | [X%] | [Xm] |
| ResNet-18 (fine-tuned) | [X] | N/A | N/A | [X%] | [X.XX] | [X%] | [Xm] |
| ResNet-18 (feature extraction) | [X] | N/A | N/A | [X%] | [X.XX] | [X%] | [Xm] |

**Selected Model:** [Model Name] - chosen because [1-2 sentence justification, e.g., "highest Sharpe ratio with acceptable drawdown and moderate training time"]

---

## 2. Equity Curve

[INSERT equity_curve.png HERE]

*Figure: Cumulative portfolio value from $1,000,000 starting capital. Blue = deep learning strategy (net of costs). Gray dashed = buy-and-hold SPY benchmark. Red shaded area = drawdown periods.*

---

## 3. Strategy Performance (Net of Costs)

| Metric | Strategy | Benchmark (SPY) | Excess |
|--------|----------|-----------------|--------|
| Total Return | [X%] | [X%] | [X%] |
| Annualized Return | [X%] | [X%] | [X%] |
| Annualized Volatility | [X%] | [X%] | [X%] |
| Sharpe Ratio | [X.XX] | [X.XX] | [X.XX] |
| Max Drawdown | [X%] | [X%] | - |
| Win Rate | [X%] | - | - |
| Number of Trades | [X] | - | - |
| Total Transaction Costs | $[X,XXX] | $0 | - |
| Final Portfolio Value | $[X,XXX,XXX] | $[X,XXX,XXX] | - |
| Dollar P&L | $[XX,XXX] | $[XX,XXX] | $[XX,XXX] |

---

## 4. Monthly Returns

| Month | Strategy | Benchmark | Excess |
|-------|----------|-----------|--------|
| [Month YYYY] | [X.XX%] | [X.XX%] | [X.XX%] |
| [Month YYYY] | [X.XX%] | [X.XX%] | [X.XX%] |
| ... | ... | ... | ... |

**Best Month:** [Month YYYY] at [X%] return
**Worst Month:** [Month YYYY] at [X%] return
**Months with Positive Excess Return:** [X] out of [Y] ([Z%])

---

## 5. Drawdown Analysis

[INSERT drawdown_chart.png HERE]

*Figure: Strategy drawdown over the test period. Red shading shows peak-to-trough decline.*

| Drawdown Event | Start | End | Depth | Recovery Time |
|----------------|-------|-----|-------|---------------|
| Largest | [Date] | [Date] | [X%] | [X] days |
| 2nd Largest | [Date] | [Date] | [X%] | [X] days |
| 3rd Largest | [Date] | [Date] | [X%] | [X] days |

---

## 6. Regime Analysis

| Market Regime | Days | Strategy Return | Dir Accuracy | Notes |
|---------------|------|-----------------|--------------|-------|
| Low Vol (VIX < 15) | [X] | [X%] ann. | [X%] | [e.g., "Best performance"] |
| Normal (VIX 15-25) | [X] | [X%] ann. | [X%] | [e.g., "Consistent"] |
| High Vol (VIX > 25) | [X] | [X%] ann. | [X%] | [e.g., "Model struggles"] |
| Crisis (VIX > 35) | [X] | [X%] ann. | [X%] | [e.g., "Below baseline"] |

---

## 7. Look-Ahead Bias Checks

Confirm that none of the following sources of data leakage are present:

- [ ] Features are computed using only past data (no future values in rolling windows)
- [ ] Train/val/test splits are strictly temporal (no shuffling across time)
- [ ] Scaler was fit on training data only
- [ ] Backtest predictions use only information available before market open
- [ ] No overlapping windows between training and test sets
- [ ] Walk-forward validation was used (not a single random split)

---

## 8. Sensitivity Analysis

| Parameter Changed | Original Value | New Value | Sharpe Impact |
|-------------------|----------------|-----------|---------------|
| Transaction costs | 10 bps | 20 bps | [X.XX] to [X.XX] |
| Transaction costs | 10 bps | 5 bps | [X.XX] to [X.XX] |
| Sequence length | 60 days | 30 days | [X.XX] to [X.XX] |
| Position sizing | Equal weight | Vol-scaled | [X.XX] to [X.XX] |

*This table shows how sensitive the strategy's Sharpe ratio is to key assumptions. If doubling transaction costs makes the strategy unprofitable, the result is fragile.*
