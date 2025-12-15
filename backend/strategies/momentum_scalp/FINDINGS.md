# Momentum Scalp Strategy - Findings

## Strategy Overview

Scalps momentum moves using RSI acceleration and price momentum.

**Signal Logic:**
- BUY CALL: Strong upward momentum detected
- BUY PUT: Strong downward momentum detected

## Backtest Results (Jan-Dec 2024)

```
Total Trades:     450
Win Rate:         34.0%
Total P&L:        -$4,124
Sharpe Ratio:     -13.69
Profit Factor:    0.23
```

## Key Findings

1. **Profit factor 0.23**: One of the worst performing strategies
2. **Low win rate (34%)**: Momentum signals unreliable
3. **Terrible Sharpe (-13.69)**: Highly volatile negative returns

## Why It Fails

- Momentum often exhausts by the time signal triggers
- Chasing momentum = buying at tops, selling at bottoms
- 0DTE options amplify losses on reversals

## Conclusion

Momentum scalping does NOT work for 0DTE options. By the time momentum is confirmed, the move is often over.
