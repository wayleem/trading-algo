# Morning Fade Strategy - Findings

## Strategy Overview

Fades the initial morning move, expecting reversal after the first 30-60 minutes.

**Signal Logic:**
- BUY CALL: Morning selloff showing exhaustion
- BUY PUT: Morning rally showing exhaustion

## Backtest Results (Jan-Dec 2024)

```
Total Trades:     22
Win Rate:         63.6%
Total P&L:        -$45
Sharpe Ratio:     -0.30
Profit Factor:    0.96
```

## Key Findings

1. **Best win rate (63.6%)**: Concept has merit
2. **Nearly breakeven**: Only -$45 loss
3. **Very few trades**: Only 22 signals in full year
4. **Profit factor ~1.0**: Wins roughly equal losses

## Why It Almost Works

- Morning fades are a real phenomenon
- First 30-60 minutes often reverse
- But gains too small to overcome slippage/commissions

## Why It Still Fails

- 0DTE options have too much slippage
- Profit targets hit but don't compensate for stop losses
- Need tighter execution for this to work

## Conclusion

Morning fade shows promise but execution costs make it unprofitable for 0DTE. Consider with longer-dated options or futures.
