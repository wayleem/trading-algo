# Gap Fade Strategy - Findings

## Strategy Overview

Trades against overnight gaps, expecting gap fill during the day.

**Signal Logic:**
- BUY CALL: Gap down at open, expecting fill higher
- BUY PUT: Gap up at open, expecting fill lower

## Backtest Results (Jan-Dec 2024)

```
Total Trades:     96
Win Rate:         21.9%
Total P&L:        -$890
Sharpe Ratio:     -5.72
Profit Factor:    0.46
```

## Key Findings

1. **Worst win rate (21.9%)**: Only 1 in 5 trades wins
2. **Gaps often don't fill**: Trend continuation more common
3. **Early morning volatility**: Wide spreads at open

## Why It Fails

- Gap fills are not reliable same-day events
- Many gaps continue in gap direction
- 0DTE options don't allow time for gap to fill
- Morning volatility = high slippage

## Conclusion

Gap fade is NOT suitable for 0DTE options. Gap fills often take multiple days.
