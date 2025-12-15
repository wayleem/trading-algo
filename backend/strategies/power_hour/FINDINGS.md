# Power Hour Strategy - Findings

## Strategy Overview

Trades momentum in the last hour of trading (3-4 PM ET).

**Signal Logic:**
- BUY CALL: Bullish momentum in power hour
- BUY PUT: Bearish momentum in power hour

## Backtest Results (Jan-Dec 2024)

```
Total Trades:     63
Win Rate:         28.6%
Total P&L:        -$360
Sharpe Ratio:     -6.36
Profit Factor:    0.40
```

## Key Findings

1. **Low win rate (28.6%)**: Power hour is choppy
2. **Small loss (-$360)**: Limited exposure due to late entry
3. **0DTE decay accelerates**: Options lose value rapidly near expiration

## Why It Fails

- 0DTE options have minimal time value in last hour
- Theta decay is extreme
- Any adverse move = total loss
- Gains capped by time remaining

## Conclusion

Power hour trading is EXTREMELY risky with 0DTE options due to accelerated theta decay.
