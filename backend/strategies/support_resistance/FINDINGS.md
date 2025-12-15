# Support/Resistance Strategy - Findings

## Strategy Overview

Trades bounces off support levels and rejections at resistance levels.

**Signal Logic:**
- BUY CALL: Price bounces off identified support
- BUY PUT: Price rejects at identified resistance

## Backtest Results (Jan-Dec 2024)

```
Total Trades:     262
Win Rate:         34.4%
Total P&L:        -$2,594
Sharpe Ratio:     -7.12
Profit Factor:    0.43
```

## Key Findings

1. **Low win rate (34.4%)**: S/R levels often break
2. **Many false signals**: Intraday S/R is noisy
3. **Profit factor 0.43**: Losses 2.3x wins

## Why It Fails

- Intraday S/R levels are unreliable
- Levels that hold on daily charts break intraday
- By the time S/R is confirmed, move is over
- Stop-losses trigger on level tests before bounce

## Conclusion

Support/resistance trading does NOT work for 0DTE options. Intraday levels are too noisy.
