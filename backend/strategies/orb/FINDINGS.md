# Opening Range Breakout (ORB) Strategy - Findings

## Strategy Overview

Trades breakouts from the first hour's high/low range on SPY.

**Signal Logic:**
- BUY CALL: Price breaks above opening range high
- BUY PUT: Price breaks below opening range low
- Opening range: First 30 minutes (9:30-10:00 AM ET)

## Backtest Results (Jan-Dec 2024)

```
Total Trades:     302
Win Rate:         37.7%
Total P&L:        -$2,298
Sharpe Ratio:     -5.17
Profit Factor:    0.56

Exit Breakdown:
  profit_target: 114 trades, 100% win rate, +$2,934
  stop_loss:     188 trades, 0% win rate, -$5,232
```

## Key Findings

1. **More stops than targets**: 62% of trades hit stop-loss
2. **False breakouts dominate**: Opening range breaks often reverse
3. **Slippage hurts entries**: Chasing breakouts = worse fills
4. **Profit factor below 1.0**: Losing strategy

## Why It Fails

- 0DTE options decay fast, need quick moves
- Breakouts often fail to follow through
- Wide spreads on entries during volatile breakout moments
- Stop-losses trigger before profits can develop

## Conclusion

ORB is NOT profitable for 0DTE options. The strategy relies on sustained momentum that rarely materializes in intraday SPY moves.
