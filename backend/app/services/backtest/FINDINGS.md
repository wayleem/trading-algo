# RSI Mean Reversion Strategy - Backtest Findings

## Strategy Overview

RSI-based mean reversion strategy for 0DTE SPY options using parallel 3-min and 5-min timeframes.

**Signal Logic:**
- BUY CALL: RSI crosses UP through its SMA while below oversold threshold
- BUY PUT: RSI crosses DOWN through its SMA while above overbought threshold

## Final Results (Jan-Dec 2024)

### Default Parameters (RSI 44/60, entry cutoff 18 UTC)
```
Total Trades:     1,263
Win Rate:         31.0%
Total P&L:        -$15,816
Sharpe Ratio:     -4.17
Profit Factor:    0.72

Exit Breakdown:
  profit_target: 391 trades, +$42,000
  stop_loss:     872 trades, -$58,000
```

### Conservative Parameters (RSI 30/70)
```
Total Trades:     215
Win Rate:         38.1%
Total P&L:        +$821
Sharpe Ratio:     0.53
Profit Factor:    1.08

Exit Breakdown:
  profit_target: 82 trades, +$11,151
  stop_loss:     133 trades, -$10,330
```

### With Late Entry Cutoff (20 UTC, allows EOD trades)
```
Total Trades:     1,435
Win Rate:         29.5%
Total P&L:        -$16,213
Sharpe Ratio:     -4.01
Profit Factor:    0.73

Exit Breakdown:
  end_of_day:    25 trades, +$268
  profit_target: 413 trades, +$42,375
  stop_loss:     997 trades, -$58,856
```

## Key Findings

1. **Stop losses are the killer**: 997 stop-loss exits lost $58,856 vs $42,375 gained from profit targets
2. **EOD expiration is NOT the main problem**: Only 25 EOD trades, actually slightly profitable (+$268)
3. **Conservative thresholds (30/70) work better**: Fewer trades but marginally profitable
4. **Aggressive thresholds (44/60) overtrade**: More signals = more stop-loss hits

## Why It Fails

1. **Slippage on 0DTE options**: 3% entry + 3% exit slippage on low-priced options
2. **Stop-loss cascade**: Options lose value fast, stop-losses trigger frequently
3. **RSI signals are noisy**: Many false signals in choppy markets
4. **Look-ahead bias removed**: Entry at next bar open, not signal bar close

## Recommendations

- RSI mean reversion is NOT viable for 0DTE options
- Stop-loss optimization cannot overcome fundamental strategy weakness
- Consider longer-dated options where slippage is less impactful
