# VWAP Reversion Strategy - Findings

## Strategy Overview

Trades pullbacks to VWAP after extended moves, expecting bounce/rejection.

**Signal Logic:**
- BUY CALL: Price pulls back to VWAP from above (support)
- BUY PUT: Price pulls back to VWAP from below (resistance)

## Backtest Results (Jan-Dec 2024)

```
Total Trades:     673
Win Rate:         44.7%
Total P&L:        -$6,071
Sharpe Ratio:     -14.64
Profit Factor:    0.36
```

## Key Findings

1. **High trade count**: 673 trades = many signals
2. **Decent win rate (44.7%)**: But losses much larger than wins
3. **Terrible Sharpe (-14.64)**: Extremely volatile returns
4. **Profit factor 0.36**: Average loss 2.8x average win

## Why It Fails

- VWAP touches are common but unreliable
- Many false signals in ranging markets
- Stop-losses hit before VWAP bounce/rejection materializes
- 0DTE options don't give enough time for setup to play out

## Conclusion

VWAP reversion generates too many signals with poor risk/reward for 0DTE options.
