# Interactive Debugger Test Guide

## Quick Start

Test the new interactive debugging feature with the SMA crossover experiment:

```bash
# Basic interactive mode - pause at each timestamp
uv run qtrader backtest experiments/sma_crossover --interactive

# Start from a specific date
uv run qtrader backtest experiments/sma_crossover --interactive --break-at 2020-06-15

# Show detailed inspection (all bar data)
uv run qtrader backtest experiments/sma_crossover --interactive --inspect full

# Show only strategy-level data (indicators, signals)
uv run qtrader backtest experiments/sma_crossover --interactive --inspect strategy
```

## Interactive Commands

Once paused at a timestamp:

- **Enter/Return** - Step to next timestamp
- **c** - Continue without pausing (run to completion)
- **q** - Quit backtest immediately
- **i** - Toggle inspection level (bars → full → strategy → bars)

## What You'll See

At each timestamp, the debugger displays:

1. **Header** - Current timestamp and timestamp number
1. **Bars Table** - OHLCV data for all symbols at current timestamp
1. **Indicators** - Strategy indicators (if tracked) - *coming soon*
1. **Signals** - Trading signals emitted - *coming soon*
1. **Portfolio** - Current positions and equity - *coming soon*

## Inspection Levels

- **bars** - Show only OHLCV bar data (minimal output)
- **full** - Show bars + all available data
- **strategy** - Show only indicators and signals (hide bar details)

## Implementation Status

### ✅ Completed

- CLI flags: `--interactive`, `--break-at DATE`, `--inspect LEVEL`
- InteractiveDebugger class with pause/display/command logic
- Engine integration (debugger threaded through to DataService)
- DataService hooks at each timestamp
- Rich console display for bars table
- User command handling (Enter, c, q, i)

### 🚧 In Progress (Future Enhancements)

- Indicator collection from StrategyService
- Signal collection from EventBus
- Portfolio state snapshots from PortfolioService
- Enhanced state inspection and filtering

## Example Session

```
$ uv run qtrader backtest experiments/sma_crossover --interactive --break-at 2020-06-15

[...backtest initialization output...]

✓ Interactive debugging enabled
  Break at: 2020-06-15
  Inspect level: bars

[Running backtest until breakpoint...]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Timestamp #42: 2020-06-15 00:00:00
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Bars (1 symbols):
┏━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━━━━━┓
┃ Symbol ┃ Open    ┃ High    ┃ Low     ┃ Close   ┃ Volume  ┃ Timestamp    ┃
┡━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━━━━━┩
│ AAPL   │ 353.68  │ 354.77  │ 351.12  │ 353.84  │ 8234567 │ 2020-06-15   │
└────────┴─────────┴─────────┴─────────┴─────────┴─────────┴──────────────┘

[Enter]=Next  [c]=Continue  [q]=Quit  [i]=Toggle Inspect
> ▊
```

## Notes

- The debugger currently displays bar data only
- Indicators, signals, and portfolio will be added in future iterations
- The debugger integrates seamlessly with existing replay_speed configuration
- Use `--break-at` to skip warmup periods and jump to interesting dates
