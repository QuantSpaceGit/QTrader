# Interactive Debugger

## Quick Start

Test the interactive debugging feature with the SMA crossover experiment:

```bash
# Basic interactive mode - pause at each timestamp
uv run qtrader backtest experiments/sma_crossover --interactive

# Start from a specific date
uv run qtrader backtest experiments/sma_crossover --interactive --break-at 2020-06-15

# Event-triggered mode - pause only on signals
uv run qtrader backtest experiments/sma_crossover --interactive --break-on signal

# Pause only on BUY signals
uv run qtrader backtest experiments/sma_crossover --interactive --break-on signal:BUY

# Combined: skip warmup, then pause on signals
uv run qtrader backtest experiments/sma_crossover --interactive --break-at 2020-06-15 --break-on signal

# Show detailed inspection (all bar data)
uv run qtrader backtest experiments/sma_crossover --interactive --inspect full

# Show only strategy-level data (indicators, signals)
uv run qtrader backtest experiments/sma_crossover --interactive --inspect strategy
```

## Debugging Modes

### Step-Through Mode (Default)

When using `--interactive` without `--break-on`, the debugger pauses at every timestamp, allowing you to step through the backtest manually:

```bash
uv run qtrader backtest experiments/sma_crossover --interactive
```

Use `--break-at DATE` to skip the warmup period and start pausing from a specific date:

```bash
uv run qtrader backtest experiments/sma_crossover --interactive --break-at 2020-06-15
```

### Event-Triggered Mode

When using `--break-on`, the debugger runs continuously and only pauses when specific events occur. This is useful for monitoring only the interesting moments without stepping through every bar:

```bash
# Pause on any signal
uv run qtrader backtest experiments/sma_crossover --interactive --break-on signal

# Pause on specific signal types
uv run qtrader backtest experiments/sma_crossover --interactive --break-on signal:BUY
uv run qtrader backtest experiments/sma_crossover --interactive --break-on signal:SELL

# Multiple breakpoints (OR logic - pauses when any triggers)
uv run qtrader backtest experiments/sma_crossover --interactive --break-on signal:BUY --break-on signal:SELL
```

**Available Signal Intentions:**

- `BUY` - Alias for OPEN_LONG (enter long position)
- `SELL` - Alias for CLOSE_LONG (exit long position)
- `SHORT` - Alias for OPEN_SHORT (enter short position)
- `COVER` - Alias for CLOSE_SHORT (exit short position)
- `OPEN_LONG`, `CLOSE_LONG`, `OPEN_SHORT`, `CLOSE_SHORT` - Explicit names

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
1. **Indicators** - Strategy indicators (if tracked)
1. **Signals** - Trading signals emitted at the current timestamp
1. **Portfolio** - Current positions and equity (in full mode)

## Inspection Levels

- **bars** - Show only OHLCV bar data (minimal output)
- **full** - Show bars + indicators + signals + portfolio
- **strategy** - Show bars + indicators + signals (hide portfolio)

## Implementation Status

### ✅ Completed

- CLI flags: `--interactive`, `--break-at DATE`, `--break-on EVENT`, `--inspect LEVEL`
- InteractiveDebugger class with pause/display/command logic
- Engine integration (debugger threaded through to DataService)
- DataService hooks at each timestamp
- Rich console display for bars, signals, indicators, and portfolio
- User command handling (Enter, c, q, i)
- Event-triggered breakpoints (signal breakpoints via EventBus subscription)
- Extensible breakpoint rule system (BreakpointRule ABC)
- Indicator collection from StrategyService
- Signal collection from EventBus
- Portfolio state display

### 🚧 Future Enhancements

- Indicator breakpoints (e.g., `--break-on indicator:RSI>70`)
- Trade/fill breakpoints (e.g., `--break-on trade`)
- Conditional expression breakpoints

## Example Session

### Step-Through Mode

```
$ uv run qtrader backtest experiments/sma_crossover --interactive --break-at 2020-06-15

[...backtest initialization output...]

✓ Interactive debugging enabled
  Mode: step-through
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
│ AAPL   │ 353.68  │ 354.77  │ 351.12  │ 353.84  │ 8.23M   │ 2020-06-15   │
└────────┴─────────┴─────────┴─────────┴─────────┴─────────┴──────────────┘

[Enter]=Next  [c]=Continue  [q]=Quit  [i]=Toggle Inspect
> ▊
```

### Event-Triggered Mode

```
$ uv run qtrader backtest experiments/sma_crossover --interactive --break-on signal:BUY

[...backtest initialization output...]

✓ Interactive debugging enabled
  Mode: event-triggered
  Breakpoints: signal:BUY
  Inspect level: bars

[Running until breakpoint triggers...]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  ⚡ Breakpoint: signal:BUY triggered
  Timestamp #87: 2020-08-03 00:00:00
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Bars (1 symbols):
┏━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━━━━━┓
┃ Symbol ┃ Open    ┃ High    ┃ Low     ┃ Close   ┃ Volume  ┃ Timestamp    ┃
┡━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━━━━━┩
│ AAPL   │ 425.04  │ 435.24  │ 424.94  │ 435.15  │ 38.52M  │ 2020-08-03   │
└────────┴─────────┴─────────┴─────────┴─────────┴─────────┴──────────────┘

Signals (1):
  ➤ AAPL: OPEN_LONG @ 435.15 | qty: 1.0 | reason: SMA crossover

[Enter]=Next  [c]=Continue  [q]=Quit  [i]=Toggle Inspect
> ▊
```

## Notes

- Use `--break-at` to skip warmup periods and jump to interesting dates
- Use `--break-on signal` to only pause when trading activity occurs
- Combine both for targeted debugging: `--break-at DATE --break-on signal`
- The debugger integrates seamlessly with existing replay_speed configuration
