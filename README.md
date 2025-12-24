# QTrader

> Event-driven Python backtesting framework for quantitative trading strategies.

QTrader helps you design, test, and iterate on trading ideas using historical market data. It provides an extensible, typed, and composable environment focused on correctness, transparency, and reproducibility.

______________________________________________________________________

## Table of Contents

1. Intro & Philosophy
1. Architecture & Workflow (Event Driven)
1. User Guide (Getting Started)
1. CLI Reference (Essentials)
1. Extending (Strategies, Indicators, Libraries, Adapters)
1. Developer Guide (Source Layout, Testing, Quality, Principles)
1. Indicator Library Overview
1. Status & Roadmap
1. License

______________________________________________________________________

## 1. Intro & Philosophy

QTrader aims to:

- Separate concerns cleanly (data, strategy, execution, portfolio, reporting).
- Make strategy iteration fast: scaffold projects in seconds with `init-project`.
- Provide strong typing and validated configuration (pydantic) for safer refactors.
- Be transparent: everything is represented as explicit events you can inspect.
- Remain extensible: plug in custom data adapters, indicators, strategies, risk policies.

> 📦 This repository is the **package source**. End users create a *project* via `qtrader init-project`. The scaffolded project has its own structure (configs, strategies, data) distinct from this source tree.

______________________________________________________________________

## 2. Architecture & Workflow (Event Driven)

The engine processes a stream of domain events. Each service reacts deterministically and may emit new events. This enables fine-grained auditing and reproducibility.

### High-Level Components

- **Data Service**: Reads raw bars via adapters, emits `PriceBarEvent`.
- **Strategy Service**: Consumes market events, computes indicators, emits `SignalEvent` (intentions).
- **Execution Service**: Translates signals → orders, applies slippage/commission, emits `OrderEvent` / `FillEvent`.
- **Portfolio Service**: Updates positions, cash, P&L on fills.
- **Metrics/Reporting Service**: Aggregates performance and writes outputs.
- **Event Bus / Store**: Dispatches & persists ordered events for replay and inspection.

### Event Flow Diagram

```
           +------------------+
           |  Data Adapter(s) |  (CSV, API, Custom)
           +---------+--------+
                     |
                     v  PriceBarEvent
                +----+----+
                |  Bus    |  (dispatch)
                +----+----+
                     |
                     v
             +---------------+
             | Strategy Svc  |  (Indicators, context)
             +-------+-------+
                     |
             SignalEvent |
                     v
            +---------------+
            | Execution Svc |  (Routing, slippage, commission)
            +-------+-------+
                    |
             OrderEvent/FillEvent
                    v
            +---------------+
            | Portfolio Svc |  (Positions, cash, NAV)
            +-------+-------+
                    |
                 Metrics
                    v
            +---------------+
            | Reporting Svc |  (Results, artifacts)
            +---------------+
```

### Design Principles

- **Event Immutability**: Events are append-only; derived state is recomputable.
- **Typed Configuration**: Backtest & system configs validated at load time.
- **Deterministic Runs**: Same inputs → same sequence of events → same results.
- **Progressive Extension**: Start with defaults, selectively override pieces.

______________________________________________________________________

### Concept: Strategies vs Backtest Experiments

QTrader treats a **strategy** and a **backtest configuration** (experiment) as distinct layers:

- **Strategy (Code + StrategyConfig)**

  - Lives in a Python module (e.g., `library/strategies/sma_crossover.py`).
  - Pairs a `Strategy` class (PROCESS / logic) with a `StrategyConfig` object (PARAMETERS / defaults).
  - Encapsulates reusable trading logic: indicator usage, signal emission rules, warmup behavior.
  - The config supplies tunable parameters (`fast_period`, `slow_period`, etc.) plus identity metadata (`name`, `display_name`, version, description).

- **Backtest Experiment (BacktestConfig YAML)**

  - Defines a single run of one or more strategies under specific market conditions.
  - Specifies: date range, initial equity, data sources + universes, adjustment modes, risk policy, and per-strategy overrides.
  - Each entry in `strategies:` references a registered strategy by `strategy_id` and may provide a `config:` block that overrides defaults from the base `StrategyConfig` (e.g., shorter lookback, different warmup).

Why separate them?

| Layer                   | Purpose                                                        | Change Frequency | Versioning Focus                    |
| ----------------------- | -------------------------------------------------------------- | ---------------- | ----------------------------------- |
| Strategy Code           | Core trading logic                                             | Lower            | Semantic version of strategy module |
| StrategyConfig Defaults | Recommended baseline parameters                                | Medium           | Align with logic evolution          |
| Backtest YAML           | Experimental scenario (dates/universe) and parameter overrides | High             | Captured per run for provenance     |

This separation enables:

- Running MANY experiments (parameter sweeps, universe variations) against ONE well-designed strategy without duplicating code.
- Clean provenance: each run stores the exact `BacktestConfig` snapshot and effective merged strategy parameters.
- Deterministic replay: event sequence depends strictly on the experiment config + code version.
- Incremental refinement: improve strategy code while preserving historical experiment definitions.

Override Mechanics:

Step 1. Base strategy declares defaults in its `StrategyConfig` subclass.

Step 2. Backtest YAML includes:

```yaml
strategies:
  - strategy_id: sma_crossover
    universe: [AAPL, MSFT]
    data_sources: [yahoo-us-equity-1d-csv]
    config:
      name: sma_crossover_fast   # overrides display identity
      fast_period: 5             # overrides default 10
      slow_period: 15            # overrides default 20
```

Step 3. Loader merges provided fields with defaults using Pydantic validation (type safety, forbidden extras if strategy enforces).

Step 4. The resulting effective config instance is passed to the strategy runtime.

Strategic Philosophy:

- “Acquire or build a robust strategy” → treat it like a model artifact with version and documentation.
- “Run multiple experiments” → vary universes, dates, and parameter grids to understand stability, sensitivity, and regime performance.
- Optimize *experiment design* separately from *strategy implementation* for clearer research iteration.

Suggested Workflow:

- Step 1. Implement or refine a strategy (add docstring, ensure parameters have meaningful defaults).
- Step 2. Create a baseline backtest YAML (canonical scenario).
- Step 3. Duplicate YAML into variants (e.g., `sma_fast.yaml`, `sma_large_universe.yaml`) changing only experiment-specific fields.
- Step 4. Use naming convention: `backtest_id` reflects purpose (`sma_crossover_fast_2020_2023`).
- Step 5. Collect outputs from each run under `experiments/{backtest_id}/runs/{timestamp}/` (with `latest` symlink for convenience) and automate summary metrics across runs.

Outcome: You maintain a high-quality library of strategies while scaling the number of experiments safely.

______________________________________________________________________

## 3. User Guide (Getting Started)

### Installation

QTrader requires Python 3.13+. We recommend using [uv](https://docs.astral.sh/uv/) for fast, reliable package management.

#### Using uv (Recommended)

```bash
# Install latest version from GitHub
uv add git+https://github.com/QuantSpaceGit/QTrader.git

# Or install a specific release
uv add git+https://github.com/QuantSpaceGit/QTrader.git@v0.2.0-beta.3
```

#### Using pip

```bash
# Install latest version from GitHub
pip install git+https://github.com/QuantSpaceGit/QTrader.git

# Or install a specific release
pip install git+https://github.com/QuantSpaceGit/QTrader.git@v0.2.0-beta.3
```

#### Verify Installation

```bash
qtrader --version
```

### Initialize a New Project

```bash
qtrader init-project <PATH>
cd <PATH>
```

and you get:

```text
|-- QTRADER_README.md              # Scaffold-specific README for this project
|-- config                         # Global system & data-source configuration
|   |-- data_sources.yaml          # Defines available datasets/adapters
|   `-- qtrader.yaml               # Engine/system settings (execution, portfolio, paths)
|-- data                           # Local market data cache
|   |-- sample-csv                 # Tiny bundled sample dataset
|   |   |-- AAPL.csv               # Example OHLCV for AAPL
|   |   `-- README.md              # Notes about the sample data
|   `-- us-equity-yahoo-csv        # Yahoo Finance daily OHLCV store
|       |-- AAPL.csv               # Cached CSV for AAPL
|       `-- universe.json          # Symbol universe used by yahoo-update CLI
|-- experiments                    # Experiment definitions (what to backtest)
|   |-- buy_hold
|   |   |-- README.md              # Notes/documentation for this experiment
|   |   `-- buy_hold.yaml          # Canonical buy & hold experiment config
|   |-- sma_crossover
|   |   |-- README.md
|   |   `-- sma_crossover.yaml     # SMA crossover experiment config
|   |-- template
|   |   |-- README.md
|   |   `-- template.yaml          # Full configuration template to copy from
|   `-- weekly_monday_friday
|       |-- README.md
|       `-- weekly_monday_friday.yaml # Weekly entry/exit example experiment
`-- library                        # Your custom code extensions
  |-- __init__.py
  |-- adapters                     # Custom data adapters
  |   |-- README.md
  |   |-- __init__.py
  |   |-- models
  |   |   |-- __init__.py
  |   |   `-- ohlcv_csv.py         # Pydantic model for OHLCV CSV rows
  |   `-- ohlcv_csv.py             # Built-in CSV adapter implementation
  |-- indicators                   # Custom technical indicators
  |   |-- README.md
  |   `-- template.py              # Indicator template to copy
  |-- risk_policies                # Position sizing / risk rules
  |   |-- README.md
  |   `-- template.yaml            # Risk policy config template
  `-- strategies                   # Custom trading strategies
    |-- README.md
    |-- __init__.py
    |-- buy_and_hold.py            # Example buy & hold strategy
    |-- sma_crossover.py           # Example SMA crossover strategy
    `-- weekly_monday_friday.py    # Example weekday-based strategy

```

### Run a Backtest (CLI)

```bash
qtrader backtest experiments/buy_hold
qtrader backtest experiments/sma_crossover
qtrader backtest --help
```

Artifacts: `experiments/{backtest_id}/runs` (metrics, equity curve, trades, config snapshot).

### Interactive Debugging

QTrader includes an interactive debugger for step-through strategy development:

```bash
# Step-through mode: pause at each timestamp
qtrader backtest experiments/sma_crossover --interactive

# Start debugging from a specific date
qtrader backtest experiments/sma_crossover --interactive --break-at 2020-06-15

# Event-triggered mode: pause only on signals
qtrader backtest experiments/sma_crossover --interactive --break-on signal

# Pause only on BUY signals
qtrader backtest experiments/sma_crossover --interactive --break-on signal:BUY

# Combined: skip warmup, then pause on signals
qtrader backtest experiments/sma_crossover --interactive --break-at 2020-06-15 --break-on signal

# Control detail level (bars, full, strategy)
qtrader backtest experiments/sma_crossover --interactive --inspect full
```

**Debugging Modes:**

- **Step-through** (default): Pause at every timestamp for manual stepping
- **Event-triggered**: Run continuously, pause only when specific events occur (e.g., signals)

**Interactive Commands:**

- `Enter` - Step to next timestamp
- `c` - Continue without pausing
- `q` - Quit backtest
- `i` - Toggle inspection level

**Signal Breakpoint Filters:**

- `signal` - Pause on any signal
- `signal:BUY` - Pause on OPEN_LONG signals
- `signal:SELL` - Pause on CLOSE_LONG signals
- `signal:SHORT` / `signal:COVER` - Short position signals

**Display Features:**

- Unified table showing OHLCV bars with indicators as columns
- Real-time strategy state and signals at each timestamp
- Clean console output with Rich formatting

For detailed usage guide, see [docs/cli/interactive.md](docs/cli/interactive.md).

### Programmatic API

```python
from qtrader.engine import BacktestEngine
from qtrader.engine.config import BacktestConfig

config = BacktestConfig.from_yaml("experiments/buy_hold/buy_hold.yaml")
engine = BacktestEngine(config)
results = engine.run()
print(results.final_value, results.total_return)
```

### Basic CLI Surface (Core Commands)

```bash
# Run a backtest
qtrader backtest experiments/sma_crossover/

# Update Yahoo CSV data incrementally (auto symbol discovery)
qtrader data yahoo-update --days 365

# Generate component templates
qtrader init-library ./library --type strategy --type indicator

# Show data source names
qtrader data list
```

______________________________________________________________________

## 4. CLI Reference (Essentials)

### Core Commands

| Command                                                      | Purpose                                      |
| ------------------------------------------------------------ | -------------------------------------------- |
| `qtrader init-project <path>`                                | Scaffold a new backtesting project           |
| `qtrader backtest <experiment dirpath>`                      | Run a configured backtest                    |
| `qtrader data yahoo-update [--days N] [--symbols AAPL MSFT]` | Download/refresh local Yahoo OHLCV CSVs      |
| `qtrader data list`                                          | List configured data adapters/sources        |
| `qtrader init-library <path> [--type ...]`                   | Generate template code for custom components |

### Interactive Debugging Options

| Flag                   | Purpose                                             |
| ---------------------- | --------------------------------------------------- |
| `--interactive` / `-i` | Enable step-through debugging mode                  |
| `--break-at DATE`      | Start pausing from specific date (YYYY-MM-DD)       |
| `--break-on EVENT`     | Event-triggered mode (e.g., `signal`, `signal:BUY`) |
| `--inspect LEVEL`      | Set detail level: `bars`, `full`, or `strategy`     |

**Examples:**

```bash
# Step-through from a specific date
qtrader backtest experiments/sma_crossover --interactive --break-at 2020-06-15 --inspect full

# Event-triggered: pause only on BUY signals
qtrader backtest experiments/sma_crossover --interactive --break-on signal:BUY

# Combined: skip warmup, then pause on any signal
qtrader backtest experiments/sma_crossover --interactive --break-at 2020-06-15 --break-on signal
```

### Documentation References

- **Backtest Command**: [docs/packages/cli/backtest.md](docs/packages/cli/backtest.md)
- **Interactive Debugging**: [docs/cli/interactive.md](docs/cli/interactive.md)
- **Strategy Development**: [docs/packages/strategy.md](docs/packages/strategy.md)
- **Indicators**: [docs/packages/indicators/indicators.md](docs/packages/indicators/indicators.md)

______________________________________________________________________

## 5. Extending

### Strategies & Indicators

Use `qtrader init-library` to create template files then implement logic in `on_bar` / indicator calculate methods.

Minimal custom strategy example:

```python
from qtrader.libraries.strategies import Strategy, StrategyConfig
from qtrader.services.strategy.models import SignalIntention

class MyStrategyConfig(StrategyConfig):
    name: str = "my_strategy"
    sma_period: int = 20

class MyStrategy(Strategy[MyStrategyConfig]):
    def on_bar(self, event, context):
        sma = context.sma(symbol=event.symbol, period=self.config.sma_period)
        if event.close > sma and not context.has_position(event.symbol):
            context.emit_signal(symbol=event.symbol,
                                intention=SignalIntention.OPEN_LONG,
                                quantity=100)

CONFIG = MyStrategyConfig()  # Required for discovery
```

### Data Adapters

Implement the adapter protocol to load proprietary data and emit events.

```python
from qtrader.services.data.adapters.protocol import IDataAdapter

class MyAdapter(IDataAdapter):
    def read_bars(self, start_date: str, end_date: str):
        # return iterable of raw bar records
        ...
    def to_price_bar_event(self, bar):
        # convert to PriceBarEvent
        ...
```

### Custom Library Layout

```
my-qtrader-extensions/
├── strategies/
├── indicators/
├── adapters/
└── risk_policies/
```

Configure paths in `config/system.yaml` (set to `null` for built-in only):

```yaml
custom_libraries:
  strategies: "./library/strategies"
  indicators: null
  adapters: null
  risk_policies: null
```

______________________________________________________________________

## 6. Developer Guide

### Source Layout (Package Repository)

```
src/qtrader/
├── engine/      # Orchestration & backtest engine
├── services/    # data, strategy, execution, portfolio, reports
├── events/      # Event definitions & bus/store
├── libraries/   # Built-in indicators, strategies, risk policies
├── cli/         # Command-line interface
└── scaffold/    # Project & library templates distributed with package
```

### Quality & Tests

```bash
make qa            # Lint + format (ruff, isort, mdformat)
make test          # Run full test suite with coverage
```

Current internal metrics (may differ in CI): ~1600+ tests, ~80% coverage.

### Principles

- Typed configs (pydantic) reduce runtime surprises.
- Indicators support streaming & batch modes.
- Services are cohesive; cross-service communication only via events.
- Deterministic sequencing enables replay & debugging.

______________________________________________________________________

## 7. Indicator Library Overview

Categories & examples:

- **Moving Averages (7)**: SMA, EMA, WMA, DEMA, TEMA, HMA, SMMA
- **Momentum (6)**: RSI, MACD, Stochastic, CCI, ROC, Williams %R
- **Volatility (3)**: ATR, Bollinger Bands, StdDev
- **Volume (4)**: VWAP, OBV, Acc/Dist, CMF
- **Trend (2)**: ADX, Aroon

Indicators are accessible through the strategy context (e.g., `context.sma(...)`).

See: `docs/packages/indicators/README.md` for parameters & formulas.

______________________________________________________________________

______________________________________________________________________

## 8. License

MIT License. See [LICENSE](LICENSE).

______________________________________________________________________

Enjoy backtesting! If you have any questions or feedback, feel free to reach out.
