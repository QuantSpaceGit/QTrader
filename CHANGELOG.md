# Changelog

All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog, and this project adheres to Semantic Versioning (pre-release identifiers included).

## [0.2.0-beta.5] - 2026-01-08

### Changed

- **Reporting Output**: Refactored timeseries and chart data exports from Parquet/CSV to JSON format
  - Converted `equity_curve.parquet` → `equity_curve.json`
  - Converted `returns.parquet` → `returns.json`
  - Converted `trades.parquet` → `trades.json`
  - Converted `drawdowns.parquet` → `drawdowns.json`
  - Converted `timeline_{strategy}.csv` → `chart_data.json` (generic filename)
  - Renamed `run_manifest.json` → `manifest.json`
  - Browser-compatible JSON with ISO timestamp format for external visualization tools

### Added

- **Event-Triggered Debugging**: Enhanced interactive debugger with breakpoint system

  - New `--break-on EVENT` option for event-triggered pausing (e.g., `signal`, `signal:BUY`)
  - Two debugging modes: step-through (pause at every timestamp) and event-triggered (pause only on matching events)
  - Extensible breakpoint system with `BreakpointRule` ABC supporting signal filters
  - Signal intention aliases: BUY, SELL, SHORT, COVER for common trading actions
  - Portfolio state and strategy indicator display in interactive mode
  - EventBus subscription for real-time signal collection
  - 37 comprehensive tests for breakpoint rules
  - See [docs/cli/interactive.md](docs/cli/interactive.md) for usage guide

- **Interactive Debugging**: Step-through debugging for backtest development

  - `--interactive` / `-i` flag to pause execution at each timestamp
  - `--break-at DATE` option to start debugging from specific date
  - `--inspect LEVEL` option to control detail level (`bars`, `full`, or `strategy`)
  - Rich console UI with unified OHLCV bars and indicators table
  - Interactive commands: `Enter` (step), `c` (continue), `q` (quit), `i` (toggle inspect)

- **EventStore API**: Added `flush()` method to EventStore base class for consistent buffered write handling

### Fixed

- **Manager Service**: Prevent duplicate CLOSE signals from opening erroneous positions

  - Framework-level duplicate detection for full close signals (confidence ≥ 1.0)
  - Partial closes (confidence < 1.0) still allowed to accumulate
  - Prevents second CLOSE from opening opposite position when first hasn't filled yet

- **Portfolio Service**: Fixed RuntimeError during stock split processing by iterating over copy of lots list

## [0.2.0-beta.4] - 2025-12-09

### Fixed

- **Yahoo Data Updater**: Prevent import of incomplete intraday data when market is still open
  - Added post-fetch filtering in `fetch_yahoo_data()` to remove rows with dates after safe end date
  - Added validation in `merge_data()` as second layer of protection against incomplete data
  - yfinance sometimes returns incomplete intraday data despite `end_date` parameter being set
  - Ensures only complete trading day data is imported when market hours are active
  - Fixes issue where partial day data (e.g., Dec 09 with volume of 31,779 vs typical ~40M) was imported during market hours

### Added

- **Release Management**: Added make targets for GitHub release workflow
  - `make version`: Show current version from pyproject.toml
  - `make release-prepare`: Run QA checks and show release preparation checklist
  - `make release VERSION=x.y.z`: Create and push git tag for GitHub releases
  - Manual version bumping workflow for better control over semantic versioning

## [0.2.0-beta.2] - 2025-11-19

### Changed

- Bumped version to 0.2.0-beta.2.

### Documentation

- Updated scaffold `QTRADER_README.md` to reflect experiment-centric structure and enhanced CLI/data acquisition guidance.

## [0.2.0-beta.1] - 2025-11-19

### Added

- Initial prerelease tag published.

### Documentation

- Baseline project README and scaffold files.

______________________________________________________________________

Earlier versions were internal and not formally tracked.
