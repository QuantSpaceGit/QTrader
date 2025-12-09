# Changelog

All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog, and this project adheres to Semantic Versioning (pre-release identifiers included).

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
