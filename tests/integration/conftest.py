"""Shared fixtures for integration tests.

Provides reusable BacktestConfig fixtures that don't depend on external files.
"""

from datetime import datetime
from decimal import Decimal
from pathlib import Path
from unittest.mock import Mock

import pytest

from qtrader.engine.config import (
    BacktestConfig,
    DataSelectionConfig,
    DataSourceConfig,
    ReportingConfigItem,
    RiskPolicyConfig,
    StrategyConfigItem,
)


@pytest.fixture
def mock_system_config(tmp_path: Path):
    """Provide mock system configuration for integration tests.

    Returns fully-configured mock that avoids file I/O during engine initialization.
    Uses temporary directories to avoid polluting user-facing folders.
    """
    mock_config = Mock()

    # Data configuration - use test fixtures
    mock_config.data = Mock()
    mock_config.data.sources_config = "tests/fixtures/config/data_sources.yaml"
    mock_config.data.default_mode = "adjusted"
    mock_config.data.default_timezone = "America/New_York"

    # Output configuration - use temp directory
    output_dir = tmp_path / "experiments"
    output_dir.mkdir(parents=True, exist_ok=True)
    mock_config.output = Mock()
    mock_config.output.experiments_root = str(output_dir)
    mock_config.output.run_id_format = "%Y%m%d_%H%M%S"
    mock_config.output.capture_git_info = False
    mock_config.output.capture_environment = False
    mock_config.output.event_store = Mock()
    mock_config.output.event_store.backend = "parquet"
    mock_config.output.event_store.filename = "events.parquet"

    # Custom libraries - use test fixtures
    mock_config.custom_libraries = Mock()
    mock_config.custom_libraries.strategies = "tests/fixtures/strategies"

    # Logging configuration
    mock_config.logging = Mock()
    mock_logger_config = Mock()
    mock_logger_config.level = "WARNING"  # Reduce noise in tests
    mock_logger_config.format = "json"
    mock_logger_config.timestamp_format = "compact"
    mock_logger_config.enable_file = False
    mock_logger_config.file_path = None
    mock_logger_config.file_level = "WARNING"
    mock_logger_config.file_rotation = False
    mock_logger_config.max_file_size_mb = 10
    mock_logger_config.backup_count = 3
    mock_logger_config.console_width = 0
    mock_config.logging.to_logger_config = Mock(return_value=mock_logger_config)

    return mock_config


@pytest.fixture
def buy_hold_backtest_config(mock_system_config) -> BacktestConfig:
    """Provide buy-and-hold backtest configuration.

    Creates minimal working config for integration tests:
    - Single symbol (AAPL)
    - Short date range (1 month)
    - Buy-and-hold strategy (from test fixtures)
    - Full reporting enabled

    Note: Uses real data from data/ directory (read-only, safe for tests).
    Uses temporary output directory from mock_system_config.
    """
    return BacktestConfig(
        backtest_id="test_buy_hold",
        start_date=datetime(2020, 8, 1),
        end_date=datetime(2020, 9, 1),
        initial_equity=Decimal("100000"),
        replay_speed=0.0,
        data=DataSelectionConfig(
            sources=[
                DataSourceConfig(
                    name="yahoo-us-equity-1d-csv",
                    universe=["AAPL"],
                )
            ]
        ),
        strategies=[
            StrategyConfigItem(
                strategy_id="buy_and_hold",
                universe=["AAPL"],
                data_sources=["yahoo-us-equity-1d-csv"],
                config={},
            )
        ],
        risk_policy=RiskPolicyConfig(name="naive", config={}),
        reporting=ReportingConfigItem(
            emit_metrics_events=False,
            event_frequency=100,
            max_equity_points=10000,
            include_trades=True,
            include_drawdowns=True,
            display_final_report=False,  # Reduce console noise
            report_detail_level="full",
            benchmark_symbol=None,
        ),
    )


@pytest.fixture
def zero_trades_backtest_config(mock_system_config) -> BacktestConfig:
    """Provide backtest configuration that generates zero trades.

    Uses a strategy that never signals (empty universe or disabled).
    Useful for testing edge case handling in reporting.
    Uses temporary output directory from mock_system_config.
    """
    return BacktestConfig(
        backtest_id="test_zero_trades",
        start_date=datetime(2020, 8, 1),
        end_date=datetime(2020, 8, 15),
        initial_equity=Decimal("100000"),
        replay_speed=0.0,
        data=DataSelectionConfig(
            sources=[
                DataSourceConfig(
                    name="yahoo-us-equity-1d-csv",
                    universe=["AAPL"],
                )
            ]
        ),
        strategies=[
            StrategyConfigItem(
                strategy_id="buy_and_hold",
                universe=[],  # Empty universe = no signals
                data_sources=["yahoo-us-equity-1d-csv"],
                config={},
            )
        ],
        risk_policy=RiskPolicyConfig(name="naive", config={}),
        reporting=ReportingConfigItem(
            emit_metrics_events=False,
            event_frequency=100,
            max_equity_points=10000,
            include_trades=True,
            include_drawdowns=True,
            display_final_report=False,
            report_detail_level="full",
            benchmark_symbol=None,
        ),
    )


@pytest.fixture
def temp_output_dir(tmp_path: Path) -> Path:
    """Provide temporary output directory for test artifacts.

    Creates clean directory for each test, automatically cleaned up after.
    """
    output_dir = tmp_path / "experiments"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir
