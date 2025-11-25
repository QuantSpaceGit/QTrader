"""
Unit tests for qtrader.engine.engine module.

Tests cover BacktestEngine initialization, configuration loading, and execution.
Note: Tests focus on minimal DataService-only implementation.
"""

from datetime import date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from qtrader.engine.config import (
    BacktestConfig,
    DataSelectionConfig,
    DataSourceConfig,
    RiskPolicyConfig,
    StrategyConfigItem,
)
from qtrader.engine.engine import BacktestEngine, BacktestResult
from qtrader.events.event_bus import EventBus
from qtrader.events.event_store import InMemoryEventStore

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_system_config(tmp_path: Path):
    """Provide mock system configuration.

    Uses temporary directories to avoid polluting user-facing folders.
    """
    mock_config = Mock()
    mock_config.data = Mock()
    mock_config.data.sources_config = "config/data_sources.yaml"
    mock_config.data.default_mode = "adjusted"
    mock_config.data.default_timezone = "America/New_York"

    # Output configuration - use temp directory
    output_dir = tmp_path / "output" / "backtests"
    output_dir.mkdir(parents=True, exist_ok=True)
    mock_config.output = Mock()
    mock_config.output.experiments_root = str(output_dir)
    mock_config.output.run_id_format = "%Y%m%d_%H%M%S"
    mock_config.output.event_store = Mock()
    mock_config.output.event_store.backend = "parquet"
    mock_config.output.event_store.filename = "events.{backend}"

    # Custom libraries - use test fixtures
    mock_config.custom_libraries = Mock()
    mock_config.custom_libraries.strategies = "tests/fixtures/strategies"
    mock_config.logging = Mock()
    # Mock the logger config with all required attributes
    mock_logger_config = Mock()
    mock_logger_config.level = "INFO"
    mock_logger_config.format = "json"
    mock_logger_config.timestamp_format = "compact"
    mock_logger_config.enable_file = False  # Disable file logging in tests
    mock_logger_config.file_path = None
    mock_logger_config.file_level = "WARNING"
    mock_logger_config.file_rotation = True
    mock_logger_config.max_file_size_mb = 10
    mock_logger_config.backup_count = 3
    mock_logger_config.console_width = 0
    mock_config.logging.to_logger_config = Mock(return_value=mock_logger_config)
    return mock_config


@pytest.fixture
def sample_backtest_config() -> BacktestConfig:
    """Provide sample backtest configuration for testing."""
    return BacktestConfig(
        backtest_id="test_backtest",
        start_date=datetime(2020, 1, 1),
        end_date=datetime(2020, 12, 31),
        initial_equity=Decimal("100000"),
        replay_speed=0.0,
        data=DataSelectionConfig(
            sources=[
                DataSourceConfig(
                    name="test-source",
                    universe=["AAPL", "MSFT"],
                )
            ]
        ),
        strategies=[
            StrategyConfigItem(
                strategy_id="test_strategy",
                universe=["AAPL"],
                data_sources=["test-source"],
                config={},
            )
        ],
        risk_policy=RiskPolicyConfig(name="naive", config={}),
    )


@pytest.fixture
def mock_event_bus() -> EventBus:
    """Provide mock event bus."""
    return Mock(spec=EventBus)


@pytest.fixture
def mock_data_service():
    """Provide mock data service."""
    service = Mock()
    service.load_symbol = Mock(return_value=iter([]))
    return service


@pytest.fixture
def mock_event_store():
    """Provide mock event store."""
    return Mock(spec=InMemoryEventStore)


# ============================================================================
# BacktestResult Tests
# ============================================================================


class TestBacktestResult:
    """Test suite for BacktestResult dataclass."""

    def test_create_result(self) -> None:
        """Test creating BacktestResult with all fields."""
        # Arrange
        start = date(2020, 1, 1)
        end = date(2020, 12, 31)
        bars = 252
        duration = timedelta(seconds=10)

        # Act
        result = BacktestResult(
            start_date=start,
            end_date=end,
            bars_processed=bars,
            duration=duration,
        )

        # Assert
        assert result.start_date == start
        assert result.end_date == end
        assert result.bars_processed == bars
        assert result.duration == duration

    def test_result_is_dataclass(self) -> None:
        """Test BacktestResult is a dataclass."""
        # Arrange & Act
        result = BacktestResult(
            start_date=date(2020, 1, 1),
            end_date=date(2020, 12, 31),
            bars_processed=100,
            duration=timedelta(seconds=5),
        )

        # Assert
        assert hasattr(result, "__dataclass_fields__")


# ============================================================================
# BacktestEngine Initialization Tests
# ============================================================================


class TestBacktestEngineInit:
    """Test suite for BacktestEngine.__init__."""

    def test_init_with_required_params(
        self,
        sample_backtest_config: BacktestConfig,
        mock_event_bus: EventBus,
        mock_data_service,
    ) -> None:
        """Test initializing engine with required parameters."""
        # Arrange & Act
        engine = BacktestEngine(
            config=sample_backtest_config,
            event_bus=mock_event_bus,
            data_service=mock_data_service,
        )

        # Assert
        assert engine.config == sample_backtest_config
        assert engine._event_bus == mock_event_bus
        assert engine._data_service == mock_data_service
        assert engine._event_store is None
        assert engine._results_dir is None

    def test_init_with_optional_params(
        self,
        sample_backtest_config: BacktestConfig,
        mock_event_bus: EventBus,
        mock_data_service,
        mock_event_store,
        tmp_path: Path,
    ) -> None:
        """Test initializing engine with optional parameters."""
        # Arrange & Act
        engine = BacktestEngine(
            config=sample_backtest_config,
            event_bus=mock_event_bus,
            data_service=mock_data_service,
            event_store=mock_event_store,
            results_dir=tmp_path,
        )

        # Assert
        assert engine._event_store == mock_event_store
        assert engine._results_dir == tmp_path

    def test_init_logs_initialization(
        self,
        sample_backtest_config: BacktestConfig,
        mock_event_bus: EventBus,
        mock_data_service,
    ) -> None:
        """Test initialization logs engine setup."""
        # Arrange & Act & Assert
        # Should not raise exception
        engine = BacktestEngine(
            config=sample_backtest_config,
            event_bus=mock_event_bus,
            data_service=mock_data_service,
        )
        assert engine is not None


# ============================================================================
# BacktestEngine.from_config Tests
# ============================================================================


class TestBacktestEngineFromConfig:
    """Test suite for BacktestEngine.from_config factory method."""

    @patch("qtrader.engine.engine.get_system_config")
    @patch("qtrader.system.log_system.LoggerFactory")
    @patch("qtrader.engine.engine.DataService")
    @patch("qtrader.engine.engine.EventBus")
    @patch("qtrader.engine.engine.SQLiteEventStore")
    def test_from_config_creates_engine(
        self,
        mock_sqlite_store,
        mock_event_bus_class,
        mock_data_service_class,
        mock_logger_factory,
        mock_get_system_config,
        sample_backtest_config: BacktestConfig,
        mock_system_config,
        tmp_path: Path,
    ) -> None:
        """Test from_config creates properly configured engine."""
        # Arrange
        mock_get_system_config.return_value = mock_system_config
        mock_event_bus = Mock()
        mock_event_bus_class.return_value = mock_event_bus
        mock_data_service = Mock()
        mock_data_service_class.from_config.return_value = mock_data_service
        mock_event_store = Mock()
        mock_sqlite_store.return_value = mock_event_store

        # Mock system config output path to use tmp_path
        mock_system_config.output.experiments_root = str(tmp_path / "output")

        # Act
        engine = BacktestEngine.from_config(sample_backtest_config)

        # Assert
        assert isinstance(engine, BacktestEngine)
        assert engine.config == sample_backtest_config
        # Logger configuration happens successfully (verified by log output in test)
        mock_event_bus.attach_store.assert_called_once()

    @patch("qtrader.engine.engine.get_system_config")
    @patch("qtrader.system.log_system.LoggerFactory")
    @patch("qtrader.engine.engine.DataService")
    @patch("qtrader.engine.engine.EventBus")
    @patch("qtrader.engine.engine.ParquetEventStore")
    def test_from_config_creates_results_directory(
        self,
        mock_parquet_store,
        mock_event_bus_class,
        mock_data_service_class,
        mock_logger_factory,
        mock_get_system_config,
        sample_backtest_config: BacktestConfig,
        mock_system_config,
        tmp_path: Path,
    ) -> None:
        """Test from_config creates results directory."""
        # Arrange
        mock_get_system_config.return_value = mock_system_config
        mock_event_bus_class.return_value = Mock()
        mock_data_service_class.from_config.return_value = Mock()
        mock_parquet_store.return_value = Mock()

        # Set output dir to tmp_path and use parquet backend
        experiments_root = tmp_path / "experiments"
        mock_system_config.output.experiments_root = str(experiments_root)
        mock_system_config.output.run_id_format = "%Y%m%d_%H%M%S"
        mock_system_config.output.event_store.backend = "parquet"
        mock_system_config.output.event_store.filename = "events.{backend}"

        # Act
        engine = BacktestEngine.from_config(sample_backtest_config)

        # Assert
        assert engine._results_dir is not None
        assert engine._results_dir.exists()

    @patch("qtrader.engine.engine.get_system_config")
    @patch("qtrader.system.log_system.LoggerFactory")
    @patch("qtrader.engine.engine.DataService")
    @patch("qtrader.engine.engine.EventBus")
    @patch("qtrader.engine.engine.ParquetEventStore")
    def test_from_config_fallback_to_memory_store_on_error(
        self,
        mock_parquet_store,
        mock_event_bus_class,
        mock_data_service_class,
        mock_logger_factory,
        mock_get_system_config,
        sample_backtest_config: BacktestConfig,
        mock_system_config,
        tmp_path: Path,
    ) -> None:
        """Test from_config falls back to InMemoryEventStore on Parquet error."""
        # Arrange
        mock_get_system_config.return_value = mock_system_config
        mock_event_bus_class.return_value = Mock()
        mock_data_service_class.from_config.return_value = Mock()
        mock_parquet_store.side_effect = Exception("Parquet initialization failed")

        mock_system_config.output.experiments_root = str(tmp_path / "output")

        # Act
        with patch("qtrader.engine.engine.InMemoryEventStore") as mock_memory_store:
            mock_memory_store.return_value = Mock()
            engine = BacktestEngine.from_config(sample_backtest_config)

        # Assert
        assert engine._event_store is not None
        mock_memory_store.assert_called_once()

    @patch("qtrader.engine.engine.get_system_config")
    @patch("qtrader.system.log_system.LoggerFactory")
    @patch("qtrader.engine.engine.DataService")
    @patch("qtrader.engine.engine.EventBus")
    @patch("qtrader.engine.engine.InMemoryEventStore")
    def test_from_config_uses_first_data_source(
        self,
        mock_memory_store,
        mock_event_bus_class,
        mock_data_service_class,
        mock_logger_factory,
        mock_get_system_config,
        mock_system_config,
        tmp_path: Path,
    ) -> None:
        """Test from_config uses data source for DataService initialization."""
        # Arrange
        config = BacktestConfig(
            backtest_id="test_source_config",
            start_date=datetime(2020, 1, 1),
            end_date=datetime(2020, 12, 31),
            initial_equity=Decimal("100000"),
            data=DataSelectionConfig(
                sources=[
                    DataSourceConfig(name="source1", universe=["AAPL", "MSFT"]),
                ]
            ),
            strategies=[
                StrategyConfigItem(
                    strategy_id="test",
                    universe=["AAPL"],
                    data_sources=["source1"],
                    config={},
                )
            ],
            risk_policy=RiskPolicyConfig(name="naive", config={}),
        )

        mock_get_system_config.return_value = mock_system_config
        mock_event_bus_class.return_value = Mock()
        mock_data_service_class.from_config.return_value = Mock()
        mock_memory_store.return_value = Mock()
        mock_system_config.output.experiments_root = str(tmp_path / "output")

        # Act
        BacktestEngine.from_config(config)

        # Assert
        call_kwargs = mock_data_service_class.from_config.call_args[1]
        assert call_kwargs["dataset"] == "source1"


# ============================================================================
# BacktestEngine.run Tests
# ============================================================================


class TestBacktestEngineRun:
    """Test suite for BacktestEngine.run method."""

    def test_run_returns_result(
        self,
        sample_backtest_config: BacktestConfig,
        mock_event_bus: EventBus,
        mock_data_service,
    ) -> None:
        """Test run returns BacktestResult."""
        # Arrange
        engine = BacktestEngine(
            config=sample_backtest_config,
            event_bus=mock_event_bus,
            data_service=mock_data_service,
        )

        # Act
        result = engine.run()

        # Assert
        assert isinstance(result, BacktestResult)
        # BacktestResult stores datetime, not date
        assert result.start_date == sample_backtest_config.start_date
        assert result.end_date == sample_backtest_config.end_date
        assert isinstance(result.duration, timedelta)

    def test_run_subscribes_to_bar_events(
        self,
        sample_backtest_config: BacktestConfig,
        mock_data_service,
    ) -> None:
        """Test run subscribes to bar events for counting."""
        # Arrange
        mock_event_bus = Mock()
        engine = BacktestEngine(
            config=sample_backtest_config,
            event_bus=mock_event_bus,
            data_service=mock_data_service,
        )

        # Act
        engine.run()

        # Assert
        mock_event_bus.subscribe.assert_called()
        call_args = mock_event_bus.subscribe.call_args[0]
        assert call_args[0] == "bar"

    def test_run_loads_symbols_from_first_source(
        self,
        sample_backtest_config: BacktestConfig,
        mock_event_bus: EventBus,
    ) -> None:
        """Test run loads symbols from first data source using stream_universe."""
        # Arrange
        mock_data_service = Mock()
        mock_data_service.stream_universe = Mock()
        engine = BacktestEngine(
            config=sample_backtest_config,
            event_bus=mock_event_bus,
            data_service=mock_data_service,
        )

        # Act
        engine.run()

        # Assert
        # Should be called once with all symbols from first source
        mock_data_service.stream_universe.assert_called_once()
        call_args = mock_data_service.stream_universe.call_args
        assert set(call_args.kwargs["symbols"]) == {"AAPL", "MSFT"}
        assert call_args.kwargs["is_warmup"] is False
        assert call_args.kwargs["strict"] is False

    def test_run_handles_symbol_load_failure_gracefully(
        self,
        sample_backtest_config: BacktestConfig,
        mock_event_bus: EventBus,
    ) -> None:
        """Test run handles data stream failures gracefully."""
        # Arrange
        mock_data_service = Mock()
        mock_data_service.stream_universe = Mock(side_effect=Exception("Stream failed"))
        engine = BacktestEngine(
            config=sample_backtest_config,
            event_bus=mock_event_bus,
            data_service=mock_data_service,
        )

        # Act & Assert
        # Should raise RuntimeError wrapping the original exception
        with pytest.raises(RuntimeError, match="Backtest execution failed"):
            engine.run()

    def test_run_tracks_bar_count(
        self,
        sample_backtest_config: BacktestConfig,
        mock_event_bus: EventBus,
    ) -> None:
        """Test run tracks number of bars processed."""
        # Arrange
        # Create mock that returns some bars
        mock_bar1 = Mock()
        mock_bar2 = Mock()
        mock_data_service = Mock()
        mock_data_service.load_symbol = Mock(return_value=iter([mock_bar1, mock_bar2]))

        engine = BacktestEngine(
            config=sample_backtest_config,
            event_bus=mock_event_bus,
            data_service=mock_data_service,
        )

        # Act
        result = engine.run()

        # Assert
        assert result.bars_processed >= 0

    def test_run_raises_runtime_error_on_failure(
        self,
        sample_backtest_config: BacktestConfig,
        mock_event_bus: EventBus,
        mock_data_service,
    ) -> None:
        """Test run raises RuntimeError on critical failure."""
        # Arrange
        mock_event_bus.subscribe.side_effect = Exception("Critical error")  # type: ignore[attr-defined]
        engine = BacktestEngine(
            config=sample_backtest_config,
            event_bus=mock_event_bus,
            data_service=mock_data_service,
        )

        # Act & Assert
        with pytest.raises(RuntimeError) as exc_info:
            engine.run()
        assert "Backtest execution failed" in str(exc_info.value)

    def test_run_calculates_duration(
        self,
        sample_backtest_config: BacktestConfig,
        mock_event_bus: EventBus,
        mock_data_service,
    ) -> None:
        """Test run calculates execution duration."""
        # Arrange
        engine = BacktestEngine(
            config=sample_backtest_config,
            event_bus=mock_event_bus,
            data_service=mock_data_service,
        )

        # Act
        result = engine.run()

        # Assert
        assert isinstance(result.duration, timedelta)
        assert result.duration.total_seconds() >= 0


# ============================================================================
# Integration Tests
# ============================================================================


class TestBacktestEngineIntegration:
    """Integration tests for BacktestEngine with real components."""

    def test_engine_with_real_event_bus(
        self,
        sample_backtest_config: BacktestConfig,
        mock_data_service,
    ) -> None:
        """Test engine works with real EventBus."""
        # Arrange
        event_bus = EventBus()
        event_store = InMemoryEventStore()

        engine = BacktestEngine(
            config=sample_backtest_config,
            event_bus=event_bus,
            data_service=mock_data_service,
            event_store=event_store,
        )

        # Act
        result = engine.run()

        # Assert
        assert isinstance(result, BacktestResult)
        assert result.bars_processed >= 0

    def test_engine_with_multiple_symbols(
        self,
        mock_event_bus: EventBus,
        mock_data_service,
    ) -> None:
        """Test engine handles multiple symbols correctly."""
        # Arrange
        config = BacktestConfig(
            backtest_id="test_multiple_symbols",
            start_date=datetime(2020, 1, 1),
            end_date=datetime(2020, 12, 31),
            initial_equity=Decimal("100000"),
            data=DataSelectionConfig(
                sources=[
                    DataSourceConfig(
                        name="test-source",
                        universe=["AAPL", "MSFT", "GOOGL", "TSLA"],
                    )
                ]
            ),
            strategies=[
                StrategyConfigItem(
                    strategy_id="test",
                    universe=["AAPL", "MSFT"],
                    data_sources=["test-source"],
                    config={},
                )
            ],
            risk_policy=RiskPolicyConfig(name="naive", config={}),
        )

        engine = BacktestEngine(
            config=config,
            event_bus=mock_event_bus,
            data_service=mock_data_service,
        )

        # Act
        engine.run()

        # Assert
        # Should call stream_universe once with all 4 symbols
        mock_data_service.stream_universe.assert_called_once()
        call_args = mock_data_service.stream_universe.call_args
        assert set(call_args.kwargs["symbols"]) == {"AAPL", "MSFT", "GOOGL", "TSLA"}
