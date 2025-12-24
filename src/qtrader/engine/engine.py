"""
Backtest Engine Implementation.

Minimal implementation focused on DataService only.
Other services suspended until refactored.

Pure event-driven orchestrator that coordinates services via EventBus.
"""

import sys
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from qtrader.engine.config import BacktestConfig
from qtrader.events.event_bus import EventBus
from qtrader.events.event_store import EventStore, InMemoryEventStore, ParquetEventStore, SQLiteEventStore
from qtrader.libraries.registry import StrategyRegistry
from qtrader.services.data.service import DataService
from qtrader.services.execution.config import ExecutionConfig
from qtrader.services.execution.service import ExecutionService
from qtrader.services.manager.service import ManagerService
from qtrader.services.portfolio.models import PortfolioConfig
from qtrader.services.portfolio.service import PortfolioService
from qtrader.services.reporting.config import ReportingConfig
from qtrader.services.reporting.service import ReportingService
from qtrader.services.strategy.service import StrategyService
from qtrader.system.config import get_system_config
from qtrader.system.log_system import LoggerFactory

if TYPE_CHECKING:
    from qtrader.libraries.strategies import Strategy

logger = structlog.get_logger(__name__)


@dataclass
class BacktestResult:
    """Results from a backtest run."""

    start_date: date
    end_date: date
    bars_processed: int
    duration: timedelta


class BacktestEngine:
    """
    Event-driven backtesting orchestrator.

    Architecture: Phase 1 - DataService Foundation
    ==============================================
    This is a minimal but complete implementation focusing on the data layer.
    The engine coordinates data streaming and event publishing via EventBus,
    with full event persistence through EventStore.

    Current Capabilities:
    - Load and validate backtest configuration
    - Create and manage EventBus with EventStore persistence
    - Initialize DataService with proper dataset configuration
    - Stream historical data with timestamp synchronization
    - Track execution metrics (bars processed, duration)

    Intentional Limitations:
    - No portfolio tracking (PortfolioService suspended)
    - No order execution simulation (ExecutionService suspended)
    - No risk management (RiskService suspended)
    - No strategy signals (StrategyService suspended)

    These services will be incrementally reintegrated following the
    lego architecture pattern once refactoring is complete.

    Event Flow (Current Phase):
    ===========================
    For each timestamp T across all symbols:
        1. DataService publishes PriceBarEvent(symbol=A, timestamp=T)
        2. DataService publishes PriceBarEvent(symbol=B, timestamp=T)
        3. ...all symbols at T before advancing to T+1
        4. EventStore persists all events

    Future Event Flow (After Service Integration):
    ==============================================
    For each timestamp T:
        Phase 1: MarketData
            - DataService publishes PriceBarEvent for ALL symbols at T
            - Services update internal state (prices, positions)

        Phase 2: Valuation (barrier)
            - Engine publishes ValuationTriggerEvent(ts=T)
            - PortfolioService calculates equity, positions, valuations

        Phase 3: RiskEvaluation (barrier)
            - Engine publishes RiskEvaluationTriggerEvent(ts=T)
            - RiskService processes signals from strategies
            - RiskService creates sized orders within risk limits

        Phase 4: Execution (next cycle)
            - ExecutionService fills orders at T+1 prices
            - FillEvent updates portfolio positions

    Resource Management:
    ====================
    The engine manages lifecycle of EventStore (SQLite or in-memory).
    Call shutdown() to properly close resources, or use as context manager:

        with BacktestEngine.from_config(config) as engine:
            result = engine.run()
    """

    def __init__(
        self,
        config: BacktestConfig,
        event_bus: EventBus,
        data_service: DataService,
        strategy_service: StrategyService | None = None,
        manager_service: ManagerService | None = None,
        portfolio_service: PortfolioService | None = None,
        execution_service: ExecutionService | None = None,
        reporting_service: ReportingService | None = None,
        event_store: EventStore | None = None,
        results_dir: Path | None = None,
        debugger: Any | None = None,
    ) -> None:
        """
        Initialize backtest engine.

        Args:
            config: Backtest configuration
            event_bus: Event bus for publishing events
            data_service: Data service for loading historical bars
            strategy_service: Optional strategy service for running trading strategies
            manager_service: Optional manager service for portfolio management and risk
            portfolio_service: Optional portfolio service for position tracking
            execution_service: Optional execution service for order simulation
            reporting_service: Optional reporting service for performance metrics
            event_store: Optional persistence backend
            results_dir: Optional directory for run artifacts
            debugger: Optional interactive debugger for step-through execution
        """
        self.config = config
        self._event_bus = event_bus
        self._data_service = data_service
        self._strategy_service = strategy_service
        self._manager_service = manager_service
        self._portfolio_service = portfolio_service
        self._execution_service = execution_service
        self._reporting_service = reporting_service
        self._event_store = event_store
        self._results_dir = results_dir
        self._debugger = debugger
        self._bar_count = 0  # Initialize for tracking bars processed

        # Get all symbols from data sources
        all_symbols = config.all_symbols

        logger.info(
            "backtest.engine.initialized",
            start_date=config.start_date,
            end_date=config.end_date,
            universe_size=len(all_symbols),
            data_sources=len(config.data.sources),
            strategies=len(strategy_service._strategies) if strategy_service else 0,
            manager_enabled=manager_service is not None,
            portfolio_enabled=portfolio_service is not None,
            execution_enabled=execution_service is not None,
            reporting_enabled=reporting_service is not None,
            event_store=getattr(event_store, "__class__", type(None)).__name__,
            results_dir=str(results_dir) if results_dir else None,
        )

    @classmethod
    def from_config(
        cls, config: BacktestConfig, results_dir: Path | None = None, debugger: Any | None = None
    ) -> "BacktestEngine":
        """
        Factory method to create engine from configuration.

        Loads SystemConfig for service configurations, uses BacktestConfig
        for run parameters (dates, data sources).

        Args:
            config: Backtest configuration loaded from YAML
            results_dir: Optional directory for run artifacts (experiment run directory)
            debugger: Optional interactive debugger for step-through execution

        Returns:
            Configured BacktestEngine instance

        Raises:
            ValueError: If configuration is invalid or services fail to initialize
        """
        # Load system configuration
        system_config = get_system_config()

        # Initialize logging from system config
        LoggerFactory.configure(system_config.logging.to_logger_config())

        # Create event bus with display configuration from backtest config
        # If replay_speed < 0 treat the run as 'silent' and disable event display
        effective_display = config.display_events
        silent_mode = getattr(config, "replay_speed", 0.0) < 0.0

        if silent_mode:
            effective_display = []

            # Disable event display in logging system to suppress direct event logs
            logging_config = LoggerFactory.get_config()
            logging_config.enable_event_display = False

            # Set console logging to WARNING+ (suppress INFO/DEBUG logs)
            # File logging remains at configured level for debugging
            import logging as stdlib_logging

            console_handler = None
            for handler in stdlib_logging.getLogger().handlers:
                if isinstance(handler, stdlib_logging.StreamHandler) and handler.stream == sys.stdout:
                    console_handler = handler
                    break
            if console_handler:
                console_handler.setLevel(stdlib_logging.WARNING)

        event_bus = EventBus(display_events=effective_display)

        # Initialize event store and attach to bus for full stream persistence
        output_cfg = system_config.output
        run_started_at = datetime.now()

        event_store: EventStore
        backend_type = output_cfg.event_store.backend
        # results_dir is passed from CLI for experiment structure, or created here as fallback

        # Create event store based on configured backend
        try:
            if backend_type == "memory":
                # Memory backend - no directory or file needed
                event_store = InMemoryEventStore()
                logger.debug(
                    "backtest.engine.event_store_initialized",
                    backend="InMemoryEventStore",
                )
            else:
                # File-based backends (sqlite, parquet) - need results directory
                # NOTE: results_dir should be provided by CLI when using experiment structure
                # This is a fallback for direct engine usage without experiments
                if not results_dir:
                    experiments_root = Path(output_cfg.experiments_root)
                    backtest_id = config.sanitized_backtest_id
                    run_timestamp = run_started_at.strftime(output_cfg.run_id_format)

                    results_dir = experiments_root / backtest_id / "runs" / run_timestamp
                    results_dir.mkdir(parents=True, exist_ok=True)

                # Auto-determine filename from backend
                store_filename = output_cfg.event_store.filename
                if "{backend}" in store_filename:
                    extension_map = {"sqlite": "sqlite", "parquet": "parquet"}
                    store_filename = store_filename.replace("{backend}", extension_map[backend_type])

                store_path = results_dir / store_filename

                if backend_type == "sqlite":
                    # Ensure .sqlite extension
                    if store_path.suffix not in [".sqlite", ".db"]:
                        store_path = store_path.with_suffix(".sqlite")
                    event_store = SQLiteEventStore(store_path)
                    logger.debug(
                        "backtest.engine.event_store_initialized",
                        backend="SQLiteEventStore",
                        path=str(store_path),
                    )
                else:  # parquet
                    # Ensure .parquet extension
                    if store_path.suffix != ".parquet":
                        store_path = store_path.with_suffix(".parquet")
                    event_store = ParquetEventStore(store_path)
                    logger.debug(
                        "backtest.engine.event_store_initialized",
                        backend="ParquetEventStore",
                        path=str(store_path),
                    )
        except Exception as exc:  # pragma: no cover - defensive fallback
            logger.warning(
                "backtest.engine.event_store_fallback",
                backend="InMemoryEventStore",
                reason=str(exc),
            )
            event_store = InMemoryEventStore()

        event_bus.attach_store(event_store)

        # Create data service using factory method
        # Extract provider from dataset name (format: provider-asset-freq-variant)
        # e.g., "yahoo-us-equity-1d-csv" -> provider="yahoo"
        first_source = config.data.sources[0]
        dataset = first_source.name

        # Build config dict that from_config() expects
        # Provider will be extracted from dataset name inside from_config()
        config_dict = {
            "dataset": dataset,
        }

        data_service = DataService.from_config(
            config_dict=config_dict,
            dataset=dataset,
            event_bus=event_bus,
            system_config=system_config,
        )

        # Initialize StrategyService if strategies configured
        strategy_service: StrategyService | None = None
        if hasattr(config, "strategies") and config.strategies:
            logger.debug(
                "backtest.engine.loading_strategies",
                strategy_count=len(config.strategies),
            )

            # Get custom strategies path from system config
            strategies_loaded: dict = {}
            if system_config.custom_libraries.strategies is not None:
                custom_strategies_path = Path(system_config.custom_libraries.strategies)

                # Discover strategies using registry
                strategy_registry = StrategyRegistry()
                try:
                    strategies_loaded = strategy_registry.load_from_directory(custom_strategies_path, recursive=False)
                    logger.debug(
                        "backtest.engine.strategies_discovered",
                        discovered_count=len(strategies_loaded),
                        strategy_names=list(strategies_loaded.keys()),
                        path=str(custom_strategies_path),
                    )
                except Exception as e:
                    logger.warning(
                        "backtest.engine.strategy_discovery_failed",
                        path=str(custom_strategies_path),
                        error=str(e),
                    )
                    strategies_loaded = {}

            # Instantiate strategies from config
            strategy_instances: dict[str, "Strategy"] = {}
            for strategy_cfg in config.strategies:
                strategy_id = strategy_cfg.strategy_id

                # Get strategy class and config from registry
                try:
                    strategy_class = strategy_registry.get_strategy_class(strategy_id)
                    base_config = strategy_registry.get_strategy_config(strategy_id)

                    # Override universe and config from backtest.yaml
                    universe = strategy_cfg.universe
                    config_overrides = strategy_cfg.config  # User-provided overrides

                    # Create new config with updated universe and config overrides
                    strategy_config_dict = base_config.model_dump()
                    strategy_config_dict["universe"] = universe

                    # Apply config overrides (e.g., fast_period, slow_period)
                    strategy_config_dict.update(config_overrides)

                    strategy_config = type(base_config)(**strategy_config_dict)

                    # Instantiate strategy
                    strategy_instance = strategy_class(strategy_config)
                    strategy_instances[strategy_id] = strategy_instance

                    logger.debug(
                        "backtest.engine.strategy_instantiated",
                        strategy_id=strategy_id,
                        strategy_class=strategy_class.__name__,
                        universe=strategy_config.universe,
                        config_overrides=config_overrides if config_overrides else None,
                    )
                except Exception as e:
                    logger.error(
                        "backtest.engine.strategy_instantiation_failed",
                        strategy_id=strategy_id,
                        error=str(e),
                        error_type=type(e).__name__,
                    )

            # Create StrategyService if we have any strategies
            if strategy_instances:
                strategy_adjustment_mode = getattr(config, "strategy_adjustment_mode", "split_adjusted")
                strategy_service = StrategyService(
                    event_bus=event_bus,
                    strategies=strategy_instances,
                    adjustment_mode=strategy_adjustment_mode,
                )
                logger.debug(
                    "backtest.engine.strategy_service_created",
                    strategy_count=len(strategy_instances),
                    adjustment_mode=strategy_adjustment_mode,
                )
            else:
                logger.warning("backtest.engine.no_strategies_loaded")

        # Initialize ManagerService if risk_policy configured
        manager_service: ManagerService | None = None
        if hasattr(config, "risk_policy") and config.risk_policy:
            logger.debug(
                "backtest.engine.loading_manager_service",
                risk_policy=config.risk_policy.name,
            )

            try:
                # Build config dict for ManagerService.from_config()
                risk_config_dict = {
                    "name": config.risk_policy.name,
                    **config.risk_policy.config,  # Merge any policy overrides
                }

                manager_service = ManagerService.from_config(
                    config_dict=risk_config_dict,
                    event_bus=event_bus,
                )

                logger.debug(
                    "backtest.engine.manager_service_created",
                    risk_policy=config.risk_policy.name,
                )
            except Exception as e:
                logger.error(
                    "backtest.engine.manager_service_failed",
                    risk_policy=config.risk_policy.name,
                    error=str(e),
                    error_type=type(e).__name__,
                    exc_info=True,  # Include full traceback
                )
                # Don't fail the entire backtest if manager service fails to load
                manager_service = None

        # Initialize PortfolioService (Phase 5)
        portfolio_service: PortfolioService | None = None
        try:
            portfolio_adjustment_mode = getattr(config, "portfolio_adjustment_mode", "split_adjusted")
            portfolio_config = PortfolioConfig(
                initial_cash=Decimal(str(config.initial_equity)),
                adjustment_mode=portfolio_adjustment_mode,
            )
            portfolio_service = PortfolioService(
                config=portfolio_config,
                event_bus=event_bus,
            )
            logger.debug(
                "backtest.engine.portfolio_service_created",
                initial_equity=config.initial_equity,
                adjustment_mode=portfolio_adjustment_mode,
            )
        except Exception as e:
            logger.error(
                "backtest.engine.portfolio_service_failed",
                error=str(e),
                error_type=type(e).__name__,
            )
            portfolio_service = None

        # Initialize ExecutionService (Phase 5)
        execution_service: ExecutionService | None = None
        try:
            portfolio_adjustment_mode = getattr(config, "portfolio_adjustment_mode", "split_adjusted")
            execution_config = ExecutionConfig()  # Uses system config defaults
            execution_service = ExecutionService(
                config=execution_config,
                event_bus=event_bus,
                adjustment_mode=portfolio_adjustment_mode,
            )
            logger.debug(
                "backtest.engine.execution_service_created",
                adjustment_mode=portfolio_adjustment_mode,
            )
        except Exception as e:
            logger.error(
                "backtest.engine.execution_service_failed",
                error=str(e),
                error_type=type(e).__name__,
            )
            execution_service = None

        # Initialize ReportingService if reporting configured (optional)
        reporting_service: ReportingService | None = None
        if hasattr(config, "reporting") and config.reporting:
            # Determine output directory: use timestamped results_dir if available, else experiments root
            reporting_output_dir = results_dir if results_dir else Path(system_config.output.experiments_root)

            logger.debug(
                "backtest.engine.loading_reporting_service",
                output_dir=str(reporting_output_dir),
                detail_level=config.reporting.report_detail_level,
            )

            try:
                # Convert ReportingConfigItem to ReportingConfig
                # Note: output_dir uses timestamped results_dir to match event store location
                reporting_config = ReportingConfig(
                    emit_metrics_events=config.reporting.emit_metrics_events,
                    event_frequency=config.reporting.event_frequency,
                    risk_free_rate=Decimal(str(config.reporting.risk_free_rate)),
                    max_equity_points=config.reporting.max_equity_points,
                    write_json=config.reporting.write_json,
                    write_parquet=config.reporting.write_parquet,
                    write_csv_timeline=config.reporting.write_csv_timeline,
                    include_equity_curve=config.reporting.include_equity_curve,
                    include_returns=config.reporting.include_returns,
                    include_trades=config.reporting.include_trades,
                    include_drawdowns=config.reporting.include_drawdowns,
                    display_final_report=config.reporting.display_final_report,
                    report_detail_level=config.reporting.report_detail_level,  # type: ignore[arg-type]
                    benchmark_symbol=config.reporting.benchmark_symbol,
                    write_html_report=config.reporting.write_html_report,
                )

                reporting_service = ReportingService(
                    event_bus=event_bus,
                    config=reporting_config,
                    output_dir=reporting_output_dir,
                    event_store=event_store,  # Pass EventStore for CSV timeline export
                )

                logger.debug(
                    "backtest.engine.reporting_service_created",
                    output_dir=str(reporting_output_dir),
                    write_json=config.reporting.write_json,
                    write_parquet=config.reporting.write_parquet,
                    write_csv_timeline=config.reporting.write_csv_timeline,
                )
            except Exception as e:
                logger.error(
                    "backtest.engine.reporting_service_failed",
                    error=str(e),
                    error_type=type(e).__name__,
                    exc_info=True,
                )
                # Don't fail the entire backtest if reporting service fails to load
                reporting_service = None

        return cls(
            config=config,
            event_bus=event_bus,
            data_service=data_service,
            strategy_service=strategy_service,
            manager_service=manager_service,
            portfolio_service=portfolio_service,
            execution_service=execution_service,
            reporting_service=reporting_service,
            event_store=event_store,
            results_dir=results_dir,
            debugger=debugger,
        )

    def shutdown(self) -> None:
        """
        Clean up resources (close EventStore).

        Call this method to properly close SQLite connections and release
        file handles. Important for long-lived daemons or repeated runs.

        Examples:
            >>> engine = BacktestEngine.from_config(config)
            >>> try:
            ...     result = engine.run()
            ... finally:
            ...     engine.shutdown()
        """
        if self._event_store is not None:
            try:
                self._event_store.close()
                logger.debug("backtest.engine.event_store_closed")
            except Exception as e:
                logger.warning(
                    "backtest.engine.event_store_close_failed",
                    error=str(e),
                )

    def __enter__(self) -> "BacktestEngine":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit - ensures cleanup."""
        self.shutdown()

    def run(self) -> BacktestResult:
        """
        Run the backtest - stream data and publish events.

        Current Implementation (DataService only):
        =========================================
        1. Stream historical data for all symbols in date range
        2. DataService publishes PriceBarEvent for each bar
        3. EventStore persists all events
        4. Return basic metrics (bars processed, duration)

        Future (After Service Refactoring):
        ==================================
        - Add warmup phase for strategies
        - Publish ValuationTriggerEvent after all bars for timestamp
        - Publish RiskEvaluationTriggerEvent for signal processing
        - Collect portfolio metrics and trade statistics
        - Generate comprehensive results

        Returns:
            BacktestResult with basic metrics

        Raises:
            RuntimeError: If backtest execution fails
        """
        start_time = datetime.now()

        # Get all symbols from all data sources
        all_symbols = list(self.config.all_symbols)

        logger.info(
            "backtest.starting",
            start_date=self.config.start_date,
            end_date=self.config.end_date,
            universe_size=len(all_symbols),
            data_sources=[s.name for s in self.config.data.sources],
        )

        try:
            # Call strategy setup before main loop
            if self._strategy_service is not None:
                logger.debug("backtest.strategy_setup.starting")
                try:
                    self._strategy_service.setup()
                except Exception as e:
                    logger.error(
                        "backtest.strategy_setup.failed",
                        error=str(e),
                        error_type=type(e).__name__,
                    )
                    raise

            # Call reporting setup before main loop
            if self._reporting_service is not None:
                logger.debug("backtest.reporting_setup.starting")
                try:
                    # Extract strategy IDs from config
                    strategy_ids = (
                        [s.strategy_id for s in self.config.strategies] if hasattr(self.config, "strategies") else []
                    )

                    context = {
                        "backtest_id": self.config.sanitized_backtest_id,
                        "start_date": self.config.start_date,
                        "end_date": self.config.end_date,
                        "strategy_ids": strategy_ids,
                    }
                    self._reporting_service.setup(context)
                except Exception as e:
                    logger.warning(
                        "backtest.reporting_setup.failed",
                        error=str(e),
                        error_type=type(e).__name__,
                    )
                    # Don't fail the backtest if reporting setup fails
                    self._reporting_service = None

            # Main event loop - stream data
            logger.debug(
                "backtest.main_phase.starting",
                end_date=self.config.end_date,
                start_date=self.config.start_date,
            )

            # Reset bar count for this run
            self._bar_count = 0

            # Setup progress bar for silent mode
            progress_bar = None
            progress_task = None
            silent_mode = getattr(self.config, "replay_speed", 0.0) < 0.0

            if silent_mode:
                console = Console()
                progress_bar = Progress(
                    SpinnerColumn(),
                    TextColumn("[bold blue]{task.description}"),
                    TextColumn("•"),
                    TextColumn("{task.completed:,} bars processed"),
                    TextColumn("•"),
                    TimeElapsedColumn(),
                    console=console,
                )

                progress_bar.start()
                progress_task = progress_bar.add_task(
                    f"Running backtest [{self.config.start_date.date()} → {self.config.end_date.date()}]",
                    total=None,  # Indeterminate progress (no percentage)
                )

            # Subscribe to price_bar events to count them and update progress
            # Keep handler reference for cleanup
            def count_bars(event) -> None:
                self._bar_count += 1
                if progress_bar and progress_task is not None:
                    progress_bar.update(progress_task, advance=1)

            self._event_bus.subscribe("bar", count_bars, priority=1000)

            # Stream data from single configured source
            # Note: BacktestConfig validation enforces single source until multi-source
            # streaming is implemented (see config.py validate_single_source)
            first_source = self.config.data.sources[0]
            source_symbols = first_source.universe

            logger.debug(
                "backtest.streaming_data",
                source=first_source.name,
                symbols=source_symbols,
                start_date=self.config.start_date,
                end_date=self.config.end_date,
            )

            # Use stream_universe to ensure all symbols publish bars at each timestamp
            # before advancing (critical for cross-symbol strategies and risk barriers)
            #
            # MEMORY WARNING: Current DataService.stream_universe implementation loads
            # all bars into a timestamp_bars dict before publishing (see service.py:575-582).
            # For large universes or long date ranges, this can consume significant RAM.
            #
            # Estimated memory: ~500 bytes/bar * symbols * trading_days
            # Example: 100 symbols * 252 days * 500 bytes = ~12.6 MB (manageable)
            #          1000 symbols * 2520 days * 500 bytes = ~1.26 GB (high)
            #
            # FUTURE OPTIMIZATION: Heap-merge streaming for memory efficiency.
            # Priority: Medium - optimize when targeting 1000+ symbol universes.
            # Current approach works well for typical use cases (10-500 symbols).
            # Heap-merge would stream bars incrementally instead of buffering all in memory.
            try:
                # Give debugger access to strategy service for collecting indicators
                if self._debugger is not None and self._strategy_service is not None:
                    self._debugger.set_strategy_service(self._strategy_service)

                # Give debugger access to portfolio service for collecting portfolio state
                if self._debugger is not None and self._portfolio_service is not None:
                    self._debugger.set_portfolio_service(self._portfolio_service)

                # Give debugger access to event bus for signal subscription
                if self._debugger is not None:
                    self._debugger.set_event_bus(self._event_bus)

                self._data_service.stream_universe(
                    symbols=list(source_symbols),
                    start_date=self.config.start_date,
                    end_date=self.config.end_date,
                    is_warmup=False,
                    strict=False,  # Continue if some symbols fail to load
                    replay_speed=self.config.replay_speed,  # Use replay_speed from config
                    debugger=self._debugger,  # Pass debugger for interactive stepping
                )
            except Exception as e:
                logger.error(
                    "backtest.data_stream_failed",
                    error=str(e),
                    error_type=type(e).__name__,
                )
                raise
            finally:
                # Always unsubscribe to prevent handler accumulation on re-runs
                self._event_bus.unsubscribe("bar", count_bars)

                # Stop progress bar if active
                if progress_bar:
                    progress_bar.stop()

            logger.info(
                "backtest.main_phase.complete",
                bars_processed=self._bar_count,
            )

            # Call strategy teardown after main loop
            if self._strategy_service is not None:
                logger.info("backtest.strategy_teardown.starting")
                try:
                    self._strategy_service.teardown()
                    strategy_metrics = self._strategy_service.get_metrics()
                    logger.info("backtest.strategy_teardown.complete", metrics=strategy_metrics)
                except Exception as e:
                    logger.warning(
                        "backtest.strategy_teardown.failed",
                        error=str(e),
                        error_type=type(e).__name__,
                    )

            # Flush event store before reporting to ensure all events are written
            # This is critical for Parquet backend which buffers events in memory
            if self._event_store is not None:
                try:
                    self._event_store.flush()
                    logger.debug("backtest.engine.event_store_flushed_before_reporting")
                except Exception as e:
                    logger.warning(
                        "backtest.engine.event_store_flush_failed",
                        error=str(e),
                    )

            # Call reporting teardown after main loop
            if self._reporting_service is not None:
                logger.info("backtest.reporting_teardown.starting")
                try:
                    # Pass full config for metadata export
                    context = {
                        "backtest_id": self.config.sanitized_backtest_id,
                        "start_date": self.config.start_date,
                        "end_date": self.config.end_date,
                        "backtest_config": self.config,
                    }
                    self._reporting_service.teardown(context)
                    logger.info("backtest.reporting_teardown.complete")
                except Exception as e:
                    logger.warning(
                        "backtest.reporting_teardown.failed",
                        error=str(e),
                        error_type=type(e).__name__,
                        exc_info=True,  # Include full traceback
                    )

            # Collect results
            duration = datetime.now() - start_time

            logger.info(
                "backtest.completed",
                bars_processed=self._bar_count,
                duration_seconds=duration.total_seconds(),
            )

            return BacktestResult(
                start_date=self.config.start_date,
                end_date=self.config.end_date,
                bars_processed=self._bar_count,
                duration=duration,
            )

        except Exception as e:
            logger.error(
                "backtest.failed",
                error=str(e),
                error_type=type(e).__name__,
            )
            raise RuntimeError(f"Backtest execution failed: {e}") from e
