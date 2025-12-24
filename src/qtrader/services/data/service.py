"""Data service implementation.

Provides concrete implementation of IDataService that coordinates
data streaming from vendor adapters via EventBus.
"""

import heapq
import time
from datetime import date, datetime
from typing import Any, Dict, List, Optional, Tuple

import structlog

from qtrader.events.event_bus import IEventBus
from qtrader.services.data.adapters.resolver import DataSourceResolver
from qtrader.services.data.config import BarSchemaConfig
from qtrader.services.data.config import DataConfig
from qtrader.services.data.config import DataConfig as ServiceDataConfig
from qtrader.services.data.models import Instrument
from qtrader.services.data.source_selector import AssetClass, DataSourceSelector

logger = structlog.get_logger()


def _normalize_date(value: date) -> date:
    """Ensure datetime inputs are converted to date objects."""

    if isinstance(value, datetime):
        return value.date()
    return value


class DataService:
    """
    Concrete implementation of data service.

    Streams market data events via adapters and EventBus.
    Provides clean interface for consumers (strategies, backtests).

    Attributes:
        config: Data configuration
        resolver: Data source resolver for adapter lookup
        dataset: Dataset name (e.g., "yahoo-us-equity-1d-csv") - REQUIRED
        _instrument_cache: Cache of Instrument objects by symbol
        _event_bus: EventBus for publishing PriceBarEvent and CorporateActionEvent

    Examples:
        >>> # Initialize with config, dataset, and EventBus
        >>> from qtrader.config import AssetClass, DataSourceSelector
        >>> from qtrader.events.event_bus import EventBus
        >>> selector = DataSourceSelector(provider="yahoo", asset_class=AssetClass.EQUITY)
        >>> config = DataConfig(
        ...     mode="unadjusted",
        ...     bar_schema=bar_schema,
        ...     source_selector=selector
        ... )
        >>> bus = EventBus()
        >>> service = DataService(
        ...     config=config,
        ...     dataset="yahoo-us-equity-1d-csv",
        ...     event_bus=bus
        ... )
        >>>
        >>> # Stream single symbol (emits PriceBarEvent and CorporateActionEvent)
        >>> service.stream_bars(
        ...     "AAPL",
        ...     date(2020, 1, 1),
        ...     date(2020, 12, 31)
        ... )
        >>>
        >>> # Stream multiple symbols with time synchronization
        >>> service.stream_universe(
        ...     ["AAPL", "MSFT", "GOOGL"],
        ...     date(2020, 1, 1),
        ...     date(2020, 12, 31)
        ... )

    Notes:
        - Event-driven architecture: publishes to EventBus
        - Dataset parameter is REQUIRED (no legacy inference)
        - Metadata comes from data_sources.yaml, not dataset name parsing
        - Handles data source resolution and adapter selection via resolver
        - Caches instrument metadata for performance
    """

    def __init__(
        self,
        config: DataConfig,
        dataset: str,
        resolver: Optional[DataSourceResolver] = None,
        event_bus: Optional[IEventBus] = None,
    ):
        """
        Initialize data service.

        Args:
            config: Data configuration
            dataset: Dataset name from data_sources.yaml (e.g., "yahoo-us-equity-1d-csv").
            resolver: Data source resolver (creates default if None)
            event_bus: Optional EventBus for publishing PriceBarEvent and CorporateActionEvent.
                      If None, DataService operates in non-event mode (pull-based).

        Examples:
            >>> # With EventBus (event-driven mode)
            >>> from qtrader.events.event_bus import EventBus
            >>> bus = EventBus()
            >>> service = DataService(config, dataset="yahoo-us-equity-1d-csv", event_bus=bus)
            >>>
            >>> # With custom resolver
            >>> resolver = DataSourceResolver("config/custom_sources.yaml")
            >>> service = DataService(config, dataset="yahoo-us-equity-1d-csv", resolver=resolver)
        """
        self.config = config
        self.resolver = resolver or DataSourceResolver()
        self._event_bus = event_bus
        self.dataset = dataset

        # Validate dataset exists
        if self.dataset not in self.resolver.sources:
            available = list(self.resolver.sources.keys())
            raise ValueError(
                f"Dataset '{self.dataset}' not found in data_sources.yaml. Available datasets: {available}"
            )

        # Cache for instrument objects
        self._instrument_cache: Dict[str, Instrument] = {}

        logger.debug(
            "data_service.initialized",
            dataset=dataset,
            mode=config.mode,
        )

    @classmethod
    def from_config(
        cls,
        config_dict: dict[str, Any],
        dataset: str,
        resolver: Optional[DataSourceResolver] = None,
        event_bus: Optional[IEventBus] = None,
        timezone: Optional[str] = None,
        system_config: Optional[Any] = None,
    ) -> "DataService":
        """
        Factory method to create DataService from backtest configuration.

        Uses dataset metadata from data_sources.yaml as the source of truth.
        No inference or parsing of dataset names - all metadata comes from config.

        Args:
                        config_dict: Dict with DataConfig fields, plus optional 'dataset'
                    (e.g., "yahoo-us-equity-1d-csv")
            dataset: Dataset name from data_sources.yaml
                    (e.g., "yahoo-us-equity-1d-csv")
            resolver: Optional DataSourceResolver instance
            event_bus: Optional EventBus for event-driven operation
            timezone: Optional timezone (from system config, defaults to "America/New_York")
            system_config: Optional SystemConfig for sources_config path

        Returns:
            Configured DataService instance

        Raises:
            KeyError: If dataset not found in data_sources.yaml

        Examples:
            >>> from qtrader.backtest.config import BacktestConfig
            >>> backtest_config = BacktestConfig(...)
            >>> service = DataService.from_config(
            ...     config_dict=backtest_config.data.model_dump(),
            ...     dataset="yahoo-us-equity-1d-csv",
            ...     event_bus=event_bus
            ... )

        Note:
            Dataset metadata (provider, asset_class, adjusted, etc.) comes from
            data_sources.yaml, not from parsing the dataset name. This ensures
            consistency and allows flexible naming.
        """
        # Create resolver if not provided, using system config path if available
        if resolver is None:
            sources_config = system_config.data.sources_config if system_config else None
            resolver = DataSourceResolver(system_sources_config=sources_config)

        # Get dataset metadata from resolver (source of truth)
        if dataset not in resolver.sources:
            available = list(resolver.sources.keys())
            raise ValueError(f"Dataset '{dataset}' not found in data_sources.yaml. Available datasets: {available}")

        source_config = resolver.sources[dataset]

        # Extract metadata from dataset config
        provider = source_config.get("provider", "unknown")
        asset_class_str = source_config.get("asset_class", "equity")
        is_adjusted = source_config.get("adjusted", False)

        # Map asset_class string to enum
        asset_class = AssetClass.EQUITY  # default
        if asset_class_str == "equity":
            asset_class = AssetClass.EQUITY
        elif asset_class_str == "options":
            asset_class = AssetClass.OPTIONS
        elif asset_class_str == "futures":
            asset_class = AssetClass.FUTURES
        elif asset_class_str == "crypto":
            asset_class = AssetClass.CRYPTO
        elif asset_class_str == "forex":
            asset_class = AssetClass.FOREX
        elif asset_class_str == "fixed_income":
            asset_class = AssetClass.FIXED_INCOME

        # Determine mode from adjusted flag (boolean from config)
        mode = "adjusted" if is_adjusted else "unadjusted"

        # Create source selector
        source_selector = DataSourceSelector(
            provider=provider,
            asset_class=asset_class,
        )

        # Create minimal bar schema
        bar_schema = BarSchemaConfig(
            ts="trade_datetime",
            symbol="symbol",
            open="open",
            high="high",
            low="low",
            close="close",
            volume="volume",
        )

        # Create service-level DataConfig
        service_config = ServiceDataConfig(
            mode=mode,
            source_selector=source_selector,
            bar_schema=bar_schema,
        )

        logger.info(
            "data_service.from_config",
            dataset=dataset,
            provider=provider,
            asset_class=asset_class.value,
            adjusted=is_adjusted,
            mode=mode,
        )

        return cls(
            config=service_config,
            dataset=dataset,
            resolver=resolver,
            event_bus=event_bus,
        )

    def _create_adapter(self, symbol: str):
        """
        Create adapter for symbol using resolver.

        Args:
            symbol: Ticker symbol

        Returns:
            Configured adapter instance (type depends on dataset configuration)

        Raises:
            ValueError: If adapter configuration missing or dataset not found
        """
        # Create minimal instrument (just symbol)
        instrument = Instrument(symbol=symbol)

        # Use resolver to create appropriate adapter for the dataset
        if not self.dataset:
            raise ValueError("Dataset not configured - cannot create adapter")

        adapter = self.resolver.resolve_by_dataset(self.dataset, instrument)
        return adapter

    def stream_bars(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
        *,
        is_warmup: bool = False,
        replay_speed: float = 0.0,
    ) -> None:
        """
        Load bars and publish PriceBarEvent for each bar (event-driven mode).

        This method combines loading and event publishing for a single symbol.
        Requires EventBus to be configured during initialization.

        REFACTORED: Now directly streams from adapter without intermediate
        MultiBar/PriceSeries transformation. Emits unadjusted PriceBarEvent
        and CorporateActionEvent.

        Args:
            symbol: Ticker symbol (e.g., 'AAPL')
            start_date: Start of date range
            end_date: End of date range (inclusive)
            is_warmup: If True, publishes bars with is_warmup=True flag.
                      Strategies should NOT generate signals during warmup.
            replay_speed: Seconds to sleep between publishing each bar (0.0 = no delay).
                         Used for visualization/debugging to slow down playback.
                         Does not affect system logic, only timing of event publication.

        Raises:
            ValueError: If EventBus not configured
            ValueError: If symbol not found or invalid date range
            FileNotFoundError: If data files missing

        Examples:
            >>> # Load warmup bars (no signals)
            >>> service.stream_bars("AAPL", date(2019, 1, 1), date(2019, 12, 31), is_warmup=True)
            >>>
            >>> # Load live bars (generate signals)
            >>> service.stream_bars("AAPL", date(2020, 1, 1), date(2020, 12, 31), is_warmup=False)
            >>>
            >>> # Visualize bars at 1 second per bar
            >>> service.stream_bars("AAPL", date(2020, 1, 1), date(2020, 12, 31), replay_speed=1.0)

        Flow:
            DataService loads bar → Publishes PriceBarEvent + CorporateActionEvent →
            PortfolioService updates prices →
            ExecutionService checks fills →
            StrategyService sees bar → Publishes SignalEvent

        Note:
            This method blocks until all bars are processed. Use in BacktestEngine event loop.
        """
        if self._event_bus is None:
            raise ValueError(
                "EventBus not configured. Initialize DataService with event_bus parameter "
                "or use load_symbol() for non-event mode."
            )

        start = _normalize_date(start_date)
        end = _normalize_date(end_date)

        logger.info(
            "data_service.stream_bars",
            symbol=symbol,
            start_date=start.isoformat(),
            end_date=end.isoformat(),
            is_warmup=is_warmup,
            replay_speed=replay_speed,
        )

        # Create adapter directly (bypass loader/series/iterator layers)
        adapter = self._create_adapter(symbol)

        # Stream bars directly from adapter
        bar_count = 0
        prev_bar = None

        for bar in adapter.read_bars(start.isoformat(), end.isoformat()):
            # Publish corporate action event FIRST (if any)
            # This ensures splits/dividends are applied before portfolio marks to market
            corp_event = adapter.to_corporate_action_event(bar, prev_bar)
            if corp_event:
                self._event_bus.publish(corp_event)

            # Then publish price bar event (after corporate actions applied)
            price_event = adapter.to_price_bar_event(bar)
            self._event_bus.publish(price_event)
            bar_count += 1

            prev_bar = bar

            # Sleep for visualization (if replay_speed > 0)
            if replay_speed > 0:
                time.sleep(replay_speed)

        logger.info(
            "data_service.stream_bars.complete",
            symbol=symbol,
            bar_count=bar_count,
            is_warmup=is_warmup,
        )

    def stream_universe(
        self,
        symbols: List[str],
        start_date: date,
        end_date: date,
        *,
        is_warmup: bool = False,
        strict: bool = False,
        replay_speed: float = 0.0,
        debugger: Any | None = None,
    ) -> None:
        """
        Load bars for multiple symbols and publish PriceBarEvent for each bar.

        Synchronizes iterators to publish all bars for a given timestamp together
        before moving to next timestamp. This ensures proper event ordering.

        REFACTORED: Now directly streams from adapters without intermediate
        MultiBar/PriceSeries transformation. Emits unadjusted PriceBarEvent
        and CorporateActionEvent.

        Args:
            symbols: List of ticker symbols
            start_date: Start of date range
            end_date: End of date range (inclusive)
            is_warmup: If True, all bars published with is_warmup=True
            strict: If True, raise ValueError if any symbol fails to load
            replay_speed: Seconds to sleep after publishing bars for each timestamp (0.0 = no delay).
                         Used for visualization/debugging to slow down playback.
                         Does not affect system logic, only timing of event publication.
            debugger: Optional interactive debugger for step-through execution at each timestamp.
                      If provided, will pause after publishing bars and display state.

        Raises:
            ValueError: If EventBus not configured
            ValueError: If any symbol not found and strict=True

        Examples:
            >>> # Stream universe in warmup mode
            >>> service.stream_universe(
            ...     ["AAPL", "MSFT", "GOOGL"],
            ...     date(2019, 1, 1),
            ...     date(2019, 12, 31),
            ...     is_warmup=True
            ... )
            >>>
            >>> # Stream universe in live mode
            >>> service.stream_universe(
            ...     ["AAPL", "MSFT", "GOOGL"],
            ...     date(2020, 1, 1),
            ...     date(2020, 12, 31),
            ...     is_warmup=False
            ... )
            >>>
            >>> # Visualize universe at 1 second per timestamp
            >>> service.stream_universe(
            ...     ["AAPL", "MSFT"],
            ...     date(2020, 1, 1),
            ...     date(2020, 12, 31),
            ...     replay_speed=1.0
            ... )

        Event Ordering:
            For each timestamp T:
            1. Publish PriceBarEvent(AAPL, T)
            2. Publish PriceBarEvent(MSFT, T)
            3. Publish PriceBarEvent(GOOGL, T)
            4. Move to next timestamp T+1

        Note:
            Bars are aligned by timestamp. If a symbol is missing data for a timestamp,
            that symbol is skipped for that timestamp.
        """
        if self._event_bus is None:
            raise ValueError(
                "EventBus not configured. Initialize DataService with event_bus parameter "
                "or use load_universe() for non-event mode."
            )

        # DEBUG: Streaming details
        start = _normalize_date(start_date)
        end = _normalize_date(end_date)

        logger.debug(
            "data_service.stream_universe",
            symbol_count=len(symbols),
            start_date=start.isoformat(),
            end_date=end.isoformat(),
            is_warmup=is_warmup,
            replay_speed=replay_speed,
        )

        # Create adapters for all symbols

        adapters: Dict[str, Any] = {}
        failed_symbols: List[str] = []

        for symbol in symbols:
            try:
                adapter = self._create_adapter(symbol)
                adapters[symbol] = adapter
            except (ValueError, FileNotFoundError) as e:
                logger.warning("data_service.stream_universe.adapter_failed", symbol=symbol, error=str(e))
                failed_symbols.append(symbol)

        if failed_symbols and strict:
            raise ValueError(
                f"Failed to create adapters for {len(failed_symbols)} symbol(s) in strict mode: {failed_symbols}"
            )

        # Initialize heap with first bar from each symbol's adapter
        # Heap entries: (timestamp, symbol, bar, bar_iterator, prev_bar)
        heap: List[Tuple[datetime, str, Any, Any, Optional[Any]]] = []

        for symbol, adapter in adapters.items():
            try:
                # Create iterator once and reuse it
                it = iter(adapter.read_bars(start.isoformat(), end.isoformat()))
                first_bar = next(it)
                ts = adapter.get_timestamp(first_bar)
                heapq.heappush(heap, (ts, symbol, first_bar, it, None))
            except StopIteration:
                # Symbol has no data in range
                logger.debug("data_service.stream_universe.no_data", symbol=symbol)
                continue
            except Exception as e:
                logger.warning("data_service.stream_universe.read_failed", symbol=symbol, error=str(e))
                continue

        # Stream bars incrementally using heap-merge
        total_bars = 0
        unique_timestamps = 0
        current_timestamp = None
        bars_at_current_ts: Dict[str, Tuple[Any, Optional[Any]]] = {}

        while heap:
            # Get next bar (earliest timestamp)
            ts, symbol, bar, bar_iterator, prev_bar = heapq.heappop(heap)

            # If we've moved to a new timestamp, publish all bars from previous timestamp
            if current_timestamp is not None and ts != current_timestamp:
                # Show debugger header BEFORE publishing events (step-through mode only)
                if debugger is not None and debugger.should_pause_before_events(current_timestamp):
                    debugger.show_header(current_timestamp)

                # Publish all bars at current_timestamp in sorted symbol order
                for sym in sorted(bars_at_current_ts.keys()):
                    bar_data, prev_data = bars_at_current_ts[sym]

                    # Get the adapter for this symbol to use mapping methods
                    adapter = adapters[sym]

                    # Publish corporate action event FIRST (if any)
                    corp_event = adapter.to_corporate_action_event(bar_data, prev_data)
                    if corp_event:
                        self._event_bus.publish(corp_event)

                    # Then publish price bar event (after corporate actions applied)
                    price_event = adapter.to_price_bar_event(bar_data)
                    self._event_bus.publish(price_event)
                    total_bars += 1

                # Sleep for visualization (if replay_speed > 0) after publishing all bars for this timestamp
                if replay_speed > 0:
                    time.sleep(replay_speed)

                # Interactive debugger display and wait (if enabled)
                # Always call on_timestamp so event-triggered mode can evaluate breakpoints
                if debugger is not None and debugger.enabled:
                    # Convert bars to dict format for display
                    bars_dict = {}
                    for sym in bars_at_current_ts.keys():
                        bar_data, _ = bars_at_current_ts[sym]
                        adapter = adapters[sym]
                        # Convert raw bar to PriceBarEvent for consistent display
                        price_event = adapter.to_price_bar_event(bar_data)
                        bars_dict[sym] = price_event

                    # Debugger decides internally whether to actually pause
                    debugger.on_timestamp(
                        timestamp=current_timestamp,
                        bars=bars_dict,
                        indicators={},
                        signals=[],
                        portfolio=None,
                    )

                unique_timestamps += 1
                bars_at_current_ts = {}

            # Collect bar for current timestamp
            current_timestamp = ts
            bars_at_current_ts[symbol] = (bar, prev_bar)

            # Try to get next bar from this symbol's iterator
            try:
                next_bar = next(bar_iterator)
                next_ts = adapters[symbol].get_timestamp(next_bar)
                heapq.heappush(heap, (next_ts, symbol, next_bar, bar_iterator, bar))
            except StopIteration:
                # This symbol's iterator is exhausted
                pass

        # Publish remaining bars from last timestamp
        if bars_at_current_ts and current_timestamp is not None:
            # Show debugger header BEFORE publishing final events (step-through mode only)
            if debugger is not None and debugger.should_pause_before_events(current_timestamp):
                debugger.show_header(current_timestamp)

            for sym in sorted(bars_at_current_ts.keys()):
                bar_data, prev_data = bars_at_current_ts[sym]

                # Get the adapter for this symbol
                adapter = adapters[sym]

                # Publish corporate action event FIRST (if any)
                corp_event = adapter.to_corporate_action_event(bar_data, prev_data)
                if corp_event:
                    self._event_bus.publish(corp_event)

                # Then publish price bar event (after corporate actions applied)
                price_event = adapter.to_price_bar_event(bar_data)
                self._event_bus.publish(price_event)
                total_bars += 1

            # Sleep for visualization (if replay_speed > 0) after publishing final timestamp
            if replay_speed > 0:
                time.sleep(replay_speed)

            # Interactive debugger display and wait for final timestamp (if enabled)
            # Always call on_timestamp so event-triggered mode can evaluate breakpoints
            if debugger is not None and debugger.enabled:
                # Convert bars to dict format for display
                bars_dict = {}
                for sym in bars_at_current_ts.keys():
                    bar_data, _ = bars_at_current_ts[sym]
                    adapter = adapters[sym]
                    # Convert raw bar to PriceBarEvent for consistent display
                    price_event = adapter.to_price_bar_event(bar_data)
                    bars_dict[sym] = price_event

                # Debugger decides internally whether to actually pause
                debugger.on_timestamp(
                    timestamp=current_timestamp,
                    bars=bars_dict,
                    indicators={},
                    signals=[],
                    portfolio=None,
                )

            unique_timestamps += 1

        # INFO: High-level streaming summary for users
        logger.info(
            "data_service.stream_universe.complete",
            symbol_count=len(adapters),
            total_bars=total_bars,
            unique_timestamps=unique_timestamps,
            is_warmup=is_warmup,
        )

    def get_instrument(self, symbol: str) -> Instrument:
        """
        Get instrument for symbol.

        Creates minimal instrument (symbol only) since dataset provides
        all metadata (provider, asset type, etc.).

        Args:
            symbol: Ticker symbol

        Returns:
            Instrument with symbol

        Examples:
            >>> service = DataService(config, dataset="yahoo-us-equity-1d-csv")
            >>> instrument = service.get_instrument("AAPL")
            >>> print(instrument.symbol)  # 'AAPL'
        """
        # Check cache first
        if symbol in self._instrument_cache:
            return self._instrument_cache[symbol]

        # Build minimal instrument (symbol only)
        # Dataset provides all metadata (no duplication)
        instrument = Instrument(symbol=symbol)

        # Cache it
        self._instrument_cache[symbol] = instrument

        logger.debug(
            "data_service.get_instrument",
            symbol=symbol,
            dataset=self.dataset,
        )

        return instrument

    def list_available_symbols(
        self,
        data_source: Optional[str] = None,
    ) -> List[str]:
        """
        Security master metadata (identifier, name, exchange, etc.).
        Format depends on the data vendor's schema.        Args:
            data_source: Filter by data source (currently unused, reserved for future multi-source support)

        Returns:
            List of available symbols (sorted)

        Raises:
            FileNotFoundError: If symbol_map file not found
            ValueError: If symbol_map not configured in adapter

        Examples:
            >>> symbols = service.list_available_symbols()
            >>> print(f"Found {len(symbols)} symbols")
            >>> print(symbols[:5])  # First 5 symbols
        """
        logger.info(
            "data_service.list_available_symbols",
            data_source=data_source,
        )

        # Get adapter config
        adapter_config = self._build_adapter_config()

        # Check if symbol_map is configured
        if "symbol_map" not in adapter_config:
            logger.error(
                "data_service.list_available_symbols.no_symbol_map",
                adapter_config_keys=list(adapter_config.keys()),
            )
            raise ValueError(
                "Symbol map not configured in adapter config. Add 'symbol_map' path to your data source configuration."
            )

        # Read symbol map file
        from pathlib import Path

        import pandas as pd

        symbol_map_path = Path(adapter_config["symbol_map"])

        if not symbol_map_path.exists():
            logger.error(
                "data_service.list_available_symbols.file_not_found",
                path=str(symbol_map_path),
            )
            raise FileNotFoundError(f"Symbol map file not found: {symbol_map_path}")

        try:
            # Read CSV
            df = pd.read_csv(symbol_map_path)

            # Extract Symbol/Tickers column (case-insensitive, try multiple variations)
            symbol_col = None
            for col in df.columns:
                col_lower = col.lower()
                if col_lower in ("symbol", "symbols", "ticker", "tickers"):
                    symbol_col = col
                    break

            if symbol_col is None:
                raise ValueError(
                    f"No symbol column found in {symbol_map_path}. "
                    f"Expected 'Symbol', 'Tickers', 'Ticker', or similar. "
                    f"Available columns: {list(df.columns)}"
                )

            # Get unique symbols - handle multi-valued cells (e.g., "AAPL,APPL")
            symbols_raw = df[symbol_col].dropna().unique()
            symbols_set: set[str] = set()

            for val in symbols_raw:
                # Handle comma-separated tickers in single cell
                if "," in str(val):
                    symbols_set.update(s.strip() for s in str(val).split(","))
                else:
                    symbols_set.add(str(val).strip())

            # Sort and return
            symbols = sorted(symbols_set)

            logger.info(
                "data_service.list_available_symbols.complete",
                symbol_count=len(symbols),
                source_file=str(symbol_map_path),
            )

            return symbols

        except Exception as e:
            logger.error(
                "data_service.list_available_symbols.error",
                error=str(e),
                path=str(symbol_map_path),
            )
            raise

    def _build_adapter_config(self) -> Dict:
        """
        Build adapter configuration dict from dataset.

        Returns:
            Configuration dict for adapter initialization

        Raises:
            ValueError: If dataset not configured or not found
        """
        if not self.dataset:
            raise ValueError("Dataset not configured - cannot build adapter config")

        try:
            config = self.resolver.get_source_config(self.dataset)
            logger.debug(
                "data_service.adapter_config_resolved",
                dataset=self.dataset,
            )
            return config.copy()
        except KeyError:
            raise ValueError(f"Dataset not found in data_sources.yaml: {self.dataset}")

    def get_corporate_actions(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
        *,
        publish_events: bool = True,
    ) -> list:
        """
        Get corporate actions for symbol in date range.

        Returns events in chronological order.
        Empty list if data source doesn't provide corp actions.

        Args:
            symbol: Ticker symbol
            start_date: Start date (inclusive)
            end_date: End date (inclusive)
            publish_events: If True and EventBus configured, publishes CorporateActionEvent
                          for each action. Default True.

        Returns:
            List of CorporateActionEvent

        Examples:
            >>> # Get actions without publishing (pull mode)
            >>> actions = service.get_corporate_actions(
            ...     "AAPL",
            ...     date(2020, 1, 1),
            ...     date(2020, 12, 31),
            ...     publish_events=False
            ... )
            >>> for action in actions:
            ...     if action.action_type == "dividend":
            ...         print(f"Dividend: ${action.dividend_amount}")
            ...     elif action.action_type == "split":
            ...         print(f"Split: {action.split_ratio}:1")
            >>>
            >>> # Get actions with event publishing (event-driven mode)
            >>> service.get_corporate_actions("AAPL", date(2020, 1, 1), date(2020, 12, 31))
            >>> # Publishes CorporateActionEvent for each action
        """
        if start_date > end_date:
            raise ValueError(f"Invalid date range: {start_date} > {end_date}")

        logger.info(
            "data_service.get_corporate_actions",
            symbol=symbol,
            start_date=start_date.isoformat(),
            end_date=end_date.isoformat(),
            publish_events=publish_events and self._event_bus is not None,
        )

        # Create adapter for this symbol using resolver
        adapter = self._create_adapter(symbol)

        # Check if adapter supports corporate actions
        if not hasattr(adapter, "get_corporate_actions"):
            logger.warning(
                "data_service.get_corporate_actions.not_supported",
                symbol=symbol,
                adapter=type(adapter).__name__,
            )
            return []

        # Get corporate actions from adapter
        try:
            actions: list = adapter.get_corporate_actions(
                start_date.isoformat(),
                end_date.isoformat(),
            )

            # Publish events if requested and EventBus configured
            if publish_events and self._event_bus is not None:
                for action in actions:
                    # action is already a CorporateActionEvent from adapter
                    self._event_bus.publish(action)

                logger.info(
                    "data_service.get_corporate_actions.published",
                    symbol=symbol,
                    count=len(actions),
                )

            logger.info(
                "data_service.get_corporate_actions.complete",
                symbol=symbol,
                count=len(actions),
            )

            return actions

        except Exception as e:
            logger.error(
                "data_service.get_corporate_actions.error",
                symbol=symbol,
                error=str(e),
            )
            raise
