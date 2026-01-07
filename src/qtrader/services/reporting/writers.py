"""Output writers for performance reports and time-series data.

Handles JSON summary reports, JSON time-series exports, and CSV timeline exports.
"""

import json
from collections import defaultdict
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from structlog import get_logger

from qtrader.events.event_store import EventStore
from qtrader.events.events import (
    FillEvent,
    IndicatorEvent,
    OrderEvent,
    PerformanceMetricsEvent,
    PriceBarEvent,
    SignalEvent,
    TradeEvent,
)
from qtrader.libraries.performance.models import DrawdownPeriod, EquityCurvePoint, FullMetrics, ReturnPoint, TradeRecord

logger = get_logger(__name__)


class DecimalEncoder(json.JSONEncoder):
    """JSON encoder that handles Decimal types."""

    def default(self, o: Any) -> Any:
        """Convert Decimal to float for JSON serialization."""
        if isinstance(o, Decimal):
            return float(o)
        return super().default(o)


def write_json_report(metrics: FullMetrics, output_path: Path) -> None:
    """
    Write performance metrics to JSON file.

    Creates human-readable JSON summary report with all metrics.

    Args:
        metrics: Complete performance metrics
        output_path: Path to write JSON file

    Example:
        >>> metrics = FullMetrics(...)
        >>> write_json_report(metrics, Path("output/run_001/performance.json"))
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert to dict and write with custom encoder
    data = metrics.model_dump(mode="json")

    with output_path.open("w") as f:
        json.dump(data, f, indent=2, cls=DecimalEncoder)

    logger.info("JSON report written", path=str(output_path), size_bytes=output_path.stat().st_size)


def write_equity_curve_json(equity_curve: list[EquityCurvePoint], output_path: Path) -> None:
    """
    Write equity curve time-series to JSON.

    Creates browser-friendly JSON file with ISO timestamp format.

    Args:
        equity_curve: List of equity curve data points
        output_path: Path to write JSON file

    Example:
        >>> points = [EquityCurvePoint(...), ...]
        >>> write_equity_curve_json(points, Path("output/run_001/timeseries/equity_curve.json"))
    """
    if not equity_curve:
        logger.warning("No equity curve data to write")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert to list of dicts with ISO timestamps
    data = [
        {
            "timestamp": p.timestamp.isoformat() if hasattr(p.timestamp, "isoformat") else str(p.timestamp),
            "equity": float(p.equity),
            "cash": float(p.cash),
            "positions_value": float(p.positions_value),
            "num_positions": p.num_positions,
            "gross_exposure": float(p.gross_exposure),
            "net_exposure": float(p.net_exposure),
            "leverage": float(p.leverage),
            "drawdown_pct": float(p.drawdown_pct),
            "underwater": p.underwater,
        }
        for p in equity_curve
    ]

    # Write JSON
    with output_path.open("w") as f:
        json.dump(data, f, indent=2, cls=DecimalEncoder)

    logger.info(
        "Equity curve written",
        path=str(output_path),
        rows=len(equity_curve),
        size_bytes=output_path.stat().st_size,
    )


def write_returns_json(returns: list[ReturnPoint], output_path: Path) -> None:
    """
    Write returns time-series to JSON.

    Creates browser-friendly JSON file with period and cumulative returns.

    Args:
        returns: List of return data points
        output_path: Path to write JSON file

    Example:
        >>> returns = [ReturnPoint(...), ...]
        >>> write_returns_json(returns, Path("output/run_001/timeseries/returns.json"))
    """
    if not returns:
        logger.warning("No returns data to write")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert to list of dicts with ISO timestamps
    data = [
        {
            "timestamp": r.timestamp.isoformat() if hasattr(r.timestamp, "isoformat") else str(r.timestamp),
            "period_return": float(r.period_return),
            "cumulative_return": float(r.cumulative_return),
            "log_return": float(r.log_return),
        }
        for r in returns
    ]

    # Write JSON
    with output_path.open("w") as f:
        json.dump(data, f, indent=2, cls=DecimalEncoder)

    logger.info(
        "Returns written",
        path=str(output_path),
        rows=len(returns),
        size_bytes=output_path.stat().st_size,
    )


def write_trades_json(trades: list[TradeRecord], output_path: Path) -> None:
    """
    Write trade records to JSON.

    Creates browser-friendly JSON file with complete trade history.

    Args:
        trades: List of completed trades
        output_path: Path to write JSON file

    Example:
        >>> trades = [TradeRecord(...), ...]
        >>> write_trades_json(trades, Path("output/run_001/timeseries/trades.json"))
    """
    if not trades:
        logger.warning("No trades to write")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert to list of dicts with ISO timestamps
    data = [
        {
            "trade_id": t.trade_id,
            "strategy_id": t.strategy_id,
            "symbol": t.symbol,
            "entry_timestamp": t.entry_timestamp.isoformat()
            if hasattr(t.entry_timestamp, "isoformat")
            else str(t.entry_timestamp),
            "exit_timestamp": t.exit_timestamp.isoformat()
            if hasattr(t.exit_timestamp, "isoformat")
            else str(t.exit_timestamp),
            "entry_price": float(t.entry_price),
            "exit_price": float(t.exit_price),
            "quantity": t.quantity,
            "side": t.side,
            "pnl": float(t.pnl),
            "pnl_pct": float(t.pnl_pct),
            "commission": float(t.commission),
            "duration_seconds": t.duration_seconds,
            "duration_days": t.duration_days,
            "is_winner": t.is_winner,
        }
        for t in trades
    ]

    # Write JSON
    with output_path.open("w") as f:
        json.dump(data, f, indent=2, cls=DecimalEncoder)

    logger.info(
        "Trades written",
        path=str(output_path),
        rows=len(trades),
        size_bytes=output_path.stat().st_size,
    )


def write_drawdowns_json(drawdowns: list[DrawdownPeriod], output_path: Path) -> None:
    """
    Write drawdown periods to JSON.

    Creates browser-friendly JSON file with drawdown history.

    Args:
        drawdowns: List of drawdown periods
        output_path: Path to write JSON file

    Example:
        >>> drawdowns = [DrawdownPeriod(...), ...]
        >>> write_drawdowns_json(drawdowns, Path("output/run_001/timeseries/drawdowns.json"))
    """
    if not drawdowns:
        logger.warning("No drawdowns to write")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert to list of dicts with ISO timestamps
    data = [
        {
            "drawdown_id": d.drawdown_id,
            "start_timestamp": d.start_timestamp.isoformat()
            if hasattr(d.start_timestamp, "isoformat")
            else str(d.start_timestamp),
            "trough_timestamp": d.trough_timestamp.isoformat()
            if hasattr(d.trough_timestamp, "isoformat")
            else str(d.trough_timestamp),
            "end_timestamp": d.end_timestamp.isoformat()
            if d.end_timestamp and hasattr(d.end_timestamp, "isoformat")
            else (str(d.end_timestamp) if d.end_timestamp else None),
            "peak_equity": float(d.peak_equity),
            "trough_equity": float(d.trough_equity),
            "depth_pct": float(d.depth_pct),
            "duration_days": d.duration_days,
            "recovery_days": d.recovery_days,
            "recovered": d.recovered,
            "total_days_underwater": d.total_days_underwater,
        }
        for d in drawdowns
    ]

    # Write JSON
    with output_path.open("w") as f:
        json.dump(data, f, indent=2, cls=DecimalEncoder)

    logger.info(
        "Drawdowns written",
        path=str(output_path),
        rows=len(drawdowns),
        size_bytes=output_path.stat().st_size,
    )


def write_strategy_chart_data(
    event_store: EventStore,
    strategy_id: str,
    output_path: Path,
    start_time: datetime | None = None,
    end_time: datetime | None = None,
) -> None:
    """
    Write comprehensive chart data in JSON format for a single strategy.

    Creates a browser-friendly JSON array with "tall format":
    - One object per ticker per timestamp (OHLC data + trading events)
    - One object per indicator per timestamp (synthetic ticker with underlying reference)
    - One object per portfolio metric per timestamp (EQUITY, CASH, etc. as synthetic tickers)

    This format is:
    - Browser/JavaScript friendly (easy filtering: data.filter(d => d.ticker === 'AAPL'))
    - No dynamic fields (uniform schema regardless of number of indicators)
    - Easy to chart (filter by ticker, plot close values over time)

    **Backward Adjustment:**
    All bar prices and price-based indicators are backward-adjusted to the last bar's scale
    using cumulative_volume_factor (split-adjusted prices). This ensures all historical data
    is comparable on the same basis as the most recent bar.

    Formula: adjusted_value = unadjusted_value / (last_factor / bar_factor)

    Args:
        event_store: EventStore to query for events
        strategy_id: Strategy identifier to filter events
        output_path: Path to write JSON file
        start_time: Optional start time filter
        end_time: Optional end time filter

    Schema:
        timestamp: ISO timestamp
        strategy_id: Strategy that generated this data
        ticker: Ticker symbol (real for price data, synthetic for indicators/metrics)
        underlying: Underlying asset (ticker for prices, AAPL for SMA_fast_10, PORTFOLIO for metrics)
        open: Open price (BACKWARD ADJUSTED, or NULL for non-OHLC rows)
        high: High price (BACKWARD ADJUSTED, or NULL)
        low: Low price (BACKWARD ADJUSTED, or NULL)
        close: Close price (BACKWARD ADJUSTED, or indicator/metric value)
        volume: Volume (adjusted for splits, or NULL for non-OHLC rows)
        signal_intention: Signal intention (OPEN_LONG, CLOSE_SHORT, etc.)
        signal_price: Signal reference price
        signal_event_id: SignalEvent UUID for event store lookup
        signal_correlation_id: Workflow correlation ID (shared across Signal→Order→Fill→Trade)
        signal_causation_id: Parent event ID (NULL for Signal as it's the root)
        signal_source_service: Service that created the signal (strategy_service)
        order_id: Order ID (aggregated if multiple)
        order_qty: Order quantity (summed if multiple)
        order_event_id: OrderEvent UUID(s) for event store lookup (comma-separated if multiple)
        order_correlation_id: Workflow correlation ID (inherited from Signal)
        order_causation_id: Parent event ID (Signal.event_id)
        order_source_service: Service that created the order (manager_service)
        fill_id: Fill ID (aggregated if multiple)
        fill_qty: Fill quantity (summed if multiple fills at same timestamp)
        fill_price: Weighted average fill price (if multiple fills)
        commission: Total commission (summed if multiple fills)
        fill_event_id: FillEvent UUID(s) for event store lookup (comma-separated if multiple)
        fill_correlation_id: Workflow correlation ID (inherited from Order)
        fill_causation_id: Parent event ID (Order.event_id)
        fill_source_service: Service that created the fill (execution_service)
        trade_id: Trade ID linking entry and exit fills (NULL if position still open)

    Example Output:
        [
            {"timestamp": "2023-01-01T09:30:00+00:00", "ticker": "AAPL", "open": 150.00, "close": 151.50, ...},
            {"timestamp": "2023-01-01T09:30:00+00:00", "ticker": "SMA_fast_10", "close": 150.20, ...},
            {"timestamp": "2023-01-01T09:30:00+00:00", "ticker": "EQUITY", "close": 100000.00, ...}
        ]
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(
        "strategy_timeline.building",
        strategy_id=strategy_id,
        output_path=str(output_path),
    )

    # Query all relevant events from EventStore
    # NOTE: In backtesting, EventStore filters by occurred_at (system time) not timestamp (simulation time)
    # For events with timestamp fields, we need to get ALL events and filter by timestamp manually
    bar_events = event_store.get_by_type("bar")  # Get all, filter below
    indicator_events = event_store.get_by_type("indicator")
    signal_events = event_store.get_by_type("signal")
    order_events = event_store.get_by_type("order")
    fill_events = event_store.get_by_type("fill")
    trade_events = event_store.get_by_type("trade")  # NEW: Get trade events for trade_id mapping
    performance_events = event_store.get_by_type("performance_metrics")

    # Filter by simulation time (timestamp field) if specified
    def in_time_range(timestamp_str: str) -> bool:
        """Check if simulation timestamp is within range."""
        if start_time is None and end_time is None:
            return True
        try:
            # Parse ISO timestamp (handles both +00:00 and Z notation)
            ts = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
            if start_time and ts < start_time:
                return False
            if end_time and ts > end_time:
                return False
            return True
        except (ValueError, AttributeError):
            return False

    bar_events = [e for e in bar_events if isinstance(e, PriceBarEvent) and in_time_range(e.timestamp)]
    indicator_events = [e for e in indicator_events if isinstance(e, IndicatorEvent) and in_time_range(e.timestamp)]
    signal_events = [e for e in signal_events if isinstance(e, SignalEvent) and in_time_range(e.timestamp)]
    order_events = [e for e in order_events if isinstance(e, OrderEvent) and in_time_range(e.timestamp)]
    fill_events = [e for e in fill_events if isinstance(e, FillEvent) and in_time_range(e.timestamp)]
    performance_events = [
        e for e in performance_events if isinstance(e, PerformanceMetricsEvent) and in_time_range(e.timestamp)
    ]

    # Filter to this strategy
    indicator_events = [e for e in indicator_events if isinstance(e, IndicatorEvent) and e.strategy_id == strategy_id]
    signal_events = [e for e in signal_events if isinstance(e, SignalEvent) and e.strategy_id == strategy_id]
    # Orders and fills are already filtered by strategy via their correlation chain

    # Build fill_id → trade_id mapping and trade metadata from TradeEvents
    # TradeEvent.fills contains list of fill_id UUIDs that belong to this trade
    fill_to_trade: dict[str, str] = {}  # fill_id → trade_id
    trade_metadata: dict[str, TradeEvent] = {}  # trade_id → TradeEvent (for status, side, P&L)
    for event in trade_events:
        if isinstance(event, TradeEvent) and event.strategy_id == strategy_id:
            event_trade_id = event.trade_id
            trade_metadata[event_trade_id] = event  # Store full TradeEvent for metadata
            for fill_id in event.fills:
                fill_to_trade[fill_id] = event_trade_id

    # Normalize timestamp format (Z notation vs +00:00 notation)
    def normalize_timestamp(ts: str) -> str:
        """Normalize timestamp to consistent format for grouping."""
        # Convert Z to +00:00 for consistent grouping
        return ts.replace("Z", "+00:00")

    # Group events by timestamp for efficient lookup
    bars_by_ts_symbol: dict[tuple[str, str], PriceBarEvent] = {}  # (timestamp, symbol) -> bar
    for event in bar_events:
        if isinstance(event, PriceBarEvent):
            bars_by_ts_symbol[(normalize_timestamp(event.timestamp), event.symbol)] = event

    indicators_by_ts: dict[str, list[IndicatorEvent]] = defaultdict(list)  # timestamp -> [indicators]
    for event in indicator_events:
        if isinstance(event, IndicatorEvent):
            indicators_by_ts[normalize_timestamp(event.timestamp)].append(event)

    signals_by_ts_symbol: dict[tuple[str, str], SignalEvent] = {}  # (timestamp, symbol) -> signal
    for event in signal_events:
        if isinstance(event, SignalEvent):
            signals_by_ts_symbol[(normalize_timestamp(event.timestamp), event.symbol)] = event

    # Group orders by timestamp and symbol (may have multiple orders at same timestamp)
    orders_by_ts_symbol: dict[tuple[str, str], list[OrderEvent]] = defaultdict(list)
    for event in order_events:
        if isinstance(event, OrderEvent):
            orders_by_ts_symbol[(normalize_timestamp(event.timestamp), event.symbol)].append(event)

    # Group fills by timestamp and symbol (may have multiple fills at same timestamp)
    fills_by_ts_symbol: dict[tuple[str, str], list[FillEvent]] = defaultdict(list)
    for event in fill_events:
        if isinstance(event, FillEvent):
            fills_by_ts_symbol[(normalize_timestamp(event.timestamp), event.symbol)].append(event)

    performance_by_ts: dict[str, PerformanceMetricsEvent] = {}  # timestamp -> metrics
    for event in performance_events:
        if isinstance(event, PerformanceMetricsEvent):
            performance_by_ts[normalize_timestamp(event.timestamp)] = event

    # Collect all unique timestamps (sorted) - normalized format
    all_timestamps: set[str] = set()
    all_timestamps.update(normalize_timestamp(e.timestamp) for e in bar_events if isinstance(e, PriceBarEvent))
    all_timestamps.update(indicators_by_ts.keys())
    all_timestamps.update(normalize_timestamp(e.timestamp) for e in signal_events if isinstance(e, SignalEvent))
    all_timestamps.update(normalize_timestamp(e.timestamp) for e in order_events if isinstance(e, OrderEvent))
    all_timestamps.update(normalize_timestamp(e.timestamp) for e in fill_events if isinstance(e, FillEvent))
    all_timestamps.update(performance_by_ts.keys())
    timestamps_sorted = sorted(all_timestamps)

    # Collect all unique symbols from bars
    all_symbols = set()
    for event in bar_events:
        if isinstance(event, PriceBarEvent):
            all_symbols.add(event.symbol)

    # Bars are already backward-adjusted from data service - write directly
    # Build CSV rows
    rows: list[dict[str, Any]] = []

    for timestamp in timestamps_sorted:
        # Add price bar rows (one per symbol)
        for symbol in sorted(all_symbols):
            bar = bars_by_ts_symbol.get((timestamp, symbol))
            if not bar:
                continue  # No bar data for this symbol at this timestamp

            signal = signals_by_ts_symbol.get((timestamp, symbol))
            orders = orders_by_ts_symbol.get((timestamp, symbol), [])
            fills = fills_by_ts_symbol.get((timestamp, symbol), [])

            # Aggregate orders (sum quantities, concatenate IDs)
            order_ids = [o.event_id for o in orders] if orders else []
            order_qty = sum(o.quantity for o in orders) if orders else None

            # Aggregate fills (sum quantities, weighted average price, sum commission)
            fill_ids = [f.event_id for f in fills] if fills else []
            fill_qty = sum(f.filled_quantity for f in fills) if fills else None
            fill_price = None
            commission = None
            if fills:
                total_value = sum(f.filled_quantity * f.fill_price for f in fills)
                total_qty = sum(f.filled_quantity for f in fills)
                fill_price = total_value / total_qty if total_qty > 0 else None
                commission = sum(f.commission for f in fills)

            # Use prices directly (already backward-adjusted)
            bar_open = float(bar.open) if bar else None
            bar_high = float(bar.high) if bar else None
            bar_low = float(bar.low) if bar else None
            bar_close = float(bar.close) if bar else None
            bar_volume = bar.volume if bar else None

            # Get trade_id and trade metadata from TradeEvent mapping (not from FillEvent)
            # FillEvent doesn't have trade_id field - trade_id comes from TradeEvent
            # which groups fills into trades
            # Note: TradeEvent.fills contains fill_id (business ID), not event_id
            trade_id: str | None = None
            trade_event: TradeEvent | None = None
            for fill in fills:
                # Look up trade_id from fill_id → trade_id mapping
                fill_trade_id = fill_to_trade.get(fill.fill_id)  # Use fill_id, not event_id
                if fill_trade_id:
                    trade_id = fill_trade_id
                    trade_event = trade_metadata.get(fill_trade_id)  # Get full TradeEvent
                    break  # Use first non-null trade_id

            row = {
                "timestamp": timestamp,
                "strategy_id": strategy_id,
                "ticker": symbol,
                "underlying": symbol,  # For real tickers, underlying == ticker
                "open": bar_open,
                "high": bar_high,
                "low": bar_low,
                "close": bar_close,
                "volume": bar_volume,
                # Signal fields
                "signal_intention": signal.intention if signal else None,
                "signal_price": float(signal.price) if signal else None,
                "signal_confidence": float(signal.confidence) if signal else None,  # NEW: Tier 1
                "signal_reason": signal.reason if signal else None,  # NEW: Tier 3
                "signal_event_id": signal.event_id if signal else None,
                "signal_correlation_id": signal.correlation_id if signal else None,
                "signal_causation_id": signal.causation_id if signal else None,
                "signal_source_service": signal.source_service if signal else None,
                # Order fields
                "order_id": ",".join(order_ids) if order_ids else None,
                "order_side": ",".join(o.side.upper() for o in orders) if orders else None,  # NEW: Tier 1 (BUY/SELL)
                "order_type": ",".join(o.order_type.upper() for o in orders) if orders else None,  # NEW: Tier 2
                "order_qty": int(order_qty) if order_qty is not None else None,
                "order_timestamp": ",".join(o.timestamp for o in orders) if orders else None,  # NEW: Tier 3
                "order_event_id": ",".join(o.event_id for o in orders) if orders else None,
                "order_correlation_id": ",".join(o.correlation_id for o in orders if o.correlation_id)
                if orders
                else None,
                "order_causation_id": ",".join(o.causation_id for o in orders if o.causation_id) if orders else None,
                "order_source_service": ",".join(o.source_service for o in orders) if orders else None,
                # Fill fields
                "fill_id": ",".join(fill_ids) if fill_ids else None,
                "fill_side": ",".join(f.side.upper() for f in fills) if fills else None,  # NEW: Tier 1 (BUY/SELL)
                "fill_qty": int(fill_qty) if fill_qty is not None else None,
                "fill_price": float(fill_price) if fill_price is not None else None,
                "fill_slippage_bps": ",".join(str(f.slippage_bps) for f in fills if f.slippage_bps is not None)
                if fills
                else None,  # NEW: Tier 2
                "fill_timestamp": ",".join(f.timestamp for f in fills) if fills else None,  # NEW: Tier 3
                "commission": float(commission) if commission is not None else None,
                "fill_event_id": ",".join(f.event_id for f in fills) if fills else None,
                "fill_correlation_id": ",".join(f.correlation_id for f in fills if f.correlation_id) if fills else None,
                "fill_causation_id": ",".join(f.causation_id for f in fills if f.causation_id) if fills else None,
                "fill_source_service": ",".join(f.source_service for f in fills) if fills else None,
                # Trade fields
                "trade_id": trade_id,  # Links entry and exit fills
                "trade_status": trade_event.status.upper() if trade_event else None,  # NEW: Tier 2 (OPEN/CLOSED)
                "trade_side": trade_event.side.upper()
                if trade_event and trade_event.side
                else None,  # NEW: Tier 2 (LONG/SHORT)
                "trade_entry_price": float(trade_event.entry_price)
                if trade_event and trade_event.entry_price
                else None,  # NEW: Tier 3
                "trade_exit_price": float(trade_event.exit_price)
                if trade_event and trade_event.exit_price
                else None,  # NEW: Tier 3
                "trade_realized_pnl": float(trade_event.realized_pnl)
                if trade_event and trade_event.realized_pnl
                else None,  # NEW: Tier 2
            }
            rows.append(row)

        # Add indicator rows (synthetic tickers)
        # Indicators are already backward-adjusted from data service - write directly
        for indicator_event in indicators_by_ts.get(timestamp, []):
            # Each indicator value becomes a separate row
            for indicator_name, indicator_value in indicator_event.indicators.items():
                # Skip non-numeric indicators (e.g., strings, None)
                if indicator_value is None or not isinstance(indicator_value, (bool, int, float, Decimal)):
                    continue

                # Convert numeric value to float (already on correct scale)
                numeric_value = float(indicator_value)

                row = {
                    "timestamp": timestamp,
                    "strategy_id": strategy_id,
                    "ticker": indicator_name,  # Display name from strategy (e.g., "fast_sma(10)")
                    "underlying": indicator_event.symbol,  # e.g., "AAPL"
                    "open": None,
                    "high": None,
                    "low": None,
                    "close": numeric_value,  # Already backward-adjusted
                    "volume": None,
                    # Signal fields
                    "signal_intention": None,
                    "signal_price": None,
                    "signal_confidence": None,
                    "signal_reason": None,
                    "signal_event_id": None,
                    "signal_correlation_id": None,
                    "signal_causation_id": None,
                    "signal_source_service": None,
                    # Order fields
                    "order_id": None,
                    "order_side": None,
                    "order_type": None,
                    "order_qty": None,
                    "order_timestamp": None,
                    "order_event_id": None,
                    "order_correlation_id": None,
                    "order_causation_id": None,
                    "order_source_service": None,
                    # Fill fields
                    "fill_id": None,
                    "fill_side": None,
                    "fill_qty": None,
                    "fill_price": None,
                    "fill_slippage_bps": None,
                    "fill_timestamp": None,
                    "commission": None,
                    "fill_event_id": None,
                    "fill_correlation_id": None,
                    "fill_causation_id": None,
                    "fill_source_service": None,
                    # Trade fields
                    "trade_id": None,
                    "trade_status": None,
                    "trade_side": None,
                    "trade_entry_price": None,
                    "trade_exit_price": None,
                    "trade_realized_pnl": None,
                }
                rows.append(row)

        # Add portfolio metric rows (synthetic tickers)
        perf = performance_by_ts.get(timestamp)
        if perf:
            metrics_to_export = {
                "EQUITY": perf.equity,
                "CASH": perf.cash,
                "POSITIONS_VALUE": perf.positions_value,
                "SHARPE": perf.sharpe_ratio,
                "SORTINO": perf.sortino_ratio,
                "CURRENT_DRAWDOWN": perf.current_drawdown_pct,
                "CAGR": perf.cagr,
                "CALMAR": perf.calmar_ratio,
                "PROFIT_FACTOR": perf.profit_factor,
                "EXPECTANCY": perf.expectancy,
            }

            for metric_name, metric_value in metrics_to_export.items():
                if metric_value is None:
                    continue

                row = {
                    "timestamp": timestamp,
                    "strategy_id": strategy_id,
                    "ticker": metric_name,
                    "underlying": "PORTFOLIO",  # All portfolio metrics have PORTFOLIO as underlying
                    "open": None,
                    "high": None,
                    "low": None,
                    "close": float(metric_value),  # Metric value in close field
                    "volume": None,
                    # Signal fields
                    "signal_intention": None,
                    "signal_price": None,
                    "signal_confidence": None,
                    "signal_reason": None,
                    "signal_event_id": None,
                    "signal_correlation_id": None,
                    "signal_causation_id": None,
                    "signal_source_service": None,
                    # Order fields
                    "order_id": None,
                    "order_side": None,
                    "order_type": None,
                    "order_qty": None,
                    "order_timestamp": None,
                    "order_event_id": None,
                    "order_correlation_id": None,
                    "order_causation_id": None,
                    "order_source_service": None,
                    # Fill fields
                    "fill_id": None,
                    "fill_side": None,
                    "fill_qty": None,
                    "fill_price": None,
                    "fill_slippage_bps": None,
                    "fill_timestamp": None,
                    "commission": None,
                    "fill_event_id": None,
                    "fill_correlation_id": None,
                    "fill_causation_id": None,
                    "fill_source_service": None,
                    # Trade fields
                    "trade_id": None,
                    "trade_status": None,
                    "trade_side": None,
                    "trade_entry_price": None,
                    "trade_exit_price": None,
                    "trade_realized_pnl": None,
                }
                rows.append(row)

    # Write JSON
    if rows:
        with output_path.open("w") as f:
            json.dump(rows, f, indent=2, cls=DecimalEncoder)

        logger.info(
            "strategy_chart_data.written",
            strategy_id=strategy_id,
            path=str(output_path),
            rows=len(rows),
            size_bytes=output_path.stat().st_size,
        )
    else:
        logger.warning(
            "strategy_chart_data.no_data",
            strategy_id=strategy_id,
            path=str(output_path),
        )


def write_backtest_metadata(backtest_config: dict[str, Any], system_config: dict[str, Any], output_path: Path) -> None:
    """
    Write backtest metadata to JSON file.

    Creates a comprehensive metadata file documenting the exact configuration
    used for the backtest run, including all strategy parameters, system settings,
    and run parameters.

    Args:
        backtest_config: Backtest configuration dict (from BacktestConfig)
        system_config: System configuration dict (from SystemConfig)
        output_path: Path to write metadata.json file

    Example output structure:
        {
            "metadata_version": "1.0",
            "generated_at": "2024-11-17T12:45:06.123456Z",
            "backtest": {
                "backtest_id": "simple_sma_crossover",
                "start_date": "2020-03-01T00:00:00",
                "end_date": "2021-01-01T00:00:00",
                "initial_equity": "100000.00",
                "data_sources": [...],
                "strategies": [
                    {
                        "strategy_id": "sma_crossover",
                        "universe": ["AAPL"],
                        "config": {
                            "fast_period": 10,
                            "slow_period": 50
                        }
                    }
                ],
                "risk_policy": {...}
            },
            "system": {
                "portfolio": {...},
                "execution": {...},
                "data": {...}
            }
        }
    """
    metadata = {
        "metadata_version": "1.0",
        "generated_at": datetime.now().isoformat(),
        "backtest": backtest_config,
        "system": system_config,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w") as f:
        json.dump(metadata, f, indent=2, cls=DecimalEncoder)

    logger.info(
        "backtest_metadata.written",
        path=str(output_path),
        size_bytes=output_path.stat().st_size,
    )
