"""ReportingService: Performance metrics calculation and reporting.

Subscribes to PortfolioStateEvent, tracks metrics incrementally,
emits PerformanceMetricsEvent (optional), generates final report.

Event Priority: 70 (after Manager at 90, before other analytics)

Architecture:
    Bar → Portfolio (100) → Manager (90) → Reporting (70)
"""

import math
import uuid
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import structlog

from qtrader.events.event_bus import EventBus
from qtrader.events.events import BaseEvent, FillEvent, PerformanceMetricsEvent, PortfolioStateEvent

if TYPE_CHECKING:
    from qtrader.events.event_store import EventStore

from qtrader.libraries.performance.calculators import (
    DrawdownCalculator,
    EquityCurveCalculator,
    PeriodAggregationCalculator,
    ReturnsCalculator,
    StrategyPerformanceCalculator,
    TradeStatisticsCalculator,
)
from qtrader.libraries.performance.metrics import (
    calculate_cagr,
    calculate_calmar_ratio,
    calculate_expectancy,
    calculate_profit_factor,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_volatility,
    calculate_win_rate,
)
from qtrader.libraries.performance.models import EquityCurvePoint, FullMetrics, ReturnPoint, TradeRecord
from qtrader.services.reporting.config import ReportingConfig
from qtrader.services.reporting.formatters import display_performance_report
from qtrader.services.reporting.writers import (
    write_backtest_metadata,
    write_drawdowns_json,
    write_equity_curve_json,
    write_json_report,
    write_returns_json,
    write_strategy_chart_data,
    write_trades_json,
)


class ReportingService:
    """
    Performance metrics calculation and reporting service.

    Tracks portfolio state incrementally during backtest execution,
    optionally emits performance events for console display,
    generates comprehensive final report at teardown.

    Design Principles:
    - Incremental: Update calculators on each portfolio state change
    - Optional events: Configurable event emission for visibility
    - Final report: Always generate complete metrics at end
    - Memory efficient: Sampling strategy for large backtests
    """

    PRIORITY = 70  # After Manager (90), before other services

    def __init__(
        self,
        event_bus: EventBus,
        config: ReportingConfig | None = None,
        output_dir: Path | None = None,
        event_store: "EventStore | None" = None,
    ):
        """
        Initialize ReportingService.

        Args:
            event_bus: Event bus for subscribing/publishing
            config: Reporting configuration (uses defaults if None)
            output_dir: Base output directory from system config (default: Path("output"))
            event_store: Optional EventStore for CSV timeline export
        """
        self.event_bus = event_bus
        self.config = config or ReportingConfig()
        self.output_dir = output_dir or Path("output")
        self._event_store = event_store
        self.logger = structlog.get_logger(self.__class__.__name__)

        # State tracking
        self._backtest_id: str | None = None
        self._start_datetime: str | None = None
        self._end_datetime: str | None = None
        self._initial_equity: Decimal | None = None
        self._bar_count = 0
        self._last_portfolio_state: PortfolioStateEvent | None = None
        self._strategy_ids: list[str] = []  # Track strategy IDs for CSV export
        self._portfolio_states_history: dict[datetime, PortfolioStateEvent] = {}  # Historical snapshots

        # Calculators (stateful, incremental updates)
        self._equity_calc = EquityCurveCalculator(max_points=self.config.max_equity_points)
        self._drawdown_calc = DrawdownCalculator()
        self._returns_calc = ReturnsCalculator()
        self._trade_stats_calc = TradeStatisticsCalculator()
        self._period_calc = PeriodAggregationCalculator()
        self._strategy_perf_calc: StrategyPerformanceCalculator | None = None  # Initialized in setup()

        # Trade tracking state
        self._positions: dict[str, Decimal] = {}  # symbol → net quantity
        self._open_trades: dict[str, dict[str, Any]] = {}  # symbol → entry info

        # Subscribe to events
        self._subscribe_to_events()

    def _subscribe_to_events(self) -> None:
        """Subscribe to relevant events."""
        self.event_bus.subscribe(
            event_type="portfolio_state",
            handler=self._handle_portfolio_state,
            priority=self.PRIORITY,
        )
        self.event_bus.subscribe(
            event_type="trade",
            handler=self._handle_trade,
            priority=self.PRIORITY,
        )
        self.event_bus.subscribe(
            event_type="corporate_action",
            handler=self._handle_corporate_action,
            priority=self.PRIORITY,
        )

    def _handle_portfolio_state(self, event: BaseEvent) -> None:
        """
        Handle PortfolioStateEvent: update calculators, optionally emit metrics.

        Args:
            event: PortfolioStateEvent from portfolio
        """
        if not isinstance(event, PortfolioStateEvent):
            return

        self._bar_count += 1
        self._last_portfolio_state = event

        # Store historical state for equity curve reconstruction
        snapshot_dt_str = event.snapshot_datetime
        snapshot_dt = datetime.fromisoformat(snapshot_dt_str.replace("Z", "+00:00"))
        self._portfolio_states_history[snapshot_dt] = event

        # Initialize on first event
        if self._start_datetime is None:
            self._start_datetime = event.snapshot_datetime
            self._initial_equity = event.current_portfolio_equity

        # Update calculators
        self._update_calculators(event)

        # Optionally emit performance event
        if self._should_emit_event():
            self._emit_performance_event(event)

    def _handle_trade(self, event: BaseEvent) -> None:
        """
        Handle TradeEvent: process completed trades for statistics.

        TradeEvent is emitted by PortfolioService and contains all information
        about a trade (fills, entry/exit prices, P&L, etc.). This is the NEW
        approach using the canonical trade_id from PortfolioService.

        Args:
            event: TradeEvent from portfolio
        """
        from qtrader.events.events import TradeEvent

        if not isinstance(event, TradeEvent):
            return

        # Only process closed trades for statistics
        if event.status != "closed":
            self.logger.debug(
                "reporting_service.trade_open",
                trade_id=event.trade_id,
                symbol=event.symbol,
                strategy_id=event.strategy_id,
                status=event.status,
            )
            return

        # Validate required fields are present
        if not all(
            [
                event.entry_price,
                event.exit_price,
                event.realized_pnl,
                event.entry_timestamp,
                event.exit_timestamp,
                event.side,
            ]
        ):
            self.logger.warning(
                "reporting_service.trade_incomplete",
                trade_id=event.trade_id,
                symbol=event.symbol,
                reason="Missing required fields for closed trade",
            )
            return

        # Extract trade data from event (already validated above)
        from typing import cast

        strategy_id = event.strategy_id if event.strategy_id != "unattributed" else "unknown"
        entry_price = cast(Decimal, event.entry_price)
        exit_price = cast(Decimal, event.exit_price)
        realized_pnl = cast(Decimal, event.realized_pnl)
        side = cast(Literal["long", "short"], event.side)

        # Parse timestamps
        from datetime import datetime

        entry_time = datetime.fromisoformat(cast(str, event.entry_timestamp).replace("Z", "+00:00"))
        exit_time = datetime.fromisoformat(cast(str, event.exit_timestamp).replace("Z", "+00:00"))

        # Calculate duration
        duration_seconds = int((exit_time - entry_time).total_seconds())

        # Calculate quantity from price difference and P&L
        # For a long trade: pnl = quantity * (exit_price - entry_price) - commission
        # For a short trade: pnl = quantity * (entry_price - exit_price) - commission
        price_diff = exit_price - entry_price if side == "long" else entry_price - exit_price

        if abs(price_diff) > Decimal("0.0001"):  # Avoid division by zero
            # Rough estimate: quantity ≈ (pnl + commission) / price_diff
            estimated_quantity = (realized_pnl + event.commission_total) / price_diff
            quantity_signed = int(estimated_quantity) if side == "long" else -int(abs(estimated_quantity))
        else:
            # Fallback if price didn't change much
            quantity_signed = 100 if side == "long" else -100

        # Calculate P&L percentage
        quantity_abs = abs(quantity_signed)
        pnl_pct = (
            (realized_pnl / (entry_price * quantity_abs)) * Decimal("100")
            if entry_price > 0 and quantity_abs > 0
            else Decimal("0")
        )

        # Create TradeRecord for statistics calculator
        trade_record = TradeRecord(
            trade_id=event.trade_id,  # ✅ Canonical trade_id from PortfolioService!
            strategy_id=strategy_id,
            symbol=event.symbol,
            entry_timestamp=entry_time,
            exit_timestamp=exit_time,
            entry_price=entry_price,
            exit_price=exit_price,
            quantity=quantity_signed,
            side=side,
            pnl=realized_pnl,
            pnl_pct=pnl_pct,
            commission=event.commission_total,
            duration_seconds=duration_seconds,
        )

        # Add to trade statistics
        self._trade_stats_calc.add_trade(trade_record)

        # Add to period aggregation
        self._period_calc.add_trade(trade_record)

        # Add to strategy performance tracking
        if self._strategy_perf_calc:
            self._strategy_perf_calc.add_trade(trade_record)

        # Trade completion is already displayed via TradeEvent Rich formatting
        # Log at DEBUG level for detailed diagnostics
        self.logger.debug(
            "reporting_service.trade_completed",
            trade_id=event.trade_id,
            symbol=event.symbol,
            strategy_id=strategy_id,
            side=event.side,
            pnl=str(event.realized_pnl),
            pnl_pct=f"{pnl_pct:.2f}%",
            duration_seconds=duration_seconds,
        )

    def _handle_fill(self, event: BaseEvent) -> None:
        """
        Handle FillEvent: track position state and detect completed trades.

        A completed trade is defined as a position that goes to flat (zero quantity).
        Tracks entry price, exit price, P&L, and duration for each completed trade.

        Args:
            event: FillEvent from execution
        """
        if not isinstance(event, FillEvent):
            return

        symbol = event.symbol

        # Calculate position quantity change (buy = positive, sell = negative)
        quantity_delta = event.filled_quantity if event.side == "buy" else -event.filled_quantity

        # Get current position
        current_qty = self._positions.get(symbol, Decimal("0"))
        new_qty = current_qty + quantity_delta

        # Check if this fill closes a position (goes to flat)
        position_closed = current_qty != 0 and new_qty == 0

        if position_closed and symbol in self._open_trades:
            # Trade completed - calculate metrics and add to statistics
            trade_entry = self._open_trades[symbol]

            # Calculate P&L
            entry_price = trade_entry["entry_price"]
            exit_price = event.fill_price
            quantity_abs = abs(current_qty)  # Absolute quantity

            # P&L depends on whether we were long or short
            side: Literal["long", "short"]
            if current_qty > 0:  # Closing long position
                pnl = (exit_price - entry_price) * quantity_abs
                side = "long"
                quantity_signed = int(quantity_abs)
            else:  # Closing short position
                pnl = (entry_price - exit_price) * quantity_abs
                side = "short"
                quantity_signed = -int(quantity_abs)

            # Calculate P&L percentage
            pnl_pct = (pnl / (entry_price * quantity_abs)) * Decimal("100")

            # Calculate duration
            entry_time = datetime.fromisoformat(trade_entry["entry_time"].replace("Z", "+00:00"))
            exit_time = datetime.fromisoformat(event.timestamp.replace("Z", "+00:00"))
            duration_seconds = int((exit_time - entry_time).total_seconds())

            # Create TradeRecord for statistics calculator
            trade_record = TradeRecord(
                trade_id=str(uuid.uuid4()),  # Generate temporary UUID for now (will use TradeEvent.trade_id later)
                strategy_id=event.strategy_id or "unknown",
                symbol=symbol,
                entry_timestamp=entry_time,
                exit_timestamp=exit_time,
                entry_price=entry_price,
                exit_price=exit_price,
                quantity=quantity_signed,
                side=side,
                pnl=pnl,
                pnl_pct=pnl_pct,
                commission=event.commission,
                duration_seconds=duration_seconds,
            )

            # Add to trade statistics
            self._trade_stats_calc.add_trade(trade_record)

            # Add to period aggregation
            self._period_calc.add_trade(trade_record)

            # Add to strategy performance tracking
            if self._strategy_perf_calc:
                self._strategy_perf_calc.add_trade(trade_record)

            # Remove from tracking
            del self._open_trades[symbol]

        # Update position tracking
        if new_qty == 0:
            # Position flat - remove from tracking
            self._positions.pop(symbol, None)
        else:
            # Position still open or just opened
            self._positions[symbol] = new_qty

            # Track entry if opening a new position
            if symbol not in self._open_trades:
                self._open_trades[symbol] = {
                    "entry_time": event.timestamp,
                    "entry_price": event.fill_price,
                }

    def _handle_corporate_action(self, event: BaseEvent) -> None:
        """
        Handle CorporateActionEvent: adjust position tracking for splits.

        When a stock split occurs, the portfolio position quantity changes
        but no fill event is generated. We need to update our position
        tracking to stay in sync with the actual portfolio state.

        Args:
            event: CorporateActionEvent with action details
        """
        # Import here to avoid circular dependency
        from qtrader.events.events import CorporateActionEvent

        if not isinstance(event, CorporateActionEvent):
            return

        # Only handle splits (dividends don't affect position quantity)
        if event.action_type.upper() != "SPLIT":
            return

        symbol = event.symbol
        split_ratio = event.split_ratio

        if split_ratio is None or split_ratio == Decimal("0"):
            return

        # Adjust position tracking for split
        if symbol in self._positions:
            old_qty = self._positions[symbol]
            new_qty = old_qty * split_ratio
            self._positions[symbol] = new_qty

            self.logger.debug(
                "reporting.split_adjusted",
                symbol=symbol,
                split_ratio=float(split_ratio),
                old_qty=float(old_qty),
                new_qty=float(new_qty),
            )

        # Note: Entry price in _open_trades doesn't need adjustment
        # because we calculate P&L using actual fill prices from events
        # The split-adjusted prices are already reflected in the exit fill event

    def _update_calculators(self, event: PortfolioStateEvent) -> None:
        """
        Update all calculators with new portfolio state.

        Args:
            event: Current portfolio state
        """
        # Parse snapshot datetime (ISO string to datetime)
        timestamp = datetime.fromisoformat(event.snapshot_datetime.replace("Z", "+00:00"))

        # Update equity curve
        self._equity_calc.update(timestamp, event.current_portfolio_equity)

        # Update drawdown tracking
        self._drawdown_calc.update(timestamp, event.current_portfolio_equity)

        # Update returns
        self._returns_calc.update(event.current_portfolio_equity)

        # Update period aggregation
        self._period_calc.update(timestamp, event.current_portfolio_equity)

        # TODO: Update trade statistics when we have trade completion events
        # self._trade_stats_calc.add_trade(trade)

    def _should_emit_event(self) -> bool:
        """
        Determine if we should emit PerformanceMetricsEvent now.

        Returns:
            True if event should be emitted
        """
        if not self.config.emit_metrics_events:
            return False

        # Emit every N bars
        return self._bar_count % self.config.event_frequency == 0

    def _emit_performance_event(self, portfolio_event: PortfolioStateEvent) -> None:
        """
        Emit PerformanceMetricsEvent with current metrics.

        Args:
            portfolio_event: Current portfolio state
        """
        if self._initial_equity is None or self._initial_equity == Decimal("0") or self._start_datetime is None:
            return

        # Calculate basic metrics
        total_return_pct = (
            ((portfolio_event.current_portfolio_equity / self._initial_equity) - Decimal("1")) * Decimal("100")
        ).quantize(Decimal("0.01"))

        # Calculate duration for CAGR
        start_dt = datetime.fromisoformat(self._start_datetime.replace("Z", "+00:00"))
        current_dt = datetime.fromisoformat(portfolio_event.snapshot_datetime.replace("Z", "+00:00"))
        duration_days = (current_dt - start_dt).days

        # Calculate risk-adjusted metrics
        cagr = (
            calculate_cagr(self._initial_equity, portfolio_event.current_portfolio_equity, duration_days)
            if duration_days > 0
            else Decimal("0")
        )

        volatility = calculate_volatility(self._returns_calc.returns) if len(self._returns_calc) >= 2 else Decimal("0")

        sharpe = (
            calculate_sharpe_ratio(self._returns_calc.returns, self.config.risk_free_rate)
            if len(self._returns_calc) >= 2
            else Decimal("0")
        )

        sortino = (
            calculate_sortino_ratio(self._returns_calc.returns, self.config.risk_free_rate)
            if len(self._returns_calc) >= 2
            else Decimal("0")
        )
        # Note: sortino can be None if there's no downside risk (infinite Sortino)

        calmar = (
            calculate_calmar_ratio(cagr, self._drawdown_calc.max_drawdown_pct)
            if self._drawdown_calc.max_drawdown_pct > Decimal("0")
            else Decimal("0")
        )

        # Calculate trade statistics
        win_rate = (
            calculate_win_rate(self._trade_stats_calc.trades)
            if self._trade_stats_calc.total_trades > 0
            else Decimal("0")
        )
        profit_factor = (
            calculate_profit_factor(self._trade_stats_calc.trades) if self._trade_stats_calc.total_trades > 0 else None
        )
        expectancy = (
            calculate_expectancy(self._trade_stats_calc.trades)
            if self._trade_stats_calc.total_trades > 0
            else Decimal("0")
        )

        # Build event
        # Count only non-zero positions (filter out closed positions)
        num_positions = sum(
            1 for sg in portfolio_event.strategies_groups for pos in sg.positions if pos.open_quantity != 0
        )

        event = PerformanceMetricsEvent(
            timestamp=portfolio_event.snapshot_datetime,  # ISO string
            equity=portfolio_event.current_portfolio_equity,
            cash=portfolio_event.cash_balance,
            positions_value=portfolio_event.total_market_value,
            total_return_pct=total_return_pct,
            max_drawdown_pct=self._drawdown_calc.max_drawdown_pct,
            current_drawdown_pct=self._drawdown_calc.current_drawdown_pct,
            num_positions=num_positions,
            gross_exposure=portfolio_event.gross_exposure,
            net_exposure=portfolio_event.net_exposure,
            leverage=portfolio_event.leverage,
            total_trades=self._trade_stats_calc.total_trades,
            winning_trades=self._trade_stats_calc.winning_trades,
            losing_trades=self._trade_stats_calc.losing_trades,
            total_commissions=portfolio_event.total_commissions_paid,
            # Risk-adjusted metrics
            cagr=cagr,
            volatility=volatility,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            win_rate=win_rate,
            profit_factor=profit_factor,
            expectancy=expectancy,
            strategy_metrics=self._build_strategy_metrics_snapshot(portfolio_event),
            monthly_returns=self._build_period_snapshots("monthly"),
            quarterly_returns=self._build_period_snapshots("quarterly"),
            annual_returns=self._build_period_snapshots("annual"),
            source_service="reporting_service",
            # No correlation_id/causation_id - performance metrics are snapshots derived from portfolio state
        )

        # Publish event
        self.event_bus.publish(event)

    def _build_strategy_metrics_snapshot(self, portfolio_event: PortfolioStateEvent) -> list:
        """Build strategy metrics snapshot for PerformanceMetricsEvent."""
        from qtrader.events.events import StrategyMetrics

        if not self._strategy_perf_calc or not self._initial_equity:
            return []

        metrics = []
        for sg in portfolio_event.strategies_groups:
            # Get allocated capital for this strategy
            # Note: This is a simplified allocation based on equal distribution
            # In production, this should come from RiskConfig
            allocated = (
                self._initial_equity / Decimal(len(portfolio_event.strategies_groups))
                if portfolio_event.strategies_groups
                else Decimal("0")
            )

            # Calculate positions value for this strategy
            positions_value = Decimal(sum(pos.gross_market_value for pos in sg.positions))

            # Calculate return
            # Note: This is simplified - full calculation happens in calculator
            return_pct = Decimal("0")
            if allocated > Decimal("0"):
                current_equity = allocated + sum(pos.unrealized_pl for pos in sg.positions)
                return_pct = ((current_equity - allocated) / allocated * Decimal("100")).quantize(Decimal("0.01"))

            # Count trades for this strategy
            strategy_trades = [t for t in self._trade_stats_calc.trades if t.strategy_id == sg.strategy_id]
            winning_trades = sum(1 for t in strategy_trades if t.is_winner)

            metric = StrategyMetrics(
                strategy_id=sg.strategy_id,
                equity_allocated=allocated,
                positions_value=positions_value,
                num_positions=len([p for p in sg.positions if p.open_quantity != 0]),
                return_pct=return_pct,
                total_trades=len(strategy_trades),
                winning_trades=winning_trades,
            )
            metrics.append(metric)

        return metrics

    def _build_period_snapshots(self, period_type: str) -> list:
        """Build period metrics snapshots for PerformanceMetricsEvent."""
        from qtrader.events.events import PeriodMetricsSnapshot

        if not self._initial_equity:
            return []

        period_metrics = self._period_calc.calculate_periods(period_type, self._initial_equity)

        snapshots = []
        for pm in period_metrics:
            snapshot = PeriodMetricsSnapshot(
                period=pm.period,
                period_type=pm.period_type,
                start_date=pm.start_date,
                end_date=pm.end_date,
                return_pct=pm.return_pct,
                num_trades=pm.num_trades,
                winning_trades=pm.winning_trades,
                losing_trades=pm.losing_trades,
            )
            snapshots.append(snapshot)

        return snapshots

    def _build_full_metrics(self, final_equity: Decimal, total_return_pct: Decimal) -> "FullMetrics":
        """
        Build complete FullMetrics report.

        Args:
            final_equity: Final portfolio equity
            total_return_pct: Total return percentage

        Returns:
            Complete FullMetrics model
        """

        # Type guards - these should already be checked in teardown()
        assert self._start_datetime is not None
        assert self._end_datetime is not None
        assert self._initial_equity is not None

        # Parse datetime strings
        start_dt = datetime.fromisoformat(self._start_datetime.replace("Z", "+00:00"))
        end_dt = datetime.fromisoformat(self._end_datetime.replace("Z", "+00:00"))
        duration_days = (end_dt - start_dt).days

        # Calculate returns
        cagr = calculate_cagr(self._initial_equity, final_equity, duration_days) if duration_days > 0 else Decimal("0")

        # Best/worst day returns
        returns = self._returns_calc.returns
        best_day = max(returns) if returns else None
        worst_day = min(returns) if returns else None

        # Calculate risk metrics
        volatility = calculate_volatility(returns) if len(returns) >= 2 else Decimal("0")
        max_drawdown_pct = self._drawdown_calc.max_drawdown_pct
        current_drawdown_pct = self._drawdown_calc.current_drawdown_pct

        # Drawdown statistics
        drawdown_periods = self._drawdown_calc.drawdown_periods
        max_dd_duration = max((dd.duration_days for dd in drawdown_periods), default=0)
        avg_dd_pct = (
            Decimal(str(sum(dd.depth_pct for dd in drawdown_periods) / len(drawdown_periods)))
            if drawdown_periods
            else Decimal("0")
        )

        # Calculate risk-adjusted returns
        sharpe = calculate_sharpe_ratio(returns, self.config.risk_free_rate) if len(returns) >= 2 else Decimal("0")
        sortino = calculate_sortino_ratio(returns, self.config.risk_free_rate) if len(returns) >= 2 else Decimal("0")
        # Note: sortino can be None if there's no downside risk (infinite Sortino)
        calmar = calculate_calmar_ratio(cagr, max_drawdown_pct) if max_drawdown_pct > Decimal("0") else Decimal("0")

        # Trade statistics
        trades = self._trade_stats_calc.trades
        total_trades = len(trades)
        winning_trades = sum(1 for t in trades if t.is_winner)
        losing_trades = total_trades - winning_trades

        win_rate = calculate_win_rate(trades) if total_trades > 0 else Decimal("0")
        profit_factor = calculate_profit_factor(trades) if total_trades > 0 else None
        expectancy = calculate_expectancy(trades) if total_trades > 0 else Decimal("0")

        # Trade amounts
        wins = [t for t in trades if t.is_winner]
        losses = [t for t in trades if not t.is_winner]

        avg_win = Decimal(str(sum(t.pnl for t in wins) / len(wins))) if wins else Decimal("0")
        avg_loss = Decimal(str(sum(t.pnl for t in losses) / len(losses))) if losses else Decimal("0")
        avg_win_pct = Decimal(str(sum(t.pnl_pct for t in wins) / len(wins))) if wins else Decimal("0")
        avg_loss_pct = Decimal(str(sum(t.pnl_pct for t in losses) / len(losses))) if losses else Decimal("0")

        largest_win = max((t.pnl for t in wins), default=Decimal("0"))
        largest_loss = min((t.pnl for t in losses), default=Decimal("0"))
        largest_win_pct = max((t.pnl_pct for t in wins), default=Decimal("0"))
        largest_loss_pct = min((t.pnl_pct for t in losses), default=Decimal("0"))

        max_consec_wins = self._trade_stats_calc.max_consecutive_wins
        max_consec_losses = self._trade_stats_calc.max_consecutive_losses

        avg_duration = Decimal(str(sum(t.duration_days for t in trades) / len(trades))) if trades else None

        # Commissions - use portfolio's total which includes ALL fills (open + closed positions)
        # This ensures buy-and-hold and other strategies with open positions show correct commissions
        if self._last_portfolio_state:
            total_commissions = self._last_portfolio_state.total_commissions_paid
        else:
            # Fallback to summing from closed trades if no portfolio state available
            total_commissions = Decimal(str(sum(t.commission for t in trades))) if trades else Decimal("0")

        # Calculate commission % of P&L using total return (realized + unrealized)
        # This gives a more accurate picture than just realized P&L from closed trades
        total_pnl = final_equity - self._initial_equity  # Total P&L including unrealized
        pnl_abs = Decimal(str(abs(total_pnl)))
        commission_pct = (total_commissions / pnl_abs) * Decimal("100") if pnl_abs != Decimal("0") else Decimal("0")

        # Build FullMetrics
        return FullMetrics(
            backtest_id=self._backtest_id or "unknown",
            start_date=start_dt.date().isoformat(),
            end_date=end_dt.date().isoformat(),
            duration_days=duration_days,
            initial_equity=self._initial_equity,
            final_equity=final_equity,
            total_return_pct=total_return_pct,
            cagr=cagr,
            best_day_return_pct=best_day,
            worst_day_return_pct=worst_day,
            volatility_annual_pct=volatility,
            max_drawdown_pct=max_drawdown_pct,
            max_drawdown_duration_days=max_dd_duration,
            avg_drawdown_pct=avg_dd_pct,
            current_drawdown_pct=current_drawdown_pct,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            risk_free_rate=self.config.risk_free_rate,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            profit_factor=profit_factor,
            avg_win=avg_win,
            avg_loss=avg_loss,
            avg_win_pct=avg_win_pct,
            avg_loss_pct=avg_loss_pct,
            largest_win=largest_win,
            largest_loss=largest_loss,
            largest_win_pct=largest_win_pct,
            largest_loss_pct=largest_loss_pct,
            expectancy=expectancy,
            max_consecutive_wins=max_consec_wins,
            max_consecutive_losses=max_consec_losses,
            avg_trade_duration_days=avg_duration,
            total_commissions=total_commissions,
            commission_pct_of_pnl=commission_pct,
            monthly_returns=self._period_calc.calculate_periods("monthly", self._initial_equity),
            quarterly_returns=self._period_calc.calculate_periods("quarterly", self._initial_equity),
            annual_returns=self._period_calc.calculate_periods("annual", self._initial_equity),
            strategy_performance=self._strategy_perf_calc.calculate_performance() if self._strategy_perf_calc else [],
            drawdown_periods=drawdown_periods,
            benchmark_symbol=self.config.benchmark_symbol,
            benchmark_return_pct=None,
            beta=None,
            alpha=None,
            correlation=None,
            tracking_error=None,
        )

    def _load_portfolio_states_from_events(self) -> dict[datetime, PortfolioStateEvent]:
        """
        Return historical portfolio state events cached during execution.

        Returns:
            Dictionary mapping snapshot_datetime to PortfolioStateEvent
        """
        if not self._portfolio_states_history:
            self.logger.warning("portfolio_states.empty", fallback="using_last_state")
        else:
            self.logger.info(
                "portfolio_states.available",
                count=len(self._portfolio_states_history),
                first=str(min(self._portfolio_states_history.keys())),
                last=str(max(self._portfolio_states_history.keys())),
            )

        return self._portfolio_states_history

    def _calculate_drawdown_at_timestamp(self, timestamp: datetime, equity: Decimal) -> Decimal:
        """
        Calculate drawdown percentage at specific timestamp.

        Args:
            timestamp: Timestamp to calculate drawdown for
            equity: Equity value at timestamp

        Returns:
            Drawdown percentage (negative value)
        """
        peak = self._get_peak_equity_at_timestamp(timestamp)
        if peak == Decimal("0"):
            return Decimal("0")

        dd = ((equity - peak) / peak) * Decimal("100")
        return dd

    def _get_peak_equity_at_timestamp(self, timestamp: datetime) -> Decimal:
        """
        Get peak equity up to given timestamp.

        Args:
            timestamp: Timestamp to get peak for

        Returns:
            Peak equity value
        """
        # Get all equity points up to this timestamp
        peak = Decimal("0")
        for ts, equity in self._equity_calc.get_curve():
            if ts > timestamp:
                break
            if equity > peak:
                peak = equity

        return peak if peak > Decimal("0") else (self._initial_equity or Decimal("0"))

    def _write_outputs(self, metrics: "FullMetrics") -> None:
        """
        Write outputs to disk based on configuration.

        Args:
            metrics: Complete performance metrics
        """
        if not self._backtest_id:
            return

        output_path = self.config.get_output_path(self.output_dir)

        # Write JSON report
        if self.config.write_json:
            json_path = output_path / "performance.json"
            write_json_report(metrics, json_path)

        # Write Parquet time-series
        if self.config.write_parquet:
            ts_path = self.config.get_timeseries_path(self.output_dir)

            # Equity curve
            if self.config.include_equity_curve and self._last_portfolio_state:
                equity_points = []

                # Load historical portfolio states from event store
                portfolio_states = self._load_portfolio_states_from_events()

                for timestamp, equity in self._equity_calc.get_curve():
                    # Find matching portfolio state by timestamp
                    portfolio_state = portfolio_states.get(timestamp)

                    if portfolio_state is None:
                        # Fallback to last state if no match (shouldn't happen normally)
                        portfolio_state = self._last_portfolio_state
                        self.logger.warning(
                            "equity_curve.missing_portfolio_state",
                            timestamp=str(timestamp),
                            using_fallback=True,
                        )

                    # Find drawdown info for this timestamp
                    dd_pct = self._calculate_drawdown_at_timestamp(timestamp, equity)
                    underwater = equity < self._get_peak_equity_at_timestamp(timestamp)

                    point = EquityCurvePoint(
                        timestamp=timestamp,
                        equity=equity,
                        cash=portfolio_state.cash_balance,
                        positions_value=portfolio_state.total_market_value,
                        num_positions=sum(len(sg.positions) for sg in portfolio_state.strategies_groups),
                        gross_exposure=portfolio_state.gross_exposure,
                        net_exposure=portfolio_state.net_exposure,
                        leverage=portfolio_state.leverage,
                        drawdown_pct=dd_pct,
                        underwater=underwater,
                    )
                    equity_points.append(point)

                write_equity_curve_json(equity_points, ts_path / "equity_curve.json")

            # Returns
            if self.config.include_returns and self._returns_calc.returns:
                returns_points: list[ReturnPoint] = []
                equity_curve = self._equity_calc.get_curve()
                cumulative = Decimal("0")

                for i, (timestamp, _) in enumerate(equity_curve[1:], 1):  # Skip first point
                    period_return = self._returns_calc.returns[i - 1]
                    cumulative += period_return
                    log_return = Decimal(str(math.log(1 + float(period_return))))

                    ret_point = ReturnPoint(
                        timestamp=timestamp,
                        period_return=period_return,
                        cumulative_return=cumulative,
                        log_return=log_return,
                    )
                    returns_points.append(ret_point)

                write_returns_json(returns_points, ts_path / "returns.json")

            # Trades
            if self.config.include_trades and metrics.total_trades > 0:
                write_trades_json(self._trade_stats_calc.trades, ts_path / "trades.json")

            # Drawdowns
            if self.config.include_drawdowns and metrics.drawdown_periods:
                write_drawdowns_json(metrics.drawdown_periods, ts_path / "drawdowns.json")

        # Write chart data JSON (one file per strategy)
        if self.config.write_csv_timeline:
            if not self._event_store:
                self.logger.warning(
                    "chart_data.skipped",
                    reason="EventStore not available - pass event_store to ReportingService.__init__()",
                )
            elif not self._strategy_ids:
                self.logger.warning(
                    "chart_data.skipped",
                    reason="No strategy IDs found in context",
                )
            else:
                ts_path = self.config.get_timeseries_path(self.output_dir)

                # Note: We don't pass start_time/end_time to EventStore queries because
                # EventStore filters by occurred_at (when event was created), not by
                # business timestamp (the market data time). The chart data writer will filter
                # by business timestamp internally if needed.

                # Export one JSON file per strategy (generic filename for easy loading)
                for strategy_id in self._strategy_ids:
                    json_path = ts_path / "chart_data.json"
                    try:
                        write_strategy_chart_data(
                            self._event_store,
                            strategy_id,
                            json_path,
                            None,  # start_time - not used, writer includes all events
                            None,  # end_time - not used, writer includes all events
                        )
                    except Exception as e:
                        self.logger.error(
                            "csv_timeline.write_failed",
                            strategy_id=strategy_id,
                            error=str(e),
                            error_type=type(e).__name__,
                        )

        # Write HTML report
        if self.config.write_html_report:
            try:
                from qtrader.services.reporting.html_reporter import HTMLReportGenerator

                html_gen = HTMLReportGenerator(output_path)
                report_path = html_gen.generate()
                self.logger.info("html_report.generated", path=str(report_path))
            except Exception as e:
                self.logger.error(
                    "html_report.generation_failed",
                    error=str(e),
                    error_type=type(e).__name__,
                )

    def _write_metadata(self, context: dict) -> None:
        """
        Write backtest metadata to JSON file.

        Exports comprehensive configuration metadata including backtest parameters,
        strategy configurations, and system settings.

        Args:
            context: Teardown context containing backtest_config
        """
        from qtrader.system.config import get_system_config

        backtest_config = context.get("backtest_config")
        if not backtest_config:
            self.logger.warning("metadata.skipped", reason="No backtest_config in context")
            return

        try:
            # Get system config
            system_config = get_system_config()

            # Convert configs to dicts for JSON serialization
            # Use model_dump() for Pydantic v2 compatibility
            if hasattr(backtest_config, "model_dump"):
                backtest_dict = backtest_config.model_dump(mode="json")
            else:
                # Fallback for Pydantic v1
                backtest_dict = backtest_config.dict()

            # Convert system config to dict
            system_dict = {
                "data": {
                    "sources_config": system_config.data.sources_config,
                    "default_timezone": system_config.data.default_timezone,
                    "price_decimals": system_config.data.price_decimals,
                    "validate_on_load": system_config.data.validate_on_load,
                },
                "output": {
                    "experiments_root": system_config.output.experiments_root,
                    "run_id_format": system_config.output.run_id_format,
                    "display_format": system_config.output.display_format,
                },
                "logging": {
                    "level": system_config.logging.level,
                    "format": system_config.logging.format,
                    "timestamp_format": system_config.logging.timestamp_format,
                    "enable_file": system_config.logging.enable_file,
                    "file_path": system_config.logging.file_path,
                },
            }

            # Write metadata.json to root output directory
            metadata_path = self.output_dir / "metadata.json"
            write_backtest_metadata(backtest_dict, system_dict, metadata_path)

        except Exception as e:
            import traceback

            self.logger.error(
                "metadata.write_failed",
                error=str(e),
                error_type=type(e).__name__,
                traceback=traceback.format_exc(),
            )

    def setup(self, context: dict) -> None:
        """
        Setup service at backtest initialization.

        Args:
            context: Backtest context with configuration (includes strategy_ids list)
        """
        self._backtest_id = context.get("backtest_id", "unknown")
        self._strategy_ids = context.get("strategy_ids", [])

        # Initialize strategy performance calculator with strategy IDs
        self._strategy_perf_calc = StrategyPerformanceCalculator(self._strategy_ids)

        self.logger.info(
            f"ReportingService initialized for backtest {self._backtest_id}",
            emit_metrics_events=self.config.emit_metrics_events,
            event_frequency=self.config.event_frequency,
            max_equity_points=self.config.max_equity_points,
            strategies=self._strategy_ids,
        )

    def teardown(self, context: dict) -> None:
        """
        Teardown service at backtest completion.

        Finalizes calculators, generates full report, writes outputs.

        Args:
            context: Backtest context
        """
        if self._start_datetime is None or self._initial_equity is None:
            self.logger.warning("No data collected, skipping final report")
            return

        # Finalize calculators
        last_timestamp = self._equity_calc.latest_timestamp()
        if last_timestamp is not None:
            self._drawdown_calc.finalize(last_timestamp)

        self._end_datetime = (
            self._last_portfolio_state.snapshot_datetime if self._last_portfolio_state else self._start_datetime
        )

        # Generate final metrics
        final_equity = self._equity_calc.latest_equity() or Decimal("0")
        total_return_pct = (((final_equity / self._initial_equity) - Decimal("1")) * Decimal("100")).quantize(
            Decimal("0.01")
        )

        self.logger.info(
            "Backtest completed",
            initial_equity=float(self._initial_equity),
            final_equity=float(final_equity),
            total_return_pct=float(total_return_pct),
            max_drawdown_pct=float(self._drawdown_calc.max_drawdown_pct),
            total_bars=self._bar_count,
        )

        # Build full metrics report
        full_metrics = self._build_full_metrics(final_equity, total_return_pct)

        # Write outputs based on configuration
        if self._backtest_id:
            self._write_outputs(full_metrics)

            # Write metadata.json if backtest_config provided
            if "backtest_config" in context:
                self._write_metadata(context)

        # Display console report
        if self.config.display_final_report:
            display_performance_report(full_metrics, detail_level=self.config.report_detail_level)

    def reset(self) -> None:
        """Reset service state for new backtest."""
        self._backtest_id = None
        self._start_datetime = None
        self._end_datetime = None
        self._initial_equity = None
        self._bar_count = 0
        self._last_portfolio_state = None
        self._portfolio_states_history = {}

        # Reset calculators
        self._equity_calc = EquityCurveCalculator(max_points=self.config.max_equity_points)
        self._drawdown_calc = DrawdownCalculator()
        self._returns_calc = ReturnsCalculator()
        self._trade_stats_calc = TradeStatisticsCalculator()
        self._period_calc = PeriodAggregationCalculator()
        self._strategy_perf_calc = None  # Will be initialized in setup()
