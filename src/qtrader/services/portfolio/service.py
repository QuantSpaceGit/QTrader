"""Portfolio service implementation.

Main service for portfolio accounting with lot-based position tracking.

Week 1: Basic fills (open positions), cash management, queries
Week 2: Lot accounting (FIFO/LIFO), realized P&L
Week 3: Corporate actions, fees, mark-to-market
Week 4: State management, polish
"""

from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Literal, Optional

from qtrader.events.event_bus import EventBus
from qtrader.events.events import (
    CorporateActionEvent,
    FillEvent,
    PortfolioStateEvent,
    PriceBarEvent,
    TradeEvent,
    ValuationTriggerEvent,
)
from qtrader.services.portfolio.lot_tracker import LotTracker
from qtrader.services.portfolio.models import (
    Ledger,
    LedgerEntry,
    LedgerEntryType,
    Lot,
    LotSide,
    PortfolioConfig,
    PortfolioState,
    Position,
)
from qtrader.system import LoggerFactory

logger = LoggerFactory.get_logger()


class PortfolioService:
    """
    Portfolio service implementation.

    Provides lot-based position accounting with complete audit trail.
    Week 1 supports basic fill processing (opens only), cash tracking, and queries.

    Example:
        >>> config = PortfolioConfig(initial_cash=Decimal("100000"))
        >>> portfolio = PortfolioService(config)
        >>>
        >>> # Open long position
        >>> portfolio.apply_fill(
        ...     fill_id="fill_001",
        ...     timestamp=datetime.now(),
        ...     symbol="AAPL",
        ...     side="buy",
        ...     quantity=Decimal("100"),
        ...     price=Decimal("150.00")
        ... )
        >>>
        >>> print(f"Cash: ${portfolio.get_cash()}")
        >>> print(f"Equity: ${portfolio.get_equity()}")
    """

    def __init__(self, config: PortfolioConfig, event_bus: Optional[EventBus] = None):
        """
        Initialize portfolio service.

        Args:
            config: Portfolio configuration
            event_bus: Optional event bus for Phase 5 event-driven mode
        """
        self.config = config
        self._event_bus = event_bus

        # Portfolio metadata
        self._portfolio_id = config.portfolio_id
        self._start_datetime = config.start_datetime
        self._reporting_currency = config.reporting_currency
        self._initial_cash = config.initial_cash
        self._adjustment_mode = config.adjustment_mode

        # Core state
        self._cash: Decimal = config.initial_cash
        self._positions: dict[tuple[str, str], Position] = {}  # (strategy_id, symbol) → Position
        self._lot_tracker: dict[tuple[str, str], LotTracker] = {}  # (strategy_id, symbol) → LotTracker

        # Ledger
        self._ledger = Ledger(max_entries=config.max_ledger_entries)

        # Cumulative metrics
        self._cumulative_realized_pnl = Decimal("0")  # Track realized P&L
        self._total_commissions = Decimal("0")
        self._total_borrow_fees = Decimal("0")
        self._total_margin_interest = Decimal("0")
        self._total_dividends_received = Decimal("0")
        self._total_dividends_paid = Decimal("0")

        # Trade tracking - group fills into trades
        self._trade_counter = 0  # Sequential counter for trade IDs
        self._active_trades: dict[tuple[str, str], dict[str, Any]] = {}  # (strategy_id, symbol) → trade data
        # Trade data structure:
        # {
        #     "trade_id": "T00001",
        #     "fills": ["fill-uuid-1", "fill-uuid-2"],
        #     "side": "long" | "short",
        #     "entry_price": Decimal("100.00"),
        #     "entry_timestamp": "2024-03-15T14:35:24.123Z",
        #     "commission_total": Decimal("1.00"),
        #     "current_quantity": Decimal("100"),
        #     "initial_quantity": Decimal("100"),  # Original position size (for P&L calc)
        #     "realized_pl_before": Decimal("0")   # Position's realized P&L when trade opened
        # }

        # Latest prices for mark-to-market (Phase 5)
        self._latest_prices: dict[str, Decimal] = {}

        # Subscribe to events if event bus provided
        if self._event_bus:
            # Portfolio service needs to run BEFORE strategies (higher priority)
            # so that PortfolioStateEvent is published before strategies emit signals
            self._event_bus.subscribe("bar", self.on_bar, priority=100)  # type: ignore[arg-type]
            self._event_bus.subscribe("valuation_trigger", self.on_valuation_trigger)  # type: ignore[arg-type]
            self._event_bus.subscribe("fill", self.on_fill)  # type: ignore[arg-type]
            # Corporate actions MUST be processed BEFORE bar marks to market (higher priority = runs first)
            self._event_bus.subscribe("corporate_action", self.on_corporate_action, priority=110)  # type: ignore[arg-type]

        logger.debug(
            "portfolio_service.initialized",
            portfolio_id=self._portfolio_id,
            start_datetime=self._start_datetime.isoformat(),
            initial_cash=str(config.initial_cash),
            reporting_currency=self._reporting_currency,
            lot_method_long=config.lot_method_long,
            lot_method_short=config.lot_method_short,
            adjustment_mode=self._adjustment_mode,
            event_driven=self._event_bus is not None,
        )

    # ==================== Fill Processing ====================

    def apply_fill(
        self,
        fill_id: str,
        timestamp: datetime,
        symbol: str,
        side: Literal["buy", "sell"],
        quantity: Decimal,
        price: Decimal,
        commission: Decimal = Decimal("0"),
        strategy_id: str | None = None,
    ) -> tuple[str | None, bool]:
        """
        Apply fill to portfolio and track trade.

        Routes to appropriate handler based on existing position:
        - Buy with no short → open/add to long position
        - Sell with no long → open/add to short position
        - Buy with existing short → close short position
        - Sell with existing long → close long position

        Args:
            fill_id: Unique identifier for this fill
            timestamp: When fill occurred
            symbol: Ticker symbol
            side: "buy" or "sell"
            quantity: Number of shares (positive)
            price: Price per share
            commission: Commission paid
            strategy_id: Strategy that generated this trade (optional)

        Returns:
            Tuple of (trade_id, trade_closed):
                - trade_id: Trade identifier (T00001, T00002, etc.) or None
                - trade_closed: True if this fill closed the position to flat

        Raises:
            ValueError: If inputs invalid
        """
        # Validate inputs
        self._validate_fill_inputs(fill_id, quantity, price, commission)

        # Use strategy_id or "unattributed" for keying
        strat_id = strategy_id if strategy_id is not None else "unattributed"
        key = (strat_id, symbol)

        # Determine action based on strategy-specific position
        has_long = self._has_long_position(symbol, strategy_id)
        has_short = self._has_short_position(symbol, strategy_id)
        is_buy = side == "buy"
        is_sell = side == "sell"

        # Get current position quantity to detect position changes
        position = self._positions.get(key)
        current_qty = position.quantity if position else Decimal("0")

        # Determine if opening new trade from flat
        is_opening_new_trade = current_qty == 0

        # Track trade data
        trade_id: str | None = None
        if is_opening_new_trade:
            # Opening new position from flat - create new trade
            self._trade_counter += 1
            trade_id = f"T{self._trade_counter:05d}"
            timestamp_str = timestamp.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")

            # Capture position's current realized P&L (baseline for this trade)
            realized_pl_before = position.realized_pl if position else Decimal("0")

            self._active_trades[key] = {
                "trade_id": trade_id,
                "fills": [fill_id],
                "side": "long" if is_buy else "short",
                "entry_price": price,
                "entry_timestamp": timestamp_str,
                "commission_total": commission,
                "current_quantity": quantity if is_buy else -quantity,
                "initial_quantity": quantity if is_buy else -quantity,
                "realized_pl_before": realized_pl_before,
            }
        elif key in self._active_trades:
            # Position already open - update existing trade
            trade_data = self._active_trades[key]
            trade_id = trade_data["trade_id"]
            trade_data["fills"].append(fill_id)
            trade_data["commission_total"] += commission

        # Apply fill to position
        if is_buy and not has_short:
            # Open or add to long position
            self._open_long_position(fill_id, timestamp, symbol, quantity, price, commission, strategy_id)
        elif is_sell and not has_long:
            # Open or add to short position
            self._open_short_position(fill_id, timestamp, symbol, quantity, price, commission, strategy_id)
        elif is_sell and has_long:
            # Close long position (FIFO)
            self._close_long_position(fill_id, timestamp, symbol, quantity, price, commission, strategy_id)
        elif is_buy and has_short:
            # Close short position (LIFO)
            self._close_short_position(fill_id, timestamp, symbol, quantity, price, commission, strategy_id)
        else:
            # Should never reach here
            raise ValueError(f"Invalid state: side={side}, has_long={has_long}, has_short={has_short}")

        # Check if position closed to flat
        position_after = self._positions.get(key)
        trade_closed = position_after is None or position_after.quantity == 0

        # Calculate realized P&L for this trade when it closes
        trade_realized_pnl: Decimal | None = None
        if trade_closed and key in self._active_trades:
            # Update trade with final quantity (should be zero)
            self._active_trades[key]["current_quantity"] = Decimal("0")

            # Calculate trade's realized P&L: change in position's realized_pl since trade opened
            if position_after:
                realized_pl_now = position_after.realized_pl
                realized_pl_before = self._active_trades[key]["realized_pl_before"]
                trade_realized_pnl = realized_pl_now - realized_pl_before
                self._active_trades[key]["realized_pnl"] = trade_realized_pnl

            # Store the exit price (last fill price when closing)
            self._active_trades[key]["exit_price"] = price

        logger.info(
            "portfolio_service.fill_applied",
            fill_id=fill_id,
            symbol=symbol,
            side=side,
            quantity=str(quantity),
            price=str(price),
            commission=str(commission),
            strategy_id=strategy_id,
            trade_id=trade_id,
            trade_closed=trade_closed,
        )

        return (trade_id, trade_closed)

    def _open_long_position(
        self,
        fill_id: str,
        timestamp: datetime,
        symbol: str,
        quantity: Decimal,
        price: Decimal,
        commission: Decimal,
        strategy_id: str | None = None,
    ) -> None:
        """
        Open or add to long position.

        Args:
            fill_id: Fill identifier
            timestamp: Fill time
            symbol: Ticker
            quantity: Shares to buy
            price: Price per share
            commission: Commission paid
            strategy_id: Strategy attribution (optional)
        """
        # Create lot
        lot = Lot(
            lot_id=f"{fill_id}_lot",
            symbol=symbol,
            side=LotSide.LONG,
            quantity=quantity,
            entry_price=price,
            entry_timestamp=timestamp,
            entry_fill_id=fill_id,
            entry_commission=commission,
        )

        # Use strategy_id or "unattributed" for keying
        strat_id = strategy_id or "unattributed"
        key = (strat_id, symbol)

        # Add to lot tracker
        if key not in self._lot_tracker:
            self._lot_tracker[key] = LotTracker()
        self._lot_tracker[key].add_lot(lot)

        # Update position
        if key not in self._positions:
            self._positions[key] = Position(
                symbol=symbol,
                quantity=Decimal("0"),
                lots=[],
                strategy_id=strat_id,
                last_updated=timestamp,
            )

        position = self._positions[key]

        position.quantity += quantity
        position.lots.append(lot)
        position.total_cost += quantity * price
        position.commission_paid += commission
        position.avg_price = position.total_cost / position.quantity if position.quantity != 0 else Decimal("0")
        position.last_updated = timestamp

        # Update cash (buy = cash out)
        cash_flow = -(quantity * price + commission)
        self._cash += cash_flow
        self._total_commissions += commission

        # Create ledger entry
        entry = LedgerEntry(
            entry_id=f"{fill_id}_ledger",
            timestamp=timestamp,
            entry_type=LedgerEntryType.FILL,
            symbol=symbol,
            quantity=quantity,
            price=price,
            cash_flow=cash_flow,
            commission=commission,
            fill_id=fill_id,
            lot_ids=[lot.lot_id],
            description=f"Buy {quantity} {symbol} @ ${price}",
        )
        self._ledger.add_entry(entry)

    def _open_short_position(
        self,
        fill_id: str,
        timestamp: datetime,
        symbol: str,
        quantity: Decimal,
        price: Decimal,
        commission: Decimal,
        strategy_id: str | None = None,
    ) -> None:
        """
        Open or add to short position.

        Args:
            fill_id: Fill identifier
            timestamp: Fill time
            symbol: Ticker
            quantity: Shares to sell short
            price: Price per share
            commission: Commission paid
            strategy_id: Strategy attribution (optional)
        """
        # Create lot (negative quantity for short)
        lot = Lot(
            lot_id=f"{fill_id}_lot",
            symbol=symbol,
            side=LotSide.SHORT,
            quantity=-quantity,  # Negative for short
            entry_price=price,
            entry_timestamp=timestamp,
            entry_fill_id=fill_id,
            entry_commission=commission,
        )

        # Use strategy_id or "unattributed" for keying
        strat_id = strategy_id or "unattributed"
        key = (strat_id, symbol)

        # Add to lot tracker
        if key not in self._lot_tracker:
            self._lot_tracker[key] = LotTracker()
        self._lot_tracker[key].add_lot(lot)

        # Update position
        if key not in self._positions:
            self._positions[key] = Position(
                symbol=symbol,
                quantity=Decimal("0"),
                lots=[],
                strategy_id=strat_id,
                last_updated=timestamp,
            )

        position = self._positions[key]

        position.quantity -= quantity  # Negative for short
        position.lots.append(lot)
        position.total_cost -= quantity * price  # Negative cost for short
        position.commission_paid += commission
        position.avg_price = position.total_cost / position.quantity if position.quantity != 0 else Decimal("0")
        position.last_updated = timestamp

        # Update cash (sell short = cash in)
        cash_flow = quantity * price - commission
        self._cash += cash_flow
        self._total_commissions += commission

        # Create ledger entry
        entry = LedgerEntry(
            entry_id=f"{fill_id}_ledger",
            timestamp=timestamp,
            entry_type=LedgerEntryType.FILL,
            symbol=symbol,
            quantity=-quantity,  # Negative for short
            price=price,
            cash_flow=cash_flow,
            commission=commission,
            fill_id=fill_id,
            lot_ids=[lot.lot_id],
            description=f"Sell short {quantity} {symbol} @ ${price}",
        )
        self._ledger.add_entry(entry)

    def _close_long_position(
        self,
        fill_id: str,
        timestamp: datetime,
        symbol: str,
        quantity: Decimal,
        price: Decimal,
        commission: Decimal,
        strategy_id: str | None = None,
    ) -> None:
        """
        Close long position using FIFO lot matching.

        Calculates realized P&L for each matched lot, updates position,
        adds cash, creates ledger entries.

        Args:
            fill_id: Fill identifier
            timestamp: Fill time
            symbol: Ticker
            quantity: Shares to sell (positive)
            price: Sale price per share
            commission: Commission paid
            strategy_id: Strategy attribution (optional)

        Raises:
            ValueError: If insufficient long quantity to close
        """
        # Use strategy_id or "unattributed" for keying
        strat_id = strategy_id or "unattributed"
        key = (strat_id, symbol)

        # Match lots using FIFO
        tracker = self._lot_tracker[key]
        matches = tracker.match_close_long(quantity)

        total_realized_pnl = Decimal("0")
        closed_lot_ids: list[str] = []

        # Calculate realized P&L for each matched lot
        for lot, qty_closed in matches:
            # Realized P&L = (exit_price - entry_price) * quantity - total_commissions
            # Both entry and exit commissions are allocated proportionally
            exit_commission = commission * (qty_closed / quantity)
            entry_commission = lot.entry_commission * (qty_closed / lot.quantity)
            total_commissions = entry_commission + exit_commission
            pnl = (price - lot.entry_price) * qty_closed - total_commissions
            total_realized_pnl += pnl
            closed_lot_ids.append(lot.lot_id)

            logger.debug(
                "portfolio_service.lot_closed",
                lot_id=lot.lot_id,
                symbol=symbol,
                side="long",
                quantity=qty_closed,
                entry_price=float(lot.entry_price),
                exit_price=float(price),
                realized_pnl=float(pnl),
            )

        # Update position
        position = self._positions[key]
        position.quantity -= quantity

        # Track realized P&L on this position
        position.realized_pl += total_realized_pnl

        # Remove closed lots from position.lots
        position.lots = [
            lot for lot in position.lots if not any(lot.lot_id == closed_id for closed_id in closed_lot_ids)
        ]

        # Also remove any "_remaining" versions that got split
        position.lots = [
            lot
            for lot in position.lots
            if not any(lot.lot_id.startswith(f"{closed_id}_remaining") for closed_id in closed_lot_ids)
        ]

        # Add any remaining lots from tracker
        tracker_lots = tracker.get_lots(LotSide.LONG)
        for tracker_lot in tracker_lots:
            if tracker_lot not in position.lots:
                position.lots.append(tracker_lot)

        # Recalculate total_cost and avg_price
        position.total_cost = sum(
            (lot.quantity * lot.entry_price for lot in position.lots),
            start=Decimal("0"),
        )
        position.avg_price = position.total_cost / position.quantity if position.quantity != 0 else Decimal("0")
        position.last_updated = timestamp

        # Update cash (sell = cash in)
        cash_flow = quantity * price - commission
        self._cash += cash_flow
        self._cumulative_realized_pnl += total_realized_pnl
        self._total_commissions += commission

        # Create ledger entry
        entry = LedgerEntry(
            entry_id=f"{fill_id}_ledger",
            timestamp=timestamp,
            entry_type=LedgerEntryType.FILL,
            symbol=symbol,
            quantity=-quantity,  # Negative for sell
            price=price,
            cash_flow=cash_flow,
            commission=commission,
            realized_pnl=total_realized_pnl,
            fill_id=fill_id,
            lot_ids=closed_lot_ids,
            description=f"Sell {quantity} {symbol} @ ${price} (FIFO close)",
        )
        self._ledger.add_entry(entry)

        logger.info(
            "portfolio_service.position_closed",
            symbol=symbol,
            side="long",
            quantity=float(quantity),
            price=float(price),
            realized_pnl=float(total_realized_pnl),
            lots_closed=len(matches),
        )

    def _close_short_position(
        self,
        fill_id: str,
        timestamp: datetime,
        symbol: str,
        quantity: Decimal,
        price: Decimal,
        commission: Decimal,
        strategy_id: str | None = None,
    ) -> None:
        """
        Close short position using LIFO lot matching.

        Calculates realized P&L for each matched lot, updates position,
        deducts cash, creates ledger entries.

        Args:
            fill_id: Fill identifier
            timestamp: Fill time
            symbol: Ticker
            quantity: Shares to buy to cover (positive)
            price: Buy price per share
            commission: Commission paid
            strategy_id: Strategy attribution (optional)

        Raises:
            ValueError: If insufficient short quantity to close
        """
        # Use strategy_id or "unattributed" for keying
        strat_id = strategy_id or "unattributed"
        key = (strat_id, symbol)

        # Match lots using LIFO
        tracker = self._lot_tracker[key]
        matches = tracker.match_close_short(quantity)

        total_realized_pnl = Decimal("0")
        closed_lot_ids: list[str] = []

        # Calculate realized P&L for each matched lot
        for lot, qty_closed in matches:
            # Realized P&L = (entry_price - exit_price) * quantity - total_commissions
            # Both entry and exit commissions are allocated proportionally
            # For shorts: profit when buy back at lower price
            exit_commission = commission * (qty_closed / quantity)
            entry_commission = lot.entry_commission * (abs(qty_closed) / abs(lot.quantity))
            total_commissions = entry_commission + exit_commission
            pnl = (lot.entry_price - price) * qty_closed - total_commissions
            total_realized_pnl += pnl
            closed_lot_ids.append(lot.lot_id)

            logger.debug(
                "portfolio_service.lot_closed",
                lot_id=lot.lot_id,
                symbol=symbol,
                side="short",
                quantity=qty_closed,
                entry_price=float(lot.entry_price),
                exit_price=float(price),
                realized_pnl=float(pnl),
            )

        # Update position
        position = self._positions[key]
        position.quantity += quantity  # Add back (shorts are negative)

        # Track realized P&L on this position
        position.realized_pl += total_realized_pnl

        # Remove closed lots from position.lots
        position.lots = [
            lot for lot in position.lots if not any(lot.lot_id == closed_id for closed_id in closed_lot_ids)
        ]

        # Also remove any "_remaining" versions that got split
        position.lots = [
            lot
            for lot in position.lots
            if not any(lot.lot_id.startswith(f"{closed_id}_remaining") for closed_id in closed_lot_ids)
        ]

        # Add any remaining lots from tracker
        tracker_lots = tracker.get_lots(LotSide.SHORT)
        for tracker_lot in tracker_lots:
            if tracker_lot not in position.lots:
                position.lots.append(tracker_lot)

        # Recalculate total_cost and avg_price
        position.total_cost = sum(
            (lot.quantity * lot.entry_price for lot in position.lots),
            start=Decimal("0"),
        )
        position.avg_price = position.total_cost / position.quantity if position.quantity != 0 else Decimal("0")
        position.last_updated = timestamp

        # Update cash (buy to cover = cash out)
        cash_flow = -(quantity * price + commission)
        self._cash += cash_flow
        self._cumulative_realized_pnl += total_realized_pnl
        self._total_commissions += commission

        # Create ledger entry
        entry = LedgerEntry(
            entry_id=f"{fill_id}_ledger",
            timestamp=timestamp,
            entry_type=LedgerEntryType.FILL,
            symbol=symbol,
            quantity=quantity,  # Positive for buy
            price=price,
            cash_flow=cash_flow,
            commission=commission,
            realized_pnl=total_realized_pnl,
            fill_id=fill_id,
            lot_ids=closed_lot_ids,
            description=f"Buy to cover {quantity} {symbol} @ ${price} (LIFO close)",
        )
        self._ledger.add_entry(entry)

        logger.info(
            "portfolio_service.position_closed",
            symbol=symbol,
            side="short",
            quantity=float(quantity),
            price=float(price),
            realized_pnl=float(total_realized_pnl),
            lots_closed=len(matches),
        )

    # ==================== Market Data ====================

    def update_prices(self, prices: dict[str, Decimal]) -> None:
        """
        Update mark-to-market prices (intraday).

        Args:
            prices: Dict mapping symbol → current price
        """
        for symbol, price in prices.items():
            # Update all positions for this symbol across all strategies
            for key in self._positions:
                strategy_id, sym = key
                if sym == symbol:
                    self._positions[key].update_market_value(price)

        logger.debug(
            "portfolio_service.prices_updated",
            count=len(prices),
        )

    def mark_to_market(self, timestamp: datetime) -> None:
        """
        Perform end-of-day mark-to-market valuation.

        This is the comprehensive EOD process that:
        1. Updates all position values with current prices
        2. Calculates unrealized P&L
        3. Accrues borrow fees on short positions
        4. Accrues margin interest on negative cash
        5. Creates ledger entries for all fees/interest

        Should be called once per trading day at market close.

        Args:
            timestamp: Time of mark (typically end of day, e.g., 16:00)

        Example:
            >>> # End of day processing
            >>> portfolio.mark_to_market(datetime(2020, 1, 2, 16, 0))
        """
        # 1. Price updates are done via update_prices() before calling this

        # 2. Accrue borrow fees on short positions
        total_borrow_fee = Decimal("0")
        for key, position in self._positions.items():
            strategy_id, symbol = key
            if position.quantity < 0 and position.current_price is not None:
                # Short position - charge borrow fee
                # Daily fee = abs(market_value) * annual_rate / day_count
                short_value = abs(position.market_value)

                # Get borrow rate (symbol-specific or default)
                borrow_rate = self.config.borrow_rate_by_symbol.get(symbol, self.config.default_borrow_rate_apr)

                # Calculate daily borrow fee
                daily_fee = short_value * borrow_rate / Decimal(str(self.config.day_count_convention))

                if daily_fee > 0:
                    # Debit cash for borrow fee
                    self._cash -= daily_fee
                    total_borrow_fee += daily_fee
                    self._total_borrow_fees += daily_fee

                    # Create ledger entry
                    entry = LedgerEntry(
                        entry_id=f"{symbol}_borrow_{timestamp.isoformat()}",
                        timestamp=timestamp,
                        entry_type=LedgerEntryType.BORROW_FEE,
                        symbol=symbol,
                        quantity=position.quantity,
                        cash_flow=-daily_fee,
                        description=f"Borrow fee on {abs(position.quantity)} shares @ {borrow_rate * 100}% APR",
                        metadata={"borrow_rate": str(borrow_rate), "short_value": str(short_value)},
                    )
                    self._ledger.add_entry(entry)

        # 3. Accrue margin interest on negative cash
        if self._cash < 0:
            # Negative cash - charge margin interest
            # Daily interest = abs(cash) * annual_rate / day_count
            daily_interest = (
                abs(self._cash) * self.config.margin_rate_apr / Decimal(str(self.config.day_count_convention))
            )

            if daily_interest > 0:
                # Debit cash for margin interest (makes it more negative)
                self._cash -= daily_interest
                self._total_margin_interest += daily_interest

                # Create ledger entry
                entry = LedgerEntry(
                    entry_id=f"margin_int_{timestamp.isoformat()}",
                    timestamp=timestamp,
                    entry_type=LedgerEntryType.MARGIN_INTEREST,
                    cash_flow=-daily_interest,
                    description=f"Margin interest on ${abs(self._cash):.2f} @ {self.config.margin_rate_apr * 100}% APR",
                    metadata={"margin_rate": str(self.config.margin_rate_apr), "negative_cash": str(abs(self._cash))},
                )
                self._ledger.add_entry(entry)

        logger.info(
            "portfolio_service.mark_to_market",
            timestamp=timestamp.isoformat(),
            total_borrow_fee=float(total_borrow_fee),
            cash_after_fees=float(self._cash),
        )

    # ==================== Corporate Actions ====================

    def process_split(
        self,
        symbol: str,
        split_date: datetime,
        ratio: Decimal,
    ) -> None:
        """
        Process stock split or reverse split.

        Adjusts all lots for this symbol:
        - quantity = quantity * ratio
        - entry_price = entry_price / ratio
        - Total value preserved

        Works for both long and short positions.

        Args:
            symbol: Symbol splitting
            split_date: Date of split
            ratio: Split ratio (4.0 for 4-for-1, 0.25 for 1-for-4 reverse)

        Raises:
            ValueError: If ratio is zero or negative
            ValueError: If no position exists for symbol

        Example:
            >>> # 4-for-1 split on 100 shares @ $400
            >>> portfolio.process_split("AAPL", datetime.now(), Decimal("4.0"))
            >>> # Result: 400 shares @ $100 (value preserved: $40,000)
        """
        # Validate ratio
        if ratio <= 0:
            raise ValueError(f"Split ratio must be positive, got {ratio}")

        # Find all positions for this symbol (across all strategies)
        positions_to_split = [(key, pos) for key, pos in self._positions.items() if key[1] == symbol]

        if not positions_to_split:
            logger.debug(
                "portfolio_service.split_skipped",
                symbol=symbol,
                reason="No position found for this symbol",
            )
            # Emit rich-formatted log showing zero impact
            impact_logger = LoggerFactory.get_logger("qtrader.events.corporate_action_impact")
            impact_logger.info(
                "event.display",
                event_type="corporate_action_impact",
                symbol=symbol,
                action_type="SPLIT",
                ratio=float(ratio),
                old_quantity=0,
                new_quantity=0,
                old_avg_price=0.0,
                new_avg_price=0.0,
                strategy_id="none",
            )
            return

        # Process each position
        for key, position in positions_to_split:
            strategy_id, _ = key

            # Adjust all lots - iterate over a COPY to avoid modifying list during iteration
            for lot in list(position.lots):
                # Adjust quantity: multiply by ratio
                new_quantity = lot.quantity * ratio
                # Adjust entry price: divide by ratio
                new_entry_price = lot.entry_price / ratio

                # Create new lot with adjusted values
                # Note: We need to update the lot in place, but lots are immutable
                # So we'll rebuild the lot with adjusted values
                adjusted_lot = Lot(
                    lot_id=lot.lot_id,
                    symbol=lot.symbol,
                    side=lot.side,
                    quantity=new_quantity,
                    entry_price=new_entry_price,
                    entry_timestamp=lot.entry_timestamp,
                    entry_fill_id=lot.entry_fill_id,
                    entry_commission=lot.entry_commission,
                    realized_pnl=lot.realized_pnl,
                )

                # Replace old lot with adjusted lot in position
                position.lots = [
                    adjusted_lot if existing_lot.lot_id == lot.lot_id else existing_lot
                    for existing_lot in position.lots
                ]

                # Update lot tracker (use tuple key)
                if key in self._lot_tracker:
                    tracker = self._lot_tracker[key]
                    if lot.side == LotSide.LONG:
                        tracker.remove_lot(lot.lot_id, LotSide.LONG)
                        tracker.add_lot(adjusted_lot)
                    elif lot.side == LotSide.SHORT:
                        tracker.remove_lot(lot.lot_id, LotSide.SHORT)
                        tracker.add_lot(adjusted_lot)

            # Adjust position aggregate values
            position.quantity = position.quantity * ratio
            # total_cost stays the same (value preserved)
            # avg_price = total_cost / new_quantity
            if position.quantity != 0:
                position.avg_price = position.total_cost / position.quantity
            else:
                position.avg_price = Decimal("0")

            # Update market value if current price exists
            if position.current_price is not None:
                new_price = position.current_price / ratio
                position.update_market_value(new_price)

            position.last_updated = split_date

            # Create ledger entry
            split_type = "split" if ratio > 1 else "reverse split"
            entry = LedgerEntry(
                entry_id=f"{strategy_id}_{symbol}_split_{split_date.isoformat()}",
                timestamp=split_date,
                entry_type=LedgerEntryType.SPLIT,
                symbol=symbol,
                quantity=position.quantity,  # New quantity after split
                price=position.avg_price,  # New avg price after split
                cash_flow=Decimal("0"),  # No cash impact
                description=f"{split_type} {ratio}:1 for {symbol} (strategy {strategy_id})",
                metadata={"ratio": str(ratio), "split_type": split_type, "strategy_id": strategy_id},
            )
            self._ledger.add_entry(entry)

            # Emit rich-formatted log for portfolio impact
            impact_logger = LoggerFactory.get_logger("qtrader.events.corporate_action_impact")
            impact_logger.info(
                "event.display",
                event_type="corporate_action_impact",
                symbol=symbol,
                action_type="SPLIT",
                ratio=float(ratio),
                old_quantity=float(position.quantity / ratio),
                new_quantity=float(position.quantity),
                old_avg_price=float(position.avg_price * ratio),
                new_avg_price=float(position.avg_price),
                strategy_id=strategy_id,
            )

    def process_dividend(
        self,
        symbol: str,
        effective_date: datetime,
        amount_per_share: Decimal,
    ) -> None:
        """
        Process cash dividend.

        Behavior depends on portfolio adjustment_mode configuration:
        - If adjustment_mode='split_adjusted': Process dividend as cash flow (prices show dividend drops)
        - If adjustment_mode='total_return': Skip dividend (prices already include dividend appreciation)

        For long positions: Cash increases (income)
        For short positions: Cash decreases (expense)

        Args:
            symbol: Symbol paying dividend
            effective_date: Effective date when dividend is applied to accounts
            amount_per_share: Dividend per share

        Raises:
            ValueError: If amount is negative
            ValueError: If no position exists for symbol

        Example:
            >>> # Portfolio using 'split_adjusted'
            >>> portfolio.process_dividend("AAPL", datetime.now(), Decimal("0.82"))
            >>> # Cash increases by $82 (dividend as separate cash flow)
            >>>
            >>> # Portfolio using 'total_return'
            >>> portfolio.process_dividend("AAPL", datetime.now(), Decimal("0.82"))
            >>> # Skipped - dividend already reflected in price appreciation
        """
        # Skip dividend processing if using total-return adjusted prices
        if self._adjustment_mode == "total_return":
            # Log at DEBUG level - corporate action already displayed via CorporateActionEvent
            logger.debug(
                "portfolio_service.dividend_skipped",
                symbol=symbol,
                effective_date=effective_date.isoformat(),
                amount_per_share=str(amount_per_share),
                reason="Using total-return adjusted prices - dividends already reflected in price appreciation",
                adjustment_mode=self._adjustment_mode,
            )

            # Emit rich-formatted impact event showing dividend is in price (no cash flow)
            impact_logger = LoggerFactory.get_logger("qtrader.events.corporate_action_impact")

            # Check if we have any positions for this symbol
            positions_for_symbol = [(key, pos) for key, pos in self._positions.items() if key[1] == symbol]

            if positions_for_symbol:
                # Show impact for each position - quantity but no cash flow
                for key, position in positions_for_symbol:
                    strategy_id, _ = key
                    impact_logger.info(
                        "event.display",
                        event_type="corporate_action_impact",
                        symbol=symbol,
                        action_type="DIVIDEND",
                        quantity=int(position.quantity),
                        amount_per_share=float(amount_per_share),
                        total_amount=0.0,  # No cash flow - dividend in price
                        position_type="LONG" if position.quantity > 0 else "SHORT",
                        strategy_id=strategy_id,
                        note="No cash flow - dividend already reflected in total-return adjusted prices",
                    )
            else:
                # No position - show zero impact
                impact_logger.info(
                    "event.display",
                    event_type="corporate_action_impact",
                    symbol=symbol,
                    action_type="DIVIDEND",
                    quantity=0,
                    amount_per_share=float(amount_per_share),
                    total_amount=0.0,
                    position_type="NONE",
                    strategy_id="none",
                )

            return

        # Validate amount
        if amount_per_share < 0:
            raise ValueError(f"Dividend amount cannot be negative, got {amount_per_share}")

        # Find all positions for this symbol (across all strategies)
        positions_for_dividend = [(key, pos) for key, pos in self._positions.items() if key[1] == symbol]

        if not positions_for_dividend:
            logger.debug(
                "portfolio_service.dividend_skipped",
                symbol=symbol,
                reason="No position found for this symbol",
            )
            # Emit rich-formatted log showing zero impact
            impact_logger = LoggerFactory.get_logger("qtrader.events.corporate_action_impact")
            impact_logger.info(
                "event.display",
                event_type="corporate_action_impact",
                symbol=symbol,
                action_type="DIVIDEND",
                quantity=0,
                amount_per_share=float(amount_per_share),
                total_amount=0.0,
                position_type="NONE",
                strategy_id="none",
            )
            return

        # Process dividend for each position
        for key, position in positions_for_dividend:
            strategy_id, _ = key

            # Calculate dividend cash flow
            # Long positions receive dividends (positive cash flow)
            # Short positions pay dividends (negative cash flow)
            cash_flow = position.quantity * amount_per_share

            # Update cash
            self._cash += cash_flow

            # Track cumulative dividends (global)
            if cash_flow > 0:
                self._total_dividends_received += cash_flow
            else:
                self._total_dividends_paid += abs(cash_flow)

            # Track dividends on position (per-symbol)
            if cash_flow > 0:
                position.dividends_received += cash_flow
            else:
                position.dividends_paid += abs(cash_flow)

            # Create ledger entry
            entry = LedgerEntry(
                entry_id=f"{strategy_id}_{symbol}_dividend_{effective_date.isoformat()}",
                timestamp=effective_date,
                entry_type=LedgerEntryType.DIVIDEND,
                symbol=symbol,
                quantity=position.quantity,
                price=amount_per_share,
                cash_flow=cash_flow,
                description=f"Dividend ${amount_per_share}/share on {symbol} (strategy {strategy_id})",
                metadata={"amount_per_share": str(amount_per_share), "strategy_id": strategy_id},
            )
            self._ledger.add_entry(entry)

            # Emit rich-formatted log for portfolio impact
            impact_logger = LoggerFactory.get_logger("qtrader.events.corporate_action_impact")
            impact_logger.info(
                "event.display",
                event_type="corporate_action_impact",
                symbol=symbol,
                action_type="DIVIDEND",
                quantity=float(abs(position.quantity)),
                amount_per_share=float(amount_per_share),
                total_amount=float(abs(cash_flow)),
                position_type="LONG" if position.quantity > 0 else "SHORT",
                strategy_id=strategy_id,
            )

    # ==================== Queries ====================

    def get_position(self, symbol: str, strategy_id: str | None = None) -> Position | None:
        """Get current position for symbol and strategy.

        Args:
            symbol: Symbol to look up
            strategy_id: Strategy ID (uses "unattributed" if None)

        Returns:
            Position if found, None otherwise
        """
        strat_id = strategy_id or "unattributed"
        key = (strat_id, symbol)
        position = self._positions.get(key)
        if position and position.quantity == 0 and not self.config.keep_position_history:
            return None
        return position

    def get_positions(self) -> dict[tuple[str, str], Position]:
        """Get all current positions keyed by (strategy_id, symbol)."""
        if self.config.keep_position_history:
            return self._positions.copy()
        else:
            # Filter out flat positions
            return {key: pos for key, pos in self._positions.items() if pos.quantity != 0}

    def get_cash(self) -> Decimal:
        """Get current cash balance."""
        return self._cash

    def get_equity(self) -> Decimal:
        """Calculate total portfolio equity."""
        market_value = sum(pos.market_value for pos in self._positions.values())
        return self._cash + market_value

    def get_state(self) -> PortfolioState:
        """
        Get complete portfolio state snapshot.

        Returns:
            Immutable state snapshot
        """
        # Calculate exposures
        long_exposure = Decimal("0")
        short_exposure = Decimal("0")
        unrealized_pnl = Decimal("0")

        for position in self._positions.values():
            unrealized_pnl += position.unrealized_pnl
            if position.quantity > 0:
                long_exposure += position.market_value
            elif position.quantity < 0:
                short_exposure += abs(position.market_value)

        net_exposure = long_exposure - short_exposure
        gross_exposure = long_exposure + short_exposure
        market_value_total = sum((pos.market_value for pos in self._positions.values()), start=Decimal("0"))
        equity = self._cash + market_value_total

        # Calculate leverage (handle zero equity)
        leverage = gross_exposure / equity if equity > 0 else Decimal("0")

        # Realized P&L from ledger
        realized_pnl = self._calculate_realized_pnl()

        return PortfolioState(
            timestamp=datetime.now(),
            cash=self._cash,
            positions=self._positions.copy(),
            equity=equity,
            market_value=market_value_total,
            realized_pnl=realized_pnl,
            unrealized_pnl=unrealized_pnl,
            total_pnl=realized_pnl + unrealized_pnl,
            long_exposure=long_exposure,
            short_exposure=short_exposure,
            net_exposure=net_exposure,
            gross_exposure=gross_exposure,
            leverage=leverage,
            total_commissions=self._total_commissions,
            total_borrow_fees=self._total_borrow_fees,
            total_margin_interest=self._total_margin_interest,
            total_dividends_received=self._total_dividends_received,
            total_dividends_paid=self._total_dividends_paid,
        )

    def get_ledger(
        self,
        since: datetime | None = None,
        entry_types: list[LedgerEntryType] | None = None,
    ) -> list[LedgerEntry]:
        """Get ledger entries."""
        entries = self._ledger.get_entries(since=since)

        # Filter by type if specified
        if entry_types is not None:
            entries = [e for e in entries if e.entry_type in entry_types]

        return entries

    def get_realized_pnl(
        self,
        symbol: str | None = None,
        since: datetime | None = None,
    ) -> Decimal:
        """Get realized P&L."""
        entries = self.get_ledger(since=since, entry_types=[LedgerEntryType.FILL])

        if symbol is not None:
            entries = [e for e in entries if e.symbol == symbol]

        # Sum realized P&L from entries
        return sum((e.realized_pnl for e in entries if e.realized_pnl is not None), start=Decimal("0"))

    def get_unrealized_pnl(
        self,
        symbol: str | None = None,
    ) -> Decimal:
        """Get unrealized P&L."""
        if symbol is not None:
            position = self.get_position(symbol)
            return position.unrealized_pnl if position else Decimal("0")

        return sum((pos.unrealized_pnl for pos in self._positions.values()), start=Decimal("0"))

    # ==================== Helper Methods ====================

    def _validate_fill_inputs(
        self,
        fill_id: str,
        quantity: Decimal,
        price: Decimal,
        commission: Decimal,
    ) -> None:
        """
        Validate fill inputs.

        Args:
            fill_id: Fill identifier
            quantity: Quantity
            price: Price
            commission: Commission

        Raises:
            ValueError: If any validation fails
        """
        if quantity <= 0:
            raise ValueError(f"Quantity must be positive, got {quantity}")

        if price <= 0:
            raise ValueError(f"Price must be positive, got {price}")

        if commission < 0:
            raise ValueError(f"Commission cannot be negative, got {commission}")

        # Check for duplicate fill_id in ledger
        existing_entries = [e for e in self._ledger.get_entries() if e.fill_id == fill_id]
        if existing_entries:
            raise ValueError(f"Fill ID {fill_id} already exists in ledger")

    def _has_long_position(self, symbol: str, strategy_id: str | None = None) -> bool:
        """Check if symbol has long position for given strategy.

        Args:
            symbol: Symbol to check
            strategy_id: Strategy ID (uses "unattributed" if None)

        Returns:
            True if long position exists for this strategy-symbol pair
        """
        strat_id = strategy_id or "unattributed"
        key = (strat_id, symbol)
        if key not in self._positions:
            return False
        return self._positions[key].quantity > 0

    def _has_short_position(self, symbol: str, strategy_id: str | None = None) -> bool:
        """Check if symbol has short position for given strategy.

        Args:
            symbol: Symbol to check
            strategy_id: Strategy ID (uses "unattributed" if None)

        Returns:
            True if short position exists for this strategy-symbol pair
        """
        strat_id = strategy_id or "unattributed"
        key = (strat_id, symbol)
        if key not in self._positions:
            return False
        return self._positions[key].quantity < 0

    def _calculate_realized_pnl(self) -> Decimal:
        """Calculate total realized P&L from ledger."""
        entries = self._ledger.get_entries(entry_type=LedgerEntryType.FILL)
        return sum((e.realized_pnl for e in entries if e.realized_pnl is not None), start=Decimal("0"))

    # ==================== State Management (Week 4) ====================

    def get_snapshot(self, timestamp: datetime | None = None) -> dict:
        """
        Get complete portfolio state snapshot for persistence.

        This captures all internal state needed to reconstruct the portfolio,
        including cash, positions, lots, ledger entries, and cumulative metrics.
        The snapshot can be serialized to JSON and later restored.

        Args:
            timestamp: Timestamp for the snapshot (defaults to current state time)

        Returns:
            Dictionary containing complete portfolio state with:
            - metadata: timestamp, config
            - cash: current cash balance
            - positions: all positions with lots
            - ledger: all ledger entries
            - cumulative_metrics: all tracked metrics

        Example:
            >>> snapshot = portfolio.get_snapshot(datetime.now())
            >>> import json
            >>> with open("portfolio_state.json", "w") as f:
            ...     json.dump(snapshot, f, default=str)
        """
        if timestamp is None:
            timestamp = datetime.now()

        # Serialize positions with their lots
        positions_data = {}
        for key, position in self._positions.items():
            strategy_id, symbol = key
            # Get lots from tracker
            lots_data = []
            if key in self._lot_tracker:
                tracker = self._lot_tracker[key]
                # Get lots for both long and short sides
                all_lots = tracker.get_lots(LotSide.LONG) + tracker.get_lots(LotSide.SHORT)
                lots_data = [
                    {
                        "lot_id": lot.lot_id,
                        "symbol": lot.symbol,
                        "quantity": str(lot.quantity),
                        "entry_price": str(lot.entry_price),
                        "entry_timestamp": lot.entry_timestamp.isoformat(),
                        "entry_fill_id": lot.entry_fill_id,
                        "side": lot.side,
                    }
                    for lot in all_lots
                ]

            # Use tuple key as string for JSON serialization
            key_str = f"{strategy_id}:{symbol}"
            positions_data[key_str] = {
                "strategy_id": strategy_id,
                "symbol": position.symbol,
                "quantity": str(position.quantity),
                "avg_price": str(position.avg_price),
                "market_value": str(position.market_value),
                "unrealized_pnl": str(position.unrealized_pnl),
                "current_price": str(position.current_price) if position.current_price is not None else None,
                "lots": lots_data,
            }

        # Serialize ledger entries
        ledger_entries = [
            {
                "entry_id": entry.entry_id,
                "timestamp": entry.timestamp.isoformat(),
                "entry_type": entry.entry_type,
                "symbol": entry.symbol,
                "quantity": str(entry.quantity) if entry.quantity is not None else None,
                "price": str(entry.price) if entry.price is not None else None,
                "cash_flow": str(entry.cash_flow),
                "commission": str(entry.commission) if entry.commission is not None else None,
                "realized_pnl": str(entry.realized_pnl) if entry.realized_pnl is not None else None,
                "fill_id": entry.fill_id,
                "description": entry.description,
                "metadata": entry.metadata,
            }
            for entry in self._ledger.get_entries()
        ]

        snapshot = {
            "metadata": {
                "timestamp": timestamp.isoformat(),
                "snapshot_version": "1.0",
                "config": {
                    "initial_cash": str(self.config.initial_cash),
                    "lot_method_long": self.config.lot_method_long,
                    "lot_method_short": self.config.lot_method_short,
                    "default_borrow_rate_apr": str(self.config.default_borrow_rate_apr),
                    "margin_rate_apr": str(self.config.margin_rate_apr),
                    "day_count_convention": self.config.day_count_convention,
                },
            },
            "cash": str(self._cash),
            "positions": positions_data,
            "ledger": ledger_entries,
            "cumulative_metrics": {
                "cumulative_realized_pnl": str(self._cumulative_realized_pnl),
                "total_commissions": str(self._total_commissions),
                "total_borrow_fees": str(self._total_borrow_fees),
                "total_margin_interest": str(self._total_margin_interest),
                "total_dividends_received": str(self._total_dividends_received),
                "total_dividends_paid": str(self._total_dividends_paid),
            },
        }

        logger.info(
            "portfolio_service.snapshot_created",
            timestamp=timestamp.isoformat(),
            num_positions=len(self._positions),
            num_ledger_entries=len(ledger_entries),
        )

        return snapshot

    def restore_from_snapshot(self, snapshot: dict) -> None:
        """
        Restore portfolio state from snapshot.

        Completely replaces current portfolio state with the saved snapshot.
        Use with caution - all current state will be lost.

        Args:
            snapshot: Dictionary from get_snapshot()

        Raises:
            ValueError: If snapshot is invalid or incompatible

        Example:
            >>> import json
            >>> with open("portfolio_state.json") as f:
            ...     snapshot = json.load(f)
            >>> portfolio.restore_from_snapshot(snapshot)
        """
        # Validate snapshot structure
        required_keys = {"metadata", "cash", "positions", "ledger", "cumulative_metrics"}
        if not required_keys.issubset(snapshot.keys()):
            missing = required_keys - snapshot.keys()
            raise ValueError(f"Invalid snapshot: missing keys {missing}")

        # Restore cash
        self._cash = Decimal(snapshot["cash"])

        # Restore positions and lots
        self._positions = {}
        self._lot_tracker = {}

        for key_str, pos_data in snapshot["positions"].items():
            # Parse the key (format: "strategy_id:symbol")
            strategy_id, symbol = key_str.split(":", 1)
            key = (strategy_id, symbol)

            # Restore position
            position = Position(
                symbol=symbol,
                quantity=Decimal(pos_data["quantity"]),
                avg_price=Decimal(pos_data["avg_price"]),
                market_value=Decimal(pos_data["market_value"]),
                unrealized_pnl=Decimal(pos_data["unrealized_pnl"]),
                current_price=Decimal(pos_data["current_price"]) if pos_data["current_price"] is not None else None,
                strategy_id=strategy_id,
            )
            self._positions[key] = position

            # Restore lots
            if pos_data["lots"]:
                lots = [
                    Lot(
                        lot_id=lot_data["lot_id"],
                        symbol=lot_data["symbol"],
                        side=lot_data["side"],
                        quantity=Decimal(lot_data["quantity"]),
                        entry_price=Decimal(lot_data["entry_price"]),
                        entry_timestamp=datetime.fromisoformat(lot_data["entry_timestamp"]),
                        entry_fill_id=lot_data["entry_fill_id"],
                    )
                    for lot_data in pos_data["lots"]
                ]

                tracker = LotTracker()
                for lot in lots:
                    tracker.add_lot(lot)
                self._lot_tracker[key] = tracker

        # Restore ledger
        self._ledger = Ledger(max_entries=self.config.max_ledger_entries)
        for entry_data in snapshot["ledger"]:
            entry = LedgerEntry(
                entry_id=entry_data["entry_id"],
                timestamp=datetime.fromisoformat(entry_data["timestamp"]),
                entry_type=entry_data["entry_type"],
                symbol=entry_data["symbol"],
                quantity=Decimal(entry_data["quantity"]) if entry_data["quantity"] is not None else None,
                price=Decimal(entry_data["price"]) if entry_data["price"] is not None else None,
                cash_flow=Decimal(entry_data["cash_flow"]),
                commission=Decimal(entry_data["commission"]) if entry_data["commission"] is not None else Decimal("0"),
                realized_pnl=Decimal(entry_data["realized_pnl"]) if entry_data["realized_pnl"] is not None else None,
                fill_id=entry_data["fill_id"],
                description=entry_data["description"],
                metadata=entry_data["metadata"],
            )
            self._ledger.add_entry(entry)

        # Restore cumulative metrics
        metrics = snapshot["cumulative_metrics"]
        self._cumulative_realized_pnl = Decimal(metrics["cumulative_realized_pnl"])
        self._total_commissions = Decimal(metrics["total_commissions"])
        self._total_borrow_fees = Decimal(metrics["total_borrow_fees"])
        self._total_margin_interest = Decimal(metrics["total_margin_interest"])
        self._total_dividends_received = Decimal(metrics["total_dividends_received"])
        self._total_dividends_paid = Decimal(metrics["total_dividends_paid"])

        logger.info(
            "portfolio_service.snapshot_restored",
            timestamp=snapshot["metadata"]["timestamp"],
            num_positions=len(self._positions),
            num_ledger_entries=len(snapshot["ledger"]),
        )

    # ==================== Query Methods ====================

    def get_fills(
        self,
        symbol: str | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
        side: Literal["buy", "sell"] | None = None,
    ) -> list[LedgerEntry]:
        """
        Get fill history with optional filters.

        Returns all fill entries from the ledger, optionally filtered by
        symbol, date range, and side.

        Args:
            symbol: Filter by symbol (optional)
            since: Filter fills on or after this time (optional)
            until: Filter fills before this time (optional)
            side: Filter by side "buy" or "sell" (optional)

        Returns:
            List of LedgerEntry objects representing fills

        Example:
            >>> # Get all AAPL buys in January
            >>> fills = portfolio.get_fills(
            ...     symbol="AAPL",
            ...     since=datetime(2024, 1, 1),
            ...     until=datetime(2024, 2, 1),
            ...     side="buy"
            ... )
        """
        entries = self._ledger.get_entries(since=since, entry_type=LedgerEntryType.FILL)

        # Filter by until time
        if until is not None:
            entries = [e for e in entries if e.timestamp < until]

        # Filter by symbol
        if symbol is not None:
            entries = [e for e in entries if e.symbol == symbol]

        # Filter by side
        if side is not None:
            if side == "buy":
                entries = [e for e in entries if e.quantity is not None and e.quantity > 0]
            elif side == "sell":
                entries = [e for e in entries if e.quantity is not None and e.quantity < 0]

        return entries

    def get_all_lots(self, symbol: str | None = None, strategy_id: str | None = None) -> list[Lot]:
        """
        Get all current lots, optionally filtered by symbol and/or strategy.

        Returns the current lot holdings for tracking purposes.
        Useful for debugging and detailed position analysis.

        Args:
            symbol: Filter by symbol (optional, returns all if None)
            strategy_id: Filter by strategy (optional, returns all if None)

        Returns:
            List of Lot objects currently held

        Example:
            >>> # Get all AAPL lots for strategy1
            >>> lots = portfolio.get_all_lots("AAPL", "strategy1")
            >>> for lot in lots:
            ...     print(f"{lot.quantity} @ ${lot.entry_price}")
        """
        if symbol is not None or strategy_id is not None:
            # Filter by symbol and/or strategy
            filtered_lots = []
            for key, tracker in self._lot_tracker.items():
                strat_id, sym = key
                # Check if matches filter criteria
                if (symbol is None or sym == symbol) and (strategy_id is None or strat_id == strategy_id):
                    filtered_lots.extend(tracker.get_lots(LotSide.LONG))
                    filtered_lots.extend(tracker.get_lots(LotSide.SHORT))
            return filtered_lots

        # Return all lots across all symbols and strategies
        all_lots = []
        for tracker in self._lot_tracker.values():
            all_lots.extend(tracker.get_lots(LotSide.LONG))
            all_lots.extend(tracker.get_lots(LotSide.SHORT))
        return all_lots

    # ==================== Utility Methods ====================

    def clear_positions(self) -> None:
        """
        Clear all positions and reset to cash-only state.

        WARNING: This is destructive and mainly useful for testing.
        Ledger history is preserved but positions and lots are cleared.

        Example:
            >>> portfolio.clear_positions()
            >>> assert len(portfolio.get_all_positions()) == 0
        """
        self._positions.clear()
        self._lot_tracker.clear()

        logger.warning(
            "portfolio_service.positions_cleared",
            cash_remaining=str(self._cash),
        )

    def validate_state(self) -> dict[str, bool]:
        """
        Validate internal state consistency.

        Performs various consistency checks on portfolio state:
        - Position quantities match lot quantities
        - Cumulative realized P&L matches ledger
        - All positions have corresponding lot trackers
        - No orphaned lot trackers

        Returns:
            Dictionary with validation results for each check

        Example:
            >>> results = portfolio.validate_state()
            >>> assert all(results.values()), "State inconsistency detected"
        """
        results = {}

        # Check: Position quantities match lot quantities
        position_lot_match = True
        for symbol, position in self._positions.items():
            if symbol in self._lot_tracker:
                tracker = self._lot_tracker[symbol]
                long_lots = tracker.get_lots(LotSide.LONG)
                short_lots = tracker.get_lots(LotSide.SHORT)
                lot_total = sum(lot.quantity for lot in long_lots) + sum(lot.quantity for lot in short_lots)
                if lot_total != position.quantity:
                    position_lot_match = False
                    logger.error(
                        "portfolio_service.validation_error",
                        check="position_lot_match",
                        symbol=symbol,
                        position_qty=str(position.quantity),
                        lot_total=str(lot_total),
                    )
        results["position_lot_match"] = position_lot_match

        # Check: Cumulative realized P&L matches ledger calculation
        ledger_realized_pnl = self._calculate_realized_pnl()
        pnl_match = self._cumulative_realized_pnl == ledger_realized_pnl
        if not pnl_match:
            logger.error(
                "portfolio_service.validation_error",
                check="realized_pnl_match",
                cumulative=str(self._cumulative_realized_pnl),
                ledger_calc=str(ledger_realized_pnl),
            )
        results["realized_pnl_match"] = pnl_match

        # Check: All positions have lot trackers
        positions_have_trackers = all(symbol in self._lot_tracker for symbol in self._positions)
        if not positions_have_trackers:
            missing = [s for s in self._positions if s not in self._lot_tracker]
            logger.error(
                "portfolio_service.validation_error",
                check="positions_have_trackers",
                missing_trackers=missing,
            )
        results["positions_have_trackers"] = positions_have_trackers

        # Check: No orphaned lot trackers
        no_orphaned_trackers = all(symbol in self._positions for symbol in self._lot_tracker)
        if not no_orphaned_trackers:
            orphaned = [s for s in self._lot_tracker if s not in self._positions]
            logger.error(
                "portfolio_service.validation_error",
                check="no_orphaned_trackers",
                orphaned_trackers=orphaned,
            )
        results["no_orphaned_trackers"] = no_orphaned_trackers

        return results

    # ==================== Phase 5: Event Handlers ====================

    def on_bar(self, event: PriceBarEvent) -> None:
        """
        Handle bar event - update latest prices for mark-to-market and publish portfolio state.

        Phase 5: Immediately publish PortfolioStateEvent after marking to market.
        This allows ManagerService to cache the latest portfolio state before processing signals.

        Args:
            event: Price bar event with symbol and OHLCV data
        """
        # Update latest price using configured adjustment mode (default to close)
        if self._adjustment_mode == "split_adjusted":
            price = event.close
        else:  # total_return
            price = event.close_adj if event.close_adj is not None else event.close
        self._latest_prices[event.symbol] = price
        logger.debug(
            "portfolio_service.price_updated",
            symbol=event.symbol,
            price=str(price),
            adjustment_mode=self._adjustment_mode,
        )

        # Mark to market and publish portfolio state (Phase 5)
        self._publish_portfolio_state(event.timestamp)

    def _publish_portfolio_state(self, timestamp: str) -> None:
        """
        Calculate portfolio metrics and publish PortfolioStateEvent.

        Phase 5: Called after each bar to provide real-time portfolio state
        to ManagerService for risk calculations.

        Args:
            timestamp: ISO8601 timestamp for the portfolio state snapshot
        """
        from datetime import timezone

        from qtrader.events.events import PortfolioPosition, StrategyGroup

        # Update all positions with latest prices for accurate valuation
        for key, position in self._positions.items():
            strategy_id, symbol = key
            if symbol in self._latest_prices:
                position.update_market_value(self._latest_prices[symbol])

        # Calculate aggregate metrics
        total_market_value = sum((pos.market_value for pos in self._positions.values()), start=Decimal("0"))
        total_unrealized_pl = sum((pos.unrealized_pnl for pos in self._positions.values()), start=Decimal("0"))
        current_portfolio_equity = self._cash + total_market_value

        # Calculate exposures
        long_exposure = Decimal("0")
        short_exposure = Decimal("0")

        for position in self._positions.values():
            if position.quantity > 0:
                long_exposure += position.market_value
            elif position.quantity < 0:
                short_exposure += abs(position.market_value)

        net_exposure = long_exposure - short_exposure
        gross_exposure = long_exposure + short_exposure

        # Calculate leverage (handle zero equity)
        leverage = gross_exposure / current_portfolio_equity if current_portfolio_equity > 0 else Decimal("0")

        # Group positions by strategy (now keyed by (strategy_id, symbol))
        strategies: dict[str, list[Position]] = {}
        for key, position in self._positions.items():
            strategy_id, symbol = key

            # Skip flat positions unless keeping history
            if position.quantity == 0 and not self.config.keep_position_history:
                continue

            if strategy_id not in strategies:
                strategies[strategy_id] = []
            strategies[strategy_id].append(position)

        # Build strategies_groups
        strategies_groups: list[StrategyGroup] = []
        for strategy_id, positions in strategies.items():
            # Build PortfolioPosition objects for each position
            portfolio_positions: list[PortfolioPosition] = []
            for pos in positions:
                # Determine side
                if pos.quantity > 0:
                    side = "long"
                elif pos.quantity < 0:
                    side = "short"
                else:
                    side = "flat"

                # Get market price (use latest price or current_price)
                market_price = pos.current_price or self._latest_prices.get(pos.symbol, Decimal("0"))

                # Calculate gross market value
                gross_market_value = pos.market_value

                # Calculate total position value including dividends
                total_position_value = gross_market_value + pos.dividends_received - pos.dividends_paid

                # Get last_updated timestamp in ISO format
                last_updated = pos.last_updated.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")

                portfolio_position = PortfolioPosition(
                    symbol=pos.symbol,
                    side=side,
                    open_quantity=int(pos.quantity),
                    average_fill_price=pos.avg_price,
                    commission_paid=pos.commission_paid,
                    cost_basis=abs(pos.total_cost),  # Always positive for schema
                    market_price=market_price,
                    gross_market_value=gross_market_value,
                    unrealized_pl=pos.unrealized_pnl,
                    realized_pl=pos.realized_pl,
                    dividends_received=pos.dividends_received,
                    dividends_paid=pos.dividends_paid,
                    total_position_value=total_position_value,
                    currency=self._reporting_currency,
                    last_updated=last_updated,
                )
                portfolio_positions.append(portfolio_position)

            # Create strategy group
            strategy_group = StrategyGroup(
                strategy_id=strategy_id,
                positions=portfolio_positions,
            )
            strategies_groups.append(strategy_group)

        # Calculate total P&L
        total_pl = self._cumulative_realized_pnl + total_unrealized_pl

        # Publish portfolio state event
        if self._event_bus:
            # Convert start_datetime to ISO8601 string with Z suffix
            start_datetime_str = self._start_datetime.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")

            state_event = PortfolioStateEvent(
                portfolio_id=self._portfolio_id,
                start_datetime=start_datetime_str,
                snapshot_datetime=timestamp,
                reporting_currency=self._reporting_currency,
                initial_portfolio_equity=self._initial_cash,
                cash_balance=self._cash,
                current_portfolio_equity=current_portfolio_equity,
                total_market_value=total_market_value,
                total_unrealized_pl=total_unrealized_pl,
                total_realized_pl=self._cumulative_realized_pnl,
                total_pl=total_pl,
                long_exposure=long_exposure,
                short_exposure=short_exposure,
                net_exposure=net_exposure,
                gross_exposure=gross_exposure,
                leverage=leverage,
                strategies_groups=strategies_groups,
                total_commissions_paid=self._total_commissions,
                total_dividends_received=self._total_dividends_received,
                total_dividends_paid=self._total_dividends_paid,
                total_borrow_fees=self._total_borrow_fees,
                total_margin_interest=self._total_margin_interest,
                source_service="portfolio_service",
                # No correlation_id/causation_id - portfolio state is a snapshot, not part of causation chain
            )
            self._event_bus.publish(state_event)

            logger.debug(
                "portfolio_service.state_published",
                timestamp=timestamp,
                portfolio_id=self._portfolio_id,
                current_portfolio_equity=str(current_portfolio_equity),
                cash_balance=str(self._cash),
                num_positions=len([p for p in self._positions.values() if p.quantity != 0]),
                num_strategies=len(strategies_groups),
            )

    def on_valuation_trigger(self, event: ValuationTriggerEvent) -> None:
        """
        Handle valuation trigger - calculate portfolio metrics and publish state.

        Legacy support: ValuationTriggerEvent can still trigger portfolio state publication.
        Phase 5: Portfolio state is now primarily published after each bar.

        Args:
            event: Valuation trigger event
        """
        # Use occurred_at from the event envelope
        timestamp = event.occurred_at.isoformat().replace("+00:00", "Z")
        self._publish_portfolio_state(timestamp)

    def on_fill(self, event: FillEvent) -> None:
        """
        Handle fill event: apply to portfolio and emit TradeEvent.

        This handler applies the fill to the portfolio, tracks which fills belong
        to the same trade, and emits TradeEvent when a trade opens or closes.

        TradeEvent provides a complete view of a trade with all its fills, serving
        as the single source of truth for trade grouping. Downstream consumers
        (reporting, analytics) use TradeEvent instead of tracking fills themselves.

        Args:
            event: Fill event from ExecutionService
        """
        from datetime import datetime

        # Convert ISO8601 string to datetime
        timestamp_dt = datetime.fromisoformat(event.timestamp.replace("Z", "+00:00"))

        # Cast side to literal type expected by apply_fill
        side: Literal["buy", "sell"] = "buy" if event.side == "buy" else "sell"

        # Apply fill and get trade tracking info
        trade_id, trade_closed = self.apply_fill(
            fill_id=event.fill_id,
            timestamp=timestamp_dt,
            symbol=event.symbol,
            side=side,
            quantity=event.filled_quantity,  # FillEvent uses filled_quantity
            price=event.fill_price,  # FillEvent uses fill_price
            commission=event.commission,
            strategy_id=event.strategy_id,  # Pass strategy attribution
        )

        logger.debug(
            "portfolio_service.fill_applied",
            fill_id=event.fill_id,
            symbol=event.symbol,
            side=event.side,
            quantity=str(event.filled_quantity),
            strategy_id=event.strategy_id,
            trade_id=trade_id,
            trade_closed=trade_closed,
        )

        # Emit TradeEvent if we have an active trade
        if trade_id and self._event_bus is not None:
            strat_id = event.strategy_id if event.strategy_id is not None else "unattributed"
            key = (strat_id, event.symbol)

            if key in self._active_trades:
                trade_data = self._active_trades[key]

                # Determine status and exit timestamp
                status = "closed" if trade_closed else "open"
                exit_timestamp = event.timestamp if trade_closed else None

                # Get realized P&L and exit price if trade is closed
                realized_pnl = trade_data.get("realized_pnl") if trade_closed else None
                exit_price = trade_data.get("exit_price") if trade_closed else None

                # Create TradeEvent
                trade_event = TradeEvent(
                    trade_id=trade_data["trade_id"],
                    timestamp=event.timestamp,
                    strategy_id=strat_id,
                    symbol=event.symbol,
                    status=status,
                    side=trade_data["side"],
                    fills=trade_data["fills"],
                    entry_price=trade_data["entry_price"],
                    exit_price=exit_price,
                    current_quantity=trade_data["current_quantity"],
                    realized_pnl=realized_pnl,
                    commission_total=trade_data["commission_total"],
                    entry_timestamp=trade_data["entry_timestamp"],
                    exit_timestamp=exit_timestamp,
                    source_service="portfolio_service",
                    correlation_id=event.correlation_id,  # Propagate workflow ID
                    causation_id=event.event_id,  # This trade was caused by the fill
                )  # Publish TradeEvent
                self._event_bus.publish(trade_event)

                # Remove closed trade from tracking
                if trade_closed:
                    del self._active_trades[key]

        # In event-driven mode, republish portfolio state after fill
        # This ensures ManagerService has up-to-date equity/positions
        if self._event_bus is not None:
            # Get the market price for mark-to-market from the current position
            # For Phase 5 MVP, we use the fill price as the current market price
            # In production, this would come from the most recent bar
            self.update_prices({event.symbol: event.fill_price})
            self._publish_portfolio_state(event.timestamp)

    def on_corporate_action(self, event: CorporateActionEvent) -> None:
        """
        Handle corporate action event - process splits and dividends.

        This handler provides an extensible dispatch mechanism for different
        corporate action types. New action types can be added by extending
        the if/elif chain and implementing corresponding processing methods.

        Currently supported:
            - split: Stock splits (forward splits with ratio > 1)
            - dividend: Cash dividends

        Future extensions could include:
            - reverse_split: Reverse splits (ratio < 1)
            - stock_dividend: Stock dividends
            - spinoff: Corporate spinoffs
            - merger: Merger adjustments
            - rights: Rights offerings

        Args:
            event: Corporate action event with action details

        Note:
            This method dispatches to specific handlers (process_split,
            process_dividend) based on action_type. If the symbol is not
            in the portfolio, the action is silently ignored (logged as debug).
        """
        from datetime import datetime

        # Convert ISO8601 effective_date to datetime for applying corporate actions
        effective_date_dt = datetime.fromisoformat(event.effective_date)

        # Dispatch based on action type - extensible design
        if event.action_type.lower() == "split":
            if event.split_ratio is None:
                logger.warning(
                    "portfolio_service.corporate_action.invalid_split",
                    symbol=event.symbol,
                    reason="Split event missing split_ratio",
                )
                return

            self.process_split(
                symbol=event.symbol,
                split_date=effective_date_dt,  # Use effective_date for splits
                ratio=event.split_ratio,
            )

        elif event.action_type.lower() == "dividend":
            if event.dividend_amount is None:
                logger.warning(
                    "portfolio_service.corporate_action.invalid_dividend",
                    symbol=event.symbol,
                    reason="Dividend event missing dividend_amount",
                )
                return

            self.process_dividend(
                symbol=event.symbol,
                effective_date=effective_date_dt,  # Use effective_date for dividends
                amount_per_share=event.dividend_amount,
            )

        else:
            # Log unsupported action types for future extension
            logger.info(
                "portfolio_service.corporate_action.unsupported",
                symbol=event.symbol,
                action_type=event.action_type,
                message="Corporate action type not yet implemented",
            )

        logger.debug(
            "portfolio_service.corporate_action.processed",
            symbol=event.symbol,
            action_type=event.action_type,
            ex_date=event.ex_date,
        )

    @classmethod
    def from_config(cls, config_dict: dict[str, Any], event_bus: EventBus) -> "PortfolioService":
        """
        Factory method to create service from configuration.

        Args:
            config_dict: Portfolio configuration dictionary
            event_bus: Event bus for communication

        Returns:
            Configured PortfolioService instance
        """
        # Convert config dict to PortfolioConfig
        portfolio_config = PortfolioConfig(
            initial_cash=Decimal(str(config_dict.get("initial_equity", "100000"))),
            max_ledger_entries=config_dict.get("max_ledger_entries", 10000),
            lot_method_long=config_dict.get("lot_method_long", "fifo"),
            lot_method_short=config_dict.get("lot_method_short", "lifo"),  # Phase 2 only supports lifo for shorts
        )

        return cls(config=portfolio_config, event_bus=event_bus)
