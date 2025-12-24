"""
ManagerService Implementation.

Event-driven portfolio management service that:
- Subscribes to SignalEvent from strategies
- Subscribes to PortfolioStateEvent for real-time equity and position tracking
- Processes each signal immediately (no batching)
- Loads risk policies from the risk library
- Uses risk library tools for sizing and limit checking
- Emits OrderEvent with intent_id and idempotency_key for audit trail

Architecture:
- Manager = Stateful orchestrator (subscribes to events, makes decisions)
- Risk Library = Stateless pure functions (sizing, limits)
- No circular dependencies (Manager → Risk, one-way only)

Event Flow:
  Strategy → SignalEvent → ManagerService → OrderEvent → ExecutionService
  Portfolio → PortfolioStateEvent → ManagerService (caches equity & positions)

Current Features:
- Fixed-fraction sizing based on portfolio equity
- Concentration limits using cached positions
- Leverage limits (gross/net exposure)
- Immediate signal processing (event-driven, no batching)
- Full portfolio state integration

Future Enhancements:
- Equal-weight sizing (needs position count logic)
- Multi-strategy capital allocation
- Advanced order types (limit, stop)
"""

from decimal import Decimal
from typing import Any

import structlog

from qtrader.events.event_bus import EventBus
from qtrader.events.events import OrderEvent, PortfolioStateEvent, SignalEvent
from qtrader.libraries.risk import load_policy
from qtrader.libraries.risk.models import RiskConfig
from qtrader.libraries.risk.tools import limits as risk_limits
from qtrader.libraries.risk.tools import sizing as risk_sizing
from qtrader.system import LoggerFactory


class ManagerService:
    """
    ManagerService: Portfolio management orchestrator.

    Responsibilities:
    - Evaluate trading signals from strategies
    - Size positions using risk library tools
    - Check limits using risk library tools
    - Emit orders with complete audit trail (intent_id, idempotency_key)
    - Cache portfolio state for real-time risk checks

    Features:
    - Fixed-fraction position sizing
    - Concentration and leverage limits
    - Immediate signal processing (event-driven, no batching)
    - Portfolio state synchronization via PortfolioStateEvent
    """

    def __init__(self, risk_config: RiskConfig, event_bus: EventBus) -> None:
        """
        Initialize ManagerService.

        Args:
            risk_config: Risk policy configuration (loaded from library)
            event_bus: Event bus for subscribing/publishing

        Side Effects:
            - Creates logger (manager.service namespace)
            - Subscribes to SignalEvent and PortfolioStateEvent
        """
        self._config = risk_config
        self._event_bus = event_bus
        self._logger = LoggerFactory.get_logger("manager.service")

        # Portfolio state cache (updated by PortfolioStateEvent)
        self._cached_equity: Decimal | None = None
        self._cached_positions: list[risk_limits.Position] = []  # Flattened for risk checks
        self._cached_strategy_positions: dict[str, dict[str, int]] = {}  # {strategy_id: {symbol: quantity}}

        # Pending close signals tracker: {(strategy_id, symbol): intention}
        # Prevents duplicate CLOSE signals while waiting for fill confirmation
        # Cleared when position goes to 0 (via PortfolioStateEvent)
        self._pending_closes: dict[tuple[str, str], str] = {}

        # Subscribe to events
        self._event_bus.subscribe("signal", self.on_signal)  # type: ignore[arg-type]
        self._event_bus.subscribe("portfolio_state", self.on_portfolio_state)  # type: ignore[arg-type]

        # Log after subscriptions
        self._logger.debug(
            "ManagerService initialized",
            extra={
                "strategies": list(risk_config.sizing.keys()),
                "concentration_max_position_pct": float(risk_config.concentration.max_position_pct),
                "leverage_max_gross": float(risk_config.leverage.max_gross),
                "leverage_max_net": float(risk_config.leverage.max_net),
                "cash_buffer_pct": float(risk_config.cash_buffer_pct),
            },
        )

    @classmethod
    def from_config(cls, config_dict: dict[str, Any], event_bus: EventBus) -> "ManagerService":
        """
        Factory method to create ManagerService from configuration.

        Loads risk policy from risk library using policy name from BacktestConfig.

        Args:
            config_dict: Configuration from BacktestConfig.risk_policy
                Expected format:
                {
                    "name": "naive",  # Policy name to load
                    "config": {}      # Optional overrides (not yet implemented)
                }
            event_bus: Event bus for service communication

        Returns:
            Configured ManagerService instance

        Example:
            >>> config_dict = {"name": "naive", "config": {}}
            >>> service = ManagerService.from_config(config_dict, event_bus)
        """
        # Extract policy name
        policy_name = config_dict.get("name", "naive")
        policy_overrides = config_dict.get("config", {})

        # Load risk policy from library
        risk_config = load_policy(policy_name)

        # Apply overrides from config (future enhancement: allow parameter overrides)
        if policy_overrides:
            logger = LoggerFactory.get_logger("manager.service")
            logger.warning(
                "Policy overrides not yet implemented",
                extra={"overrides": policy_overrides},
            )

        return cls(risk_config=risk_config, event_bus=event_bus)

    def _get_position_quantity(self, strategy_id: str, symbol: str) -> int:
        """
        Get current position quantity for a strategy-symbol pair.

        Args:
            strategy_id: Strategy identifier
            symbol: Symbol to check

        Returns:
            Quantity held by strategy (positive=long, negative=short, 0=flat)

        Example:
            >>> quantity = manager._get_position_quantity("sma_crossover", "AAPL")
            >>> # Returns 100 if strategy has long position of 100 shares
        """
        strategy_positions = self._cached_strategy_positions.get(strategy_id, {})
        return strategy_positions.get(symbol, 0)

    def on_signal(self, event: SignalEvent) -> None:
        """
        Handle incoming trading signal (immediate processing).

        Architecture:
        - No batching - process each signal immediately
        - Use cached portfolio state for sizing and limits
        - Use risk library tools for calculations
        - Emit OrderEvent with intent_id and idempotency_key for audit trail

        Args:
            event: Trading signal from strategy

        Side Effects:
            - Publishes OrderEvent if approved
            - Logs rejection if signal rejected
            - Does NOT emit rejection events (orders simply don't exist)

        Flow:
            1. Get cached portfolio equity from PortfolioStateEvent
            2. Get sizing configuration for strategy
            3. Calculate position size using risk library
            4. Check limits using risk library and cached positions
            5. If approved: emit OrderEvent with intent_id and idempotency_key
            6. If rejected: log reason (no event emitted)

        SignalEvent fields (actual structure):
            - signal_id: str
            - timestamp: str (ISO8601)
            - strategy_id: str
            - symbol: str
            - intention: str (OPEN_LONG, CLOSE_LONG, OPEN_SHORT, CLOSE_SHORT)
            - price: Decimal
            - confidence: Decimal [0.0, 1.0]
            - metadata: Optional[dict]

        Example:
            >>> signal = SignalEvent(
            ...     signal_id="sig-001",
            ...     timestamp="2020-01-02T16:00:00Z",
            ...     strategy_id="momentum",
            ...     symbol="AAPL",
            ...     intention="OPEN_LONG",
            ...     price=Decimal("150.0"),
            ...     confidence=Decimal("0.75"),
            ...     metadata={"equity": 100000.0}
            ... )
            >>> manager.on_signal(signal)  # Emits OrderEvent
        """
        self._logger.debug(
            "Signal received",
            extra={
                "timestamp": event.timestamp,
                "signal_id": event.signal_id,
                "strategy_id": event.strategy_id,
                "symbol": event.symbol,
                "intention": event.intention,
                "confidence": float(event.confidence),
            },
        )

        # Step 1: Get portfolio equity from cache
        # Manager must receive PortfolioStateEvent before processing signals
        current_equity = self._cached_equity

        if current_equity is None:
            self._logger.warning(
                "Signal rejected: no cached equity (PortfolioStateEvent not received)",
                extra={
                    "signal_id": event.signal_id,
                    "strategy_id": event.strategy_id,
                    "symbol": event.symbol,
                    "hint": "Ensure PortfolioService publishes PortfolioStateEvent before signals",
                },
            )
            return

        current_price = event.price
        current_positions_list = self._cached_positions

        # Map intention to side for OrderEvent
        # OPEN_LONG: Buy to open long position
        # CLOSE_SHORT: Buy to cover short position
        # CLOSE_LONG: Sell to close long position
        # OPEN_SHORT: Sell to open short position
        if event.intention in ("OPEN_LONG", "CLOSE_SHORT"):
            side = "buy"  # OrderEvent schema requires lowercase
        elif event.intention in ("CLOSE_LONG", "OPEN_SHORT"):
            side = "sell"  # OrderEvent schema requires lowercase
        else:
            self._logger.warning(
                "Signal rejected: unknown intention",
                extra={
                    "signal_id": event.signal_id,
                    "strategy_id": event.strategy_id,
                    "intention": event.intention,
                },
            )
            return

        # Step 1.5: Check shorting policy before proceeding
        # Reject OPEN_SHORT signals if shorting is not allowed by risk policy
        if event.intention == "OPEN_SHORT":
            if self._config.shorting and not self._config.shorting.allow_short_positions:
                # Emit Rich-formatted display event (INFO level shown in console)
                policy_logger = structlog.get_logger("qtrader.events.policy")
                policy_logger.info(
                    "event.display",
                    event_type="policy_violation",
                    symbol=event.symbol,
                    strategy_id=event.strategy_id,
                    intention=event.intention,
                    reason="shorting not allowed",
                    policy="shorting.allow_short_positions = false",
                    timestamp=event.timestamp,
                )
                # Also log as WARNING for file/debugging (will be filtered from console if event display enabled)
                self._logger.warning(
                    "manager.signal.rejected.policy_violation",
                    extra={
                        "signal_id": event.signal_id,
                        "strategy_id": event.strategy_id,
                        "symbol": event.symbol,
                        "intention": event.intention,
                        "policy_setting": "shorting.allow_short_positions = false",
                    },
                )
                return

        # Step 2: Get sizing configuration for this strategy
        sizing_config = self._config.sizing.get(event.strategy_id) or self._config.sizing.get("default")

        if sizing_config is None:
            self._logger.warning(
                "Signal rejected: no sizing config for strategy",
                extra={
                    "signal_id": event.signal_id,
                    "strategy_id": event.strategy_id,
                    "symbol": event.symbol,
                },
            )
            return

        # Step 3: Determine quantity based on intention
        # For CLOSE signals: use actual position size (optionally scaled by confidence)
        # For OPEN signals: calculate size using risk library
        quantity: int = 0

        if event.intention in ("CLOSE_LONG", "CLOSE_SHORT"):
            # Check for duplicate FULL close signal (confidence=1.0, already pending)
            # Partial closes (confidence < 1.0) are allowed to accumulate
            pending_key = (event.strategy_id, event.symbol)
            if pending_key in self._pending_closes and event.confidence >= Decimal("1.0"):
                # Duplicate full close signal - reject silently (common strategy mistake)
                # This prevents: signal1 → CLOSE_LONG, signal2 → CLOSE_LONG (same bar)
                # Without fill in between, second CLOSE would open a short position
                self._logger.debug(
                    "manager.signal.rejected.duplicate_close",
                    extra={
                        "signal_id": event.signal_id,
                        "strategy_id": event.strategy_id,
                        "symbol": event.symbol,
                        "intention": event.intention,
                        "pending_intention": self._pending_closes[pending_key],
                    },
                )
                return

            # Get current position for this strategy-symbol pair
            current_quantity = self._get_position_quantity(event.strategy_id, event.symbol)

            if current_quantity == 0:
                # Emit Rich-formatted rejection event
                rejection_logger = structlog.get_logger("qtrader.events.rejection")
                rejection_logger.info(
                    "event.display",
                    event_type="signal_rejected",
                    symbol=event.symbol,
                    strategy_id=event.strategy_id,
                    intention=event.intention,
                    reason="no position to close",
                    timestamp=event.timestamp,
                )
                # Also log for file/debugging
                self._logger.warning(
                    "manager.signal.rejected.no_position",
                    extra={
                        "signal_id": event.signal_id,
                        "strategy_id": event.strategy_id,
                        "symbol": event.symbol,
                        "intention": event.intention,
                    },
                )
                return

            # Validate position direction matches intention
            if event.intention == "CLOSE_LONG" and current_quantity <= 0:
                # Emit Rich-formatted rejection event
                rejection_logger = structlog.get_logger("qtrader.events.rejection")
                rejection_logger.info(
                    "event.display",
                    event_type="signal_rejected",
                    symbol=event.symbol,
                    strategy_id=event.strategy_id,
                    intention=event.intention,
                    reason=f"position not long (qty: {current_quantity})",
                    timestamp=event.timestamp,
                )
                # Also log for file/debugging
                self._logger.warning(
                    "manager.signal.rejected.wrong_direction",
                    extra={
                        "signal_id": event.signal_id,
                        "strategy_id": event.strategy_id,
                        "symbol": event.symbol,
                        "current_quantity": current_quantity,
                    },
                )
                return

            if event.intention == "CLOSE_SHORT" and current_quantity >= 0:
                # Emit Rich-formatted rejection event
                rejection_logger = structlog.get_logger("qtrader.events.rejection")
                rejection_logger.info(
                    "event.display",
                    event_type="signal_rejected",
                    symbol=event.symbol,
                    strategy_id=event.strategy_id,
                    intention=event.intention,
                    reason=f"position not short (qty: {current_quantity})",
                    timestamp=event.timestamp,
                )
                # Also log for file/debugging
                self._logger.warning(
                    "manager.signal.rejected.wrong_direction",
                    extra={
                        "signal_id": event.signal_id,
                        "strategy_id": event.strategy_id,
                        "symbol": event.symbol,
                        "current_quantity": current_quantity,
                    },
                )
                return

            # Use absolute value of position (will be sized correctly by side later)
            base_quantity = abs(current_quantity)

            # Optionally scale by confidence for partial exits
            # confidence=1.0 → close full position
            # confidence=0.5 → close 50% of position
            if event.confidence < Decimal("1.0"):
                # Calculate raw scaled quantity
                raw_quantity = base_quantity * float(event.confidence)

                # Apply lot size constraints (same logic as risk sizer)
                lot_size = sizing_config.lot_size
                quantity = int(raw_quantity / lot_size) * lot_size

                # If lot rounding floors to 0 but we have at least 1 lot available,
                # round UP to 1 lot instead of discarding the order
                # Signal contract: confidence is a strength hint, not a hard cap
                # Manager should close "about" the requested amount, not suppress entirely
                if quantity == 0 and base_quantity >= lot_size:
                    quantity = lot_size  # Round up to 1 lot
                    self._logger.debug(
                        "Close signal rounded up to minimum lot",
                        extra={
                            "signal_id": event.signal_id,
                            "strategy_id": event.strategy_id,
                            "symbol": event.symbol,
                            "position_quantity": base_quantity,
                            "confidence": float(event.confidence),
                            "raw_scaled_quantity": int(raw_quantity),
                            "lot_size": lot_size,
                            "rounded_quantity": quantity,
                            "reason": "Confidence scaling floored to 0, rounded up to 1 lot",
                        },
                    )
                # If we still have quantity < min_quantity, apply min_quantity constraint
                elif quantity < sizing_config.min_quantity and base_quantity >= sizing_config.min_quantity:
                    # Round up to minimum, but don't exceed position
                    quantity = min(base_quantity, sizing_config.min_quantity)

                    # CRITICAL: Re-apply lot size rounding after min_quantity adjustment
                    # This prevents violating lot constraints (e.g., lot_size=100, min_quantity=1 → 1 share)
                    quantity = int(quantity / lot_size) * lot_size

                    # If lot rounding zeros out again, round up to 1 lot if available
                    if quantity == 0 and base_quantity >= lot_size:
                        quantity = lot_size
                        self._logger.debug(
                            "Close signal min_quantity rounded up to lot size",
                            extra={
                                "signal_id": event.signal_id,
                                "strategy_id": event.strategy_id,
                                "symbol": event.symbol,
                                "position_quantity": base_quantity,
                                "min_quantity": sizing_config.min_quantity,
                                "lot_size": lot_size,
                                "rounded_quantity": quantity,
                            },
                        )
                    elif quantity == 0:
                        # Position too small for even 1 lot - skip order
                        self._logger.debug(
                            "Close signal position too small for lot constraints",
                            extra={
                                "signal_id": event.signal_id,
                                "strategy_id": event.strategy_id,
                                "symbol": event.symbol,
                                "position_quantity": base_quantity,
                                "lot_size": lot_size,
                                "reason": "Position < 1 lot, cannot close",
                            },
                        )
                        return  # Skip order emission

                self._logger.debug(
                    "Close signal scaled by confidence",
                    extra={
                        "signal_id": event.signal_id,
                        "strategy_id": event.strategy_id,
                        "symbol": event.symbol,
                        "position_quantity": base_quantity,
                        "confidence": float(event.confidence),
                        "raw_scaled_quantity": int(raw_quantity),
                        "lot_size": lot_size,
                        "scaled_quantity": quantity,
                    },
                )
            else:
                # Full position close
                quantity = base_quantity

        else:  # OPEN_LONG or OPEN_SHORT
            # Calculate position size using risk library
            # Step 3: Calculate position size using risk library
            # Use strategy's allocated capital from risk policy budgets
            # Note: get_allocated_capital will use strategy-specific budget or fall back to "default"
            # The loader ensures a "default" budget always exists (auto-created at 95% if not specified)
            allocated_capital = self._config.get_allocated_capital(event.strategy_id, current_equity)
            self._logger.debug(
                "Using allocated capital from risk policy",
                extra={
                    "strategy_id": event.strategy_id,
                    "allocated_capital": float(allocated_capital),
                    "current_equity": float(current_equity),
                },
            )

            try:
                if sizing_config.model == "fixed_fraction":
                    quantity = risk_sizing.calculate_fixed_fraction_size(
                        allocated_capital=allocated_capital,
                        signal_strength=float(event.confidence),  # Use confidence as signal strength
                        current_price=current_price,
                        fraction=sizing_config.fraction,
                        lot_size=sizing_config.lot_size,
                        min_quantity=sizing_config.min_quantity,
                    )
                elif sizing_config.model == "equal_weight":
                    # Equal weight needs position count - not yet implemented
                    # Fallback to fixed fraction for now
                    self._logger.warning(
                        "Equal weight sizing not yet supported, using fixed fraction",
                        extra={"strategy_id": event.strategy_id},
                    )
                    quantity = risk_sizing.calculate_fixed_fraction_size(
                        allocated_capital=allocated_capital,
                        signal_strength=float(event.confidence),
                        current_price=current_price,
                        fraction=Decimal("0.10"),  # Default fallback
                        lot_size=sizing_config.lot_size,
                        min_quantity=sizing_config.min_quantity,
                    )
                # Note: No else needed - sizing_config.model is Literal["fixed_fraction", "equal_weight"]

            except (ValueError, TypeError) as e:
                # Sizing calculation failed - reject signal
                self._logger.warning(
                    "Signal rejected: sizing calculation failed",
                    extra={
                        "signal_id": event.signal_id,
                        "strategy_id": event.strategy_id,
                        "error": str(e),
                    },
                )
                return

        # If we reach here, quantity was successfully calculated
        if quantity == 0:
            self._logger.debug(
                "Signal rejected: position size rounded to zero",
                extra={
                    "signal_id": event.signal_id,
                    "strategy_id": event.strategy_id,
                    "symbol": event.symbol,
                },
            )
            return

        # Step 4: Check limits using risk library with cached positions
        proposed_order = risk_limits.ProposedOrder(
            symbol=event.symbol,
            side=side,
            quantity=quantity,
        )

        violations = risk_limits.check_all_limits(
            order=proposed_order,
            current_positions=current_positions_list,
            equity=current_equity,
            current_price=current_price,
            max_position_pct=(
                float(self._config.concentration.max_position_pct) if self._config.concentration else None
            ),
            max_gross_leverage=float(self._config.leverage.max_gross) if self._config.leverage else None,
            max_net_leverage=float(self._config.leverage.max_net) if self._config.leverage else None,
        )

        if violations:
            reasons = [v.message for v in violations]
            self._logger.warning(
                "Signal rejected: limit violations",
                extra={
                    "signal_id": event.signal_id,
                    "strategy_id": event.strategy_id,
                    "symbol": event.symbol,
                    "violations": reasons,
                },
            )
            return

        # Step 5: Generate audit trail fields
        idempotency_key = f"{event.strategy_id}-{event.signal_id}-{event.timestamp}"
        intent_id = event.signal_id  # Link order back to signal

        # Step 6: Emit OrderEvent with causation chain
        order_event = OrderEvent(
            intent_id=intent_id,
            idempotency_key=idempotency_key,
            timestamp=event.timestamp,
            symbol=event.symbol,
            side=side,
            quantity=Decimal(str(quantity)),
            order_type="market",  # Future: support limit/stop orders
            time_in_force="GTC",
            source_strategy_id=event.strategy_id,
            source_service="manager_service",
            correlation_id=event.correlation_id,  # Propagate workflow ID
            causation_id=event.event_id,  # This order was caused by the signal
        )

        self._event_bus.publish(order_event)

        # Track pending FULL close signals to prevent duplicates
        # Only track full closes (confidence=1.0) - partial closes can accumulate
        if event.intention in ("CLOSE_LONG", "CLOSE_SHORT") and event.confidence >= Decimal("1.0"):
            pending_key = (event.strategy_id, event.symbol)
            self._pending_closes[pending_key] = event.intention

        self._logger.info(
            "Order emitted",
            extra={
                "timestamp": event.timestamp,
                "signal_id": event.signal_id,
                "strategy_id": event.strategy_id,
                "symbol": event.symbol,
                "side": side,
                "quantity": quantity,
                "intent_id": intent_id,
                "idempotency_key": idempotency_key,
            },
        )

    def on_portfolio_state(self, event: "PortfolioStateEvent") -> None:
        """
        Cache portfolio state for use in risk checks.

        Subscribes to PortfolioStateEvent published by PortfolioService
        after mark-to-market on each bar. Extracts and caches equity
        and positions for subsequent signal processing.

        Args:
            event: Portfolio state snapshot

        Side Effects:
            - Updates _cached_equity
            - Updates _cached_positions (converted from PortfolioPosition format)

        Flow:
            Bar → PortfolioService.on_bar() → mark_to_market()
                → PortfolioStateEvent → ManagerService.on_portfolio_state()
                → cache equity/positions for next signal

        Example:
            >>> state = PortfolioStateEvent(
            ...     portfolio_id="portfolio-123",
            ...     start_datetime="2020-01-01T00:00:00Z",
            ...     snapshot_datetime="2020-01-02T16:00:00Z",
            ...     reporting_currency="USD",
            ...     initial_portfolio_equity=Decimal("100000"),
            ...     cash_balance=Decimal("50000"),
            ...     current_portfolio_equity=Decimal("100000"),
            ...     total_market_value=Decimal("50000"),
            ...     total_unrealized_pl=Decimal("0"),
            ...     total_realized_pl=Decimal("0"),
            ...     total_pl=Decimal("0"),
            ...     long_exposure=Decimal("50000"),
            ...     short_exposure=Decimal("0"),
            ...     net_exposure=Decimal("50000"),
            ...     gross_exposure=Decimal("50000"),
            ...     leverage=Decimal("0.5"),
            ...     strategies_groups=[],
            ... )
            >>> manager.on_portfolio_state(state)
            >>> # _cached_equity = 100000
        """
        self._cached_equity = event.current_portfolio_equity

        # Convert Portfolio positions to risk_limits.Position format
        # Flatten all strategy positions into a single list for risk checks
        converted_positions: list[risk_limits.Position] = []

        # Also maintain strategy-grouped positions for close signal processing
        strategy_positions_map: dict[str, dict[str, int]] = {}

        for strategy_group in event.strategies_groups:
            strategy_id = strategy_group.strategy_id
            strategy_positions_map[strategy_id] = {}

            for portfolio_pos in strategy_group.positions:
                # Only include open positions (skip flat/closed positions)
                if portfolio_pos.open_quantity != 0:
                    # Add to flattened list for risk checks
                    converted_positions.append(
                        risk_limits.Position(
                            symbol=portfolio_pos.symbol,
                            quantity=portfolio_pos.open_quantity,
                            market_value=portfolio_pos.gross_market_value,
                        )
                    )

                    # Add to strategy-grouped map for close signal lookup
                    strategy_positions_map[strategy_id][portfolio_pos.symbol] = portfolio_pos.open_quantity
                else:
                    # Position is flat/closed - clear any pending close signal
                    pending_key = (strategy_id, portfolio_pos.symbol)
                    if pending_key in self._pending_closes:
                        del self._pending_closes[pending_key]

        self._cached_positions = converted_positions
        self._cached_strategy_positions = strategy_positions_map

        self._logger.debug(
            "Portfolio state cached",
            extra={
                "snapshot_datetime": event.snapshot_datetime,
                "current_portfolio_equity": str(event.current_portfolio_equity),
                "num_strategies": len(event.strategies_groups),
                "num_positions": len(converted_positions),
                "gross_exposure": str(event.gross_exposure),
                "net_exposure": str(event.net_exposure),
            },
        )
