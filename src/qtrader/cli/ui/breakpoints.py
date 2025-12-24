"""
Breakpoint Rules for Interactive Debugging.

Provides an extensible breakpoint system for the interactive debugger.
Each rule defines a condition that triggers a pause during backtest execution.

Supported breakpoint types:
- DateBreakpoint: Pause from a specific date onward
- SignalBreakpoint: Pause when trading signals are emitted

Future extensions:
- IndicatorBreakpoint: Pause when indicator crosses threshold
- DrawdownBreakpoint: Pause when drawdown exceeds limit
- TradeBreakpoint: Pause when trades are executed

Usage:
    from qtrader.cli.ui.breakpoints import parse_breakpoint, SignalBreakpoint

    # Parse from CLI string
    bp = parse_breakpoint("signal:BUY")

    # Create directly
    bp = SignalBreakpoint(intention="BUY")

    # Check if should trigger
    context = BreakpointContext(timestamp=ts, signals=[signal])
    if bp.should_trigger(context):
        # Pause execution
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Optional

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class BreakpointContext:
    """
    Context passed to breakpoint rules for evaluation.

    Contains all state available at a given timestamp, allowing rules
    to make decisions based on any combination of factors.

    Attributes:
        timestamp: Current simulation timestamp
        signals: List of signals emitted at this timestamp
        bars: Dict of symbol -> bar data
        indicators: Dict of strategy_id -> indicator values
        portfolio: Portfolio state snapshot
    """

    timestamp: datetime
    signals: list[Any] = field(default_factory=list)
    bars: dict[str, Any] = field(default_factory=dict)
    indicators: dict[str, dict[str, Any]] = field(default_factory=dict)
    portfolio: Optional[dict[str, Any]] = None


class BreakpointRule(ABC):
    """
    Base class for breakpoint rules.

    Subclasses implement should_trigger() to define when the debugger
    should pause execution. Rules can be combined and evaluated together.
    """

    @abstractmethod
    def should_trigger(self, context: BreakpointContext) -> bool:
        """
        Check if this breakpoint should trigger a pause.

        Args:
            context: Current state context for evaluation

        Returns:
            True if debugger should pause at this point
        """
        pass

    @abstractmethod
    def description(self) -> str:
        """
        Human-readable description of the breakpoint.

        Returns:
            Short description for display in CLI output
        """
        pass

    @property
    def requires_events(self) -> bool:
        """
        Whether this breakpoint needs to evaluate after events are published.

        Date-based breakpoints can be evaluated before events (to show header).
        Event-based breakpoints (signals) must wait until after events.

        Returns:
            True if must evaluate after events, False if can evaluate before
        """
        return False


class DateBreakpoint(BreakpointRule):
    """
    Break from a specific date onward.

    This is a "start condition" - once the date is reached, execution
    pauses at every timestamp. Used with --break-at DATE.

    Attributes:
        from_date: Date to start pausing from (inclusive)
    """

    def __init__(self, from_date: date):
        """
        Initialize date breakpoint.

        Args:
            from_date: Date to start pausing from
        """
        self.from_date = from_date

    def should_trigger(self, context: BreakpointContext) -> bool:
        """Check if current timestamp is on or after the breakpoint date."""
        return context.timestamp.date() >= self.from_date

    def description(self) -> str:
        """Return description like 'from 2020-06-15'."""
        return f"from {self.from_date.isoformat()}"

    @property
    def requires_events(self) -> bool:
        """Date breakpoints can be evaluated before events."""
        return False


class SignalBreakpoint(BreakpointRule):
    """
    Break when trading signals are emitted.

    Can match any signal or filter by intention type (BUY/SELL/OPEN_LONG/etc).
    Used with --break-on signal or --break-on signal:BUY.

    Attributes:
        intention: Optional intention type to match (None = any signal)
    """

    # Map common aliases to full intention names
    INTENTION_ALIASES = {
        "BUY": ["OPEN_LONG", "BUY"],
        "SELL": ["CLOSE_LONG", "SELL"],
        "SHORT": ["OPEN_SHORT", "SHORT"],
        "COVER": ["CLOSE_SHORT", "COVER"],
    }

    def __init__(self, intention: Optional[str] = None):
        """
        Initialize signal breakpoint.

        Args:
            intention: Optional intention type to filter by.
                      None = pause on any signal.
                      "BUY" = pause on OPEN_LONG or BUY signals.
                      "SELL" = pause on CLOSE_LONG or SELL signals.
                      Or exact match like "OPEN_LONG", "CLOSE_SHORT", etc.
        """
        self.intention = intention.upper() if intention else None

    def should_trigger(self, context: BreakpointContext) -> bool:
        """Check if any matching signals were emitted."""
        if not context.signals:
            return False

        if self.intention is None:
            # Any signal triggers
            return True

        # Get possible matching intentions (handle aliases)
        match_intentions = self.INTENTION_ALIASES.get(self.intention, [self.intention])

        for signal in context.signals:
            signal_intention = getattr(signal, "intention", None)
            if signal_intention:
                signal_intention_str = str(signal_intention).upper()
                if signal_intention_str in match_intentions:
                    return True

        return False

    def description(self) -> str:
        """Return description like 'signal' or 'signal:BUY'."""
        if self.intention:
            return f"signal:{self.intention}"
        return "signal"

    @property
    def requires_events(self) -> bool:
        """Signal breakpoints must wait for events to be published."""
        return True


def parse_breakpoint(value: str) -> BreakpointRule:
    """
    Parse a breakpoint rule from CLI string.

    Supported formats:
        "signal"           -> SignalBreakpoint(intention=None)
        "signal:BUY"       -> SignalBreakpoint(intention="BUY")
        "signal:SELL"      -> SignalBreakpoint(intention="SELL")
        "signal:OPEN_LONG" -> SignalBreakpoint(intention="OPEN_LONG")

    Future formats:
        "indicator:RSI>70" -> IndicatorBreakpoint(name="RSI", op=">", value=70)
        "drawdown:5%"      -> DrawdownBreakpoint(percent=5)
        "trade"            -> TradeBreakpoint()

    Args:
        value: CLI string like "signal" or "signal:BUY"

    Returns:
        Parsed BreakpointRule instance

    Raises:
        ValueError: If format is not recognized
    """
    value = value.strip().lower()

    # Signal breakpoints
    if value == "signal":
        return SignalBreakpoint()
    elif value.startswith("signal:"):
        intention = value.split(":", 1)[1].strip().upper()
        if not intention:
            raise ValueError("signal: requires an intention type (e.g., signal:BUY)")
        return SignalBreakpoint(intention=intention)

    # Future: Add more breakpoint types here
    # elif value == "trade":
    #     return TradeBreakpoint()
    # elif value.startswith("indicator:"):
    #     ...
    # elif value.startswith("drawdown:"):
    #     ...

    raise ValueError(
        f"Unknown breakpoint type: '{value}'. "
        f"Supported types: signal, signal:BUY, signal:SELL, signal:OPEN_LONG, signal:CLOSE_LONG"
    )


def parse_breakpoints(values: list[str]) -> list[BreakpointRule]:
    """
    Parse multiple breakpoint rules from CLI strings.

    Args:
        values: List of CLI strings

    Returns:
        List of parsed BreakpointRule instances

    Raises:
        ValueError: If any format is not recognized
    """
    return [parse_breakpoint(v) for v in values]
