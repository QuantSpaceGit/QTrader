"""
Unit tests for Breakpoint Rules.

Tests the breakpoint abstraction including BreakpointContext, BreakpointRule base class,
and concrete implementations (DateBreakpoint, SignalBreakpoint).
"""

from datetime import date, datetime
from unittest.mock import Mock

import pytest

from qtrader.cli.ui.breakpoints import (
    BreakpointContext,
    DateBreakpoint,
    SignalBreakpoint,
    parse_breakpoint,
    parse_breakpoints,
)


class TestBreakpointContext:
    """Test BreakpointContext dataclass."""

    def test_context_with_defaults(self):
        """Test context initializes with default empty values."""
        # Arrange
        timestamp = datetime(2020, 6, 15, 12, 0, 0)

        # Act
        context = BreakpointContext(timestamp=timestamp)

        # Assert
        assert context.timestamp == timestamp
        assert context.signals == []
        assert context.bars == {}
        assert context.indicators == {}
        assert context.portfolio is None

    def test_context_with_all_fields(self):
        """Test context with all fields populated."""
        # Arrange
        timestamp = datetime(2020, 6, 15, 12, 0, 0)
        signals = [Mock()]
        bars = {"AAPL": Mock()}
        indicators = {"strategy1": {"SMA": 100.0}}
        portfolio = {"equity": 100000}

        # Act
        context = BreakpointContext(
            timestamp=timestamp,
            signals=signals,
            bars=bars,
            indicators=indicators,
            portfolio=portfolio,
        )

        # Assert
        assert context.signals == signals
        assert context.bars == bars
        assert context.indicators == indicators
        assert context.portfolio == portfolio


class TestDateBreakpoint:
    """Test DateBreakpoint rule."""

    def test_should_trigger_before_date(self):
        """Test DateBreakpoint returns False before the breakpoint date."""
        # Arrange
        bp = DateBreakpoint(from_date=date(2020, 6, 15))
        context = BreakpointContext(timestamp=datetime(2020, 6, 10, 12, 0, 0))

        # Act
        result = bp.should_trigger(context)

        # Assert
        assert result is False

    def test_should_trigger_on_date(self):
        """Test DateBreakpoint returns True on the breakpoint date."""
        # Arrange
        bp = DateBreakpoint(from_date=date(2020, 6, 15))
        context = BreakpointContext(timestamp=datetime(2020, 6, 15, 12, 0, 0))

        # Act
        result = bp.should_trigger(context)

        # Assert
        assert result is True

    def test_should_trigger_after_date(self):
        """Test DateBreakpoint returns True after the breakpoint date."""
        # Arrange
        bp = DateBreakpoint(from_date=date(2020, 6, 15))
        context = BreakpointContext(timestamp=datetime(2020, 6, 20, 12, 0, 0))

        # Act
        result = bp.should_trigger(context)

        # Assert
        assert result is True

    def test_description(self):
        """Test DateBreakpoint description format."""
        # Arrange
        bp = DateBreakpoint(from_date=date(2020, 6, 15))

        # Act
        desc = bp.description()

        # Assert
        assert desc == "from 2020-06-15"

    def test_requires_events_is_false(self):
        """Test DateBreakpoint can be evaluated before events."""
        # Arrange
        bp = DateBreakpoint(from_date=date(2020, 6, 15))

        # Assert
        assert bp.requires_events is False


class TestSignalBreakpoint:
    """Test SignalBreakpoint rule."""

    def test_should_trigger_no_signals(self):
        """Test SignalBreakpoint returns False when no signals present."""
        # Arrange
        bp = SignalBreakpoint()
        context = BreakpointContext(timestamp=datetime(2020, 6, 15, 12, 0, 0))

        # Act
        result = bp.should_trigger(context)

        # Assert
        assert result is False

    def test_should_trigger_any_signal(self):
        """Test SignalBreakpoint without filter triggers on any signal."""
        # Arrange
        bp = SignalBreakpoint()  # No intention filter
        mock_signal = Mock()
        mock_signal.intention = "OPEN_LONG"
        context = BreakpointContext(
            timestamp=datetime(2020, 6, 15, 12, 0, 0),
            signals=[mock_signal],
        )

        # Act
        result = bp.should_trigger(context)

        # Assert
        assert result is True

    def test_should_trigger_matching_intention(self):
        """Test SignalBreakpoint triggers on matching intention."""
        # Arrange
        bp = SignalBreakpoint(intention="BUY")
        mock_signal = Mock()
        mock_signal.intention = "OPEN_LONG"  # BUY alias
        context = BreakpointContext(
            timestamp=datetime(2020, 6, 15, 12, 0, 0),
            signals=[mock_signal],
        )

        # Act
        result = bp.should_trigger(context)

        # Assert
        assert result is True

    def test_should_trigger_exact_intention(self):
        """Test SignalBreakpoint triggers on exact intention match."""
        # Arrange
        bp = SignalBreakpoint(intention="OPEN_LONG")
        mock_signal = Mock()
        mock_signal.intention = "OPEN_LONG"
        context = BreakpointContext(
            timestamp=datetime(2020, 6, 15, 12, 0, 0),
            signals=[mock_signal],
        )

        # Act
        result = bp.should_trigger(context)

        # Assert
        assert result is True

    def test_should_not_trigger_non_matching_intention(self):
        """Test SignalBreakpoint doesn't trigger on non-matching intention."""
        # Arrange
        bp = SignalBreakpoint(intention="BUY")
        mock_signal = Mock()
        mock_signal.intention = "CLOSE_LONG"  # SELL, not BUY
        context = BreakpointContext(
            timestamp=datetime(2020, 6, 15, 12, 0, 0),
            signals=[mock_signal],
        )

        # Act
        result = bp.should_trigger(context)

        # Assert
        assert result is False

    def test_should_trigger_sell_alias(self):
        """Test SignalBreakpoint handles SELL alias for CLOSE_LONG."""
        # Arrange
        bp = SignalBreakpoint(intention="SELL")
        mock_signal = Mock()
        mock_signal.intention = "CLOSE_LONG"
        context = BreakpointContext(
            timestamp=datetime(2020, 6, 15, 12, 0, 0),
            signals=[mock_signal],
        )

        # Act
        result = bp.should_trigger(context)

        # Assert
        assert result is True

    def test_should_trigger_multiple_signals_one_match(self):
        """Test SignalBreakpoint triggers if any signal matches."""
        # Arrange
        bp = SignalBreakpoint(intention="BUY")
        signal1 = Mock()
        signal1.intention = "CLOSE_LONG"
        signal2 = Mock()
        signal2.intention = "OPEN_LONG"  # BUY match
        context = BreakpointContext(
            timestamp=datetime(2020, 6, 15, 12, 0, 0),
            signals=[signal1, signal2],
        )

        # Act
        result = bp.should_trigger(context)

        # Assert
        assert result is True

    def test_description_no_filter(self):
        """Test SignalBreakpoint description without filter."""
        # Arrange
        bp = SignalBreakpoint()

        # Act
        desc = bp.description()

        # Assert
        assert desc == "signal"

    def test_description_with_filter(self):
        """Test SignalBreakpoint description with filter."""
        # Arrange
        bp = SignalBreakpoint(intention="BUY")

        # Act
        desc = bp.description()

        # Assert
        assert desc == "signal:BUY"

    def test_requires_events_is_true(self):
        """Test SignalBreakpoint must wait for events."""
        # Arrange
        bp = SignalBreakpoint()

        # Assert
        assert bp.requires_events is True

    def test_intention_case_insensitive(self):
        """Test intention matching is case-insensitive."""
        # Arrange
        bp = SignalBreakpoint(intention="buy")  # lowercase
        mock_signal = Mock()
        mock_signal.intention = "OPEN_LONG"
        context = BreakpointContext(
            timestamp=datetime(2020, 6, 15, 12, 0, 0),
            signals=[mock_signal],
        )

        # Act
        result = bp.should_trigger(context)

        # Assert
        assert result is True


class TestParseBreakpoint:
    """Test parse_breakpoint function."""

    def test_parse_signal(self):
        """Test parsing 'signal' breakpoint."""
        # Act
        bp = parse_breakpoint("signal")

        # Assert
        assert isinstance(bp, SignalBreakpoint)
        assert bp.intention is None

    def test_parse_signal_buy(self):
        """Test parsing 'signal:BUY' breakpoint."""
        # Act
        bp = parse_breakpoint("signal:BUY")

        # Assert
        assert isinstance(bp, SignalBreakpoint)
        assert bp.intention == "BUY"

    def test_parse_signal_sell(self):
        """Test parsing 'signal:SELL' breakpoint."""
        # Act
        bp = parse_breakpoint("signal:sell")

        # Assert
        assert isinstance(bp, SignalBreakpoint)
        assert bp.intention == "SELL"

    def test_parse_signal_open_long(self):
        """Test parsing 'signal:OPEN_LONG' breakpoint."""
        # Act
        bp = parse_breakpoint("signal:OPEN_LONG")

        # Assert
        assert isinstance(bp, SignalBreakpoint)
        assert bp.intention == "OPEN_LONG"

    def test_parse_case_insensitive(self):
        """Test parsing is case-insensitive."""
        # Act
        bp = parse_breakpoint("SIGNAL:buy")

        # Assert
        assert isinstance(bp, SignalBreakpoint)
        assert bp.intention == "BUY"

    def test_parse_with_whitespace(self):
        """Test parsing handles whitespace."""
        # Act
        bp = parse_breakpoint("  signal  ")

        # Assert
        assert isinstance(bp, SignalBreakpoint)

    def test_parse_unknown_raises_error(self):
        """Test parsing unknown type raises ValueError."""
        # Act & Assert
        with pytest.raises(ValueError) as exc_info:
            parse_breakpoint("unknown")

        assert "Unknown breakpoint type" in str(exc_info.value)

    def test_parse_signal_empty_intention_raises_error(self):
        """Test parsing 'signal:' with empty intention raises ValueError."""
        # Act & Assert
        with pytest.raises(ValueError) as exc_info:
            parse_breakpoint("signal:")

        assert "requires an intention type" in str(exc_info.value)


class TestParseBreakpoints:
    """Test parse_breakpoints function."""

    def test_parse_empty_list(self):
        """Test parsing empty list returns empty list."""
        # Act
        result = parse_breakpoints([])

        # Assert
        assert result == []

    def test_parse_single(self):
        """Test parsing single breakpoint."""
        # Act
        result = parse_breakpoints(["signal"])

        # Assert
        assert len(result) == 1
        assert isinstance(result[0], SignalBreakpoint)

    def test_parse_multiple(self):
        """Test parsing multiple breakpoints."""
        # Act
        result = parse_breakpoints(["signal:BUY", "signal:SELL"])

        # Assert
        assert len(result) == 2
        assert all(isinstance(bp, SignalBreakpoint) for bp in result)
        assert result[0].intention == "BUY"
        assert result[1].intention == "SELL"


class TestInteractiveDebuggerBreakpoints:
    """Test InteractiveDebugger with breakpoint rules."""

    def test_init_with_break_on(self):
        """Test debugger initializes with break_on rules."""
        from qtrader.cli.ui.interactive import InteractiveDebugger

        # Act
        debugger = InteractiveDebugger(
            break_on=["signal", "signal:BUY"],
            enabled=False,
        )

        # Assert
        assert len(debugger.breakpoints) == 2
        assert debugger._event_triggered_mode is True

    def test_init_without_break_on_is_step_through(self):
        """Test debugger without break_on is in step-through mode."""
        from qtrader.cli.ui.interactive import InteractiveDebugger

        # Act
        debugger = InteractiveDebugger(enabled=False)

        # Assert
        assert len(debugger.breakpoints) == 0
        assert debugger._event_triggered_mode is False

    def test_should_pause_before_events_in_event_mode(self):
        """Test should_pause_before_events returns False in event-triggered mode."""
        from qtrader.cli.ui.interactive import InteractiveDebugger

        # Arrange
        debugger = InteractiveDebugger(break_on=["signal"], enabled=True)
        timestamp = datetime(2020, 6, 15, 12, 0, 0)

        # Act
        result = debugger.should_pause_before_events(timestamp)

        # Assert
        assert result is False

    def test_should_pause_after_events_with_signal(self):
        """Test should_pause_after_events triggers on signal."""
        from qtrader.cli.ui.interactive import InteractiveDebugger

        # Arrange
        debugger = InteractiveDebugger(break_on=["signal"], enabled=True)
        mock_signal = Mock()
        mock_signal.intention = "OPEN_LONG"
        context = BreakpointContext(
            timestamp=datetime(2020, 6, 15, 12, 0, 0),
            signals=[mock_signal],
        )

        # Act
        result = debugger.should_pause_after_events(context)

        # Assert
        assert result is True

    def test_should_pause_after_events_without_signal(self):
        """Test should_pause_after_events returns False when no signals."""
        from qtrader.cli.ui.interactive import InteractiveDebugger

        # Arrange
        debugger = InteractiveDebugger(break_on=["signal"], enabled=True)
        context = BreakpointContext(
            timestamp=datetime(2020, 6, 15, 12, 0, 0),
            signals=[],
        )

        # Act
        result = debugger.should_pause_after_events(context)

        # Assert
        assert result is False

    def test_set_event_bus(self):
        """Test set_event_bus stores event bus reference."""
        from qtrader.cli.ui.interactive import InteractiveDebugger

        # Arrange
        debugger = InteractiveDebugger(break_on=["signal"], enabled=True)
        mock_bus = Mock()

        # Act
        debugger.set_event_bus(mock_bus)

        # Assert
        assert debugger._event_bus is mock_bus
        mock_bus.subscribe.assert_called_once()

    def test_on_signal_event_collects_signals(self):
        """Test _on_signal_event collects signals."""
        from qtrader.cli.ui.interactive import InteractiveDebugger

        # Arrange
        debugger = InteractiveDebugger(enabled=False)
        mock_signal = Mock()

        # Act
        debugger._on_signal_event(mock_signal)
        debugger._on_signal_event(mock_signal)

        # Assert
        assert len(debugger._pending_signals) == 2

    def test_combined_break_at_and_break_on(self):
        """Test break_at and break_on work together."""
        from qtrader.cli.ui.interactive import InteractiveDebugger

        # Arrange
        debugger = InteractiveDebugger(
            break_at=date(2020, 6, 15),
            break_on=["signal"],
            enabled=True,
        )
        mock_signal = Mock()
        mock_signal.intention = "OPEN_LONG"

        # Context before break_at date - should not pause
        context_before = BreakpointContext(
            timestamp=datetime(2020, 6, 10, 12, 0, 0),
            signals=[mock_signal],
        )
        assert debugger.should_pause_after_events(context_before) is False

        # Context on break_at date with signal - should pause
        context_on = BreakpointContext(
            timestamp=datetime(2020, 6, 15, 12, 0, 0),
            signals=[mock_signal],
        )
        assert debugger.should_pause_after_events(context_on) is True

        # Context after break_at date without signal - should not pause
        context_no_signal = BreakpointContext(
            timestamp=datetime(2020, 6, 20, 12, 0, 0),
            signals=[],
        )
        assert debugger.should_pause_after_events(context_no_signal) is False
